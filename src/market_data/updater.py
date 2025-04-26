"""Main orchestrator for updating and validating historical market data."""

import time
import warnings
from typing import Any, List, Optional

import pandas as pd  # type: ignore

from market_data.enriched_data import EnrichedData
from market_data.raw_data import RawData
from market_data.summarizer import Summarizer
from market_data.symbol_processor import SymbolProcessor
from market_data.validator import Validator
from utils.google_drive_manager import GoogleDriveManager
from utils.interval import Interval, IntervalConverter
from utils.json_manager import JsonManager
from utils.logger import Logger
from utils.marketdata_structs import MarketDataVersion
from utils.parameters import ParameterLoader

warnings.simplefilter("ignore", ResourceWarning)


# pylint: disable=too-few-public-methods
class Updater:
    """Coordinates symbol updates, retry logic, and post-update validation."""

    _GOOGLE_DRIVE = GoogleDriveManager()
    _PARAMS = ParameterLoader()
    _SYMBOLS = _PARAMS.get("all_symbols")
    _UPDATER_RETRIES = _PARAMS.get("updater_retries")
    _RAW_MARKETDATA_FILEPATH = _PARAMS.get("raw_marketdata_filepath")
    _RAW_DATA_INTERVAL: str = Interval.market_raw_data()

    @staticmethod
    def is_valid_marketdata(data: dict) -> bool:
        """
        Validate the structure of a marketdata dictionary.

        This method checks whether the input data conforms to the expected schema
        used for storing market data, including presence of a symbol list with
        required keys in each entry.

        Args:
            data (dict): Dictionary parsed from the marketdata JSON file.

        Returns:
            bool: True if the data structure is valid and contains non-empty
                  symbol entries with 'symbol' and 'historical_prices' keys;
                  False otherwise.
        """
        return (
            isinstance(data, dict)
            and "symbols" in data
            and isinstance(data["symbols"], list)
            and all("symbol" in s and "historical_prices" in s for s in data["symbols"])
        )

    @staticmethod
    def _perform_single_update(symbols: List[str], local_start_time: float) -> dict:
        """Executes a single update attempt and logs summary."""
        symbol_repo = Updater._PARAMS.symbol_repo
        invalid_symbols = set(symbol_repo.get_invalid_symbols())

        result = SymbolProcessor.process_symbols(symbols, invalid_symbols)
        symbol_repo.set_invalid_symbols(result["invalid_symbols"])

        Summarizer.print_sumary(symbols, result, local_start_time)
        Logger.separator()
        return result

    @staticmethod
    def _load_valid_marketdata(data: Any) -> Any:
        """Load marketdata JSON and return data and timestamp only if symbols are non-empty."""
        file_id: Optional[str] = None
        symbols: Optional[list[Any]] = None
        ts: Optional[pd.Timestamp] = None
        if data is not None:
            if not Updater.is_valid_marketdata(data):
                Logger.warning("Local marketdata is invalid. Ignoring file.")
                data = None
                file_id = None
                ts = None
            else:
                file_id = data.get("id", None)
                symbols = data.get("symbols", []) if isinstance(data, dict) else []
                ts = (
                    pd.to_datetime(data.get("last_updated"), utc=True)
                    if data and "last_updated" in data
                    else pd.Timestamp.min.tz_localize("UTC")
                )
                if len(symbols) == 0:
                    data = None
                    file_id = None
                    ts = None
        return data, ts, file_id

    @staticmethod
    def _synchronize_marketdata_with_drive(filepath: Optional[str] = None) -> Any:
        """Synchronize local and Drive JSONs by preserving the most recent version locally."""

        filepath = (
            Updater._RAW_MARKETDATA_FILEPATH
            if not filepath or not filepath.strip()
            else filepath.strip()
        )
        if not filepath:
            raise ValueError("Market data filepath is empty")

        local_data = MarketDataVersion(*Updater._load_local_marketdata(filepath))
        drive_data = MarketDataVersion(*Updater._load_drive_marketdata(filepath))

        Updater._resolve_data_conflict(local_data, drive_data, filepath)

        local_data = MarketDataVersion(*Updater._load_local_marketdata(filepath))
        if (
            local_data is not None
            and local_data.data is None
            and local_data.timestamp is None
            and local_data.file_id is None
        ):
            local_data = None
        if (
            drive_data is not None
            and drive_data.data is None
            and drive_data.timestamp is None
            and drive_data.file_id is None
        ):
            drive_data = None

        Logger.separator()
        return {
            "local_data": local_data is not None,
            "drive_data": drive_data is not None,
        }

    @staticmethod
    def _load_local_marketdata(
        filepath: str,
    ) -> tuple[Any, Optional[pd.Timestamp], Optional[str]]:
        try:
            data = JsonManager.load(filepath)
            return Updater._load_valid_marketdata(data)
        except Exception:  # pylint: disable=broad-exception-caught
            return None, None, None

    @staticmethod
    def _load_drive_marketdata(
        filepath: str,
    ) -> tuple[Any, Optional[pd.Timestamp], Optional[str]]:
        try:
            temp_path = f"{filepath}.tmp"
            if Updater._GOOGLE_DRIVE.download_file(filepath, temp_path):
                data = JsonManager.load(temp_path)
                result = Updater._load_valid_marketdata(data)
                Logger.debug(
                    "Marketdata JSON successfully retrieved from Google Drive file."
                )
                if JsonManager.delete(temp_path):
                    Logger.debug(f"Temporary file '{temp_path}' deleted successfully.")
                return result
        except Exception as e:  # pylint: disable=broad-exception-caught
            Logger.warning(f"Failed to load marketdata from Drive: {e}")
        return None, None, None

    @staticmethod
    def _resolve_data_conflict(
        local_version: MarketDataVersion,
        drive_version: MarketDataVersion,
        filepath: str,
    ) -> None:
        local_data, local_ts, local_id = (
            local_version.data,
            local_version.timestamp,
            local_version.file_id,
        )
        drive_data, drive_ts, drive_id = (
            drive_version.data,
            drive_version.timestamp,
            drive_version.file_id,
        )

        if (
            local_data is not None
            and local_ts is not None
            and drive_data is not None
            and drive_ts is not None
        ):
            if local_id and drive_id and local_id == drive_id:
                Logger.info("Using local JSON (It's same to Google Drive JSON).")
            elif drive_ts > local_ts:
                Logger.info("Using Google Drive JSON (newer).")
                JsonManager.save(drive_data, filepath)
            elif drive_ts < local_ts:
                Logger.info("Using local JSON (newer).")
            else:
                Logger.info("Timestamps are equal. Retaining local JSON by default.")
        elif drive_data is not None and drive_ts is not None:
            Logger.info("Using Google Drive JSON (only available).")
            JsonManager.save(drive_data, filepath)
        elif local_data is not None and local_ts is not None:
            Logger.info("Using local JSON (only available).")
        else:
            Logger.warning(
                "No valid JSON found in either local or Drive sources. A new marketdata file will "
                "be created from scratch, and all symbol data will be downloaded."
            )
            now = pd.Timestamp.now(tz="UTC").isoformat()
            JsonManager.save(
                {"id": None, "last_updated": now, "last_check": now, "symbols": []},
                filepath,
            )

    @staticmethod
    def _enrich_data() -> None:
        """Enrich and persist market symbol data with logging and error handling."""
        Logger.debug("Starting symbol data enrichment process...")
        try:
            EnrichedData.generate()
            Logger.debug("Symbol data was enriched successfully")
        except Exception as e:  # pylint: disable=broad-exception-caught
            Logger.error(f"Symbol data enrichment failed: {e}")

    @staticmethod
    def _check_and_prompt_interval_mismatch() -> bool:
        """
        Validates consistency between file and configured intervals.

        Prompts user on mismatch.

        Returns:
            bool: True if the process should continue; False if aborted by the user.
        """
        file_interval: Optional[str] = None
        interval1 = RawData.get_interval()
        if interval1 is not None and len(interval1.strip()) > 0:
            file_interval = interval1.strip()

        prop_interval: Optional[str] = None
        interval2 = Updater._RAW_DATA_INTERVAL
        if interval2 is not None and len(interval2.strip()) > 0:
            prop_interval = interval2.strip()

        try:
            Interval.validate_market_interval_hierarchy()
        except Exception as ex:  # pylint: disable=broad-exception-caught
            Logger.error(str(ex))
            return False

        file_interval_minutes: int = IntervalConverter.to_minutes(file_interval)
        prop_interval_minutes: int = IntervalConverter.to_minutes(prop_interval)
        if file_interval is not None and file_interval_minutes != prop_interval_minutes:
            Logger.warning(
                f"The current market raw data has interval '{file_interval}', "
                f"but the system is configured to use '{prop_interval}'.\n"
                "Proceeding will delete the existing data and download new data from scratch.\n"
                "Do you wish to continue? [y/N]"
            )
            response = input().strip().lower()
            if response != "y":
                Logger.info("Process aborted by user. Existing data is retained.")
                return False
            RawData.set_interval(None)
            RawData.set_latest_price_date(None)
            RawData.set_symbols({})
            RawData.set_new_id()
            RawData.set_last_update(pd.Timestamp.now(tz="UTC"))
            RawData.save()
        return True

    @staticmethod
    def update_data(in_max_retries: int = 0, filepath: Optional[str] = None) -> None:
        """Orchestrates the entire update process with retry logic."""
        start_time = time.time()
        if filepath is None or len(filepath.strip()) == 0:
            filepath = Updater._RAW_MARKETDATA_FILEPATH

        found_data = Updater._synchronize_marketdata_with_drive(filepath)

        max_retries = (
            Updater._UPDATER_RETRIES
            if in_max_retries <= 0 or in_max_retries is None
            else in_max_retries
        )
        max_retries = max(1, max_retries)

        RawData.load()

        if not Updater._check_and_prompt_interval_mismatch():
            return

        updated: bool = False
        for current_try in range(1, max_retries + 1):
            local_start_time = time.time()
            if max_retries > 1:
                Logger.info(f"Update try No. {current_try}/{max_retries}:")

            processed = Updater._perform_single_update(
                Updater._SYMBOLS, local_start_time
            )

            if current_try >= max_retries or processed.get("updated", 0) == 0:
                total_time = time.time() - start_time
                Logger.success(
                    f"Update process is completed. Tries: {current_try}. "
                    f"Time: {int(total_time // 60)}m {int(total_time % 60)}s"
                )
                Logger.debug(f"Latest price date: {RawData.get_latest_price_date()}")
                Logger.debug(f"Stale symbols: {RawData.get_stale_symbols()}")
                Logger.separator()
                break
            if processed.get("updated", 0) > 0:
                updated = True

        if Validator.validate_market_data(True):
            if updated is True or found_data["drive_data"] is False:
                Updater._GOOGLE_DRIVE.upload_file(filepath)
            else:
                Logger.warning(
                    f"Upload to Google Drive skipped '{filepath}': "
                    f"No changes detected in market data."
                )
                Logger.separator()
            Updater._enrich_data()


if __name__ == "__main__":
    Updater.update_data()
