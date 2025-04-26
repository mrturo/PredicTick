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
from utils.json_manager import JsonManager
from utils.logger import Logger
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
    _INTERVAL = _PARAMS.get("interval")

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
        ts: Optional[pd.Timestamp] = None
        symbols: Optional[list[Any]] = None
        if data is not None:
            if not Updater.is_valid_marketdata(data):
                Logger.warning("Local marketdata is invalid. Ignoring file.")
                data = None
                ts = None
            else:
                ts = (
                    pd.to_datetime(data.get("last_updated"), utc=True)
                    if data and "last_updated" in data
                    else pd.Timestamp.min.tz_localize("UTC")
                )
                symbols = data.get("symbols", []) if isinstance(data, dict) else []
                if len(symbols) == 0:
                    ts = None
                    data = None
        return data, ts

    @staticmethod
    def _synchronize_marketdata_with_drive(filepath: Optional[str] = None) -> None:
        """Synchronize local and Drive JSONs by preserving the most recent version locally."""

        filepath = (
            Updater._RAW_MARKETDATA_FILEPATH
            if filepath is None or len(filepath.strip()) == 0
            else filepath.strip()
        )
        if filepath is None or len(filepath.strip()) == 0:
            raise ValueError("Market data filepath is empty")

        local_ts: Optional[pd.Timestamp] = None
        local_data: Any = None
        try:
            local_data = JsonManager.load(filepath)
            local_data, local_ts = Updater._load_valid_marketdata(local_data)
        except Exception:  # pylint: disable=broad-exception-caught
            local_ts = None
            local_data: Any = None

        drive_ts: Optional[pd.Timestamp] = None
        drive_data: Any = None
        try:
            temp_drive_path = filepath + ".tmp"
            if Updater._GOOGLE_DRIVE.download_file(filepath, temp_drive_path):
                drive_data = JsonManager.load(temp_drive_path)
                drive_data, drive_ts = Updater._load_valid_marketdata(drive_data)
                Logger.debug(
                    "Marketdata JSON successfully retrieved from Google Drive file."
                )
                if JsonManager.delete(temp_drive_path):
                    Logger.debug(
                        f"Temporary file '{temp_drive_path}' deleted successfully."
                    )

        except Exception:  # pylint: disable=broad-exception-caught
            drive_ts = None
            drive_data: Any = None

        if (local_data is not None or local_ts is not None) and (
            drive_data is not None and drive_ts is not None
        ):
            if drive_ts > local_ts:
                Logger.info("Using Google Drive JSON (newer).")
                JsonManager.save(drive_data, filepath)
                local_ts = None
                local_data = None
            elif drive_ts < local_ts:
                Logger.info("Using local JSON (newer).")
                drive_ts = None
                drive_data = None
            else:
                Logger.info("Timestamps are equal. Retaining local JSON by default.")
                drive_ts = None
                drive_data = None
        elif drive_data is not None and drive_ts is not None:
            Logger.info("Using Google Drive JSON (only available).")
            JsonManager.save(drive_data, filepath)
        elif local_data is not None or local_ts is not None:
            Logger.info("Using local JSON (only available).")
        else:
            Logger.warning(
                "No valid JSON found in either local or Drive sources. A new marketdata file will "
                "be created from scratch, and all symbol data will be downloaded."
            )
            JsonManager.save(
                {"last_updated": pd.Timestamp.now(tz="UTC").isoformat(), "symbols": []},
                filepath,
            )
        Logger.separator()

    @staticmethod
    def _enrich_data() -> None:
        """Enrich and persist market symbol data with logging and error handling."""
        Logger.separator()
        Logger.debug("Starting symbol data enrichment process...")
        try:
            EnrichedData.generate()
            Logger.debug("Symbol data was enriched successfully")
        except Exception as e:  # pylint: disable=broad-exception-caught
            Logger.error(f"Symbol data enrichment failed: {e}")

    @staticmethod
    def update_data(in_max_retries: int = 0, filepath: Optional[str] = None) -> None:
        """Orchestrates the entire update process with retry logic."""
        start_time = time.time()
        if filepath is None or len(filepath.strip()) == 0:
            filepath = Updater._RAW_MARKETDATA_FILEPATH

        Updater._synchronize_marketdata_with_drive(filepath)

        max_retries = (
            Updater._UPDATER_RETRIES
            if in_max_retries <= 0 or in_max_retries is None
            else in_max_retries
        )
        max_retries = max(1, max_retries)

        RawData.load()
        if (
            RawData.get_interval() is not None
            and RawData.get_interval() != Updater._INTERVAL
        ):
            Logger.warning(
                f"The current market data has interval '{RawData.get_interval()}', "
                f"but the system is configured to use '{Updater._INTERVAL}'.\n"
                "Proceeding will delete the existing data and download new data from scratch.\n"
                "Do you wish to continue? [y/N]"
            )
            response = input().strip().lower()
            if response != "y":
                Logger.info("Process aborted by user. Existing data is retained.")
                return
            RawData.set_interval(None)
            RawData.set_latest_price_date(None)
            RawData.set_symbols({})
            RawData.save()

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

        if Validator.validate_market_data(True):
            Updater._GOOGLE_DRIVE.upload_file(filepath)
            Updater._enrich_data()


if __name__ == "__main__":
    Updater.update_data()
