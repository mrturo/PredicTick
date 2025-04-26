"""Main orchestrator for ingesting and validating historical market data."""

import time
import warnings
from typing import Any, List, Optional

import pandas as pd  # type: ignore

from src.market_data.ingestion.raw.raw_data import RawData
from src.market_data.ingestion.summarizers.summarizer import Summarizer
from src.market_data.ingestion.symbol.symbol_processor import SymbolProcessor
from src.market_data.utils.intervals.interval import (Interval,
                                                      IntervalConverter)
from src.market_data.utils.storage.market_data_sync_manager import \
    MarketDataSyncManager
from src.utils.config.parameters import ParameterLoader
from src.utils.io.logger import Logger

warnings.simplefilter("ignore", ResourceWarning)


class Ingester:
    """Coordinates symbol ingest and post-ingest validation."""

    _PARAMS = ParameterLoader()
    _RAW_MARKETDATA_FILEPATH = _PARAMS.get("raw_marketdata_filepath")
    _RAW_DATA_INTERVAL: Optional[str] = Interval.market_raw_data()
    _SYMBOLS = _PARAMS.get("all_symbols")
    _INGESTER_RETRIES = _PARAMS.get("ingester_retries")

    @staticmethod
    def _check_and_prompt_interval_mismatch() -> bool:
        """Validates consistency between file and configured intervals.

        Prompts user on mismatch.
        """
        file_interval: Optional[str] = None
        interval1 = RawData.get_interval()
        if interval1 is not None and len(interval1.strip()) > 0:
            file_interval = interval1.strip()
        prop_interval: Optional[str] = None
        interval2 = Ingester._RAW_DATA_INTERVAL
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
            RawData.set_last_updated(pd.Timestamp.now(tz="UTC"))
            RawData.save()
        return True

    @staticmethod
    def _perform_single_update(symbols: List[str]) -> dict:
        """Executes a single update attempt and logs summary."""
        symbol_repo = Ingester._PARAMS.symbol_repo
        invalid_symbols = set(symbol_repo.get_invalid_symbols())
        result = SymbolProcessor.process_symbols(symbols, invalid_symbols)
        symbol_repo.set_invalid_symbols(result["invalid_symbols"])
        return result

    @staticmethod
    def ingest_raw_data(
        in_max_retries: int = 0, filepath_raw_data: Optional[str] = None
    ) -> tuple[bool, Optional[str], Any]:
        """Orchestrates the entire update process with retry logic."""
        start_time = time.time()
        if filepath_raw_data is None or len(filepath_raw_data.strip()) == 0:
            filepath_raw_data = Ingester._RAW_MARKETDATA_FILEPATH
        found_raw_data = MarketDataSyncManager.synchronize_marketdata_with_drive(
            filepath_raw_data
        )
        max_retries = (
            Ingester._INGESTER_RETRIES
            if in_max_retries <= 0 or in_max_retries is None
            else in_max_retries
        )
        max_retries = max(1, max_retries)
        RawData.load()
        updated: bool = False
        if not Ingester._check_and_prompt_interval_mismatch():
            return updated, filepath_raw_data, found_raw_data
        for current_try in range(1, max_retries + 1):
            local_start_time = time.time()
            Logger.separator()
            if max_retries > 1:
                Logger.info(f"Update try No. {current_try}/{max_retries}:")
            processed = Ingester._perform_single_update(Ingester._SYMBOLS)
            Logger.separator()
            Summarizer.print_sumary(Ingester._SYMBOLS, processed, local_start_time)
            Logger.separator()
            if current_try >= max_retries or processed.get("updated", 0) == 0:
                total_time = time.time() - start_time
                Logger.success(
                    f"Update process is completed. Tries: {current_try}. "
                    f"Time: {int(total_time // 60)}m {int(total_time % 60)}s"
                )
                remaining_str: Optional[str] = processed.get("remaining_str")
                if remaining_str is not None and len(remaining_str.strip()) > 0:
                    Logger.info(f"Next suggested update in {remaining_str}")
                if RawData.get_latest_price_date() is not None:
                    Logger.debug(
                        f"Latest price date: {RawData.get_latest_price_date()}"
                    )
                Logger.debug(f"Stale symbols: {RawData.get_stale_symbols()}")
                break
            if processed.get("updated", 0) > 0:
                updated = True
        return updated, filepath_raw_data, found_raw_data
