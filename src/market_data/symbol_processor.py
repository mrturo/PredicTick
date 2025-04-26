"""Handles symbol-level market data update logic."""

import time
from typing import Any, List, Optional

import pandas as pd  # type: ignore

from market_data.downloader import Downloader
from market_data.raw_data import RawData
from utils.interval import Interval
from utils.logger import Logger
from utils.parameters import ParameterLoader


# pylint: disable=too-few-public-methods
class SymbolProcessor:
    """Class for orchestrating raw symbol-level market data updates.

    This includes validation, and incremental downloading via RawData.
    """

    _PARAMS = ParameterLoader()
    _BLOCK_DAYS = _PARAMS.get("block_days")
    _DOWNLOAD_RETRIES = _PARAMS.get("download_retries")
    _RETRY_SLEEP_SECONDS = _PARAMS.get("retry_sleep_seconds")
    _RAW_DATA_INTERVAL = Interval.market_raw_data()

    _DOWNLOADER = Downloader(_BLOCK_DAYS, _DOWNLOAD_RETRIES, _RETRY_SLEEP_SECONDS)

    @staticmethod
    def _is_valid_symbol(symbol: str) -> bool:
        """Check if the given symbol has valid metadata indicating it is active and tradable."""
        try:
            info = SymbolProcessor._DOWNLOADER.get_metadata(symbol)
            return info is not None and info.get("name") is not None
        except (KeyError, ValueError):
            return False

    @staticmethod
    def _should_skip_symbol(
        symbol: str, entry: Optional[dict], invalid_symbols: set
    ) -> bool:
        """Check if a symbol should be skipped due to missing or invalid data."""
        if not (entry and entry.get("historical_prices")):
            if symbol in invalid_symbols:
                Logger.warning(f"    Skipping {symbol}: previously marked as invalid.")
                return True
            if not SymbolProcessor._is_valid_symbol(symbol):
                Logger.warning(f"    Skipping {symbol}: appears invalid or delisted.")
                invalid_symbols.add(symbol)
                return True
        return False

    @staticmethod
    def _process_single_symbol(symbol: str, entry: Optional[dict]) -> tuple[str, str]:
        """Process historical price update for a single symbol."""
        existing_df = (
            pd.DataFrame(entry["historical_prices"]) if entry else pd.DataFrame()
        )
        existing_metadata: dict[str, Any] = (
            {
                k: entry.get(k)
                for k in ["name", "type", "sector", "industry", "currency", "exchange"]
            }
            if entry
            else {}
        )

        if not existing_df.empty and "datetime" in existing_df.columns:
            last_dt_ts = pd.to_datetime(
                existing_df["datetime"], utc=True, errors="coerce"
            )
            if isinstance(last_dt_ts, pd.Series):
                last_dt = (
                    last_dt_ts.dropna().max()
                    if isinstance(last_dt_ts, pd.Series)
                    else (last_dt_ts if pd.notnull(last_dt_ts) else None)  # type: ignore
                )
        else:
            last_dt = None

        if not (isinstance(last_dt, pd.Timestamp) or last_dt is None):
            last_dt = None

        incremental = SymbolProcessor._DOWNLOADER.get_historical_prices(symbol, last_dt)

        if not incremental.empty and not existing_df.empty:
            incremental = incremental[
                ~incremental["datetime"].isin(existing_df["datetime"])
            ].reset_index(drop=True)

        if incremental.empty:
            updated_entry: dict[str, Any] = {
                "historical_prices": existing_df.to_dict(orient="records")
            }
            RawData.set_symbol(
                symbol, updated_entry["historical_prices"], existing_metadata
            )
            return "no_new", existing_metadata.get("name", "")

        combined_df = (
            pd.concat([incremental, existing_df], axis=0)
            .drop_duplicates("datetime")
            .sort_values("datetime")
        )

        combined_df["datetime"] = pd.to_datetime(
            combined_df["datetime"], utc=True, errors="coerce"
        )

        metadata: Optional[dict[str, Any]] = existing_metadata
        if not existing_metadata.get("name"):
            metadata = SymbolProcessor._DOWNLOADER.get_metadata(symbol)
            Logger.debug(f"    Metadata fetched for {symbol}")
        metadata = metadata if metadata is not None else {}

        updated_entry: dict[str, Any] = {
            "historical_prices": combined_df.to_dict(orient="records")
        }

        RawData.set_symbol(symbol, updated_entry["historical_prices"], metadata)
        Logger.debug(f"    Updated data for {symbol} with {len(incremental)} new rows.")
        return "updated", metadata.get("name", "")

    @staticmethod
    def process_symbols(symbols: List[str], invalid_symbols: set) -> dict:
        """Batch-process a list of symbols, updating market data and feature sets."""
        counts = {
            "updated": 0,
            "skipped": 0,
            "no_new": 0,
            "failed": 0,
            "invalid_symbols": [],
        }
        RawData.load()
        invalid_symbols_set = set(invalid_symbols)

        for idx, symbol in enumerate(symbols, 1):
            Logger.info(f" * Processing symbol ({idx}/{len(symbols)}): {symbol}")
            entry = RawData.get_symbol(symbol)
            symbol_start_time = time.time()

            if SymbolProcessor._should_skip_symbol(symbol, entry, invalid_symbols_set):
                counts["skipped"] += 1
                continue

            try:
                result, name = SymbolProcessor._process_single_symbol(symbol, entry)
                counts[result] += 1
                display_name = f"{symbol} ({name.strip()})" if name else symbol
                Logger.success(
                    f"    Completed {display_name} in {(time.time() - symbol_start_time):.2f}s"
                )
            except Exception as err:  # pylint: disable=broad-exception-caught
                Logger.error(f"  Failed to update {symbol}: {err}")
                counts["failed"] += 1

        RawData.set_interval(SymbolProcessor._RAW_DATA_INTERVAL)
        if counts["updated"] > 0:
            RawData.set_new_id()
            RawData.set_last_update(pd.Timestamp.now(tz="UTC"))
        RawData.save()
        counts["invalid_symbols"] = invalid_symbols_set
        return counts
