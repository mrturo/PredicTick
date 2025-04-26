"""Handles symbol-level market data update logic."""

import time
from typing import List

import pandas as pd

from market_data.downloader import Downloader
from market_data.enrichment import Enrichment
from market_data.gateway import Gateway
from market_data.schedule_builder import ScheduleBuilder
from utils.logger import Logger
from utils.parameters import ParameterLoader


# pylint: disable=too-few-public-methods
class SymbolProcessor:
    """Class for orchestrating symbol-level market data updates and feature enrichment.

    This includes validation, incremental downloading, enrichment with technical and temporal
    features, and persistence via Gateway.
    """

    _PARAMS = ParameterLoader()
    _BLOCK_DAYS = _PARAMS.get("block_days")
    _DOWNLOAD_RETRIES = _PARAMS.get("download_retries")
    _RETRY_SLEEP_SECONDS = _PARAMS.get("retry_sleep_seconds")

    _DOWNLOADER = Downloader(_BLOCK_DAYS, _DOWNLOAD_RETRIES, _RETRY_SLEEP_SECONDS)

    @staticmethod
    def _is_valid_symbol(symbol: str) -> bool:
        """Check if the given symbol has valid metadata indicating it is active and tradable."""
        try:
            info = SymbolProcessor._DOWNLOADER.get_metadata(symbol)
            return info["name"] is not None
        except (KeyError, ValueError):
            return False

    @staticmethod
    def _should_skip_symbol(symbol: str, entry: dict, invalid_symbols: set) -> bool:
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
    def _process_single_symbol(symbol: str, entry: dict) -> tuple:
        """Process historical price update and feature enrichment for a single symbol."""
        existing_df = (
            pd.DataFrame(entry["historical_prices"]) if entry else pd.DataFrame()
        )
        existing_metadata = (
            {
                k: entry.get(k)
                for k in ["name", "type", "sector", "industry", "currency", "exchange"]
            }
            if entry
            else {}
        )

        last_dt = (
            pd.to_datetime(existing_df["datetime"], utc=True, errors="coerce").max()
            if not existing_df.empty and "datetime" in existing_df.columns
            else None
        )

        incremental = SymbolProcessor._DOWNLOADER.get_historical_prices(symbol, last_dt)

        if not incremental.empty and not existing_df.empty:
            incremental = incremental[
                ~incremental["datetime"].isin(existing_df["datetime"])
            ].reset_index(drop=True)

        if incremental.empty:
            updated_entry = {"historical_prices": existing_df.to_dict(orient="records")}
            updated_entry["schedule"] = ScheduleBuilder.build_schedule(updated_entry)
            existing_metadata["schedule"] = updated_entry["schedule"]
            Gateway.set_symbol(
                symbol, updated_entry["historical_prices"], existing_metadata
            )
            return "no_new", existing_metadata.get("name", "")

        combined_df = (
            pd.concat([incremental, existing_df], axis=0)
            .drop_duplicates("datetime")
            .sort_values("datetime")
        )
        combined_df = Enrichment.enrich_historical_prices(combined_df)

        combined_df["datetime"] = pd.to_datetime(
            combined_df["datetime"], utc=True, errors="coerce"
        )
        combined_df["date"] = combined_df["datetime"].dt.date
        combined_df["records_day_current"] = combined_df.groupby("date").cumcount() + 1
        combined_df["records_day_total"] = combined_df.groupby("date")[
            "datetime"
        ].transform("count")
        combined_df["weekdays_current"] = combined_df["datetime"].dt.isocalendar().day
        combined_df["weekdays_total"] = combined_df["datetime"].dt.dayofweek.nunique()

        active_weekdays = combined_df["datetime"].dt.dayofweek.unique()
        calendar_type = (
            7
            if set(active_weekdays) == set(range(7))
            else (6 if set(active_weekdays) == set(range(6)) else 5)
        )
        combined_df = Enrichment.add_month_and_year_days(combined_df, calendar_type)

        combined_df["pct_day"] = (
            combined_df["records_day_current"] / combined_df["records_day_total"]
        )
        combined_df["pct_week"] = (
            (combined_df["records_day_total"] * (combined_df["weekdays_current"] - 1))
            + combined_df["records_day_current"]
        ) / (combined_df["records_day_total"] * combined_df["weekdays_total"])
        combined_df["pct_month"] = (
            (combined_df["records_day_total"] * (combined_df["month_days_current"] - 1))
            + combined_df["records_day_current"]
        ) / (combined_df["records_day_total"] * combined_df["month_days_total"])
        combined_df["pct_year"] = (
            (combined_df["records_day_total"] * (combined_df["year_days_current"] - 1))
            + combined_df["records_day_current"]
        ) / (combined_df["records_day_total"] * combined_df["year_days_total"])

        combined_df["datetime"] = combined_df["datetime"].dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        combined_df = combined_df.drop(columns=["date"])

        metadata = (
            SymbolProcessor._DOWNLOADER.get_metadata(symbol)
            if not existing_metadata.get("name")
            else existing_metadata
        )
        updated_entry = {"historical_prices": combined_df.to_dict(orient="records")}
        updated_entry["schedule"] = ScheduleBuilder.build_schedule(updated_entry)
        metadata["schedule"] = updated_entry["schedule"]

        Gateway.set_symbol(symbol, updated_entry["historical_prices"], metadata)
        Logger.debug(f"    Updated data for {symbol} with {len(incremental)} new rows.")
        return "updated", metadata.get("name", "")

    @staticmethod
    def process_symbols(symbols: List[str], invalid_symbols: set) -> dict:
        """Batch-process a list of symbols, updating market data and feature sets."""
        counts = {"updated": 0, "skipped": 0, "no_new": 0, "failed": 0}
        Gateway.load()
        invalid_symbols_set = set(invalid_symbols)

        for idx, symbol in enumerate(symbols, 1):
            Logger.info(f" * Processing symbol ({idx}/{len(symbols)}): {symbol}")
            entry = Gateway.get_symbol(symbol)
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

        Gateway.save()
        counts["invalid_symbols"] = invalid_symbols_set
        return counts
