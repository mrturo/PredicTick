"""
Provides utilities for summarizing symbol update operations and exporting.

metadata and statistics about symbols and historical price data to CSV files.
"""

import time
from typing import List

import pandas as pd

from market_data.gateway import Gateway
from utils.google_drive_manager import GoogleDriveManager
from utils.logger import Logger
from utils.parameters import ParameterLoader


class Summarizer:
    """Summarizer logs symbol update stats and exports metadata and historical data to CSV."""

    _PARAMS = ParameterLoader()
    _WEEKDAYS: List[str] = _PARAMS.get("weekdays")
    _MARKETDATA_SUMMARY_FILEPATH = _PARAMS.get("marketdata_summary_filepath")
    _MARKETDATA_DETAILED_FILEPATH = _PARAMS.get("marketdata_detailed_filepath")
    _GOOGLE_DRIVE = GoogleDriveManager()

    @staticmethod
    def print_sumary(symbols, procesed_symbols, start_time) -> None:
        """
        Print a summary report of the symbol update process.

        Args:
            symbols (List[str]): List of all symbols processed.
            procesed_symbols (dict): Dictionary with update counts including keys:
                                    'updated', 'skipped', 'no_new', 'failed'.
            start_time (float): Timestamp indicating when the process started.
        """
        updated = procesed_symbols["updated"]
        skipped = procesed_symbols["skipped"]
        no_new = procesed_symbols["no_new"]
        failed = procesed_symbols["failed"]

        Logger.separator()
        Logger.debug("Summary")
        Logger.debug(f"  * Symbols processed: {len(symbols)}")
        Logger.debug(f"  * Symbols updated: {updated}")
        Logger.debug(f"  * Symbols skipped: {skipped}")
        Logger.debug(f"  * Symbols with no new data: {no_new}")
        if failed > 0:
            Logger.error(f"  * Symbols failed: {failed}")
        total_time = time.time() - start_time
        Logger.debug(f"  Total time: {int(total_time // 60)}m {int(total_time % 60)}s")

    @staticmethod
    def export_symbol_summary_to_csv() -> None:
        """Exports symbol metadata into a CSV file."""
        Gateway.load()
        index = Gateway.get_symbols()
        rows = []

        for symbol, data in index.items():
            row = {
                "symbol": symbol,
                "name": data.get("name"),
                "type": data.get("type"),
                "sector": data.get("sector"),
                "industry": data.get("industry"),
                "currency": data.get("currency"),
                "exchange": data.get("exchange"),
            }

            historical = data.get("historical_prices", [])
            datetimes = pd.to_datetime(
                [p.get("datetime") for p in historical if p.get("datetime")],
                utc=True,
                errors="coerce",
            ).dropna()

            row["num_prices"] = len(datetimes)
            row["min_datetime"] = datetimes.min() if not datetimes.empty else None
            row["max_datetime"] = datetimes.max() if not datetimes.empty else None

            schedule = data.get("schedule", {})
            for day in Summarizer._WEEKDAYS:
                day_lower = day.lower()
                day_data = schedule.get(day_lower, {})
                day_short = day_lower[:3]
                row[f"all_day_{day_short}"] = day_data.get("all_day")
                row[f"open_{day_short}"] = day_data.get("min_open")
                row[f"close_{day_short}"] = day_data.get("max_close")

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(Summarizer._MARKETDATA_SUMMARY_FILEPATH, index=False)
        Summarizer._GOOGLE_DRIVE.upload_file(Summarizer._MARKETDATA_SUMMARY_FILEPATH)

    @staticmethod
    def export_symbol_detailed_to_csv() -> None:
        """Exports symbol historical price into a CSV file."""
        Gateway.load()
        index = Gateway.get_symbols()
        rows = []

        for symbol, data in index.items():
            historical = data.get("historical_prices", [])
            for entry in historical:
                row = {"symbol": symbol}
                row.update(entry)
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(Summarizer._MARKETDATA_DETAILED_FILEPATH, index=False)
        Summarizer._GOOGLE_DRIVE.upload_file(Summarizer._MARKETDATA_DETAILED_FILEPATH)
