"""Data summarizar.

Provides utilities for summarizing symbol update operations and exporting metadata and
statistics about symbols and historical price data to CSV files.
"""

import time
from typing import List

from src.market_data.utils.storage.google_drive_manager import \
    GoogleDriveManager
from src.utils.config.parameters import ParameterLoader
from src.utils.io.logger import Logger


# pylint: disable=too-few-public-methods
class Summarizer:
    """Summarizer logs symbol update stats and exports metadata and historical data to CSV."""

    _PARAMS = ParameterLoader()
    _WEEKDAYS: List[str] = _PARAMS.get("weekdays")
    _GOOGLE_DRIVE = GoogleDriveManager()

    @staticmethod
    def print_sumary(symbols, procesed_symbols, start_time) -> None:
        """Print a summary report of the symbol update process."""
        updated = procesed_symbols["updated"]
        skipped = procesed_symbols["skipped"]
        no_new = procesed_symbols["no_new"]
        failed = procesed_symbols["failed"]
        Logger.debug("Summary")
        Logger.debug(f"  * Symbols processed: {len(symbols)}")
        Logger.debug(f"  * Symbols updated: {updated}")
        Logger.debug(f"  * Symbols skipped: {skipped}")
        Logger.debug(f"  * Symbols with no new data: {no_new}")
        if failed > 0:
            Logger.error(f"  * Symbols failed: {failed}")
        total_time = time.time() - start_time
        Logger.debug(f"  Total time: {int(total_time // 60)}m {int(total_time % 60)}s")
