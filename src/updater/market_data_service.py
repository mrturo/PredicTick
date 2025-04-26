"""Service layer to manage incremental data downloads and update ranges."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from updater.file_manager import FileManager
from updater.market_data_downloader import MarketDataDownloader
from utils.parameters import ParameterLoader

# Global parameter instance
parameters = ParameterLoader()


class MarketDataService:
    """Service layer for managing market data update operations."""

    def __init__(
        self, interval: str, block_days: int, downloader: MarketDataDownloader
    ):
        self.interval = interval
        self.block_days = block_days
        self.downloader = downloader

    def get_incremental_data(
        self, symbol: str, last_datetime: Optional[datetime]
    ) -> pd.DataFrame:
        """Downloads new data incrementally from last known datetime."""
        if last_datetime:
            last_datetime = (
                last_datetime.astimezone(timezone.utc)
                if last_datetime.tzinfo
                else last_datetime.replace(tzinfo=timezone.utc)
            )

        today = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = today + timedelta(days=1)
        start = min(
            today,
            (
                last_datetime + timedelta(hours=1)
                if last_datetime
                else datetime.now(timezone.utc)
                - timedelta(days=parameters["historical_days_fallback"])
            ),
        )

        all_data_frames = []
        current_start = start

        while current_start < end:
            current_end = min(current_start + timedelta(days=self.block_days), end)
            df = self.downloader.download(
                symbol, current_start, current_end, self.interval
            )
            if not df.empty:
                all_data_frames.append(df)
            current_start = current_end

        if not all_data_frames:
            return pd.DataFrame()

        full_df = pd.concat(all_data_frames)
        full_df = full_df[~full_df.index.duplicated()]
        full_df.columns = [
            col[0] if isinstance(col, tuple) else col for col in full_df.columns
        ]
        full_df.reset_index(inplace=True)
        full_df.rename(columns={full_df.columns[0]: "Datetime"}, inplace=True)

        return full_df

    def symbol_last_update_range(self, symbol: str) -> Optional[str]:
        """Returns the date range of the last known data for a symbol."""
        entry = FileManager.find_symbol(symbol)
        if not entry or "historical_prices" not in entry:
            return None

        df = pd.DataFrame(entry["historical_prices"])
        if df.empty:
            return None

        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
        min_date = df["Datetime"].min().strftime("%Y-%m-%d")
        max_date = df["Datetime"].max().strftime("%Y-%m-%d")
        return f"{min_date} → {max_date}"
