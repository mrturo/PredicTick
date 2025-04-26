"""Downloader module for historical market data using yfinance."""

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from updater.output_suppressor import OutputSuppressor
from utils.logger import Logger
from utils.parameters import ParameterLoader
from utils.timezone import localize_to_market_time

# Global parameter instance
parameters = ParameterLoader()


class MarketDataDownloader:
    """Handles downloading of historical market data."""

    def __init__(self, retries: int, sleep_seconds: int):
        self.retries = retries
        self.sleep_seconds = sleep_seconds

    def download(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        """Attempts to download market data with retries."""
        attempt = 0
        while attempt < self.retries:
            try:
                with OutputSuppressor.suppress():
                    df = yf.download(
                        symbol,
                        start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"),
                        interval=interval,
                        auto_adjust=False,
                        progress=False,
                    )
                if not df.empty:
                    df = localize_to_market_time(df, parameters["market_tz"])
                    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
                    df = df[(df.index >= start) & (df.index <= end)]
                    return df
            except (ValueError, IOError) as error:
                Logger.error(
                    f"  Error downloading {symbol} on attempt {attempt + 1}: {error}"
                )
            attempt += 1
            time.sleep(self.sleep_seconds)
        return pd.DataFrame()

    def is_symbol_available(self, symbol: str) -> bool:
        """Check if a given symbol has valid downloadable data."""
        try:
            df = self.download(
                symbol,
                datetime.now(timezone.utc) - timedelta(days=7),
                datetime.now(timezone.utc),
                "1d",
            )
            return not df.empty
        except (ValueError, IOError, KeyError) as error:
            Logger.error(f"  Failed to update {symbol}: {error}")
            return False
