"""Downloader module for fetching market data and metadata.

Provides utilities for retrieving historical price data, metadata, and availability
checks for financial instruments, with retry logic, parametrizable windows, and
timezone normalization.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from market_data.provider import PriceDataConfig, Provider, TickerMetadata
from utils.logger import Logger
from utils.output_suppressor import OutputSuppressor
from utils.parameters import ParameterLoader
from utils.timezone_normalizer import TimezoneNormalizer


class Downloader:
    """Retrieves historical price data and metadata for financial instruments."""

    _PARAMS = ParameterLoader()
    _AVAILABILITY_DAYS_WINDOW = _PARAMS.get("availability_days_window")
    _DEFAULT_CURRENCY = _PARAMS.get("default_currency")
    _HISTORICAL_DAYS_FALLBACK = _PARAMS.get("historical_days_fallback")
    _INTERVAL = _PARAMS.get("interval")
    _MARKET_TZ = _PARAMS.get("market_tz")

    _PROVIDER = Provider()

    def __init__(
        self,
        block_days: Optional[int] = None,
        retries: Optional[int] = None,
        sleep_seconds: Optional[int] = None,
    ):
        """
        Initialize Downloader instance.

        Args:
            block_days (Optional[int]): Number of days to request per data block.
            retries (Optional[int]): Number of retry attempts for failed downloads.
            sleep_seconds (Optional[int]): Sleep time between retries in seconds.
        """
        self.block_days = block_days
        self.retries = retries
        self.sleep_seconds = sleep_seconds

    def _build_ticker_name(self, ticker: TickerMetadata) -> str:
        """
        Builds a readable ticker name using available metadata.

        Args:
            ticker (TickerMetadata): The ticker metadata object.

        Returns:
            str: The best available name for the ticker.
        """
        return (
            (
                ticker.display_name.strip()
                if ticker.display_name and len(ticker.display_name.strip()) > 0
                else None
            )
            or (
                ticker.short_name.strip()
                if ticker.short_name and len(ticker.short_name.strip()) > 0
                else None
            )
            or (
                ticker.long_name.strip()
                if ticker.long_name and len(ticker.long_name.strip()) > 0
                else None
            )
            or (
                ticker.symbol.strip().upper()
                if ticker.symbol and len(ticker.symbol.strip()) > 0
                else None
            )
        )

    def _build_ticker_type(self, ticker: TickerMetadata) -> str:
        """
        Builds a ticker type using available metadata.

        Args:
            ticker (TickerMetadata): The ticker metadata object.

        Returns:
            str: The type of the ticker.
        """
        return (
            ticker.type_disp.strip().lower()
            if ticker.type_disp and len(ticker.type_disp.strip()) > 0
            else None
        ) or (
            ticker.quote_type.strip()
            if ticker.quote_type and len(ticker.quote_type.strip()) > 0
            else None
        )

    def _build_ticker_sector(self, ticker: TickerMetadata) -> str:
        """
        Builds a sector name using available metadata.

        Args:
            ticker (TickerMetadata): The ticker metadata object.

        Returns:
            str: The sector of the ticker.
        """
        return (
            (
                ticker.sector_key.strip()
                if ticker.sector_key and len(ticker.sector_key.strip()) > 0
                else None
            )
            or (
                ticker.sector_disp.strip()
                if ticker.sector_disp and len(ticker.sector_disp.strip()) > 0
                else None
            )
            or (
                ticker.sector.strip()
                if ticker.sector and len(ticker.sector.strip()) > 0
                else None
            )
        )

    def _build_ticker_industry(self, ticker: TickerMetadata) -> str:
        """
        Builds an industry name using available metadata.

        Args:
            ticker (TickerMetadata): The ticker metadata object.

        Returns:
            str: The industry of the ticker.
        """
        return (
            (
                ticker.industry_key.strip()
                if ticker.industry_key and len(ticker.industry_key.strip()) > 0
                else None
            )
            or (
                ticker.industry_disp.strip()
                if ticker.industry_disp and len(ticker.industry_disp.strip()) > 0
                else None
            )
            or (
                ticker.industry.strip()
                if ticker.industry and len(ticker.industry.strip()) > 0
                else None
            )
        )

    def _build_ticker_currency(self, ticker: TickerMetadata) -> str:
        """
        Builds a currency name using available metadata.

        Args:
            ticker (TickerMetadata): The ticker metadata object.

        Returns:
            str: The currency of the ticker.
        """
        return (
            (
                ticker.currency.strip()
                if ticker.currency and len(ticker.currency.strip()) > 0
                else None
            )
            or (
                ticker.financial_currency.strip()
                if ticker.financial_currency
                and len(ticker.financial_currency.strip()) > 0
                else None
            )
            or self._DEFAULT_CURRENCY
        )

    def _build_ticker_exchange(self, ticker: TickerMetadata) -> str:
        """
        Builds the exchange name from metadata.

        Args:
            ticker (TickerMetadata): The ticker metadata object.

        Returns:
            str: The exchange of the ticker.
        """
        return (
            ticker.exchange.strip()
            if ticker.exchange and len(ticker.exchange.strip()) > 0
            else None
        )

    def get_price_data(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        """
        Download price data for a given symbol and date range.

        Args:
            symbol (str): The ticker symbol to download data for.
            start (datetime): Start date for data download.
            end (datetime): End date for data download.
            interval (str): Data frequency interval.

        Returns:
            pd.DataFrame: DataFrame containing price data.
        """
        attempt = 0
        symbol = symbol.strip().upper() if symbol and symbol.strip() else None

        config = PriceDataConfig(
            symbols=symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        while attempt < self.retries:
            try:
                with OutputSuppressor.suppress():
                    df = self._PROVIDER.get_price_data(config)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df = TimezoneNormalizer.localize_to_market_time(df, self._MARKET_TZ)
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

    def get_metadata(self, symbol: str) -> Optional[TickerMetadata]:
        """
        Fetch ticker metadata for a given symbol.

        Args:
            symbol (str): The ticker symbol to fetch metadata for.

        Returns:
            Optional[TickerMetadata]: TickerMetadata object if available, else None.
        """
        symbol = symbol.strip().upper() if symbol and symbol.strip() else None
        attempt = 0
        metadata = {
            "name": None,
            "type": None,
            "sector": None,
            "industry": None,
            "currency": self._DEFAULT_CURRENCY,
            "exchange": None,
        }
        while attempt < self.retries:
            try:
                with OutputSuppressor.suppress():
                    ticker = self._PROVIDER.get_metadata(symbol)
                if ticker:
                    metadata = {
                        "name": self._build_ticker_name(ticker),
                        "type": self._build_ticker_type(ticker),
                        "sector": self._build_ticker_sector(ticker),
                        "industry": self._build_ticker_industry(ticker),
                        "currency": self._build_ticker_currency(ticker),
                        "exchange": self._build_ticker_exchange(ticker),
                    }
                    Logger.debug(f"    Metadata fetched for {symbol}")
                    attempt = self.retries
                else:
                    Logger.warning(f"No metadata found for {symbol}")
            except (KeyError, ValueError) as error:
                Logger.error(f"    Failed to fetch metadata for {symbol}: {error}")
                time.sleep(self.sleep_seconds)
            attempt += 1
        return metadata

    def is_symbol_available(self, symbol: str) -> bool:
        """
        Check if price data is available for the given symbol in the recent window.

        Args:
            symbol (str): Stock or asset symbol.

        Returns:
            bool: True if data is available, False otherwise.
        """
        try:
            now = datetime.now(timezone.utc)
            start = now - timedelta(days=self._AVAILABILITY_DAYS_WINDOW)
            df = self.get_price_data(symbol, start, now, self._INTERVAL)
            return not df.empty
        except (ValueError, IOError, KeyError) as error:
            Logger.error(f"  Failed to symbol availability check {symbol}: {error}")
            return False

    def get_historical_prices(
        self, symbol: str, last_datetime: Optional[datetime]
    ) -> pd.DataFrame:
        """
        Fetch historical price data from the last recorded datetime up to today.

        Args:
            symbol (str): Stock or asset symbol.
            last_datetime (Optional[datetime]): Last datetime to resume data fetching.

        Returns:
            pd.DataFrame: Historical price data.
        """
        if last_datetime:
            last_datetime = (
                last_datetime.astimezone(timezone.utc)
                if last_datetime.tzinfo
                else last_datetime.replace(tzinfo=timezone.utc)
            )

        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = today + timedelta(days=1)
        start = min(
            today,
            (
                (last_datetime + timedelta(hours=1))
                if last_datetime
                else now - timedelta(days=self._HISTORICAL_DAYS_FALLBACK)
            ),
        ).replace(hour=0, minute=0, second=0, microsecond=0)

        all_data_frames = []
        current_start = start

        while current_start < end:
            current_end = min(current_start + timedelta(days=self.block_days), end)
            df = self.get_price_data(symbol, current_start, current_end, self._INTERVAL)
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
        full_df.columns = [
            "datetime",
            "adj_close",
            "close",
            "high",
            "low",
            "open",
            "volume",
        ]
        return full_df
