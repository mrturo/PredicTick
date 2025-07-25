"""Downloader module for fetching market data and metadata.

Provides utilities for retrieving historical price data, metadata, and availability
checks for financial instruments, with retry logic, parametrizable windows, and
timezone normalization.
"""

import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, NamedTuple, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore

from market_data.ingestion.providers.provider import (
    PriceDataConfig,
    Provider,
    TickerMetadata,
)
from market_data.utils.datetime.timezone_normalizer import TimezoneNormalizer
from market_data.utils.intervals.interval import Interval
from market_data.utils.intervals.interval_converter import IntervalConverter
from utils.calendar_manager import CalendarManager
from utils.hours import Hours
from utils.logger import Logger
from utils.output_suppressor import OutputSuppressor
from utils.parameters import ParameterLoader
from utils.sessions_days import SessionsDays
from utils.sessions_hours import SessionsHours
from utils.stock_exchange import StockExchange


class Downloader:
    """Retrieves historical price data and metadata for financial instruments."""

    _PARAMS = ParameterLoader()
    _AVAILABILITY_DAYS_WINDOW = _PARAMS.get("availability_days_window")
    _DEFAULT_CURRENCY = _PARAMS.get("default_currency")
    _HISTORICAL_DAYS_FALLBACK = _PARAMS.get("historical_days_fallback")
    _RAW_DATA_INTERVAL: str = (Interval.market_raw_data() or "").strip()
    _MARKET_TZ = _PARAMS.get("market_tz")
    _REQUIRED_MARKET_RAW_COLUMNS: list[str] = _PARAMS.get("required_market_raw_columns")

    _PROVIDER = Provider()

    def __init__(
        self,
        block_days: Optional[int] = None,
        retries: Optional[int] = None,
        sleep_seconds: Optional[int] = None,
    ):
        """Initialize Downloader instance."""
        self.block_days = block_days or 0
        self.retries = retries or 1
        self.sleep_seconds = sleep_seconds or 0
        self.holidays: list[date] = []
        if len(Downloader._RAW_DATA_INTERVAL) == 0:
            raise ValueError(
                "Interval parameter is not defined. Please set it before proceeding."
            )
        holidays: Optional[list[date]] = None
        if not hasattr(Downloader, "_CALENDAR"):
            (
                Downloader._CALENDAR,
                holidays,
                _event_days,
            ) = CalendarManager.build_market_calendars()
        self.holidays = holidays if holidays is not None else []

    def _build_ticker_name(self, ticker: TickerMetadata) -> Optional[str]:
        """Builds a readable ticker name using available metadata."""
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

    def _build_ticker_type(self, ticker: TickerMetadata) -> Optional[str]:
        """Builds a ticker type using available metadata."""
        return (
            ticker.type_disp.strip().lower()
            if ticker.type_disp and len(ticker.type_disp.strip()) > 0
            else None
        ) or (
            ticker.quote_type.strip()
            if ticker.quote_type and len(ticker.quote_type.strip()) > 0
            else None
        )

    def _build_ticker_sector(self, ticker: TickerMetadata) -> Optional[str]:
        """Builds a sector name using available metadata."""
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

    def _build_ticker_industry(self, ticker: TickerMetadata) -> Optional[str]:
        """Builds an industry name using available metadata."""
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

    def _build_ticker_currency(self, ticker: TickerMetadata) -> Optional[str]:
        """Builds a currency name using available metadata."""
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

    def _build_ticker_exchange(self, ticker: TickerMetadata) -> Optional[str]:
        """Builds the exchange name from metadata."""
        return (
            ticker.exchange.strip()
            if ticker.exchange and len(ticker.exchange.strip()) > 0
            else None
        )

    def get_price_data(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        """Download price data for a given symbol and date range."""
        attempt = 0
        symbol = (
            symbol.strip().upper()
            if symbol is not None and len(symbol.strip()) > 0
            else ""
        )
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
                df = self._PROVIDER.get_price_data(config)
                if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                    df = TimezoneNormalizer.localize_to_market_time(df, self._MARKET_TZ)
                    df = df.set_index(
                        pd.to_datetime(df.index, utc=True, errors="coerce")
                    )
                    df = df[(df.index >= start) & (df.index <= end)]
                    return df
                raise ValueError("Empty data")
            except (ValueError, IOError) as error:
                Logger.error(
                    f"  Error downloading {symbol} on attempt {attempt + 1}: {error}"
                )
            attempt += 1
            time.sleep(self.sleep_seconds)
        return pd.DataFrame()

    def get_metadata(self, symbol: str) -> Optional[dict[str, Any]]:
        """Fetch ticker metadata for a given symbol."""
        symbol = symbol.strip().upper() if symbol and symbol.strip() else ""
        attempt = 0
        metadata: dict[str, Optional[str]] = {
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
                    attempt = self.retries
                else:
                    Logger.warning(f"No metadata found for {symbol}")
            except (KeyError, ValueError) as error:
                Logger.error(f"    Failed to fetch metadata for {symbol}: {error}")
                time.sleep(self.sleep_seconds)
            attempt += 1
        return metadata

    def is_symbol_available(self, symbol: str) -> bool:
        """Check if price data is available for the given symbol in the recent window."""
        try:
            now = datetime.now(timezone.utc)
            start = now - timedelta(days=self._AVAILABILITY_DAYS_WINDOW)
            df = self.get_price_data(symbol, start, now, Downloader._RAW_DATA_INTERVAL)
            return not df.empty
        except (ValueError, IOError, KeyError) as error:
            Logger.error(f"  Failed to symbol availability check {symbol}: {error}")
            return False

    def _parse_time(
        self, time_str: Optional[str]
    ) -> tuple[Optional[int], Optional[int]]:
        """
        Split a string of the form 'hh:mm' into hour and minute integers.

        and validate that both lie in the range 0‒59 (inclusive).
        """
        if time_str is not None and len(time_str.strip()) > 0:
            try:
                hour, minute = map(int, time_str.strip().split(":"))
            except ValueError as exc:
                raise ValueError("Expected format hh:mm with numeric values.") from exc
            except AttributeError as exc:
                raise TypeError("time_str must be a str.") from exc
            if not (0 <= hour <= 59 and 0 <= minute <= 59):
                raise ValueError("Hour and minute must be in the range 0–59.")
            return hour, minute
        return None, None

    def _get_regular_session_specs(self, exchange_id: str) -> tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[ZoneInfo],
        Optional[SessionsDays],
    ]:
        """Retrieve the hour–minute components of the regular trading session."""
        stock_exchange: Optional[StockExchange] = Downloader._PARAMS.exchange(
            exchange_id
        ).to_utc()
        if stock_exchange is None:
            raise ValueError(f"Stock exchange not found for {exchange_id}")
        sessions_hours: SessionsHours = stock_exchange.sessions_hours
        regular_hours: Hours = sessions_hours.regular
        hour_open, minute_open = self._parse_time(regular_hours.open)
        hour_close, minute_close = self._parse_time(regular_hours.close)
        tz_market = ZoneInfo(sessions_hours.timezone)
        sessions_days: SessionsDays = stock_exchange.sessions_days
        return (
            hour_open,
            minute_open,
            hour_close,
            minute_close,
            tz_market,
            sessions_days,
        )

    def _compute_next_valid_datetime(
        self, exchange_id: str, last_datetime: datetime
    ) -> Optional[datetime]:
        """Return the next valid UTC datetime for price retrieval."""
        candidate = last_datetime + self._get_step()
        h_open, m_open, h_close, m_close, tz_market, sessions_days = (
            self._get_regular_session_specs(exchange_id)
        )
        always_open_market = sessions_days is not None and sessions_days.all_true()
        no_session_hours_configured = all(
            x is None for x in (h_open, m_open, h_close, m_close)
        )
        if always_open_market and no_session_hours_configured:
            return candidate
        while True:
            sess_date = candidate.astimezone(tz_market).date()
            if not self._is_valid_trading_day(sess_date, sessions_days):
                candidate = self._next_session_start(
                    sess_date + timedelta(days=1), h_open, m_open, tz_market
                )
                continue
            open_dt, close_dt = self._utc_session_bounds(
                sess_date,
                self.MarketTime(h_open, m_open),
                self.MarketTime(h_close, m_close),
                tz_market,
            )
            candidate = max(candidate, open_dt)
            if candidate <= close_dt:
                break
            candidate = self._next_session_start(
                sess_date + timedelta(days=1), h_open, m_open, tz_market
            )
        return candidate

    def _get_step(self) -> timedelta:
        return timedelta(minutes=IntervalConverter.to_minutes(self._RAW_DATA_INTERVAL))

    def _is_valid_trading_day(self, sess_date: date, sessions_days) -> bool:
        return sessions_days is None or (
            sessions_days.is_trading_day(sess_date) and sess_date not in self.holidays
        )

    def _next_session_start(
        self, sess_date: date, h_open: Optional[int], m_open: Optional[int], tz_market
    ) -> datetime:
        return datetime(
            sess_date.year,
            sess_date.month,
            sess_date.day,
            h_open or 0,
            m_open or 0,
            tzinfo=tz_market,
        ).astimezone(timezone.utc)

    class MarketTime(NamedTuple):
        """Represents a local market time using optional hour and minute components."""

        hour: Optional[int]
        minute: Optional[int]

    def _utc_session_bounds(
        self,
        sess_date: date,
        open_time: MarketTime,
        close_time: MarketTime,
        tz_market,
    ) -> tuple[datetime, datetime]:
        """Computes UTC session boundaries given local open and close times."""
        open_dt = datetime(
            sess_date.year,
            sess_date.month,
            sess_date.day,
            open_time.hour or 0,
            open_time.minute or 0,
            tzinfo=tz_market,
        ).astimezone(timezone.utc)
        close_dt = datetime(
            sess_date.year,
            sess_date.month,
            sess_date.day,
            close_time.hour or 0,
            close_time.minute or 0,
            tzinfo=tz_market,
        ).astimezone(timezone.utc)
        if close_dt == open_dt:
            close_dt += timedelta(days=1) - self._get_step()
        return open_dt, close_dt

    def _build_consolidated_price_data(
        self, symbol: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Download and concatenate historical price data by block."""
        current_start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        all_data_frames = []
        while current_start < end:
            current_end = min(current_start + timedelta(days=self.block_days), end)
            df = self.get_price_data(
                symbol, current_start, current_end, Downloader._RAW_DATA_INTERVAL
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
        full_df.columns = Downloader._REQUIRED_MARKET_RAW_COLUMNS
        return full_df

    def _format_timedelta(self, td: timedelta) -> Tuple[str, int]:  # type: ignore
        """Return («human-readable», total_seconds)."""
        total = int(td.total_seconds())
        if total <= 0:
            return "0s", 0
        days, rem = divmod(total, 86_400)
        hrs, rem = divmod(rem, 3_600)
        mins, secs = divmod(rem, 60)
        parts: list[str] = []
        if days:
            parts.append(f"{days}d")
        if hrs:
            parts.append(f"{hrs}h")
        if mins:
            parts.append(f"{mins}m")
        if secs:
            parts.append(f"{secs}s")
        return " ".join(parts), total

    def get_historical_prices(
        self,
        symbol: str,
        exchange_id: str,
        last_datetime: Optional[datetime] = None,
        now: Optional[datetime] = None,
    ) -> tuple[pd.DataFrame, Optional[str], Optional[int]]:
        """Fetch historical price data from the last recorded datetime up to today."""
        if not Downloader._RAW_DATA_INTERVAL:
            raise ValueError("Raw data interval is not defined.")
        if not Downloader._HISTORICAL_DAYS_FALLBACK.get(Downloader._RAW_DATA_INTERVAL):
            raise ValueError(
                f"Days fallback parameter missing for interval '{Downloader._RAW_DATA_INTERVAL}'."
            )
        now = datetime.now(timezone.utc) if now is None else now
        start: datetime
        if last_datetime:
            last_datetime = (
                last_datetime.astimezone(timezone.utc)
                if last_datetime.tzinfo
                else last_datetime.replace(tzinfo=timezone.utc)
            )
            next_datetime = self._compute_next_valid_datetime(
                exchange_id, last_datetime
            )
            if next_datetime is None or next_datetime > now:
                remaining: str = ""
                seconds: int = 0
                if next_datetime is not None:
                    remaining, seconds = self._format_timedelta(next_datetime - now)
                    Logger.debug(
                        f"    Last register: {last_datetime} -> "
                        f"Next update: {next_datetime} "
                        f"(in {remaining}) [Now: {now}]"
                    )
                return pd.DataFrame(), remaining, seconds
            Logger.debug(
                f"    Last register: {last_datetime} -> Next register: {next_datetime} "
                f"[Now: {now}]"
            )
            start = next_datetime
        else:
            fallback_days = Downloader._HISTORICAL_DAYS_FALLBACK[
                Downloader._RAW_DATA_INTERVAL
            ]
            start = (now - timedelta(days=fallback_days)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        return self._build_consolidated_price_data(symbol, start, now), None, None
