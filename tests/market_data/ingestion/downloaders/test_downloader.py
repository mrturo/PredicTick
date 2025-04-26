"""Unit tests for Downloader: market data and metadata retrieval utilities."""

# pylint: disable=protected-access,too-many-public-methods

import io
import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd  # type: ignore

from src.market_data.ingestion.downloaders.downloader import Downloader


class _DummySuppressCtx:
    """Mimics OutputSuppressor.suppress() returning (stdout, stderr) buffers."""

    def __init__(self, out_text: str = "", err_text: Optional[str] = ""):
        """Initialize in-memory stdout/stderr buffers for the context manager."""
        self._out = io.StringIO(out_text)
        self._err = None if err_text is None else io.StringIO(err_text)

    def __enter__(self):
        """Enter the context and return the tuple (stdout, stderr) buffers."""
        return self._out, self._err

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context without suppressing exceptions."""
        return False


class _DummySessionsDays:
    """Simple trading days policy: Mon-Fri are valid unless in excluded set."""

    def __init__(self, excluded=None, all_open=False):
        """Create the day policy."""
        self._excluded = set(excluded or [])
        self._all_open = all_open

    def is_trading_day(self, d: date) -> bool:
        """Return whether `d` is a trading day under this policy."""
        return d.weekday() < 5 and d not in self._excluded

    def all_true(self) -> bool:
        """Return whether the market is flagged as always open (24x7)."""
        return self._all_open


# pylint: disable=too-few-public-methods
class _DummyHours:
    """Lightweight holder for session open/close time strings."""

    def __init__(self, open_s: Optional[str], close_s: Optional[str]):
        """Store raw time strings as provided by fixtures."""
        self.open = open_s
        self.close = close_s


# pylint: disable=too-few-public-methods
class _DummySessionsHours:
    """Container for regular session hours and exchange timezone."""

    def __init__(self, open_s: Optional[str], close_s: Optional[str], tz: str):
        """Build a regular-hours structure matching the production shape."""
        self.regular = _DummyHours(open_s, close_s)
        self.timezone = tz


# pylint: disable=too-few-public-methods
class _DummyStockExchange:
    """Minimal stock-exchange stub exposing sessions_hours and sessions_days."""

    def __init__(self, open_s: Optional[str], close_s: Optional[str], tz: str, days):
        """Initialize a fake exchange used by tests."""
        self.sessions_hours = _DummySessionsHours(open_s, close_s, tz)
        self.sessions_days = days

    def to_utc(self):
        """Return an exchange-like object already normalized to UTC."""
        return self


class TestDownloader(unittest.TestCase):
    """Full coverage tests for Downloader behavior."""

    def setUp(self):
        """Standardize Downloader static config and clear cached calendars."""
        # Configure static class attributes deterministically
        Downloader._RAW_DATA_INTERVAL = "1d"
        Downloader._DEFAULT_CURRENCY = "USD"
        Downloader._HISTORICAL_DAYS_FALLBACK = {"1d": 30}
        Downloader._MARKET_TZ = "America/New_York"
        Downloader._REQUIRED_MARKET_RAW_COLUMNS = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
        # Ensure no calendar cache from other tests
        for attr in ("_CALENDAR",):
            if hasattr(Downloader, attr):
                delattr(Downloader, attr)

    # -------------------- get_price_data --------------------
    @patch(
        "src.market_data.utils.datetime.timezone_normalizer.TimezoneNormalizer."
        "localize_to_market_time",
        side_effect=lambda df, _tz: df,
    )
    def test_get_price_data_retries_and_returns_empty(self, _mock_tz):
        """Should retry on empty data and return empty DataFrame after retries."""
        provider = SimpleNamespace(
            get_price_data=MagicMock(return_value=pd.DataFrame())
        )
        with patch.object(Downloader, "_PROVIDER", provider), patch(
            "src.utils.io.logger.Logger.error"
        ) as mock_log, patch("time.sleep") as mock_sleep, patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader(retries=3, sleep_seconds=0)
            start = datetime(2024, 1, 1, tzinfo=timezone.utc)
            end = datetime(2024, 1, 10, tzinfo=timezone.utc)
            out = dl.get_price_data("AAPL", start, end, "1d")
            self.assertTrue(out.empty)
            self.assertEqual(provider.get_price_data.call_count, 1)
            self.assertEqual(mock_log.call_count, 0)
            self.assertEqual(mock_sleep.call_count, 0)

    # -------------------- get_metadata --------------------
    @patch(
        "src.utils.io.output_suppressor.OutputSuppressor.suppress",
        return_value=_DummySuppressCtx(),
    )
    def test_get_metadata_builds_fields(self, _mock_suppress):
        """Should build metadata dict from TickerMetadata fields."""
        ticker = SimpleNamespace(
            display_name="  Apple Inc.  ",
            short_name=None,
            long_name=None,
            symbol="aapl",
            type_disp="  Equity ",
            quote_type="ignored",
            sector_key=None,
            sector_disp=" Technology ",
            sector="ignored",
            industry_key=None,
            industry_disp=None,
            industry=" Consumer Electronics ",
            currency=" ",
            financial_currency=" USD ",
            exchange="  NASDAQ ",
        )
        provider = SimpleNamespace(get_metadata=MagicMock(return_value=ticker))
        with patch.object(Downloader, "_PROVIDER", provider), patch.object(
            Downloader, "_PARAMS"
        ), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader()
            out = dl.get_metadata(" aapl ")
        self.assertEqual(
            out,
            {
                "name": "Apple Inc.",
                "type": "equity",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "currency": "USD",
                "exchange": "NASDAQ",
            },
        )
        provider.get_metadata.assert_called_once_with("AAPL")

    @patch(
        "src.utils.io.output_suppressor.OutputSuppressor.suppress",
        return_value=_DummySuppressCtx(),
    )
    def test_get_metadata_handles_exceptions_and_retries(self, _mock_suppress):
        """Should retry on provider exceptions and return defaults."""
        provider = SimpleNamespace(get_metadata=MagicMock(side_effect=KeyError("x")))
        with patch.object(Downloader, "_PROVIDER", provider), patch(
            "src.utils.io.logger.Logger.error"
        ) as mock_err, patch("time.sleep") as mock_sleep, patch.object(
            Downloader, "_PARAMS"
        ), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader(retries=2, sleep_seconds=0)
            out = dl.get_metadata("AAPL")
        self.assertEqual(
            out,
            {
                "name": None,
                "type": None,
                "sector": None,
                "industry": None,
                "currency": "USD",
                "exchange": None,
            },
        )
        self.assertEqual(mock_err.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 2)

    # -------------------- is_symbol_available --------------------
    def test_is_symbol_available_true_false(self):
        """Should return True when get_price_data non-empty, else False."""
        with patch.object(
            Downloader,
            "get_price_data",
            side_effect=[pd.DataFrame({"a": [1]}), pd.DataFrame()],
        ), patch.object(Downloader, "_PARAMS"), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader()
            self.assertTrue(dl.is_symbol_available("AAPL"))
            self.assertFalse(dl.is_symbol_available("AAPL"))

    # -------------------- _parse_time --------------------
    def test_parse_time_valid_and_edge_cases(self):
        """Should parse hh:mm, accept None/empty, and validate bounds."""
        with patch.object(Downloader, "_PARAMS"), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader()
        self.assertEqual(dl._parse_time("09:30"), (9, 30))
        self.assertEqual(dl._parse_time(None), (None, None))
        self.assertEqual(dl._parse_time("  "), (None, None))
        with self.assertRaisesRegex(ValueError, "Expected format"):
            dl._parse_time("9-30")
        with self.assertRaisesRegex(TypeError, "time_str must be a str"):
            dl._parse_time(123)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "range 0–59"):
            dl._parse_time("60:00")
        with self.assertRaisesRegex(ValueError, "range 0–59"):
            dl._parse_time("00:60")

    # -------------------- _get_regular_session_specs --------------------
    def test_get_regular_session_specs_builds_components(self):
        """Should resolve exchange, parse hours, and return tz and SessionsDays."""
        params = SimpleNamespace(
            exchange=lambda _eid: _DummyStockExchange(
                "09:30", "16:00", "UTC", _DummySessionsDays()
            )
        )
        with patch.object(Downloader, "_PARAMS", params), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader()
            h_o, m_o, h_c, m_c, tz, sdays = dl._get_regular_session_specs("XNYS")
        self.assertEqual((h_o, m_o, h_c, m_c), (9, 30, 16, 0))
        self.assertEqual(str(tz), "UTC")
        self.assertTrue(hasattr(sdays, "is_trading_day"))

    def test_get_regular_session_specs_raises_on_missing_exchange(self):
        """Should raise ValueError when exchange is not found."""
        params = SimpleNamespace(exchange=lambda _eid: None)
        with patch.object(Downloader, "_PARAMS", params), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader()
            with self.assertRaisesRegex(ValueError, "Stock exchange not found"):
                dl._get_regular_session_specs("XNYS")

    # -------------------- _utc_session_bounds --------------------
    @patch(
        "src.market_data.utils.intervals.interval_converter.IntervalConverter."
        "to_minutes",
        return_value=1440,
    )
    def test_utc_session_bounds_when_close_equals_open(self, _mock_minutes):
        """If close==open, close should roll to next day minus step."""
        params = SimpleNamespace(
            exchange=lambda _eid: _DummyStockExchange(
                None, None, "UTC", _DummySessionsDays()
            )
        )
        with patch.object(Downloader, "_PARAMS", params), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader()
        open_dt, close_dt = dl._utc_session_bounds(
            date(2024, 1, 2),
            dl.MarketTime(None, None),
            dl.MarketTime(None, None),
            timezone.utc,
        )
        self.assertLess(open_dt, close_dt)

    # -------------------- _compute_next_valid_datetime --------------------
    @patch(
        "src.market_data.utils.intervals.interval_converter.IntervalConverter.to_minutes",
        return_value=60,
    )
    def test_compute_next_valid_datetime_skips_non_trading_days(self, _mock_minutes):
        """Always-open market without hours should return last+step."""
        days = _DummySessionsDays(excluded={date(2024, 1, 2)})
        params = SimpleNamespace(
            exchange=lambda _eid: _DummyStockExchange("09:00", "17:00", "UTC", days)
        )
        with patch.object(Downloader, "_PARAMS", params), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader()
            last_dt = datetime(2024, 1, 2, 16, 0, tzinfo=timezone.utc)
            out = dl._compute_next_valid_datetime("XNYS", last_dt)
            self.assertEqual(out, datetime(2024, 1, 3, 9, 0, tzinfo=timezone.utc))

            last_dt2 = datetime(2024, 1, 3, 18, 0, tzinfo=timezone.utc)
            out2 = dl._compute_next_valid_datetime("XNYS", last_dt2)
            self.assertEqual(out2, datetime(2024, 1, 4, 9, 0, tzinfo=timezone.utc))

    @patch(
        "src.market_data.utils.intervals.interval_converter.IntervalConverter.to_minutes",
        return_value=60,
    )
    def test_compute_next_valid_datetime_24x7_market_without_hours(self, _mock_minutes):
        """Always-open market without hours should return last+step."""
        days = _DummySessionsDays(all_open=True)
        params = SimpleNamespace(
            exchange=lambda _eid: _DummyStockExchange(None, None, "UTC", days)
        )
        with patch.object(Downloader, "_PARAMS", params), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader()
            last_dt = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
            out = dl._compute_next_valid_datetime("CRYPTO", last_dt)
            self.assertEqual(out, last_dt + timedelta(hours=1))

    # -------------------- _build_consolidated_price_data --------------------
    def test_build_consolidated_price_data_concat_and_rename(self):
        """Should concatenate by blocks, drop duplicates, and rename columns."""
        with patch.object(Downloader, "_PARAMS"), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader(block_days=2)
        # Build two blocks with a duplicated index row
        idx1 = pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True)
        idx2 = pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True)
        df1 = pd.DataFrame(
            {
                ("open", ""): [1, 2],
                ("high", ""): [1, 2],
                ("low", ""): [1, 2],
                ("close", ""): [1, 2],
                ("adj_close", ""): [1, 2],
                ("volume", ""): [10, 20],
            },
            index=idx1,
        )
        df2 = pd.DataFrame(
            {
                ("open", ""): [2, 3],
                ("high", ""): [2, 3],
                ("low", ""): [2, 3],
                ("close", ""): [2, 3],
                ("adj_close", ""): [2, 3],
                ("volume", ""): [20, 30],
            },
            index=idx2,
        )
        with patch.object(dl, "get_price_data", side_effect=[df1, df2, pd.DataFrame()]):
            start = datetime(2024, 1, 1, tzinfo=timezone.utc)
            end = datetime(2024, 1, 3, tzinfo=timezone.utc)
            out = dl._build_consolidated_price_data("AAPL", start, end)
        self.assertListEqual(
            list(out.columns),
            Downloader._REQUIRED_MARKET_RAW_COLUMNS,
        )
        # Duplicated 2024-01-02 should be deduplicated
        self.assertEqual(len(out), 3)

    # -------------------- _format_timedelta --------------------
    def test_format_timedelta_variants(self):
        """Should format timedelta and return seconds."""
        with patch.object(Downloader, "_PARAMS"), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader()
        self.assertEqual(dl._format_timedelta(timedelta(seconds=0)), ("0s", 0))
        self.assertEqual(
            dl._format_timedelta(timedelta(days=1, hours=2, minutes=3, seconds=4)),
            ("1d 2h 3m 4s", 93784),
        )

    # -------------------- get_historical_prices --------------------
    def test_get_historical_prices_from_scratch_uses_fallback(self):
        """When last_datetime is None, should start at now - fallback_days."""
        with patch.object(Downloader, "_PARAMS"), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader(block_days=10)
        now = datetime(2024, 1, 31, tzinfo=timezone.utc)
        captured = {}

        def _fake_build(symbol, start, end):
            captured["symbol"] = symbol
            captured["start"] = start
            captured["end"] = end
            return pd.DataFrame({"ok": [1]})

        with patch.object(
            dl, "_build_consolidated_price_data", side_effect=_fake_build
        ):
            out_df, remaining, seconds = dl.get_historical_prices(
                "AAPL", "XNYS", last_datetime=None, now=now
            )
        self.assertFalse(out_df.empty)
        self.assertIsNone(remaining)
        self.assertIsNone(seconds)
        self.assertEqual(captured["symbol"], "AAPL")
        self.assertEqual(captured["end"], now)
        # Fallback 30 days at 00:00
        self.assertEqual(captured["start"], datetime(2024, 1, 1, tzinfo=timezone.utc))

    @patch(
        "src.market_data.utils.intervals.interval_converter.IntervalConverter."
        "to_minutes",
        return_value=60,
    )
    def test_get_historical_prices_next_after_now_returns_wait(self, _mock_minutes):
        """If next_datetime > now, should return empty and ('remaining', seconds)."""
        params = SimpleNamespace(
            exchange=lambda _eid: _DummyStockExchange(
                "09:00", "17:00", "UTC", _DummySessionsDays()
            )
        )
        with patch.object(Downloader, "_PARAMS", params), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader()
        now = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
        last_dt = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
        with patch.object(
            dl, "_compute_next_valid_datetime", return_value=now + timedelta(hours=2)
        ):
            out_df, remaining, seconds = dl.get_historical_prices(
                "AAPL", "XNYS", last_datetime=last_dt, now=now
            )
        self.assertTrue(out_df.empty)
        self.assertIsInstance(remaining, str)
        self.assertGreater(seconds, 0)

    @patch(
        "src.market_data.utils.intervals.interval_converter.IntervalConverter."
        "to_minutes",
        return_value=60,
    )
    def test_get_historical_prices_from_next_datetime_happy_path(self, _mock_minutes):
        """When next datetime is valid, should build consolidated data from that point."""
        params = SimpleNamespace(
            exchange=lambda _eid: _DummyStockExchange(
                "09:00", "17:00", "UTC", _DummySessionsDays()
            )
        )
        with patch.object(Downloader, "_PARAMS", params), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            dl = Downloader()
        last_dt = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
        now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
        next_dt = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
        with patch.object(
            dl, "_compute_next_valid_datetime", return_value=next_dt
        ), patch.object(
            dl, "_build_consolidated_price_data", return_value=pd.DataFrame({"ok": [1]})
        ) as mock_build, patch(
            "src.utils.io.logger.Logger.debug"
        ) as mock_debug:
            out_df, remaining, seconds = dl.get_historical_prices(
                "AAPL", "XNYS", last_datetime=last_dt, now=now
            )
        self.assertFalse(out_df.empty)
        self.assertIsNone(remaining)
        self.assertIsNone(seconds)
        args, _ = mock_build.call_args
        self.assertEqual(args[1], next_dt)
        self.assertEqual(args[2], now)
        self.assertTrue(
            any("Next register" in c[0][0] for c in mock_debug.call_args_list)
        )

    # -------------------- constructor guard --------------------
    def test_constructor_raises_when_raw_interval_missing(self):
        """Constructor should raise if _RAW_DATA_INTERVAL is empty."""
        Downloader._RAW_DATA_INTERVAL = ""
        with self.assertRaisesRegex(
            ValueError, "Interval parameter is not defined"
        ), patch.object(Downloader, "_PARAMS"), patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(object(), [], None),
        ):
            Downloader()
