"""Unit tests for the timezone localization utilities in market data."""

import unittest
from typing import cast
from unittest.mock import patch

import pandas as pd  # type: ignore
from pytz import timezone  # type: ignore

from src.market_data.utils.datetime.timezone_normalizer import \
    TimezoneNormalizer


class TestTimezoneLocalization(unittest.TestCase):
    """Test cases for converting DataFrame index to UTC timezone."""

    def test_empty_dataframe_returns_itself(self):
        """Ensure empty DataFrame returns unchanged and logs warning."""
        df = pd.DataFrame()
        result = TimezoneNormalizer.localize_to_market_time(df, "America/New_York")
        self.assertTrue(result.empty)

    def test_naive_datetime_localization(self):
        """Test that naive datetime index is correctly localized to UTC."""
        index = pd.date_range("2024-01-01 09:00", periods=3, freq="h")
        df = pd.DataFrame({"price": [1, 2, 3]}, index=index)
        result = TimezoneNormalizer.localize_to_market_time(df, "America/New_York")
        self.assertIsInstance(result.index, pd.DatetimeIndex)
        idx = cast(pd.DatetimeIndex, result.index)
        self.assertEqual(str(idx.tz), "UTC")

    def test_already_localized_datetime(self):
        """Test that already localized datetimes are converted to UTC."""
        ny_tz = timezone("America/New_York")
        index = pd.date_range("2024-01-01 09:00", periods=3, freq="h", tz=ny_tz)
        df = pd.DataFrame({"price": [1, 2, 3]}, index=index)
        result = TimezoneNormalizer.localize_to_market_time(df, "America/New_York")
        self.assertIsInstance(result.index, pd.DatetimeIndex)
        idx = cast(pd.DatetimeIndex, result.index)
        self.assertEqual(str(idx.tz), "UTC")

    def test_invalid_timezone_raises_and_logs(self):
        """Test that passing an invalid timezone raises an exception and logs the error."""
        index = pd.date_range("2024-01-01 09:00", periods=1, freq="h")
        df = pd.DataFrame({"price": [1]}, index=index)
        invalid_tz = "America/NO_EXISTE"
        with patch("src.utils.io.logger.Logger.error") as mock_log:
            with self.assertRaises(Exception) as _cm:
                TimezoneNormalizer.localize_to_market_time(df, invalid_tz)
            self.assertIn(
                "Invalid timezone: America/NO_EXISTE", mock_log.call_args[0][0]
            )
