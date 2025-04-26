"""Unit tests for the timezone localization utilities in market data."""

import unittest
from unittest.mock import patch

import pandas as pd
from pytz import timezone

from utils.timezone import localize_to_market_time


class TestTimezoneLocalization(unittest.TestCase):
    """Test cases for converting DataFrame index to UTC timezone."""

    @patch("utils.timezone.Logger.warning")
    def test_empty_dataframe_returns_itself(self, mock_warn):
        """Ensure empty DataFrame returns unchanged and logs warning."""
        df = pd.DataFrame()
        result = localize_to_market_time(df, "America/New_York")
        self.assertTrue(result.empty)
        mock_warn.assert_called_once()

    @patch("utils.timezone.Logger.info")
    def test_naive_datetime_localization(self, mock_info):
        """Test that naive datetime index is correctly localized to UTC."""
        index = pd.date_range("2024-01-01 09:00", periods=3, freq="h")
        df = pd.DataFrame({"price": [1, 2, 3]}, index=index)

        result = localize_to_market_time(df, "America/New_York")
        self.assertEqual(str(result.index.tz), "UTC")
        self.assertTrue(mock_info.called)

    def test_already_localized_datetime(self):
        """Test that already localized datetimes are converted to UTC."""
        ny_tz = timezone("America/New_York")
        index = pd.date_range("2024-01-01 09:00", periods=3, freq="h", tz=ny_tz)
        df = pd.DataFrame({"price": [1, 2, 3]}, index=index)

        result = localize_to_market_time(df, "America/New_York")
        self.assertEqual(str(result.index.tz), "UTC")
