"""Unit tests for the EnrichedData class which handles enriched market data operations.

This module tests methods responsible for:
- Path resolution
- Symbol setting and retrieval
- Metadata handling
- File loading operations

All tests avoid direct `assert` usage for compliance with Bandit security rules.
"""

# pylint: disable=protected-access

from unittest.mock import patch

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pytest  # type: ignore

from market_data.enriched_data import EnrichedData, TechnicalIndicators
from utils.parameters import ParameterLoader

_PARAMS = ParameterLoader()
_TEST_RAW_MARKETDATA_FILEPATH = _PARAMS.get("test_enriched_marketdata_filepath")


@pytest.fixture(autouse=True)
def clear_symbols():
    """Reset symbols dictionary before and after each test to ensure isolation."""
    EnrichedData.set_symbols({})
    yield
    EnrichedData.set_symbols({})


@pytest.fixture(scope="module")  # pylint: disable=redefined-outer-name
def ohlcv_test_data():
    """Provide a consistent sample OHLCV dataset for indicator computation tests."""
    index = pd.date_range(start="2024-01-01", periods=200, freq="1min")
    close = pd.Series(np.linspace(100, 120, 200), index=index)
    high = close + 1
    low = close - 1
    open_ = close - 0.5
    volume = pd.Series(np.random.randint(1000, 5000, size=200), index=index)
    return close, high, low, open_, volume


def test_get_filepath_with_valid():
    """Test that a valid filepath is returned unchanged."""
    expected = "/some/path.json"
    actual = EnrichedData._get_filepath("/some/path.json")
    if actual != expected:
        raise AssertionError(f"Expected {expected}, got {actual}")


def test_get_filepath_with_none():
    """Test that default filepath is used when input is None."""
    expected = "default.json"
    actual = EnrichedData._get_filepath(None, "default.json")
    if actual != expected:
        raise AssertionError(f"Expected {expected}, got {actual}")


def test_get_symbols_empty():
    """Verify that get_symbols returns an empty dictionary initially."""
    if EnrichedData.get_symbols():
        raise AssertionError("Expected empty symbols dictionary")


def test_set_and_get_symbols():
    """Test setting and retrieving the full symbol dictionary."""
    sample = {"AAPL": {"symbol": "AAPL"}}
    EnrichedData.set_symbols(sample)
    if EnrichedData.get_symbols() != sample:
        raise AssertionError("Symbols mismatch after setting")


def test_get_symbol_existing():
    """Verify correct retrieval of an existing symbol's metadata."""
    sample = {"symbol": "TSLA", "sector": "Auto"}
    EnrichedData.set_symbols({"TSLA": sample})
    result = EnrichedData.get_symbol("TSLA")
    if result != sample:
        raise AssertionError(f"Expected {sample}, got {result}")


def test_get_symbol_missing():
    """Test retrieval of a missing symbol, expecting None."""
    if EnrichedData.get_symbol("MSFT") is not None:
        raise AssertionError("Expected None for missing symbol")


def test_load_with_valid_data():
    """Test loading enriched data from a mock JSON file and validate conversion."""
    sample_data = {
        "id": "mock_id",
        "interval": "1day",
        "last_updated": "2024-01-01T00:00:00Z",
        "ranges": {
            "price": {"min": 1.0182262659072876, "max": 111944.5},
            "volume": {"min": 34217.0, "max": 35653705728.0},
        },
        "symbols": [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "type": "Equity",
                "sector": "Tech",
                "industry": "Consumer Electronics",
                "currency": "USD",
                "exchange": "NASDAQ",
                "historical_prices": [
                    {"datetime": "2024-01-01T00:00:00Z", "close": 150.0}
                ],
            }
        ],
    }

    with patch("utils.json_manager.JsonManager.load", return_value=sample_data):
        result = EnrichedData.load(filepath=_TEST_RAW_MARKETDATA_FILEPATH)
        symbols = result["symbols"]
        if "AAPL" not in symbols:
            raise AssertionError("Missing AAPL in loaded symbols")
        if symbols["AAPL"]["name"] != "Apple Inc.":
            raise AssertionError("Incorrect name loaded for AAPL")
        if not isinstance(symbols["AAPL"]["historical_prices"], list):
            raise AssertionError("Historical prices should be a list")
        dt = pd.to_datetime(
            symbols["AAPL"]["historical_prices"][0]["datetime"], utc=True
        )
        if not isinstance(dt, pd.Timestamp):
            raise AssertionError("Datetime conversion failed")


def test_load_with_none_data():
    """Test behavior when JSON load returns None; expect empty symbols dictionary."""
    with patch("utils.json_manager.JsonManager.load", return_value=None):
        result = EnrichedData.load(filepath=_TEST_RAW_MARKETDATA_FILEPATH)
        if result["symbols"]:
            raise AssertionError("Expected empty symbols when loading None")


def test_get_ranges_returns_expected_data():
    """Test that get_ranges returns the expected range object."""
    expected_ranges = {
        "price": {"min": 100.0, "max": 200.0},
        "volume": {"min": 1000.0, "max": 5000.0},
    }
    EnrichedData.set_ranges(expected_ranges)
    actual_ranges = EnrichedData.get_ranges()
    if actual_ranges != expected_ranges:
        raise AssertionError(f"Expected {expected_ranges}, got {actual_ranges}")


def test_get_indicator_parameters_returns_expected_keys():
    """Validate get_indicator_parameters returns a complete and well-structured dict."""
    result = EnrichedData.get_indicator_parameters()
    expected_keys = {
        "rsi_window",
        "macd_fast",
        "macd_slow",
        "macd_signal",
        "bollinger_window",
        "bollinger_band_method",
        "stoch_rsi_window",
        "stoch_rsi_min_periods",
        "obv_fill_method",
        "atr_window",
        "williams_r_window",
    }
    if not isinstance(result, dict):
        raise AssertionError("Expected result to be a dictionary")
    if set(result.keys()) != expected_keys:
        missing = expected_keys - set(result.keys())
        extra = set(result.keys()) - expected_keys
        raise AssertionError(f"Mismatch in keys. Missing: {missing}, Extra: {extra}")


def test_filter_prices_from_global_min_returns_none_if_no_min_datetimes():
    """Verify that filter_prices_from_global_min returns None if no symbol has historical_prices."""
    symbols = {
        "AAPL": {"symbol": "AAPL", "name": "Apple"},
        "GOOG": {"symbol": "GOOG", "name": "Google"},
    }
    result = EnrichedData.filter_prices_from_global_min(symbols)
    if result is not None:
        raise AssertionError("Expected None when no historical_prices are present")


def test_filter_prices_from_global_min_skips_symbols_without_prices():
    """Verify that symbols without 'historical_prices' are skipped during filtering."""
    symbols = {
        "AAPL": {
            "symbol": "AAPL",
            "historical_prices": [
                {"datetime": "2024-01-01T00:00:00Z"},
                {"datetime": "2024-01-03T00:00:00Z"},
            ],
        },
        "GOOG": {
            "symbol": "GOOG",
            "historical_prices": [
                {"datetime": "2024-01-02T00:00:00Z"},
                {"datetime": "2024-01-04T00:00:00Z"},
            ],
        },
        "MSFT": {
            "symbol": "MSFT",
            "note": "missing historical_prices",
        },
    }
    result = EnrichedData.filter_prices_from_global_min(symbols)
    if result is None:
        raise AssertionError("Expected filtered result, got None")
    if "MSFT" not in result:
        raise AssertionError("Expected MSFT to remain in symbols")
    if len(result["AAPL"]["historical_prices"]) != 1:
        raise AssertionError("AAPL should be filtered to one record from 2024-01-03")
    if result["AAPL"]["historical_prices"][0]["datetime"] != "2024-01-03T00:00:00Z":
        raise AssertionError("Filtered datetime is incorrect for AAPL")


def test_compute_return(ohlcv_test_data):  # pylint: disable=redefined-outer-name
    """Test compute_return produces a Series of percent changes from close prices."""
    close, *_ = ohlcv_test_data
    result = TechnicalIndicators.compute_return(close)
    if not isinstance(result, pd.Series):
        raise AssertionError("Return must be a Series")


def test_compute_volatility(ohlcv_test_data):  # pylint: disable=redefined-outer-name
    """Test compute_volatility returns a Series from high, low, and open prices."""
    _, high, low, open_, _ = ohlcv_test_data
    result = TechnicalIndicators.compute_volatility(high, low, open_)
    if not isinstance(result, pd.Series):
        raise AssertionError("Volatility must be a Series")


def test_compute_price_change(ohlcv_test_data):  # pylint: disable=redefined-outer-name
    """Test compute_price_change returns a Series representing close - open."""
    close, _, _, open_, _ = ohlcv_test_data
    result = TechnicalIndicators.compute_price_change(close, open_)
    if not isinstance(result, pd.Series):
        raise AssertionError("Price change must be a Series")


def test_compute_volume_change(ohlcv_test_data):  # pylint: disable=redefined-outer-name
    """Test compute_volume_change returns percentage change in volume as a Series."""
    *_, volume = ohlcv_test_data
    result = TechnicalIndicators.compute_volume_change(volume)
    if not isinstance(result, pd.Series):
        raise AssertionError("Volume change must be a Series")


def test_compute_typical_price(ohlcv_test_data):  # pylint: disable=redefined-outer-name
    """Test compute_typical_price returns average of high, low, and close as a Series."""
    close, high, low, _, _ = ohlcv_test_data
    result = TechnicalIndicators.compute_typical_price(high, low, close)
    if not isinstance(result, pd.Series):
        raise AssertionError("Typical price must be a Series")


def test_compute_average_price(ohlcv_test_data):  # pylint: disable=redefined-outer-name
    """Test compute_average_price returns average of high and low as a Series."""
    _, high, low, _, _ = ohlcv_test_data
    result = TechnicalIndicators.compute_average_price(high, low)
    if not isinstance(result, pd.Series):
        raise AssertionError("Average price must be a Series")


def test_compute_range(ohlcv_test_data):  # pylint: disable=redefined-outer-name
    """Test compute_range returns difference between high and low as a Series."""
    _, high, low, _, _ = ohlcv_test_data
    result = TechnicalIndicators.compute_range(high, low)
    if not isinstance(result, pd.Series):
        raise AssertionError("Range must be a Series")


def test_compute_relative_volume(
    ohlcv_test_data,
):  # pylint: disable=redefined-outer-name
    """Test compute_relative_volume returns ratio of volume to rolling average."""
    *_, volume = ohlcv_test_data
    result = TechnicalIndicators.compute_relative_volume(volume, window=20)
    if not isinstance(result, pd.Series):
        raise AssertionError("Relative volume must be a Series")


def test_compute_overnight_return(
    ohlcv_test_data,
):  # pylint: disable=redefined-outer-name
    """Test compute_overnight_return returns Series of open-to-open returns."""
    *_, open_ = ohlcv_test_data
    result = TechnicalIndicators.compute_overnight_return(open_)
    if not isinstance(result, pd.Series):
        raise AssertionError("Overnight return must be a Series")


def test_compute_intraday_return(
    ohlcv_test_data,
):  # pylint: disable=redefined-outer-name
    """Test compute_intraday_return returns Series of intraday returns (close/open - 1)."""
    close, *_, open_ = ohlcv_test_data
    result = TechnicalIndicators.compute_intraday_return(close, open_)
    if not isinstance(result, pd.Series):
        raise AssertionError("Intraday return must be a Series")


def test_compute_rsi(ohlcv_test_data):  # pylint: disable=redefined-outer-name
    """Test compute_rsi returns the Relative Strength Index over given window."""
    close, *_ = ohlcv_test_data
    result = TechnicalIndicators.compute_rsi(close, window=14)
    if not isinstance(result, pd.Series):
        raise AssertionError("RSI must be a Series")


def test_compute_bb_width(ohlcv_test_data):  # pylint: disable=redefined-outer-name
    """Test compute_bb_width returns Bollinger Band width over a rolling window."""
    close, *_ = ohlcv_test_data
    result = TechnicalIndicators.compute_bb_width(close, window=20)
    if not isinstance(result, pd.Series):
        raise AssertionError("BB width must be a Series")


def test_compute_stoch_rsi(ohlcv_test_data):  # pylint: disable=redefined-outer-name
    """Test compute_stoch_rsi returns normalized RSI values over a window."""
    close, *_ = ohlcv_test_data
    result = TechnicalIndicators.compute_stoch_rsi(close, window=14)
    if not isinstance(result, pd.Series):
        raise AssertionError("Stoch RSI must be a Series")


def test_compute_atr(ohlcv_test_data):  # pylint: disable=redefined-outer-name
    """Test compute_atr returns average true range over given rolling window."""
    close, high, low, _, _ = ohlcv_test_data
    result = TechnicalIndicators.compute_atr(high, low, close, window=14)
    if not isinstance(result, pd.Series):
        raise AssertionError("ATR must be a Series")


def test_compute_williams_r(ohlcv_test_data):  # pylint: disable=redefined-outer-name
    """Test compute_williams_r returns Williams %R indicator as a Series."""
    close, high, low, _, _ = ohlcv_test_data
    result = TechnicalIndicators.compute_williams_r(high, low, close, window=14)
    if not isinstance(result, pd.Series):
        raise AssertionError("Williams %R must be a Series")
