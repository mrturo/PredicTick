"""Unit tests for the IndicatorCalculator class and indicator utilities."""

from unittest.mock import patch

import pandas as pd
import pytest

from utils.indicators import IndicatorCalculator
from utils.parameters import ParameterLoader

_PARAMS = ParameterLoader()
_FEATURES = _PARAMS.get("features")


def check_contains(container, item, message=""):
    """Assert that an item is in a container."""
    if item not in container:
        raise AssertionError(message or f"{item} not found in {container}")


def check_isinstance(obj, cls, message=""):
    """Assert that an object is instance of a given class."""
    if not isinstance(obj, cls):
        raise AssertionError(message or f"Expected {obj} to be instance of {cls}")


def check_eq(actual, expected, message=""):
    """Assert that two values are equal."""
    if actual != expected:
        raise AssertionError(message or f"Expected {expected}, got {actual}")


@pytest.fixture
def ohlcv_sample():
    """Sample OHLCV DataFrame for testing."""
    data = {
        "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
    }
    return pd.DataFrame(data)


def test_compute_technical_indicators(
    ohlcv_sample,
):  # pylint: disable=redefined-outer-name, duplicate-code
    """Test that all expected technical indicators are computed."""
    df = ohlcv_sample.copy()
    result = IndicatorCalculator.compute_technical_indicators(df)

    expected_features = [
        "return_1h",
        "volatility_3h",
        "momentum_3h",
        "rsi",
        "macd",
        "bb_width",
        "stoch_rsi",
        "obv",
        "atr",
        "williams_r",
    ]

    for col in expected_features:
        check_contains(result.columns, col)


@patch("utils.indicators.IndicatorCalculator._MACD_FAST", 2)
@patch("utils.indicators.IndicatorCalculator._MACD_SLOW", 5)
@patch("utils.indicators.IndicatorCalculator._MACD_SIGNAL", 3)
@patch("utils.indicators.IndicatorCalculator._RSI_WINDOW", 3)
@patch("utils.indicators.IndicatorCalculator._BOLLINGER_WINDOW", 3)
@patch("utils.indicators.IndicatorCalculator._BOLLINGER_BAND_METHOD", "max-min")
@patch("utils.indicators.IndicatorCalculator._STOCH_RSI_WINDOW", 3)
@patch("utils.indicators.IndicatorCalculator._STOCH_RSI_MIN_PERIODS", 1)
@patch("utils.indicators.IndicatorCalculator._OBV_FILL_METHOD", 0.0)
@patch("utils.indicators.IndicatorCalculator._ATR_WINDOW", 2)
@patch("utils.indicators.IndicatorCalculator._WILLIAMS_R_WINDOW", 3)
def test_get_indicator_parameters():
    """Test that parameter dictionary includes expected keys."""
    params = IndicatorCalculator.get_indicator_parameters()
    check_isinstance(params, dict)
    check_contains(params, "rsi_window")
    check_eq(params["macd_fast"], 2)


def test_compute_rsi_logic():
    """Validate RSI computation logic on price series."""
    prices = pd.Series([100, 101, 102, 101, 100, 99, 98])
    result = IndicatorCalculator.compute_rsi(prices, window=2)
    check_isinstance(result, pd.Series)
    check_eq(result.shape[0], prices.shape[0])


def test_compute_macd_logic():
    """Validate MACD diff computation and series shape."""
    prices = pd.Series([100, 102, 101, 105, 107, 110])
    result = IndicatorCalculator.compute_macd(prices, fast=2, slow=4, signal=2)[
        "histogram"
    ]
    check_isinstance(result, pd.Series)
    check_eq(result.shape[0], prices.shape[0])


def test_compute_stoch_rsi_logic():
    """Validate Stochastic RSI computation and output length."""
    series = pd.Series([50, 55, 53, 52, 60])
    result = IndicatorCalculator.compute_stoch_rsi(series, window=3)
    check_isinstance(result, pd.Series)
    check_eq(result.shape[0], series.shape[0])


def test_check_contains_pass():
    """Test check_contains when item exists."""
    check_contains(["a", "b"], "a")


def test_check_contains_fail():
    """Test check_contains raises error when item is missing."""
    with pytest.raises(AssertionError, match="c not found in"):
        check_contains(["a", "b"], "c")


def test_check_isinstance_pass():
    """Test check_isinstance when type matches."""
    check_isinstance("hello", str)


def test_check_isinstance_fail():
    """Test check_isinstance raises error on type mismatch."""
    with pytest.raises(
        AssertionError, match="Expected 123 to be instance of <class 'str'>"
    ):
        check_isinstance(123, str)


def test_check_eq_pass():
    """Test check_eq when values match."""
    check_eq(5, 5)


def test_check_eq_fail():
    """Test check_eq raises error on mismatch."""
    with pytest.raises(AssertionError, match="Expected 1, got 2"):
        check_eq(2, 1)
