"""Unit tests for the IndicatorCalculator class and indicator utilities."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from utils.indicators import IndicatorCalculator


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
    """Provide synthetic OHLCV data for indicator tests."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Close": np.linspace(100, 110, 10),
            "High": np.linspace(101, 111, 10),
            "Low": np.linspace(99, 109, 10),
            "Volume": np.linspace(1000, 2000, 10),
        }
    )


@patch(
    "utils.indicators.parameters",
    {
        "rsi_window_backtest": 3,
        "macd_fast": 2,
        "macd_slow": 5,
        "macd_signal": 3,
        "bollinger_window": 3,
        "bollinger_band_method": "max-min",
        "stoch_rsi_window": 3,
        "stoch_rsi_min_periods": 1,
        "obv_fill_method": 0.0,
        "atr_window": 2,
        "williams_r_window": 3,
    },
)
def test_compute_technical_indicators(
    ohlcv_sample,
):  # pylint: disable=redefined-outer-name
    """Check that all technical indicators are computed and return series."""
    df = ohlcv_sample.copy()
    result = IndicatorCalculator.compute_technical_indicators(df)

    expected_columns = [
        "return_1h",
        "volatility_3h",
        "momentum_3h",
        "rsi",
        "macd",
        "volume",
        "bb_width",
        "stoch_rsi",
        "obv",
        "atr",
        "williams_r",
    ]
    for col in expected_columns:
        check_contains(result.columns, col)
        check_isinstance(result[col], pd.Series)


@patch(
    "utils.indicators.parameters",
    {
        "rsi_window_backtest": 3,
        "macd_fast": 2,
        "macd_slow": 5,
        "macd_signal": 3,
        "bollinger_window": 3,
        "bollinger_band_method": "max-min",
        "stoch_rsi_window": 3,
        "stoch_rsi_min_periods": 1,
        "obv_fill_method": 0.0,
        "atr_window": 2,
        "williams_r_window": 3,
    },
)
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


def test_compute_macd_diff_logic():
    """Validate MACD diff computation and series shape."""
    prices = pd.Series([100, 102, 101, 105, 107, 110])
    result = IndicatorCalculator.compute_macd_diff(prices, fast=2, slow=4, signal=2)
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
