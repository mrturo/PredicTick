"""Unit tests for the Candle and MultiCandlePattern classes.

This module tests the functionality of candlestick pattern recognition, including single-candle
patterns and multi-candle patterns. All tests use mocked parameters to isolate behavior.
"""

from unittest.mock import patch

import pytest  # type: ignore

from market_data.transform.candle import Candle, MultiCandlePattern

CANDLE_METRICS = {
    "doji_body_threshold": 0.1,
    "hammer_shadow_ratio": 2.0,
    "upper_shadow_max_ratio": 0.3,
    "hammer_body_threshold": 0.3,
    "shooting_star_shadow_ratio": 2.0,
    "lower_shadow_max_ratio": 0.3,
    "doji_max_upper_shadow": 0.1,
    "doji_max_lower_shadow": 0.1,
    "shooting_star_body_threshold": 0.3,
}

CANDLE_SIMPLE_SCORE = {
    "hammer_bullish": 0.75,
    "hammer_bearish": 0.25,
    "shooting_star_bullish": 0.3,
    "shooting_star_bearish": 0.2,
    "marubozu_bullish": 0.85,
    "marubozu_bearish": 0.15,
    "bullish": 0.6,
    "bearish": 0.4,
}

CANDLE_MULTIPLE_SCORE = {
    "bullish_engulfing": 0.9,
    "bearish_engulfing": 0.1,
    "morning_star": 0.88,
    "evening_star": 0.12,
    "piercing_line": 0.82,
    "dark_cloud_cover": 0.18,
    "tweezer_bottom": 0.8,
    "tweezer_top": 0.2,
    "three_white_soldiers": 0.95,
    "three_black_crows": 0.05,
}


@pytest.fixture(autouse=True)
def patch_params():
    """Patch _PARAMS access for Candle and MultiCandlePattern classes during test execution."""
    with patch("market_data.candle.Candle._PARAMS.get") as single, patch(
        "market_data.candle.MultiCandlePattern._PARAMS.get"
    ) as multi:
        single.side_effect = lambda key: (
            CANDLE_METRICS if key == "candle_metrics" else CANDLE_SIMPLE_SCORE
        )
        multi.side_effect = lambda key: CANDLE_MULTIPLE_SCORE
        yield


def test_body():
    """Test the calculation of candle body size."""
    candle = Candle(open=10, high=15, low=9, close=12)
    if candle.body() != 2.0:
        raise AssertionError("Expected candle.body() == 2.0")


def test_upper_shadow():
    """Test the calculation of upper shadow size."""
    candle = Candle(open=10, high=15, low=9, close=12)
    if candle.upper_shadow() != 3.0:
        raise AssertionError("Expected candle.upper_shadow() == 3.0")


def test_lower_shadow():
    """Test the calculation of lower shadow size."""
    candle = Candle(open=10, high=15, low=9, close=12)
    if candle.lower_shadow() != 1.0:
        raise AssertionError("Expected candle.lower_shadow() == 1.0")


@pytest.mark.parametrize(
    "open_,close_,expected",
    [
        (10, 12, True),
        (12, 10, False),
    ],
)
def test_is_bullish(open_, close_, expected):
    """Test bullish pattern detection based on open and close prices."""
    candle = Candle(open=open_, high=13, low=9, close=close_)
    if candle.is_bullish() is not expected:
        raise AssertionError(f"Expected is_bullish() to be {expected}")


@pytest.mark.parametrize(
    "open_,close_,expected",
    [
        (12, 10, True),
        (10, 12, False),
    ],
)
def test_is_bearish(open_, close_, expected):
    """Test bearish pattern detection based on open and close prices."""
    candle = Candle(open=open_, high=13, low=9, close=close_)
    if candle.is_bearish() is not expected:
        raise AssertionError(f"Expected is_bearish() to be {expected}")


def test_is_hammer():
    """Test that a candle is not identified as hammer incorrectly."""
    c = Candle(open=10, high=10.3, low=9.0, close=10.2)
    if c.is_hammer() is not False:
        raise AssertionError("Expected is_hammer() to be False")


def test_is_shooting_star():
    """Test that a candle is not identified as shooting star incorrectly."""
    c = Candle(open=10.2, high=11.0, low=10.1, close=10.4)
    if c.is_shooting_star() is not False:
        raise AssertionError("Expected is_shooting_star() to be False")


def test_is_marubozu():
    """Test that a candle is not identified as marubozu incorrectly."""
    c = Candle(open=10, high=10.2, low=9.8, close=10.1)
    if c.is_marubozu() is not False:
        raise AssertionError("Expected is_marubozu() to be False")


def test_close_position():
    """Test the result of the close_position method on a Candle instance."""
    c = Candle(open=10, high=12, low=8, close=11)
    result = c.close_position()
    if abs(result - 0.75) >= 1e-6:
        raise AssertionError("Expected close_position() to be approximately 0.75")


def test_score_detection():
    """Test the score detection based on candle attributes."""
    c = Candle(open=10, high=10.3, low=9.0, close=10.2)
    if c.score() != 0.8:
        raise AssertionError("Expected score to be 0.8")


def test_multicandle_detect_pattern():
    """Test multi-candle pattern detection with a known bullish engulfing pattern."""
    c0 = Candle(open=11.0, high=11.5, low=10.5, close=11.4)
    c1 = Candle(open=12.0, high=12.0, low=11.0, close=11.1)
    c2 = Candle(open=11.0, high=12.2, low=11.0, close=12.1)
    if c1.detect_pattern() != "marubozu_bearish":
        raise AssertionError("Expected marubozu_bearish pattern")
    if c2.detect_pattern() != "marubozu_bullish":
        raise AssertionError("Expected marubozu_bullish pattern")
    result = MultiCandlePattern.detect_pattern([c0, c1, c2])
    if result != "bullish_engulfing":
        raise AssertionError("Expected bullish_engulfing pattern")


def test_score_default_when_no_pattern():
    """Test fallback score and pattern name when no strong pattern is detected."""
    c = Candle(open=10, high=10.1, low=9.9, close=10)
    if c.detect_pattern() != "dragonfly_doji":
        raise AssertionError("Expected dragonfly_doji pattern")
    if c.score() != 0.85:
        raise AssertionError("Expected score to be 0.85")


def test_multicandle_detect_pattern_fewer_than_3():
    """Test that patterns are not detected with fewer than three candles."""
    c1 = Candle(open=10, high=11, low=9, close=10.5)
    c2 = Candle(open=10.2, high=11.1, low=9.5, close=10.8)
    result_1 = MultiCandlePattern.detect_pattern([])
    result_2 = MultiCandlePattern.detect_pattern([c1])
    result_3 = MultiCandlePattern.detect_pattern([c1, c2])
    if result_1 is not None:
        raise AssertionError("Expected result_1 to be None")
    if result_2 is not None:
        raise AssertionError("Expected result_2 to be None")
    if result_3 is not None:
        raise AssertionError("Expected result_3 to be None")
