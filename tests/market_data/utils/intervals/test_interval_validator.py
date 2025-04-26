"""Unit tests for IntervalValidator (no direct asserts)."""

# pylint: disable=protected-access

import pytest  # type: ignore

from src.market_data.utils.intervals.interval_validator import \
    IntervalValidator


@pytest.mark.parametrize(
    "valid_interval",
    [
        "1min",
        "5hour",
        "10day",
        "3week",
        "2month",
        "1year",
        "15m",
        "2h",
        "30d",
        "1wk",
        "4mo",
        "6y",
    ],
)
def test_valid_intervals(valid_interval):
    """Valid interval strings must be accepted."""
    result = IntervalValidator.is_valid(valid_interval)
    if result is not True:  # noqa: PT011
        raise AssertionError(f"Expected True for valid interval: {valid_interval}")


@pytest.mark.parametrize(
    "invalid_interval",
    [
        "0min",
        "-1day",
        "10",
        "hour5",
        "",
        " ",
        "5minute",
        "1seconds",
        "min",
        "5 d",
        "5days",
        "1.5h",
        "1.0min",
        "abcmin",
        "5hr",
        "h1",
    ],
)
def test_invalid_intervals(invalid_interval):
    """Invalid interval strings must be rejected."""
    result = IntervalValidator.is_valid(invalid_interval)
    if result is not False:  # noqa: PT011
        raise AssertionError(f"Expected False for invalid interval: {invalid_interval}")


def test_interval_with_spaces():
    """Leading and trailing whitespace is ignored."""
    if IntervalValidator.is_valid(" 5d ") is not True:  # noqa: PT011
        raise AssertionError("Expected True for ' 5d ' with surrounding spaces")


def test_empty_string():
    """An empty string returns False."""
    if IntervalValidator.is_valid("") is not False:  # noqa: PT011
        raise AssertionError("Expected False for empty string")


def test_non_digit_prefix():
    """A non-digit prefix returns False."""
    if IntervalValidator.is_valid("xday") is not False:  # noqa: PT011
        raise AssertionError("Expected False for 'xday'")
