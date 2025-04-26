"""Test suite for IntervalProvider logic with fallback and validation scenarios."""

# pylint: disable=protected-access

from unittest.mock import patch

import pytest  # type: ignore

from src.market_data.utils.intervals.interval_provider import IntervalProvider
from src.market_data.utils.intervals.interval_validator import \
    IntervalValidator


@patch.object(IntervalValidator, "is_valid", return_value=True)
@patch(
    "src.market_data.utils.intervals.interval_provider.IntervalProvider._INTERVAL",
    new_callable=dict,
)
@pytest.mark.parametrize(
    "config,primary,fallback,expected",
    [
        ({"primary": "60min", "fallback": "120min"}, "primary", "fallback", "1hour"),
        ({"primary": "", "fallback": "120min"}, "primary", "fallback", "2hour"),
        ({"primary": None, "fallback": "1440min"}, "primary", "fallback", "1day"),
    ],
)
def test_resolve_with_fallback(
    mock_interval, _mock_is_valid, config, primary, fallback, expected
):
    """Ensure resolve() returns fallback interval if primary is missing or invalid."""
    mock_interval.update(config)
    result = IntervalProvider.resolve(primary, fallback)
    if result != expected:  # noqa: PT011
        raise AssertionError(f"Expected {expected}, got {result}")


@patch(
    "src.market_data.utils.intervals.interval_provider.IntervalProvider._INTERVAL",
    new={"primary": "", "fallback": "  "},
)
def test_resolve_invalid_primary_and_fallback():
    """Raise ValueError when both primary and fallback intervals are invalid."""
    with pytest.raises(ValueError, match="Undefined interval: primary"):
        IntervalProvider.resolve("primary", "fallback")


@patch.object(IntervalValidator, "is_valid", return_value=False)
@patch(
    "src.market_data.utils.intervals.interval_provider.IntervalProvider._INTERVAL",
    new={"primary": "badint"},
)
def test_invalid_interval_raises(_mock_is_valid):
    """Raise ValueError when interval exists but fails validation."""
    with pytest.raises(ValueError, match="Invalid Interval for 'primary'"):
        IntervalProvider._get_interval_from_key("primary")


@patch.object(IntervalValidator, "is_valid", side_effect=ValueError("Invalid interval"))
@patch(
    "src.market_data.utils.intervals.interval_provider.IntervalProvider._INTERVAL",
    new={"primary": "invalid"},
)
def test_get_interval_handles_value_error(_mock_is_valid):
    """Return None when _get_interval_from_key raises ValueError."""
    result = IntervalProvider._get_interval("primary")
    if result is not None:  # noqa: PT011
        raise AssertionError(f"Expected None, got {result}")
