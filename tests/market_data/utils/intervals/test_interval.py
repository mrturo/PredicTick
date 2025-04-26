"""Test suite for the Interval module in market_data.utils.intervals.

Validates the functionality of:
- Retrieving raw and enriched market intervals via IntervalProvider.
- Enforcing correct hierarchical relationships between intervals using IntervalConverter.

All test cases mock external dependencies and raise AssertionError explicitly
to comply with Bandit security guidelines and avoid use of direct 'assert' statements.
"""

from unittest.mock import patch

import pytest  # type: ignore

from src.market_data.utils.intervals.interval import Interval


@patch("src.market_data.utils.intervals.interval_provider.IntervalProvider.resolve")
def test_market_raw_data_returns_expected(mock_resolve):
    """Test market_raw_data returns the expected value from IntervalProvider."""
    mock_resolve.return_value = "15min"
    result = Interval.market_raw_data()
    if result != "15min":
        raise AssertionError(f"Expected '15min', got {result}")
    mock_resolve.assert_called_once_with("market_raw_data", "market_enriched_data")


@patch("src.market_data.utils.intervals.interval_provider.IntervalProvider.resolve")
def test_market_enriched_data_returns_expected(mock_resolve):
    """Test market_enriched_data returns the expected value from IntervalProvider."""
    mock_resolve.return_value = "1h"
    result = Interval.market_enriched_data()
    if result != "1h":
        raise AssertionError(f"Expected '1h', got {result}")
    mock_resolve.assert_called_once_with("market_enriched_data", "market_raw_data")


@patch("src.market_data.utils.intervals.interval_provider.IntervalProvider.resolve")
@patch(
    "src.market_data.utils.intervals.interval_converter.IntervalConverter.to_minutes"
)
def test_validate_market_interval_hierarchy_valid(mock_to_minutes, mock_resolve):
    """Test validation passes when raw < enriched and enriched is divisible by raw."""
    mock_resolve.side_effect = ["15min", "1h"]
    mock_to_minutes.side_effect = [15, 60]
    Interval.validate_market_interval_hierarchy()


@patch("src.market_data.utils.intervals.interval_provider.IntervalProvider.resolve")
@patch(
    "src.market_data.utils.intervals.interval_converter.IntervalConverter.to_minutes"
)
def test_validate_market_interval_hierarchy_raises_on_raw_greater(
    mock_to_minutes, mock_resolve
):
    """Test validation raises when raw interval is longer than enriched."""
    mock_resolve.side_effect = ["1h", "15min"]
    mock_to_minutes.side_effect = [60, 15]
    with pytest.raises(ValueError, match="Raw interval.*greater.*enriched"):
        Interval.validate_market_interval_hierarchy()


@patch("src.market_data.utils.intervals.interval_provider.IntervalProvider.resolve")
@patch(
    "src.market_data.utils.intervals.interval_converter.IntervalConverter.to_minutes"
)
def test_validate_market_interval_hierarchy_raises_on_non_divisible(
    mock_to_minutes, mock_resolve
):
    """Test validation raises when enriched is not divisible by raw."""
    mock_resolve.side_effect = ["20min", "1h"]
    mock_to_minutes.side_effect = [20, 65]
    with pytest.raises(ValueError, match="Enriched interval.*divisible.*raw"):
        Interval.validate_market_interval_hierarchy()
