"""Unit tests for the IntervalConverter and Validator classes.

This module contains parameterized and individual unit tests that validate the behavior
of interval parsing, conversion, simplification, and validation functions using the
IntervalConverter and Validator utilities. The tests avoid direct assert statements and
instead raise AssertionError with explicit messages upon failure."""

# pylint: disable=protected-access

from unittest.mock import patch

import pandas as pd  # type: ignore
import pytest  # type: ignore

from src.market_data.utils.intervals.interval_converter import \
    IntervalConverter
from src.market_data.utils.validation.validator import Validator


@pytest.mark.parametrize(
    "interval,expected",
    [
        ("60min", 60),
        ("120m", 120),
        ("1440min", 1440),
        ("10080m", 10080),
        ("43200m", 43200),
        ("525600m", 525600),
    ],
)
def test_to_minutes(interval, expected):
    """Convert interval string to total minutes."""
    result = IntervalConverter.to_minutes(interval)
    if result != expected:  # noqa: PT011
        raise AssertionError(f"{interval=} → {result}, expected {expected}")


@pytest.mark.parametrize(
    "interval,expected",
    [
        ("60min", "1hour"),
        ("120m", "2h"),
        ("1440min", "1day"),
        ("10080m", "1wk"),
        ("43200m", "1mo"),
        ("525600m", "1y"),
    ],
)
def test_simplify_to_full_units(interval, expected):
    """Simplify interval string to a more human-readable format."""
    result = IntervalConverter.simplify(interval)
    if result != expected:  # noqa: PT011
        raise AssertionError(f"{interval=} simplified to {result}, expected {expected}")


@pytest.mark.parametrize(
    "interval_a,interval_b,expected_label",
    [
        ("2h", "1h", "2:1"),
        ("90min", "30min", "3:1"),
        ("1d", "12h", "2:1"),
        ("3wk", "1wk", "3:1"),
    ],
)
def test_get_ratio(interval_a, interval_b, expected_label):
    """Compute ratio label based on the GCD of two intervals."""
    ratio = IntervalConverter.get_ratio(interval_a, interval_b)
    if ratio["label"] != expected_label:  # noqa: PT011
        raise AssertionError(f"Expected label {expected_label}, got {ratio['label']}")


@pytest.mark.parametrize(
    "interval,expected",
    [
        ("60min", "min"),
        ("120m", "min"),
        ("1440min", "min"),
        ("10080m", "min"),
        ("43200m", "min"),
        ("525600m", "min"),
    ],
)
def test_to_pandas_floor_freq(interval, expected):
    """Convert interval string to a pandas-compatible frequency alias."""
    result = IntervalConverter.to_pandas_floor_freq(interval)
    if result != expected:  # noqa: PT011
        raise AssertionError(f"{interval=} mapped to {result}, expected {expected}")


def test_extract_suffix_valid():
    """Extract suffix from a well-formed interval."""
    suffix = IntervalConverter._extract_suffix("15min")
    if suffix != "min":  # noqa: PT011
        raise AssertionError(f"Expected 'min', got {suffix}")


def test_extract_suffix_none():
    """Return None when interval is None."""
    if IntervalConverter._extract_suffix(None) is not None:  # noqa: PT011
        raise AssertionError("Expected None when interval is None")


def test_extract_suffix_invalid():
    """Raise ValueError for malformed interval."""
    with pytest.raises(ValueError):
        IntervalConverter._extract_suffix("bad")


def test_extract_suffix_match_none():
    """Raise ValueError when regex match fails for numeric-only input."""
    with pytest.raises(ValueError, match="Invalid interval format: 9999"):
        IntervalConverter._extract_suffix("9999")


def test_to_minutes_invalid_suffix():
    """Raise ValueError for unsupported unit suffix in interval."""
    with pytest.raises(ValueError):
        IntervalConverter.to_minutes("3hr")


def test_to_pandas_floor_freq_invalid():
    """Raise ValueError for unrecognized pandas frequency suffix."""
    with pytest.raises(ValueError):
        IntervalConverter.to_pandas_floor_freq("7xyz")


def test_check_volume_logs_warning():
    """Log a warning when volume equals zero for any row."""
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=3, freq="1D"),
            "volume": [100, 0, 150],
        }
    )
    symbol = "TESTSYM"
    with patch("src.utils.io.logger.Logger.warning") as mock_warning:
        Validator._check_volume_and_time(symbol, df, "1d")
        mock_warning.assert_called_once()
        called_msg = mock_warning.call_args[0][0]
        if "zero volume" not in called_msg or symbol not in called_msg:  # noqa: PT011
            raise AssertionError("Expected warning message not found or incomplete")


@pytest.mark.parametrize("interval", [None, "", "foo", "123xyz"])
def test_parse_suffix_unable_to_parse(interval):
    """Raise ValueError when interval suffix cannot be parsed."""
    with pytest.raises(ValueError, match="Unable to parse interval"):
        IntervalConverter._parse_suffix(interval)


@pytest.mark.parametrize("interval", [None, "", "   "])
def test_to_minutes_returns_zero_when_interval_or_suffix_none(interval):
    """Return 0 when interval or its suffix is None or empty."""
    result = IntervalConverter.to_minutes(interval)
    if result != 0:  # noqa: PT011
        raise AssertionError(f"Expected 0 for {interval=}, got {result}")


def test_to_minutes_keyerror_unsupported_suffix(monkeypatch):
    """Force KeyError by removing a valid suffix to test exception handling."""
    if "h" not in IntervalConverter._UNIT_TO_MINUTES:
        raise AssertionError("Key 'h' does not exist in _UNIT_TO_MINUTES")
    monkeypatch.delitem(IntervalConverter._UNIT_TO_MINUTES, "h", raising=True)
    with pytest.raises(ValueError, match="Unsupported unit in interval: h"):
        IntervalConverter.to_minutes("3h")


def test_simplify_returns_none_when_no_divisor(monkeypatch):
    """Force simplify to return None when no divisor matches."""
    monkeypatch.setattr(IntervalConverter, "_SIMPLIFICATION_ORDER", [("hour", 7)])
    result = IntervalConverter.simplify("5min")
    if result is not None:  # noqa: PT011
        raise AssertionError(f"Expected None, got {result}")


@pytest.mark.parametrize("interval", [None, "", "   "])
def test_to_pandas_floor_freq_invalid_format(interval):
    """Raise ValueError when interval format is invalid or empty."""
    with pytest.raises(ValueError, match="Invalid interval format"):
        IntervalConverter.to_pandas_floor_freq(interval)


def test_to_pandas_floor_freq_keyerror_unsupported_suffix(monkeypatch):
    """Remove valid suffix from mapping to provoke a KeyError and validate handling."""
    if "h" not in IntervalConverter._UNIT_TO_PANDAS_FREQ:
        raise AssertionError("'h' not in _UNIT_TO_PANDAS_FREQ")
    monkeypatch.delitem(IntervalConverter._UNIT_TO_PANDAS_FREQ, "h", raising=True)
    with pytest.raises(ValueError, match="Unsupported unit: h"):
        IntervalConverter.to_pandas_floor_freq("3h")
