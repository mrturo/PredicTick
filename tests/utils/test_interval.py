"""
Unit tests for the interval module.

This test suite verifies the behavior of interval validation,
resolution, and access using mocked configuration parameters.
"""

# pylint: disable=protected-access

import re

import pytest

from utils import interval as interval_module
from utils.interval import (
    Interval,
    IntervalConverter,
    IntervalProvider,
    IntervalValidator,
)


@pytest.mark.parametrize(
    "interval_str,expected",
    [
        ("1min", True),
        ("5m", True),
        ("2hour", True),
        ("3h", True),
        ("1day", True),
        ("10d", True),
        ("1week", True),
        ("2wk", True),
        ("1month", True),
        ("6mo", True),
        ("1year", True),
        ("3y", True),
        ("-3y", False),
        ("3.5y", False),
        ("0y", False),
        ("60sec", False),
        ("h1", False),
        ("1", False),
        ("", False),
    ],
)
def test_interval_validator(interval_str, expected):
    """Test that IntervalValidator correctly validates interval strings."""
    result = IntervalValidator.is_valid(interval_str)
    if result is not expected:
        raise AssertionError(f"Expected {expected} for '{interval_str}', got {result}")


class MockLoader:
    """
    Helper mock class for interval parameter loading.

    Emulates the behavior of ParameterLoader in unit tests.
    """

    def __init__(self, values: dict):
        """Initializes the mock with a dictionary of simulated parameters."""
        self._values = values

    def get(self, key: str):
        """
        Returns the intervals dictionary if the key is 'interval',
        or an empty dictionary for other keys.
        """
        if key == "interval":
            return self._values
        return {}

    def __repr__(self) -> str:
        """Human-readable representation of the mock contents."""
        return f"MockLoader(values={self._values})"


def test_provider_fallback_interval(monkeypatch):
    """Test valid fallback interval when primary is missing."""
    monkeypatch.setattr(
        "src.utils.interval.ParameterLoader",
        lambda: MockLoader({"market_enriched_data": "1h"}),
    )
    provider = IntervalProvider()
    result = provider.resolve("market_raw_data", "market_enriched_data")
    if result != "1h":
        raise AssertionError(f"Expected '1h', got {result}")


def test_provider_invalid_primary(monkeypatch):
    """Test fallback used when primary interval is invalid."""
    monkeypatch.setattr(
        "src.utils.interval.ParameterLoader",
        lambda: MockLoader(
            {"market_raw_data": "invalid", "market_enriched_data": "1h"}
        ),
    )
    provider = IntervalProvider()
    result = provider.resolve("market_raw_data", "market_enriched_data")
    if result != "1h":
        raise AssertionError(f"Expected '1h', got {result}")


def test_interval_api_enriched(monkeypatch):
    """Test Interval.market_enriched_data API method."""
    monkeypatch.setattr(
        "utils.interval.ParameterLoader",
        lambda: MockLoader({"market_enriched_data": "1h"}),
    )
    interval_module.Interval._provider = interval_module.IntervalProvider()
    result = interval_module.Interval.market_enriched_data()
    if result != "1h":
        raise AssertionError(f"Expected '1h', got {result}")


@pytest.mark.parametrize(
    "interval_str,expected_minutes",
    [
        ("1min", 1),
        ("5m", 5),
        ("1hour", 60),
        ("2h", 120),
        ("1day", 1440),
        ("3d", 4320),
        ("1week", 10080),
        ("2wk", 20160),
        ("1month", 43200),
        ("3mo", 129600),
        ("1year", 525600),
        ("2y", 1051200),
    ],
)
def test_interval_converter_valid(interval_str, expected_minutes):
    """Test valid conversions of interval strings to minutes."""
    result = IntervalConverter.to_minutes(interval_str)
    if result != expected_minutes:
        raise AssertionError(
            f"{interval_str} -> Expected {expected_minutes}, got {result}"
        )


@pytest.mark.parametrize(
    "invalid_interval",
    [
        "0m",  # zero
        "3.5h",  # float
        "h1",  # malformed
        "10sec",  # unsupported unit
        "1",  # missing unit
        "-1h",  # negative
    ],
)
def test_interval_converter_invalid(invalid_interval):
    """Test invalid interval strings raise ValueError in IntervalConverter."""
    with pytest.raises(ValueError):
        IntervalConverter.to_minutes(invalid_interval)


def test_interval_converter_empty_returns_zero():
    """Test that empty interval string returns 0 minutes (not an error)."""
    result = IntervalConverter.to_minutes("")
    if result != 0:
        raise AssertionError(f"Expected 0 for empty interval, got {result}")


def test_to_minutes_invalid_format_rejects_unknown_suffix():
    """Checks that '10sec' raises error due to invalid format, not unknown unit."""
    with pytest.raises(ValueError, match="Invalid interval format: 10sec"):
        IntervalConverter.to_minutes("10sec")


def test_simplify_invalid_format():
    """Forces a ValueError due to invalid format in simplify()."""
    with pytest.raises(ValueError, match="Invalid interval format: bad"):
        IntervalConverter.simplify("bad")


def test_simplify_returns_original_if_no_simplification():
    """Checks that simplify returns the same value if no simplification is possible."""
    result = IntervalConverter.simplify("1min")
    if result != "1min":
        raise AssertionError(f"Expected '1min', got {result}")


def test_get_ratio_invalid_format_rejected_early():
    """Checks that '0m' raises ValueError for invalid format instead of returning 0 minutes."""
    with pytest.raises(ValueError, match="Invalid interval format: 0m"):
        IntervalConverter.get_ratio("0m", "1h")


def test_provider_invalid_interval_raises(monkeypatch):
    """Exercises the ValueError raise in _get_interval_from_key with an invalid interval."""
    monkeypatch.setattr(
        "utils.interval.ParameterLoader", lambda: {"interval": {"bad_key": "xxx"}}
    )
    provider = IntervalProvider()
    with pytest.raises(ValueError, match="Invalid Interval for 'bad_key': xxx"):
        provider._get_interval_from_key("bad_key")


def test_provider_returns_none(monkeypatch):
    """Exercises the return of None in _get_interval_from_key when the key does not exist."""
    monkeypatch.setattr("utils.interval.ParameterLoader", lambda: {"interval": {}})
    provider = IntervalProvider()
    result = provider._get_interval_from_key("missing_key")
    if result is not None:
        raise AssertionError("Expected None when key is missing")


def test_provider_resolve_primary_invalid_raises(monkeypatch):
    """Checks that an invalid primary raises ValueError directly, without trying the fallback."""
    monkeypatch.setattr(
        "utils.interval.ParameterLoader", lambda: {"interval": {"bad": "xxx"}}
    )
    provider = IntervalProvider()
    with pytest.raises(ValueError, match="Invalid Interval for 'bad': xxx"):
        provider.resolve("bad", "also_bad")


def test_validate_hierarchy_raw_greater(monkeypatch):
    """Forces ValueError when raw > enriched in minutes."""
    monkeypatch.setattr(
        "utils.interval.ParameterLoader",
        lambda: {"interval": {"market_raw_data": "2h", "market_enriched_data": "1h"}},
    )
    Interval._provider = IntervalProvider()
    with pytest.raises(ValueError, match="Raw interval.*must not be greater"):
        Interval.validate_market_interval_hierarchy()


def test_validate_hierarchy_not_divisible(monkeypatch):
    """Forces ValueError when enriched % raw != 0."""
    monkeypatch.setattr(
        "utils.interval.ParameterLoader",
        lambda: {"interval": {"market_raw_data": "45m", "market_enriched_data": "2h"}},
    )
    Interval._provider = IntervalProvider()
    with pytest.raises(ValueError, match="must be divisible by raw interval"):
        Interval.validate_market_interval_hierarchy()


def test_to_minutes_unsupported_unit_raises_keyerror(monkeypatch):
    """Forces KeyError for suffix recognized by regex but not defined."""
    interval = "1fortnight"
    monkeypatch.setattr(
        IntervalValidator,
        "PATTERN",
        re.compile(r"^\d+(min|hour|day|week|month|year|m|h|d|wk|mo|y|fortnight)$"),
    )
    with pytest.raises(ValueError, match="Unsupported unit in interval: fortnight"):
        IntervalConverter.to_minutes(interval)


def test_simplify_falls_back_to_original_when_no_unit_fits():
    """Checks that simplify returns the original if no exact unit match is found."""
    result = IntervalConverter.simplify("7m")
    if result != "7m":
        raise AssertionError(f"Expected '7m', got {result}")


def test_provider_resolve_fallback_invalid_raises(monkeypatch):
    """
    Checks that if the primary interval does not exist and the fallback is invalid,
    the ValueError is caught and a final one is raised due to resolution failure.
    """
    monkeypatch.setattr(
        "utils.interval.ParameterLoader",
        lambda: {"interval": {"market_enriched_data": "invalid"}},
    )
    provider = IntervalProvider()
    with pytest.raises(ValueError, match="Undefined interval: market_raw_data"):
        provider.resolve("market_raw_data", "market_enriched_data")


def test_provider_resolve_uses_fallback_successfully(monkeypatch):
    """
    Checks that if the primary does not exist and the fallback is valid,
    it is simplified and returned correctly.
    """
    monkeypatch.setattr(
        "utils.interval.ParameterLoader",
        lambda: {"interval": {"market_enriched_data": "2h"}},
    )
    provider = IntervalProvider()
    result = provider.resolve("market_raw_data", "market_enriched_data")
    if result != "2h":
        raise AssertionError(f"Expected '2h' from fallback, got {result}")


@pytest.mark.parametrize(
    "interval_a, interval_b, exp_label, exp_ant, exp_cons, exp_value",
    [
        # minutos_a < minutos_b
        ("30m", "1h", "1:2", 1, 2, 0.5),
        ("45m", "90m", "1:2", 1, 2, 0.5),
        # minutos_a > minutos_b
        ("2h", "30m", "4:1", 4, 1, 4.0),
        ("90m", "45m", "2:1", 2, 1, 2.0),
        # minutos_a == minutos_b
        ("1d", "1day", "1:1", 1, 1, 1.0),
    ],
)
def test_get_ratio_success(
    interval_a, interval_b, exp_label, exp_ant, exp_cons, exp_value
):
    """
    Verifica que get_ratio:
    - Simplifique correctamente la razón usando el MCD.
    - Devuelva las claves y valores esperados.
    """
    result = IntervalConverter.get_ratio(interval_a, interval_b)

    if result["label"] != exp_label:
        raise AssertionError(
            f"label -> esperado '{exp_label}', obtenido '{result['label']}'"
        )
    if result["antecedent"] != exp_ant:
        raise AssertionError(
            f"antecedent -> esperado {exp_ant}, obtenido {result['antecedent']}"
        )
    if result["consequent"] != exp_cons:
        raise AssertionError(
            f"consequent -> esperado {exp_cons}, obtenido {result['consequent']}"
        )
    # para evitar problemas de punto flotante comparamos con una tolerancia pequeña
    if abs(result["value"] - exp_value) > 1e-9:
        raise AssertionError(
            f"value -> esperado {exp_value}, obtenido {result['value']}"
        )
