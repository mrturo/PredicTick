"""
Unit tests for the interval module.

This test suite verifies the behavior of interval validation,
resolution, and access using mocked configuration parameters.
"""

# pylint: disable=protected-access

import pytest

from utils import interval as interval_module
from utils.interval import IntervalConverter, IntervalProvider, IntervalValidator


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

    Emula el comportamiento de ParameterLoader en pruebas unitarias.
    """

    def __init__(self, values: dict):
        """Inicializa el mock con un diccionario de parámetros simulados."""
        self._values = values

    def get(self, key: str):
        """
        Devuelve el diccionario de intervalos si la clave es 'interval',
        o un diccionario vacío para otras claves.
        """
        if key == "interval":
            return self._values
        return {}

    def __repr__(self) -> str:
        """Representación legible del contenido del mock."""
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
        "src.utils.interval.ParameterLoader",
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
        "",  # empty
        "-1h",  # negative
    ],
)
def test_interval_converter_invalid(invalid_interval):
    """Test invalid interval strings raise ValueError in IntervalConverter."""
    with pytest.raises(ValueError):
        IntervalConverter.to_minutes(invalid_interval)
