"""Unit tests for the EventDates utility class used in economic event date handling.

This test module covers date normalization, parsing, and logic for past and future
event determination based on a JSON-configured calendar of economic events."""

# pylint: disable=protected-access

from datetime import date, datetime
from typing import Any, Optional

import pytest  # type: ignore

from src.market_data.utils.datetime.event_dates import EventDates
from src.utils.io.json_manager import JsonManager


@pytest.fixture
def fed_reference_date():
    """Fixture providing a consistent reference datetime for FED date tests."""
    return datetime(2025, 5, 26)


class DummyJsonManager:
    """Mock JsonManager to inject controlled data for testing."""

    @staticmethod
    def exists(_filepath: str) -> bool:
        """Simulate that a given file path exists, always returning True."""
        return True

    @staticmethod
    def load(filepath: Optional[str]) -> Any:
        """Mock loading JSON data, returning valid or invalid FED event days."""
        if filepath:
            if "valid" in filepath:
                return {"fed_event_days": ["2024-01-01", "2025-12-31"]}
            if "invalid" in filepath:
                return {"fed_event_days": ["2024-01-01", "bad-date"]}
        return {}


def monkeypatch_json_manager(monkeypatch, mock_class):
    """Monkeypatch JsonManager's static methods with those from a mock class."""
    monkeypatch.setattr(JsonManager, "load", staticmethod(mock_class.load))
    monkeypatch.setattr(JsonManager, "exists", staticmethod(mock_class.exists))


def test_init_valid_dates(monkeypatch):
    """Test initialization with valid date strings."""
    monkeypatch_json_manager(monkeypatch, DummyJsonManager)
    ed = EventDates("valid_path.json")
    if not ed.fed_event_days == [date(2024, 1, 1), date(2025, 12, 31)]:
        raise AssertionError("Failed to parse and sort valid event dates")


def test_resolve_reference_date_normalization():
    """Test that date and datetime inputs are normalized to midnight."""
    dt = datetime(2025, 5, 26, 15, 45)
    d = datetime(2025, 5, 26)
    result_dt = EventDates._resolve_reference_date(dt)
    result_d = EventDates._resolve_reference_date(d)
    expected = datetime(2025, 5, 26, 0, 0)
    if not (result_dt == expected and result_d == expected):
        raise AssertionError("Reference date normalization failed")


def test_convert_str_dates_to_date_objects_valid():
    """Test conversion of valid date strings to sorted date objects."""
    date_strs = ["2025-12-31", "2024-01-01"]
    result = EventDates._convert_str_dates_to_date_objects(date_strs)
    if not result == [date(2024, 1, 1), date(2025, 12, 31)]:
        raise AssertionError("Date conversion or sorting incorrect")


def test_convert_str_dates_to_date_objects_invalid():
    """Test conversion with invalid date strings returns empty list."""
    result = EventDates._convert_str_dates_to_date_objects(["bad-date"])
    if not result == []:
        raise AssertionError("Expected empty list for invalid date string")


def test_resolve_reference_date_with_date_object():
    """Test normalization when reference date is a `date` (not datetime)."""
    d = date(2025, 5, 26)
    result = EventDates._resolve_reference_date(d)
    expected = datetime(2025, 5, 26, 0, 0)
    if result != expected:
        raise AssertionError("Expected datetime from date with time zeroed")


def test_get_all_fed_event_days(monkeypatch):
    """Test retrieving all FED event days."""
    monkeypatch_json_manager(monkeypatch, DummyJsonManager)
    ed = EventDates("valid_path.json")
    result = ed.get_all_fed_event_days()
    expected = [date(2024, 1, 1), date(2025, 12, 31)]
    if result != expected:
        raise AssertionError("Expected all fed event days as parsed from input")


def test_resolve_reference_date_none(monkeypatch):
    """Test normalization when reference date is None uses current datetime."""
    fixed_now = datetime(2025, 5, 26, 12, 34, 56)

    class MockDatetime(datetime):
        """Subclass of `datetime` that overrides the `now` method to return a fixed datetime."""

        @classmethod
        def now(cls, tz=None):
            """Return a fixed `datetime` instance used to simulate the current time."""
            return fixed_now

    monkeypatch.setattr(
        "src.market_data.utils.datetime.event_dates.datetime", MockDatetime
    )
    result = EventDates._resolve_reference_date(None)
    expected = datetime(2025, 5, 26, 0, 0)
    if result != expected:
        raise AssertionError("Expected datetime.now() normalized to midnight")
