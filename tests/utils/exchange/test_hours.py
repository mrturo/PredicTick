"""Unit tests for the Hours class in utils.exchange.hours.

This module verifies the correct validation and encapsulation of trading session times
by the Hours class, including format checking, logical constraints, and error handling
for invalid input scenarios.
"""

import json

import pytest  # type: ignore

from src.utils.exchange.hours import Hours


def test_hours_valid_open_and_close():
    """Valid session with both open and close times."""
    hours = Hours("09:30", "16:00")
    if hours.open != "09:30":
        raise AssertionError("Expected open to be '09:30'")
    if hours.close != "16:00":
        raise AssertionError("Expected close to be '16:00'")
    received_str: str = json.dumps(hours.to_json(), ensure_ascii=False)
    expected_str: str = '{"open": "09:30", "close": "16:00"}'
    if received_str != expected_str:
        raise AssertionError(
            f"Expected json to be: {expected_str}. Received was: {received_str}"
        )


def test_hours_valid_open_only():
    """Valid session with only open time."""
    hours = Hours("09:30", None)
    if hours.open != "09:30":
        raise AssertionError("Expected open to be '09:30'")
    if hours.close != "00:00":
        raise AssertionError("Expected close to be '00:00'")


def test_hours_valid_close_only1():
    """Valid session with only close time."""
    hours = Hours(None, "16:00")
    if hours.open != "00:00":
        raise AssertionError("Expected open to be '00:00'")
    if hours.close != "16:00":
        raise AssertionError("Expected close to be '16:00'")


def test_hours_valid_close_only2():
    """Valid session with only close time."""
    hours = Hours(" ", "16:00")
    if hours.open != "00:00":
        raise AssertionError("Expected open to be '00:00'")
    if hours.close != "16:00":
        raise AssertionError("Expected close to be '16:00'")


def test_hours_close_midnight_allows_open_after():
    """Close at 00:00 allows open to be after."""
    hours = Hours("22:00", "00:00")
    if hours.open != "22:00":
        raise AssertionError("Expected open to be '22:00'")
    if hours.close != "00:00":
        raise AssertionError("Expected close to be '00:00'")


def test_hours_invalid_both_none():
    """Raises error if both open and close are None."""
    hours = Hours(None, None)
    if hours.open != "00:00":
        raise AssertionError("Expected open to be '00:00'")
    if hours.close != "00:00":
        raise AssertionError("Expected close to be '00:00'")


def test_hours_open_after_close_raises():
    """Raises error if open is after close (and close != 00:00)."""
    with pytest.raises(ValueError, match="`open` must be <= `close`"):
        Hours("16:00", "09:30")


def test_hours_invalid_format():
    """Raises error for invalid time format."""
    with pytest.raises(ValueError, match="must be in 'HH:MM' format"):
        Hours("930", "16:00")


def test_hours_out_of_bounds():
    """Raises error for times outside valid hour/minute bounds."""
    with pytest.raises(ValueError, match="valid time between 00:00 and 23:59"):
        Hours("24:00", "16:00")


def test_hours_non_string_input():
    """Raises error if time inputs are not strings or None."""
    with pytest.raises(TypeError, match="`open` must be a string or None"):
        Hours(930, "16:00")  # type: ignore
    with pytest.raises(TypeError, match="`close` must be a string or None"):
        Hours("09:30", 1600)  # type: ignore
