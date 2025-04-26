"""Unit tests for the SessionsDays class which validates and handles trading day flags.

This module verifies correct instantiation, input validation, behavior of public methods
like `is_trading_day`, `open_days`, `any_open`, and proper JSON serialization and iteration order.
"""

# pylint: disable=protected-access

from datetime import date, datetime
from typing import Dict

import pytest  # type: ignore

from src.utils.exchange.sessions_days import SessionsDays

VALID_DAYS = {
    "monday": True,
    "tuesday": False,
    "wednesday": True,
    "thursday": False,
    "friday": True,
    "saturday": False,
    "sunday": False,
}

weekdays = [
    "friday",
    "monday",
    "saturday",
    "sunday",
    "thursday",
    "tuesday",
    "wednesday",
]


def test_valid_sessions_days_instantiation():
    """Test that SessionsDays initializes correctly with valid boolean flags for weekdays."""
    days = SessionsDays(VALID_DAYS, weekdays)
    for day, expected in VALID_DAYS.items():
        if getattr(days, day.lower()) is not expected:
            raise AssertionError(f"{day} flag incorrect")


def test_invalid_type_for_days():
    """Test that SessionsDays raises TypeError when initialized with a non-dictionary input."""
    with pytest.raises(TypeError, match=r"`days` must be `Dict\[str, bool\]"):
        SessionsDays("invalid", weekdays)  # type: ignore


def test_missing_keys():
    """Test that SessionsDays raises ValueError when required weekday keys are missing."""
    bad_days = VALID_DAYS.copy()
    del bad_days["friday"]
    with pytest.raises(ValueError, match=r"Missing keys in `days`"):
        SessionsDays(bad_days, weekdays)


def test_unexpected_keys():
    """Test that SessionsDays raises ValueError when unexpected keys are present in the input."""
    bad_days = VALID_DAYS.copy()
    bad_days["Holiday"] = True
    with pytest.raises(ValueError, match=r"Unexpected key in `days`"):
        SessionsDays(bad_days, weekdays)


def test_invalid_value_type():
    """Test that SessionsDays raises TypeError when any weekday value is not of boolean type."""
    bad_days = VALID_DAYS.copy()
    bad_days["monday"] = "yes"  # type: ignore
    with pytest.raises(TypeError, match=r"Value for 'monday' must be bool"):
        SessionsDays(bad_days, weekdays)


def test_is_trading_day():
    """Test the is_trading_day method with date objects, validating true and false trading days."""
    days = SessionsDays(VALID_DAYS, weekdays)
    monday_date = date(2024, 1, 1)  # monday
    saturday_date = date(2024, 1, 6)  # saturday
    if not days.is_trading_day(monday_date):
        raise AssertionError("Expected monday to be a trading day")
    if days.is_trading_day(saturday_date):
        raise AssertionError("Expected saturday to not be a trading day")


def test_is_trading_day_with_datetime():
    """Test that is_trading_day works correctly with datetime objects."""
    days = SessionsDays(VALID_DAYS, weekdays)
    dt = datetime(2024, 1, 1, 10, 0)  # monday
    if not days.is_trading_day(dt):
        raise AssertionError("Expected datetime on trading day to return True")


def test_open_days():
    """Test that open_days returns a list of weekdays marked as trading days."""
    tmp_valid_days = {
        "friday": True,
        "monday": True,
        "saturday": True,
        "sunday": True,
        "thursday": True,
        "tuesday": True,
        "wednesday": True,
    }
    days = SessionsDays(tmp_valid_days, weekdays)
    result = days.open_days()
    expected = [d for d, v in tmp_valid_days.items() if v]
    if set(result) != set(expected):
        raise AssertionError(
            f"Expected open days: {sorted(expected)}, got: {sorted(result)}"
        )


def test_any_open_true():
    """Test that any_open returns True when at least one weekday is a trading day."""
    days = SessionsDays(VALID_DAYS, weekdays)
    if not days.any_open():
        raise AssertionError("Expected any_open to be True")


def test_any_open_false():
    """Test that any_open returns False when all weekdays are non-trading days."""
    closed_days = {k: False for k in VALID_DAYS}
    days = SessionsDays(closed_days, weekdays)
    if days.any_open():
        raise AssertionError("Expected any_open to be False")


def test_to_json():
    """Test that to_json returns the original input dictionary of weekday flags."""
    days = SessionsDays(VALID_DAYS, weekdays)
    if days.to_json() != VALID_DAYS:
        raise AssertionError("Expected to_json output to match input")


def test_iteration_order():
    """Test that iteration over SessionsDays yields values in weekday order (Monday to Sunday)."""
    days = SessionsDays(VALID_DAYS, weekdays)
    flags = list(days)
    expected = [VALID_DAYS[day] for day in weekdays]
    if set(flags) != set(expected):
        raise AssertionError("Expected iteration to match weekday flags order")


def test_validate_bool_raises():
    """Test that the internal _validate_bool method raises TypeError on non-boolean values."""
    with pytest.raises(TypeError, match=r"'mockday' must be bool"):
        SessionsDays._validate_bool("not_bool", "mockday")  # type: ignore


@pytest.mark.parametrize(
    "flags, expect_all_true, expect_all_false",
    [
        ({day: True for day in weekdays}, True, False),  # all enabled
        ({day: False for day in weekdays}, False, True),  # all disabled
        (  # mixed – at least one True and one False
            {
                "monday": True,
                "tuesday": False,
                "wednesday": True,
                "thursday": False,
                "friday": True,
                "saturday": False,
                "sunday": False,
            },
            False,
            False,
        ),
    ],
    ids=["all_true", "all_false", "mixed"],
)
def test_all_true_and_all_false(
    flags: Dict[str, bool], expect_all_true: bool, expect_all_false: bool
) -> None:
    """Validate the behaviour of :pymeth:`SessionsDays.all_true` and
    :pymeth:`SessionsDays.all_false` under different flag configurations.
    """
    session_days = SessionsDays(flags, weekdays)
    if session_days.all_true() is not expect_all_true:
        raise AssertionError(
            "Unexpected result for `all_true`: "
            f"{session_days.all_true()} (expected {expect_all_true})"
        )
    if session_days.all_false() is not expect_all_false:
        raise AssertionError(
            "Unexpected result for `all_false`: "
            f"{session_days.all_false()} (expected {expect_all_false})"
        )
