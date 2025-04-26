"""Tests for the calendar utility module."""

import datetime

import holidays
import pandas_market_calendars as mcal
import pytest

from utils.calendar_manager import CalendarManager


def test_build_market_calendars_return_types():
    """Test that build_market_calendars returns expected object types."""
    fed_dates = ["2023-01-25", "2023-03-22"]
    calendar, us_holidays, fed_events = CalendarManager.build_market_calendars(
        fed_dates
    )

    expected_calendar_type = type(mcal.get_calendar("NYSE"))

    assert isinstance(calendar, expected_calendar_type)  # nosec
    assert isinstance(us_holidays, holidays.HolidayBase)  # nosec
    assert isinstance(fed_events, list)  # nosec
    assert all(isinstance(date, datetime.date) for date in fed_events)  # nosec


def test_fed_event_dates_conversion():
    """Test that FED date strings are converted to datetime.date objects."""
    fed_dates = ["2023-06-14", "2023-07-26"]
    _, _, fed_events = CalendarManager.build_market_calendars(fed_dates)

    expected_dates = [datetime.date(2023, 6, 14), datetime.date(2023, 7, 26)]

    assert fed_events == expected_dates  # nosec


def test_empty_fed_event_list():
    """Test that an empty input list returns an empty result."""
    fed_dates = []
    _, _, fed_events = CalendarManager.build_market_calendars(fed_dates)

    assert not fed_events  # nosec


def test_invalid_date_format():
    """Test that invalid date strings raise a ValueError."""
    fed_dates = ["2023/01/25", "not-a-date"]
    with pytest.raises(ValueError):
        CalendarManager.build_market_calendars(fed_dates)
