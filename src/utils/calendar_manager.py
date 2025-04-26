"""
Utilities for building market calendars and handling important event dates.

Provides a function to construct trading calendars and extract structured holiday/event metadata.
"""

import datetime
from typing import Any, List, Tuple

import holidays
import pandas_market_calendars as mcal

from utils.parameters import ParameterLoader


# pylint: disable=too-few-public-methods
class CalendarManager:
    """Builds market calendars and parses important event dates for holiday-aware pipelines."""

    _PARAMS = ParameterLoader()
    _CALENDAR_NAME = _PARAMS.get("market_calendar_name")
    _HOLIDAY_COUNTRY = _PARAMS.get("holiday_country")
    _DATE_FORMAT = _PARAMS.get("date_format")

    @staticmethod
    def build_market_calendars(
        fed_event_days: List[str],
    ) -> Tuple[Any, holidays.HolidayBase, List[datetime.date]]:
        """
        Build market calendar, holiday set, and convert FED event days to datetime.date objects.

        Args:
            fed_event_days (List[str]): List of FED event date strings.

        Returns:
            Tuple[Any, holidays.HolidayBase, List[datetime.date]]: Market calendar,
                national holidays, and parsed FED event dates.
        """

        calendar = mcal.get_calendar(CalendarManager._CALENDAR_NAME)
        us_holidays = holidays.country_holidays(CalendarManager._HOLIDAY_COUNTRY)

        fed_events = []
        for date_str in fed_event_days:
            try:
                parsed_date = datetime.datetime.strptime(
                    date_str, CalendarManager._DATE_FORMAT
                ).date()
                fed_events.append(parsed_date)
            except ValueError as error:
                raise ValueError(
                    f"Invalid date format for FED event: {date_str}"
                ) from error

        return calendar, us_holidays, fed_events
