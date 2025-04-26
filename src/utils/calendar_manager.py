"""
Utilities for building market calendars and handling important event dates.

Provides a function to construct trading calendars and extract structured holiday/event metadata.
"""

import datetime
from typing import Any, List, Tuple

import holidays  # type: ignore
import pandas_market_calendars as mcal  # type: ignore

from utils.parameters import ParameterLoader


# pylint: disable=too-few-public-methods
class CalendarManager:
    """Builds market calendars and parses important event dates for holiday-aware pipelines."""

    _PARAMS = ParameterLoader()
    _CALENDAR_NAME = _PARAMS.get("market_calendar_name")
    _EVENT_DATES_FILEPATH = _PARAMS.get("event_dates_filepath")
    _HOLIDAY_COUNTRY = _PARAMS.get("holiday_country")
    _DATE_FORMAT = _PARAMS.get("date_format")

    @staticmethod
    def build_market_calendars() -> (
        Tuple[Any, list[datetime.date], List[datetime.date]]
    ):
        """
        Build market calendar, holiday set, and convert FED event days to datetime.date objects.

        Returns:
            Tuple[Any, holidays.HolidayBase, List[datetime.date]]: Market calendar,
                national holidays, and parsed FED event dates.
        """
        # pylint: disable=import-outside-toplevel
        from utils.event_dates import EventDates

        current_year = datetime.datetime.now().year
        years = list(range(current_year - 20, current_year + 1))
        calendar = mcal.get_calendar(CalendarManager._CALENDAR_NAME)
        us_holidays = list(
            holidays.country_holidays(
                CalendarManager._HOLIDAY_COUNTRY, years=years
            ).keys()
        )

        event_dates = EventDates(CalendarManager._EVENT_DATES_FILEPATH)
        fed_event_dates = event_dates.get_all_fed_event_days()

        return calendar, us_holidays, fed_event_dates
