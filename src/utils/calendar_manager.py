"""Utilities for building market calendars and handling important event dates.

Provides a function to construct trading calendars and extract structured event metadata.
"""

import datetime
from typing import Any, List, Tuple

import holidays  # type: ignore
import pandas_market_calendars as mcal  # type: ignore

from market_data.utils.datetime.event_dates import EventDates
from utils.parameters import ParameterLoader


# pylint: disable=too-few-public-methods
class CalendarManager:
    """Builds market calendars and parses important event dates for holiday-aware pipelines."""

    _PARAMS = ParameterLoader()
    _EXCHANGE_DEFAULT = _PARAMS.get("exchange_default")
    _EVENT_DATES_FILEPATH = _PARAMS.get("event_dates_filepath")
    _HOLIDAY_COUNTRY = _PARAMS.get("holiday_country")
    _DATE_FORMAT = _PARAMS.get("date_format")

    @staticmethod
    def build_market_calendars() -> (
        Tuple[Any, list[datetime.date], List[datetime.date]]
    ):
        """Build market calendar, holiday set, and convert event days to datetime.date objects."""
        current_year = datetime.datetime.now().year
        years = list(range(current_year - 20, current_year + 1))
        calendar = mcal.get_calendar(CalendarManager._EXCHANGE_DEFAULT)
        us_holidays = list(
            holidays.country_holidays(
                CalendarManager._HOLIDAY_COUNTRY, years=years
            ).keys()
        )
        event_dates = EventDates(CalendarManager._EVENT_DATES_FILEPATH)
        fed_event_dates = event_dates.get_all_fed_event_days()
        return calendar, us_holidays, fed_event_dates
