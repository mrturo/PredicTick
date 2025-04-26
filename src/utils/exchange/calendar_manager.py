"""Utilities for building market calendars and handling important event dates.

Provides a function to construct trading calendars and extract structured event metadata.
"""

import datetime
from typing import Any, List, Tuple

import holidays  # type: ignore
import pandas_market_calendars as mcal  # type: ignore
from pandas_market_calendars.market_calendar import \
    MarketCalendar  # type: ignore

from src.market_data.utils.datetime.event_dates import EventDates
from src.utils.config.parameters import ParameterLoader


# pylint: disable=too-few-public-methods
class CalendarManager:
    """Builds market calendars and parses important event dates for holiday-aware pipelines."""

    _PARAMS = ParameterLoader()
    _EXCHANGE_DEFAULT = _PARAMS.get("exchange_default")
    _EXCHANGE_CODE_MAP = _PARAMS.get("exchange_code_map")
    _EVENT_DATES_FILEPATH = _PARAMS.get("event_dates_filepath")
    _HOLIDAY_COUNTRY = _PARAMS.get("holiday_country")
    _DATE_FORMAT = _PARAMS.get("date_format")

    @staticmethod
    def _find_exchange() -> str:
        if CalendarManager._EXCHANGE_DEFAULT is None:
            raise ValueError("Exchange default parameter is not defined")
        if CalendarManager._EXCHANGE_CODE_MAP is None:
            raise ValueError("Exchange code map parameter is not defined")
        if not isinstance(CalendarManager._EXCHANGE_DEFAULT, str):
            raise ValueError(
                f"Exchange default parameter is invalid: {CalendarManager._EXCHANGE_DEFAULT}"
            )
        if not isinstance(CalendarManager._EXCHANGE_CODE_MAP, dict):
            raise ValueError(
                f"Exchange code map parameter is invalid: {CalendarManager._EXCHANGE_CODE_MAP}"
            )
        _exchange_default: str = CalendarManager._EXCHANGE_DEFAULT.strip()
        if len(_exchange_default) == 0:
            raise ValueError("Exchange default parameter is empty")
        if len(CalendarManager._EXCHANGE_CODE_MAP) == 0:
            raise ValueError("Exchange code map parameter is empty")
        exchange: Any = CalendarManager._EXCHANGE_CODE_MAP[_exchange_default]
        if exchange is None:
            raise ValueError(
                "Exchange default is not defined in exchange code map parameter"
            )
        if not isinstance(exchange, str):
            raise ValueError(
                f"Exchange code mapped value of '{_exchange_default}' is invalid: {exchange}"
            )
        exchange = exchange.strip()
        if len(exchange) == 0:
            raise ValueError(
                f"Exchange code mapped value of '{_exchange_default}' is empty"
            )
        return exchange

    @staticmethod
    def build_market_calendars() -> (
        Tuple[MarketCalendar, list[datetime.date], List[datetime.date]]
    ):
        """Build market calendar, holiday set, and convert event days to datetime.date objects."""
        current_year = datetime.datetime.now().year
        years = list(range(current_year - 20, current_year + 1))
        calendar: MarketCalendar = mcal.get_calendar(CalendarManager._find_exchange())
        us_holidays = list(
            holidays.country_holidays(
                CalendarManager._HOLIDAY_COUNTRY, years=years
            ).keys()
        )
        event_dates = EventDates(CalendarManager._EVENT_DATES_FILEPATH)
        fed_event_dates = event_dates.get_all_fed_event_days()
        return calendar, us_holidays, fed_event_dates
