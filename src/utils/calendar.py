"""
Utilities for building US market calendars and handling FED event dates.

Provides a function to construct trading calendars and extract structured holiday/event metadata.
"""

import datetime
from typing import Any, List, Tuple

import holidays
import pandas_market_calendars as mcal


def build_market_calendars(
    fed_event_days: List[str],
) -> Tuple[Any, holidays.HolidayBase, List[datetime.date]]:
    """
    Build US market calendar, holiday set, and convert FED event days to datetime.date objects.

    Args:
        fed_event_days (List[str]): List of FED event date strings.

    Returns:
        Tuple[Any, holidays.HolidayBase, List[datetime.date]]: NYSE calendar,
            US holidays, and parsed FED event dates.
    """
    calendar = mcal.get_calendar("NYSE")
    us_holidays = holidays.country_holidays("US")
    fed_events = [
        datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in fed_event_days
    ]
    return calendar, us_holidays, fed_events
