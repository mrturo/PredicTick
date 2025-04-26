"""Utilities for handling dates tied to key macro-economic events, such as US FED meetings.

The module exposes the :class:`EventDates` helper used throughout the code-base to query event
calendars in a type-safe manner.
"""

from datetime import date, datetime
from typing import List, Optional, Union

from src.utils.io.json_manager import JsonManager
from src.utils.io.logger import Logger


# pylint: disable=too-few-public-methods
class EventDates:
    """Encapsulates and operates on dates of predefined economic events."""

    def __init__(self, filepath: str):
        """Load event dates from *filepath* and prepare internal structures."""
        dates = JsonManager().load(filepath) or {}
        self.fed_event_days: List[date] = EventDates._convert_str_dates_to_date_objects(
            dates.get("fed_event_days", [])
        )

    @staticmethod
    def _resolve_reference_date(
        reference_date: Optional[Union[datetime, date]],
    ) -> datetime:
        """Normalize *reference_date* to a ``datetime`` instance at midnight.

        If *reference_date* is ``None`` the current system time is used. When a
        :class:`datetime.date` instance is supplied it is combined with midnight
        (00:00:00).
        """
        if reference_date is None:
            reference_date = datetime.now()
        if isinstance(reference_date, date) and not isinstance(
            reference_date, datetime
        ):
            reference_date = datetime.combine(reference_date, datetime.min.time())
        return reference_date.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def _convert_str_dates_to_date_objects(date_strs: List[str]) -> List[date]:
        """Convert a list of ISO strings (``YYYY-MM-DD``) to ``date`` objects."""
        date_objs = []
        for d in date_strs:
            try:
                date_objs.append(datetime.strptime(d, "%Y-%m-%d").date())
            except ValueError:
                Logger.warning(f"Skipping invalid date format: {d}")
        return sorted(date_objs)

    def get_all_fed_event_days(self) -> List[date]:
        """Return a copy of all loaded FED event dates."""
        return [d for d in self.fed_event_days if d]
