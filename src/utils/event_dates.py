"""EventDates — Manage key economic event dates used in model pipelines.

This module loads key calendar dates (e.g., FED event days) from a JSON file
and provides utility methods to assess temporal relationships with these dates.
"""

from datetime import date, datetime
from typing import List, Optional, Union

from utils.json_manager import JsonManager
from utils.logger import Logger


class EventDates:
    """Handles important event dates, such as FED meetings, and provides utilities."""

    def __init__(self, filepath: str):
        dates = JsonManager().load(filepath) or {}
        self.fed_event_days: List[date] = EventDates._convert_str_dates_to_date_objects(
            dates.get("fed_event_days", [])
        )

    @staticmethod
    def _resolve_reference_date(
        reference_date: Optional[Union[datetime, date]],
    ) -> datetime:
        """Ensure reference date is a datetime object, normalized to midnight."""
        if reference_date is None:
            reference_date = datetime.now()
        if isinstance(reference_date, date) and not isinstance(
            reference_date, datetime
        ):
            reference_date = datetime.combine(reference_date, datetime.min.time())
        return reference_date.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def _convert_str_dates_to_date_objects(date_strs: List[str]) -> List[date]:
        """Convert a list of date strings into sorted date objects, skipping invalid ones."""
        date_objs = []
        for d in date_strs:
            try:
                date_objs.append(datetime.strptime(d, "%Y-%m-%d").date())
            except ValueError:
                Logger.warning(f"Skipping invalid date format: {d}")
        return sorted(date_objs)

    def get_all_fed_event_days(self) -> List[date]:
        """Return a list of all FED event days."""
        return [d for d in self.fed_event_days if d]

    def get_past_fed_event_days(
        self, reference_date: Optional[datetime] = None
    ) -> List[date]:
        """Return a list of FED event days that occurred before the reference date."""
        ref_date = EventDates._resolve_reference_date(reference_date).date()
        return [d for d in self.fed_event_days if d < ref_date]

    def get_future_fed_event_days(
        self, reference_date: Optional[datetime] = None
    ) -> List[date]:
        """Return a list of FED event days that occur after the reference date."""
        ref_date = EventDates._resolve_reference_date(reference_date).date()
        return [d for d in self.fed_event_days if d >= ref_date]
