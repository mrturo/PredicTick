"""Handles trading schedule construction and calendar day analysis."""

import warnings
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Hashable, List, Optional, Tuple, Union

import pandas as pd  # type: ignore
import pandas_market_calendars as mcal  # type: ignore

from utils.parameters import ParameterLoader

warnings.simplefilter("error")


class ScheduleBuilder:
    """Builds trading schedules and business day indexes from intraday data."""

    _PARAMS = ParameterLoader()
    _WEEKDAYS: List[str] = _PARAMS.get("weekdays")
    _CUTOFF_MINUTES = _PARAMS.get("cutoff_minutes")
    _CALENDAR_NAME = _PARAMS.get("market_calendar_name")

    @staticmethod
    def build_schedule_sumary(
        open_str: Optional[str], close_str: Optional[str], schedules: Any
    ) -> Dict[str, Any]:
        """
        Builds a summary dictionary for a trading schedule.

        Args:
            open_str (str): Minimum opening time in format "HH:MM:SS".
            close_str (str): Maximum closing time in format "HH:MM:SS".
            schedules (Any): List or structure of detailed daily schedules.

        Returns:
            dict: Summary with all_day flag, min/max times, open hours, and schedules.
        """
        all_day = (
            open_str == close_str and open_str == "00:00:00"
            if open_str and close_str
            else None
        )
        open_hours = ScheduleBuilder.calc_open_hours(open_str, close_str, all_day)
        return {
            "all_day": all_day,
            "min_open": open_str,
            "max_close": close_str,
            "schedules": schedules,
            "open_hours": open_hours,
        }

    @staticmethod
    def calc_open_hours(
        open_str: Optional[str],
        close_str: Optional[str],
        all_day: Optional[bool] = None,
    ) -> Optional[float]:
        """
        Calculates the number of open trading hours between two time strings.

        Args:
            open_str (str): Opening time string in format "HH:MM:SS".
            close_str (str): Closing time string in format "HH:MM:SS".

        Returns:
            float: Number of open hours.
        """
        if not open_str or not close_str:
            return None

        try:
            open_time = datetime.strptime(open_str, "%H:%M:%S")
            close_time = datetime.strptime(close_str, "%H:%M:%S")
            delta = (close_time - open_time).total_seconds() / 3600.0
            return 24.0 if all_day is True else delta if delta >= 0 else 24.0 + delta
        except ValueError:
            return None

    @staticmethod
    def build_schedule(
        entry: dict[str, list[dict[Hashable, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Constructs a weekly trading schedule from historical intraday prices.

        Args:
            entry (dict): Dictionary containing a "historical_prices" DataFrame.

        Returns:
            Dict[str, Dict[str, Any]]: Weekday schedule with open/close times and variants.
        """
        df = pd.DataFrame(entry["historical_prices"])
        result: Dict[str, Dict[str, Any]] = {}

        if "datetime" in df.columns and not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df.dropna(subset=["datetime"], inplace=True)
            now = datetime.now(timezone.utc)
            latest_dt = df["datetime"].max()

            if latest_dt.date() == now.date():
                elapsed_minutes = (now - latest_dt).total_seconds() / 60.0
                if elapsed_minutes < ScheduleBuilder._CUTOFF_MINUTES:
                    df = df[df["datetime"].dt.date < now.date()]

            df["weekday"] = df["datetime"].dt.day_name()
            df["date"] = df["datetime"].dt.date

            for day in ScheduleBuilder._WEEKDAYS:
                lower_day = day.lower()
                day_data = df[df["weekday"] == day]

                if day_data.empty:
                    result[lower_day] = {
                        "all_day": None,
                        "min_open": None,
                        "max_close": None,
                        "schedules": [],
                        "open_hours": None,
                    }
                    continue

                daily_ranges = ScheduleBuilder._generate_daily_ranges(day_data)
                min_open, max_close = ScheduleBuilder._compute_open_close_extremes(
                    daily_ranges
                )
                schedule_variants = ScheduleBuilder._group_schedule_variants(
                    daily_ranges
                )

                result[lower_day] = ScheduleBuilder.build_schedule_sumary(
                    min_open, max_close, schedule_variants
                )

        return result

    @staticmethod
    def _generate_daily_ranges(df: Union[pd.DataFrame, pd.Series]) -> Dict[date, dict]:
        daily_ranges = defaultdict(lambda: {"open": None, "close": None})
        for local_date, group in df.groupby("date"):
            open_dt = group["datetime"].min()
            close_dt = group["datetime"].max() + timedelta(hours=1)
            daily_ranges[local_date]["open"] = open_dt.time()
            daily_ranges[local_date]["close"] = close_dt.time()
        return daily_ranges

    @staticmethod
    def _compute_open_close_extremes(
        daily_ranges: dict,
    ) -> Tuple[Optional[str], Optional[str]]:
        base_date = datetime(2000, 1, 1)
        opens = []
        closes = []

        for v in daily_ranges.values():
            open_dt = datetime.combine(base_date, v["open"])
            close_dt = datetime.combine(base_date, v["close"])
            if close_dt <= open_dt:
                close_dt += timedelta(days=1)
            opens.append(open_dt)
            closes.append(close_dt)

        min_open = min(opens).time().strftime("%H:%M:%S") if opens else None
        max_close = max(closes).time().strftime("%H:%M:%S") if closes else None
        return min_open, max_close

    @staticmethod
    def _has_future_valid_days(
        idx: int, all_trading_days: list, all_dates_set: set
    ) -> bool:
        """
        Check if there are valid trading days within the next 14 days from the given index.

        Args:
            idx (int): Current index in the all_trading_days list.
            all_trading_days (list): List of all possible trading days.
            all_dates_set (set): Set of actual trading dates to validate against.

        Returns:
            bool: True if a valid future date exists within 14 days, False otherwise.
        """
        for lookahead in range(1, 15):
            if idx + lookahead < len(all_trading_days):
                if all_trading_days[idx + lookahead] in all_dates_set:
                    return True
        return False

    @staticmethod
    def group_continuous_date_ranges(dates: List[str]) -> List[List[str]]:
        """
        Groups continuous trading dates, allowing short gaps due to market holidays.

        Args:
            dates (List[str]): List of trading dates in 'YYYY-MM-DD' format.

        Returns:
            List[List[str]]: List of grouped date ranges.
        """
        if not dates:
            return []

        params = ParameterLoader()
        calendar_name = params.get("market_calendar_name")
        market_calendar = mcal.get_calendar(calendar_name)

        # Convert and sort dates
        sorted_dates = sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in dates)
        all_dates_set = set(sorted_dates)

        # Retrieve full valid trading schedule
        full_schedule = market_calendar.schedule(
            start_date=sorted_dates[0], end_date=sorted_dates[-1] + timedelta(days=7)
        )
        all_trading_days = [d.date() for d in full_schedule.index.to_list()]

        groups: List[List[str]] = []
        current_group: List[str] = []

        for day in all_trading_days:
            if day in all_dates_set:
                current_group.append(day.strftime("%Y-%m-%d"))
            else:
                if current_group:
                    idx = all_trading_days.index(day)
                    if not ScheduleBuilder._has_future_valid_days(
                        idx, all_trading_days, all_dates_set
                    ):
                        groups.append(current_group)
                        current_group = []

        return groups

    @staticmethod
    def _group_schedule_variants(daily_ranges: dict) -> List[dict]:
        range_groups = defaultdict(list)
        for local_date, times in daily_ranges.items():
            key = (
                times["open"].strftime("%H:%M:%S"),
                times["close"].strftime("%H:%M:%S"),
            )
            range_groups[key].append(str(local_date))

        total_days = sum(len(dates) for dates in range_groups.values())
        threshold = 0.2 * total_days

        result: List[dict] = []
        for (open_str, close_str), dates in range_groups.items():
            all_day = (
                open_str == close_str and open_str == "00:00:00"
                if open_str and close_str
                else None
            )
            habitual = len(dates) >= threshold
            open_hours = ScheduleBuilder.calc_open_hours(open_str, close_str, all_day)
            market_time = (
                (all_day is False)
                and habitual
                and (open_hours is not None)
                and 6 <= open_hours <= 8
            )
            result.append(
                {
                    "all_day": all_day,
                    "open": open_str,
                    "close": close_str,
                    "dates": ScheduleBuilder.group_continuous_date_ranges(dates),
                    "habitual": habitual,
                    "open_hours": open_hours,
                    "market_time": market_time,
                }
            )
        return result

    @staticmethod
    def _get_business_days(
        start: pd.Timestamp, end: pd.Timestamp, calendar_type: int
    ) -> List[date]:
        if calendar_type == 5:
            cal = mcal.get_calendar(ScheduleBuilder._CALENDAR_NAME)
            schedule = cal.schedule(start_date=start, end_date=end)
            return [d.date() for d in pd.to_datetime(schedule.index)]
        if calendar_type == 6:
            return [
                (start + pd.Timedelta(days=i)).date()
                for i in range((end - start).days + 1)
                if (start + pd.Timedelta(days=i)).weekday() < 6
            ]
        if calendar_type == 7:
            return [
                (start + pd.Timedelta(days=i)).date()
                for i in range((end - start).days + 1)
            ]
        return []

    @staticmethod
    def _estimate_total_business_days_month(
        year: int, month: int, calendar_type: int
    ) -> int:
        start = pd.Timestamp(f"{year}-{month:02d}-01")
        end = (start + pd.offsets.MonthEnd(1)).normalize()

        if calendar_type == 5:
            schedule = mcal.get_calendar(ScheduleBuilder._CALENDAR_NAME).schedule(
                start_date=start, end_date=end
            )
            return len(schedule)
        if calendar_type == 6:
            return sum(
                (start + pd.Timedelta(days=i)).weekday() < 6
                for i in range((end - start).days + 1)
            )
        if calendar_type == 7:
            return (end - start).days + 1
        return 0

    @staticmethod
    def _estimate_total_business_days_year(year: int, calendar_type: int) -> int:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")

        if calendar_type == 5:
            schedule = mcal.get_calendar(ScheduleBuilder._CALENDAR_NAME).schedule(
                start_date=start, end_date=end
            )
            return len(schedule)
        if calendar_type == 6:
            return sum(
                (start + pd.Timedelta(days=i)).weekday() < 6
                for i in range((end - start).days + 1)
            )
        if calendar_type == 7:
            return (end - start).days + 1
        return 0

    @staticmethod
    def build_day_index_maps(
        unique_dates: pd.DataFrame, calendar_type: int
    ) -> Tuple[dict, dict]:
        """
        Maps each business day to its sequential index in the month and year.

        Args:
            unique_dates (pd.DataFrame): DataFrame with 'local_year' and 'local_month' columns.
            calendar_type (int): Calendar type (5=market, 6=Mon-Sat, 7=all days).

        Returns:
            Tuple[dict, dict]: (month-level index map, year-level index map).
        """
        month_day_map = {}
        year_day_map = {}

        for key, _group_df in unique_dates.groupby(["local_year", "local_month"]):
            # key may be a tuple (year, month) or a single value if only one groupby column
            if isinstance(key, tuple) and len(key) == 2:
                year, month = key
            else:
                year = key
                month = None
            if month is not None:
                start = pd.Timestamp(f"{year}-{month:02d}-01")
                end = (start + pd.offsets.MonthEnd(1)).normalize()
                business_days = ScheduleBuilder._get_business_days(
                    start, end, calendar_type
                )
                month_day_map[(year, month)] = {
                    d: i + 1 for i, d in enumerate(sorted(business_days))
                }

        for key, _ in unique_dates.groupby("local_year"):
            # key will be the year
            year = key
            start = pd.Timestamp(f"{year}-01-01")
            end = pd.Timestamp(f"{year}-12-31")
            business_days = ScheduleBuilder._get_business_days(
                start, end, calendar_type
            )
            year_day_map[year] = {d: i + 1 for i, d in enumerate(sorted(business_days))}

        return month_day_map, year_day_map

    @staticmethod
    def merge_and_estimate_totals(
        unique_dates: pd.DataFrame, calendar_type: int
    ) -> pd.DataFrame:
        """
        Merges daily business day information with estimated monthly and yearly totals.

        Args:
            unique_dates (pd.DataFrame): DataFrame with current business day counts.
            calendar_type (int): Calendar type (5=market, 6=Mon-Sat, 7=all days).

        Returns:
            pd.DataFrame: Updated DataFrame with estimated total days per month and year.
        """
        first_period = unique_dates[["local_year", "local_month"]].iloc[0]
        last_period = unique_dates[["local_year", "local_month"]].iloc[-1]

        total_counts = (
            unique_dates.groupby(["local_year", "local_month"])[
                "local_month_days_current"
            ]
            .max()
            .reset_index()
            .rename(columns={"local_month_days_current": "local_month_days_total"})
        )
        for period in [first_period, last_period]:
            y, m = period["local_year"], period["local_month"]
            total_counts.loc[
                (total_counts["local_year"] == y) & (total_counts["local_month"] == m),
                "local_month_days_total",
            ] = ScheduleBuilder._estimate_total_business_days_month(y, m, calendar_type)

        year_counts = (
            unique_dates.groupby("local_year")["local_year_days_current"]
            .max()
            .reset_index()
            .rename(columns={"local_year_days_current": "local_year_days_total"})
        )
        for y in unique_dates["local_year"].unique():
            year_counts.loc[year_counts["local_year"] == y, "local_year_days_total"] = (
                ScheduleBuilder._estimate_total_business_days_year(y, calendar_type)
            )

        merged = pd.merge(
            unique_dates, total_counts, on=["local_year", "local_month"], how="left"
        )
        return pd.merge(merged, year_counts, on="local_year", how="left")
