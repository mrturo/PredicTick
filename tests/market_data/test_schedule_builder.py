"""
Unit tests for the ScheduleBuilder class from the market_data module.
These tests validate calendar-based business day calculations,
open/close time estimations, and schedule aggregation logic.
"""

# pylint: disable=protected-access

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from market_data.gateway import Gateway
from market_data.schedule_builder import ScheduleBuilder


def test_group_schedule_variants():
    """Test grouping of schedule variants to identify habitual open/close patterns."""
    ranges = {
        datetime(2024, 5, 6).date(): {
            "open": datetime.strptime("09:00:00", "%H:%M:%S").time(),
            "close": datetime.strptime("16:00:00", "%H:%M:%S").time(),
        },
        datetime(2024, 5, 7).date(): {
            "open": datetime.strptime("09:00:00", "%H:%M:%S").time(),
            "close": datetime.strptime("16:00:00", "%H:%M:%S").time(),
        },
        datetime(2024, 5, 8).date(): {
            "open": datetime.strptime("10:00:00", "%H:%M:%S").time(),
            "close": datetime.strptime("17:00:00", "%H:%M:%S").time(),
        },
    }
    result = ScheduleBuilder._group_schedule_variants(ranges)
    if not isinstance(result, list):
        raise AssertionError("Result is not a list")
    if not any(r["habitual"] for r in result):
        raise AssertionError("No habitual variant found")


def test_compute_open_close_extremes():
    """Test computation of earliest open and latest close times from a daily schedule."""
    sample = {
        datetime(2024, 5, 6).date(): {
            "open": datetime.strptime("23:00:00", "%H:%M:%S").time(),
            "close": datetime.strptime("01:00:00", "%H:%M:%S").time(),
        },
        datetime(2024, 5, 7).date(): {
            "open": datetime.strptime("22:00:00", "%H:%M:%S").time(),
            "close": datetime.strptime("04:00:00", "%H:%M:%S").time(),
        },
    }
    open_time, close_time = ScheduleBuilder._compute_open_close_extremes(sample)
    if open_time != "22:00:00":
        raise AssertionError("Incorrect open_time")
    if close_time != "04:00:00":
        raise AssertionError("Incorrect close_time")


def test_get_business_days_calendar_type_5():
    """Test business day generation for calendar type 5."""
    start = pd.Timestamp("2024-05-01")
    end = pd.Timestamp("2024-05-03")
    days = ScheduleBuilder._get_business_days(start, end, calendar_type=5)
    days = list(days)
    if not isinstance(days, list):
        raise AssertionError("Result is not a list")
    if not all(isinstance(day, date) for day in days):
        raise AssertionError("Some items are not dates")
    if not all(start.date() <= day <= end.date() for day in days):
        raise AssertionError("Dates out of range")


def test_get_business_days_calendar_type_7():
    """Test full-day calendar returns with calendar type 7."""
    start = pd.Timestamp("2024-05-01")
    end = pd.Timestamp("2024-05-03")
    days = ScheduleBuilder._get_business_days(start, end, calendar_type=7)
    expected = [
        datetime(2024, 5, 1).date(),
        datetime(2024, 5, 2).date(),
        datetime(2024, 5, 3).date(),
    ]
    if days != expected:
        raise AssertionError("Incorrect full-day calendar result")


def test_estimate_total_business_days_year():
    """Test estimation of business days in a leap year using calendar type 7."""
    days = ScheduleBuilder._estimate_total_business_days_year(2024, calendar_type=7)
    if days != 366:
        raise AssertionError("Expected 366 days in leap year")


def test_merge_and_estimate_totals():
    """Test merging logic and total day estimation from raw data."""
    data = pd.DataFrame(
        {
            "local_year": [2023, 2023, 2023],
            "local_month": [1, 1, 1],
            "local_month_days_current": [1, 2, 3],
            "local_year_days_current": [1, 2, 3],
        }
    )
    result = ScheduleBuilder.merge_and_estimate_totals(data, calendar_type=7)
    if "local_month_days_total" not in result.columns:
        raise AssertionError("Missing 'local_month_days_total' column")
    if "local_year_days_total" not in result.columns:
        raise AssertionError("Missing 'local_year_days_total' column")
    if result["local_month_days_total"].iloc[0] != 31:
        raise AssertionError("Incorrect month total days")
    if result["local_year_days_total"].iloc[0] != 365:
        raise AssertionError("Incorrect year total days")


def test_build_day_index_maps():
    """Test construction of day index maps grouped by month and year."""
    df = pd.DataFrame({"local_year": [2023, 2023], "local_month": [1, 1]})
    month_map, year_map = ScheduleBuilder.build_day_index_maps(df, calendar_type=7)
    if (2023, 1) not in month_map:
        raise AssertionError("Missing (2023, 1) in month_map")
    if 2023 not in year_map:
        raise AssertionError("Missing 2023 in year_map")


def test_generate_daily_ranges():
    """Test generation of open/close times for each trading day."""
    df = pd.DataFrame(
        {
            "datetime": [
                "2024-05-06T09:00:00Z",
                "2024-05-06T15:00:00Z",
                "2024-05-07T10:00:00Z",
                "2024-05-07T16:00:00Z",
            ]
        }
    )
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["date"] = df["datetime"].dt.date
    result = ScheduleBuilder._generate_daily_ranges(df)
    if len(result) != 2:
        raise AssertionError("Incorrect number of daily ranges")
    if not all("open" in r and "close" in r for r in result.values()):
        raise AssertionError("Missing 'open' or 'close' in some ranges")


def test_build_schedule_returns_expected_keys():
    """Test that built schedule contains all expected weekday keys."""
    data = {
        "historical_prices": [
            {"datetime": "2024-05-06T09:00:00Z"},
            {"datetime": "2024-05-06T15:00:00Z"},
            {"datetime": "2024-05-07T09:30:00Z"},
            {"datetime": "2024-05-07T16:00:00Z"},
        ]
    }
    result = ScheduleBuilder.build_schedule(data)
    if not isinstance(result, dict):
        raise AssertionError("Schedule result is not a dictionary")
    expected_keys = [day.lower() for day in ScheduleBuilder._WEEKDAYS]
    missing = [k for k in expected_keys if k not in result]
    if missing:
        raise AssertionError(f"Missing keys in schedule result: {missing}")


def test_build_schedule_empty_input():
    """Test schedule builder with empty input returns an empty dictionary."""
    result = ScheduleBuilder.build_schedule({"historical_prices": []})
    if not isinstance(result, dict):
        raise AssertionError("Schedule result is not a dictionary")
    if result:
        raise AssertionError("Expected empty dictionary for empty input")


def test_get_business_days_calendar_type_6():
    """Test business days filtering for Mon-Sat with calendar type 6."""
    start = pd.Timestamp("2024-05-01")
    end = pd.Timestamp("2024-05-07")
    days = ScheduleBuilder._get_business_days(start, end, calendar_type=6)
    if not all(day.weekday() < 6 for day in days):
        raise AssertionError("Non-Mon-Sat weekday found")
    if datetime(2024, 5, 5).date() in days:
        raise AssertionError("Sunday should not be included")


def test_get_business_days_calendar_type_invalid():
    """Test handling of an invalid calendar type returns an empty list."""
    start = pd.Timestamp("2024-05-01")
    end = pd.Timestamp("2024-05-03")
    days = ScheduleBuilder._get_business_days(start, end, calendar_type=0)
    if days != []:
        raise AssertionError("Expected empty list for invalid calendar type")


def test_estimate_total_business_days_month_type_6():
    """Test monthly business day count does not exceed 31 for type 6."""
    total = ScheduleBuilder._estimate_total_business_days_month(
        2024, 5, calendar_type=6
    )
    if not isinstance(total, int):
        raise AssertionError("Result is not an integer")
    if total > 31:
        raise AssertionError("Business days exceed possible days in month")


def test_estimate_total_business_days_month_type_5():
    """Test monthly business days are positive with calendar type 5."""
    total = ScheduleBuilder._estimate_total_business_days_month(
        2024, 5, calendar_type=5
    )
    if not isinstance(total, int):
        raise AssertionError("Result is not an integer")
    if total <= 0:
        raise AssertionError("Total business days should be greater than zero")


def test_estimate_total_business_days_month_type_invalid():
    """Test invalid calendar type returns zero monthly business days."""
    total = ScheduleBuilder._estimate_total_business_days_month(
        2024, 5, calendar_type=0
    )
    if total != 0:
        raise AssertionError("Expected zero for invalid calendar type")


def test_estimate_total_business_days_year_type_5():
    """Test annual business day estimation with calendar type 5."""
    total = ScheduleBuilder._estimate_total_business_days_year(2024, calendar_type=5)
    if not isinstance(total, int):
        raise AssertionError("Result is not an integer")
    if total <= 0:
        raise AssertionError("Total business days should be greater than zero")


def test_estimate_total_business_days_year_type_invalid():
    """Test invalid calendar type returns zero annual business days."""
    total = ScheduleBuilder._estimate_total_business_days_year(2024, calendar_type=0)
    if total != 0:
        raise AssertionError("Expected zero for invalid calendar type")


def test_build_schedule_intraday_truncated():
    """Test intraday data truncation still results in valid weekday keys."""
    now = datetime.now(timezone.utc)
    last = now - timedelta(minutes=5)
    data = {
        "historical_prices": [
            {"datetime": (last - timedelta(hours=6)).isoformat()},
            {"datetime": last.isoformat()},
        ]
    }
    result = ScheduleBuilder.build_schedule(data)
    if not isinstance(result, dict):
        raise AssertionError("Schedule result is not a dictionary")
    missing_keys = [
        k for k in [d.lower() for d in ScheduleBuilder._WEEKDAYS] if k not in result
    ]
    if missing_keys:
        raise AssertionError(f"Missing expected weekday keys in result: {missing_keys}")


def test_build_schedule_non_datetime_column():
    """Test non-datetime historical input results in empty dictionary."""
    data = {"historical_prices": [{"value": 1}, {"value": 2}]}
    result = ScheduleBuilder.build_schedule(data)
    if result:
        raise AssertionError("Expected empty dictionary for empty input")


def test_estimate_total_business_days_year_type_6():
    """Test expected count of Mon-Sat days for full year using type 6."""
    year = 2024
    total = ScheduleBuilder._estimate_total_business_days_year(year, calendar_type=6)
    expected = sum(
        (datetime(year, 1, 1) + timedelta(days=i)).weekday() < 6
        for i in range((datetime(year, 12, 31) - datetime(year, 1, 1)).days + 1)
    )
    if total != expected:
        raise AssertionError("Mismatch in Mon-Sat business day count")


def test_build_schedule_filters_recent_intraday_data():
    """Test that intraday data within cutoff window is excluded from same-day results."""
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    cutoff = ScheduleBuilder._CUTOFF_MINUTES
    recent = now - timedelta(minutes=cutoff - 1)
    older = now - timedelta(days=1)

    data = {
        "historical_prices": [
            {"datetime": older.isoformat()},
            {"datetime": recent.isoformat()},
        ]
    }

    result = ScheduleBuilder.build_schedule(data)
    today = now.strftime("%A").lower()

    if today not in result:
        raise AssertionError(f"{today} not in result keys")
    if result[today]["min_open"] is not None:
        raise AssertionError("Expected today's data to be excluded due to cutoff")


def test_calc_open_hours_invalid_time_string():
    """
    Test calc_open_hours handles invalid time string format gracefully.

    This test ensures that passing an invalid time string as input
    causes calc_open_hours to return None due to ValueError.
    """
    invalid_open = "25:61:00"
    close = "16:00:00"
    result = ScheduleBuilder.calc_open_hours(invalid_open, close, False)
    if result is not None:
        raise AssertionError("Expected None for invalid open_str format")


def test_has_future_valid_days_returns_true():
    """
    Test _has_future_valid_days returns True when a future valid day exists.

    This test ensures that the function correctly identifies a valid
    trading day within lookahead range in all_trading_days.
    """
    all_trading_days = [
        date(2024, 5, 6),
        date(2024, 5, 7),
        date(2024, 5, 8),
        date(2024, 5, 9),
    ]
    all_dates_set = {date(2024, 5, 9)}
    idx = 0
    result = ScheduleBuilder._has_future_valid_days(
        idx, all_trading_days, all_dates_set
    )
    if result is not True:
        raise AssertionError("Expected True when a valid day is within lookahead")


def test_has_future_valid_days_returns_false():
    """
    Test _has_future_valid_days returns False when no valid future days exist.

    This test ensures that the function correctly returns False if no
    trading day in all_dates_set is found within lookahead range.
    """
    all_trading_days = [date(2024, 5, 6), date(2024, 5, 7), date(2024, 5, 8)]
    all_dates_set = {date(2024, 6, 1)}
    idx = 0
    result = ScheduleBuilder._has_future_valid_days(
        idx, all_trading_days, all_dates_set
    )
    if result is not False:
        raise AssertionError("Expected False when no valid day in lookahead range")


def test_group_continuous_date_ranges_empty_list():
    """
    Test group_continuous_date_ranges returns empty list on empty input.

    This test ensures that if an empty list is passed to group_continuous_date_ranges,
    the function returns an empty list as expected.
    """
    result = ScheduleBuilder.group_continuous_date_ranges([])
    if result:
        raise AssertionError("Expected empty list for empty input")


def test_get_symbol_returns_none_for_missing_symbol():
    """
    Test get_symbol returns None when the requested symbol does not exist.

    Covers the path where the symbol is not present in Gateway._symbols.
    """
    Gateway._symbols = {"AAPL": {"metadata": "foo"}}
    result = Gateway.get_symbol("FAKE_SYMBOL")
    if result is not None:
        raise AssertionError("Expected None for missing symbol")
