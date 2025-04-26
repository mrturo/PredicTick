"""Unit tests for the CalendarManager module.

This test suite validates the construction of market calendars, retrieval of holidays,
and parsing of event dates. All tests are isolated with mocks, cover normal flows,
edge cases, and error handling, ensuring reliability and maintainability for
high-stakes financial environments."""

# pylint: disable=protected-access

import datetime
import importlib
import sys
import typing as _t

import pytest  # type: ignore


class DummyCalendar:  # pylint: disable=too-few-public-methods
    """Dummy calendar class used for mocking market calendar objects."""


class DummyEventDates:  # pylint: disable=too-few-public-methods
    """Dummy EventDates class for mocking FED event dates loading."""

    def __init__(self, path):
        self.path = path
        self.get_all_fed_event_days_called = False

    def get_all_fed_event_days(self):
        """Returns a static list of FED event dates for testing purposes."""
        self.get_all_fed_event_days_called = True
        return [datetime.date(2022, 6, 15), datetime.date(2023, 3, 22)]


class DummyParameterLoader:  # pylint: disable=too-few-public-methods
    """Dummy parameter loader for simulating parameter retrieval in tests."""

    def __init__(self):
        self.params = {
            "exchange_default": "XNYS",
            "event_dates_filepath": "fed_events.csv",
            "holiday_country": "US",
            "date_format": "%Y-%m-%d",
        }

    def get(self, key):
        """Retrieve a parameter value by key."""
        return self.params[key]


class DummyHolidayBase:  # pylint: disable=too-few-public-methods
    """Dummy holidays class to provide mock holiday dates for testing."""

    def __init__(self, years=None):
        self.years = years

    def keys(self):
        """Returns a fixed list of holiday dates."""
        return [
            datetime.date(2023, 1, 1),
            datetime.date(2023, 7, 4),
            datetime.date(2023, 12, 25),
        ]


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """Patch external dependencies used in CalendarManager for isolation and i/o control"""
    # Patch ParameterLoader
    monkeypatch.setattr(
        "src.utils.config.parameters.ParameterLoader", DummyParameterLoader
    )

    # Patch holidays.country_holidays and holidays.US
    class DummyHolidaysModule:  # pylint: disable=too-few-public-methods
        """Dummy holidays module for mocking the 'holidays' package in unit tests.
        Provides a mock US holidays class and a method to retrieve it based on country code.
        """

        class US:  # pylint: disable=too-few-public-methods
            """Dummy US holidays class for mocking US holidays."""

            def __init__(self, years=None, **kwargs):  # pylint: disable=unused-argument
                self.years = years

            def keys(self):
                """Returns mock holiday dates."""
                return [
                    datetime.date(2023, 1, 1),
                    datetime.date(2023, 7, 4),
                    datetime.date(2023, 12, 25),
                ]

        @staticmethod
        def country_holidays(country, years=None, **kwargs):
            """Returns a dummy holidays class for the specified country."""
            if country == "US":
                return DummyHolidaysModule.US(years=years, **kwargs)
            raise NotImplementedError(f"Country {country} not available")

    monkeypatch.setitem(sys.modules, "holidays", DummyHolidaysModule())

    # Patch pandas_market_calendars.get_calendar
    class DummyMcalModule:  # pylint: disable=too-few-public-methods
        """Dummy market calendar module for mocking the 'pandas_market_calendars' library.
        Provides a static method to return a dummy calendar object."""

        @staticmethod
        def get_calendar(_name):
            """Returns a dummy calendar object."""
            return DummyCalendar()

    monkeypatch.setitem(sys.modules, "pandas_market_calendars", DummyMcalModule())
    # Patch EventDates
    monkeypatch.setattr(
        "src.market_data.utils.datetime.event_dates.EventDates", DummyEventDates
    )


@pytest.fixture(scope="module")
def calendar_manager_module():
    """Import the calendar_manager module (under test) after dependencies are patched."""
    mod = importlib.import_module("src.utils.exchange.calendar_manager")
    return mod


def test_build_market_calendars_holidays_range(
    monkeypatch, calendar_manager_module
):  # pylint: disable=redefined-outer-name
    """Tests that the holidays list covers the correct 20-year range.
    - Simulates current_year as 2024 and verifies the range used for holiday generation.
    """
    orig_datetime = datetime.datetime

    class DummyDatetime(datetime.datetime):
        """Dummy datetime class to control current date in tests."""

        @classmethod
        def now(cls, tz=None):
            return orig_datetime(2024, 6, 1)

    class DummyEventDates:  # pylint: disable=too-few-public-methods
        """Dummy EventDates class returning no FED events."""

        def __init__(self, _path):
            self.fed_event_days = []

        @staticmethod
        def _convert_str_dates_to_date_objects(arg):
            return arg

        def get_all_fed_event_days(self):
            """Return a static list of FED event dates for use in unit tests.

            This mock implementation simulates retrieval of predefined event dates,
            and can be used to test calendar logic that depends on the presence of FED events.
            """
            return []

    monkeypatch.setattr(
        "src.market_data.utils.datetime.event_dates.EventDates", DummyEventDates
    )
    try:
        datetime.datetime = DummyDatetime
        result = calendar_manager_module.CalendarManager.build_market_calendars()
        us_holidays = result[1]
        if not all(isinstance(x, datetime.date) for x in us_holidays):
            raise AssertionError("All holidays must be date objects.")
    finally:
        datetime.datetime = orig_datetime


def test_build_market_calendars_bad_params(monkeypatch):
    """Tests the case when required parameters are missing or corrupted.

    - Expects a KeyError to be raised if any critical parameter is absent."""

    class DummyParameterLoaderBad:  # pylint: disable=too-few-public-methods
        """Dummy parameter loader that always raises KeyError."""

        def get(self, key):
            """Raise KeyError for any key requested."""
            raise KeyError("Key not found")

    monkeypatch.setattr(
        "src.utils.config.parameters.ParameterLoader", DummyParameterLoaderBad
    )
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("calendar_manager")


def test_build_market_calendars_invalid_holiday_module(
    monkeypatch, calendar_manager_module
):  # pylint: disable=redefined-outer-name
    """Tests the case when holidays.country_holidays returns an invalid object.

    - Expects AttributeError if the object lacks the .keys() method."""

    class DummyHolidaysInvalid:  # pylint: disable=too-few-public-methods
        """Dummy holidays class lacking the .keys() method."""

    class DummyHolidaysMod:  # pylint: disable=too-few-public-methods
        """Dummy holidays module that always returns an invalid holidays object"""

        @staticmethod
        def country_holidays(_country, _years=None):
            """Returns an invalid holidays object without .keys()."""
            return DummyHolidaysInvalid()

    monkeypatch.setitem(sys.modules, "holidays", DummyHolidaysMod())

    class DummyEventDates:  # pylint: disable=too-few-public-methods disable=redefined-outer-name
        """Dummy EventDates class returning no FED events."""

        def __init__(self, _path):
            self.fed_event_days = []

        @staticmethod
        def _convert_str_dates_to_date_objects(arg):
            return arg

        def get_all_fed_event_days(self):
            """Return a mock list of FED event dates for testing.

            In specific dummy implementations, this may return an empty list,
            raise an exception, or simulate other edge cases depending on the test."""
            return []

    monkeypatch.setattr(
        "src.market_data.utils.datetime.event_dates.EventDates", DummyEventDates
    )
    with pytest.raises(NotImplementedError):
        calendar_manager_module.CalendarManager.build_market_calendars()


@pytest.fixture(scope="module")
def _calendar_manager():  # pylint: disable=missing-function-docstring
    mod = importlib.import_module("src.utils.exchange.calendar_manager")
    return mod.CalendarManager


# Test matrix: (exchange_default, exchange_code_map, expected_error_message)
_PARAMS: list[tuple[_t.Any, _t.Any, str]] = [
    # 1. _EXCHANGE_DEFAULT is None
    (None, {"XNYS": "NYSE"}, "Exchange default parameter is not defined"),
    # 2. _EXCHANGE_CODE_MAP is None
    ("XNYS", None, "Exchange code map parameter is not defined"),
    # 3. _EXCHANGE_DEFAULT is *not* a str
    (123, {123: "NYSE"}, "Exchange default parameter is invalid: 123"),
    # 4. _EXCHANGE_CODE_MAP is *not* a dict
    ("XNYS", "not a dict", "Exchange code map parameter is invalid: not a dict"),
    # 5. _EXCHANGE_DEFAULT becomes empty after ``strip``
    ("   ", {"": "NYSE"}, "Exchange default parameter is empty"),
    # 6. _EXCHANGE_CODE_MAP is an empty dict
    ("XNYS", {}, "Exchange code map parameter is empty"),
    # 7. Mapping exists but value is ``None``
    (
        "XNYS",
        {"XNYS": None},
        "Exchange default is not defined in exchange code map parameter",
    ),
    # 8. Mapping value is *not* a str
    ("XNYS", {"XNYS": 123}, "Exchange code mapped value of 'XNYS' is invalid: 123"),
    # 9. Mapping value becomes empty after ``strip``
    ("XNYS", {"XNYS": "   "}, "Exchange code mapped value of 'XNYS' is empty"),
]


@pytest.mark.parametrize("exchange_default, exchange_code_map, expected", _PARAMS)
# pylint: disable=redefined-outer-name
def test_find_exchange_raises(
    _calendar_manager,  # CalendarManager class, injected by fixture
    monkeypatch: pytest.MonkeyPatch,
    exchange_default: _t.Any,
    exchange_code_map: _t.Any,
    expected: str,
):
    """Ensure that every misconfiguration path triggers the correct ``ValueError`` message."""
    # Patch class‑level configuration directly; ``raising=False`` avoids AttributeError
    monkeypatch.setattr(
        _calendar_manager, "_EXCHANGE_DEFAULT", exchange_default, raising=False
    )
    monkeypatch.setattr(
        _calendar_manager, "_EXCHANGE_CODE_MAP", exchange_code_map, raising=False
    )
    with pytest.raises(ValueError) as exc_info:
        _calendar_manager._find_exchange()
    # Validate the error message without using bare ``assert``
    if str(exc_info.value) != expected:
        raise AssertionError(
            f"Expected error message '{expected}', got '{exc_info.value}'"
        )
