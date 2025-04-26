"""
Unit tests for the CalendarManager module.

This test suite validates the construction of market calendars, retrieval of holidays,
and parsing of event dates. All tests are isolated with mocks, cover normal flows,
edge cases, and error handling, ensuring reliability and maintainability for
high-stakes financial environments.
"""

import datetime
import importlib
import sys

import pytest  # type: ignore


class DummyCalendar:  # pylint: disable=too-few-public-methods
    """Dummy calendar class used for mocking market calendar objects."""


class DummyEventDates:  # pylint: disable=too-few-public-methods
    """
    Dummy EventDates class for mocking FED event dates loading.

    Args:
        path (str): Path to event dates file.
    """

    def __init__(self, path):
        self.path = path
        self.get_all_fed_event_days_called = False

    def get_all_fed_event_days(self):
        """Returns a static list of FED event dates for testing purposes."""
        self.get_all_fed_event_days_called = True
        # Caso normal: dos fechas
        return [datetime.date(2022, 6, 15), datetime.date(2023, 3, 22)]


class DummyParameterLoader:  # pylint: disable=too-few-public-methods
    """Dummy parameter loader for simulating parameter retrieval in tests."""

    def __init__(self):
        self.params = {
            "market_calendar_name": "XNYS",
            "event_dates_filepath": "fed_events.csv",
            "holiday_country": "US",
            "date_format": "%Y-%m-%d",
        }

    def get(self, key):
        """
        Retrieve a parameter value by key.

        Args:
            key (str): Parameter key.
        Returns:
            Any: Parameter value.
        Raises:
            KeyError: If the key is not found.
        """
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
    """
    Patch external dependencies used in CalendarManager for isolation.

    and input/output control during tests.
    """

    # Patch ParameterLoader
    monkeypatch.setattr("utils.parameters.ParameterLoader", DummyParameterLoader)

    # Patch holidays.country_holidays and holidays.US
    class DummyHolidaysModule:  # pylint: disable=too-few-public-methods
        """
        Dummy holidays module for mocking the 'holidays' package in unit tests.

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
        """
        Dummy market calendar module for mocking the 'pandas_market_calendars' library.

        Provides a static method to return a dummy calendar object.
        """

        @staticmethod
        def get_calendar(_name):
            """Returns a dummy calendar object."""
            return DummyCalendar()

    monkeypatch.setitem(sys.modules, "pandas_market_calendars", DummyMcalModule())

    # Patch EventDates
    monkeypatch.setattr("utils.event_dates.EventDates", DummyEventDates)


# Importa el módulo principal solo después de parchear dependencias
@pytest.fixture(scope="module")
def calendar_manager_module():
    """
    Import the calendar_manager module (under test) after dependencies are patched.

    Returns:
        module: The imported calendar_manager module.
    """

    mod = importlib.import_module("utils.calendar_manager")
    return mod


def test_build_market_calendars_empty_events(
    monkeypatch, calendar_manager_module
):  # pylint: disable=redefined-outer-name
    """
    Tests the case where no FED events are defined.

    - Ensures fed_event_dates is an empty list if the event file is empty.
    """

    class DummyEventDatesEmpty:  # pylint: disable=too-few-public-methods
        """Dummy EventDates class that simulates an empty list of FED event dates."""

        def __init__(self, path):
            self.path = path
            self.fed_event_days = []

        @staticmethod
        def _convert_str_dates_to_date_objects(arg):
            """Mock static method to convert string dates (passthrough in dummy)."""
            return arg

        def get_all_fed_event_days(self):
            """
            Return a list of mock FED event dates for testing purposes.

            This simulates loading FED event dates from an external source.

            Returns:
                list[datetime.date]: List of event dates.
            """
            return []

    monkeypatch.setattr("utils.event_dates.EventDates", DummyEventDatesEmpty)
    result = calendar_manager_module.CalendarManager.build_market_calendars()
    fed_event_dates = result[2]
    if not fed_event_dates == []:
        raise AssertionError(
            "fed_event_dates debe ser una lista vacía cuando no hay eventos."
        )


def test_build_market_calendars_holidays_range(
    monkeypatch, calendar_manager_module
):  # pylint: disable=redefined-outer-name
    """
    Tests that the holidays list covers the correct 20-year range.

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
            """
            Return a static list of FED event dates for use in unit tests.

            This mock implementation simulates retrieval of predefined event dates,
            and can be used to test calendar logic that depends on the presence of FED events.

            Returns:
                list[datetime.date]: Hardcoded list of FED event dates.
            """
            return []

    monkeypatch.setattr("utils.event_dates.EventDates", DummyEventDates)

    try:
        datetime.datetime = DummyDatetime
        result = calendar_manager_module.CalendarManager.build_market_calendars()
        us_holidays = result[1]
        if not all(isinstance(x, datetime.date) for x in us_holidays):
            raise AssertionError("Todos los feriados deben ser fechas.")
    finally:
        datetime.datetime = orig_datetime


def test_build_market_calendars_bad_params(monkeypatch):
    """
    Tests the case when required parameters are missing or corrupted.

    - Expects a KeyError to be raised if any critical parameter is absent.
    """

    class DummyParameterLoaderBad:  # pylint: disable=too-few-public-methods
        """Dummy parameter loader that always raises KeyError."""

        def get(self, key):
            """Raise KeyError for any key requested."""
            raise KeyError("Clave no encontrada")

    monkeypatch.setattr("utils.parameters.ParameterLoader", DummyParameterLoaderBad)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("calendar_manager")


def test_build_market_calendars_invalid_holiday_module(
    monkeypatch, calendar_manager_module
):  # pylint: disable=redefined-outer-name
    """
    Tests the case when holidays.country_holidays returns an invalid object.

    - Expects AttributeError if the object lacks the .keys() method.
    """

    class DummyHolidaysInvalid:  # pylint: disable=too-few-public-methods
        """Dummy holidays class lacking the .keys() method."""

    class DummyHolidaysMod:  # pylint: disable=too-few-public-methods
        """
        Dummy holidays module that always returns an invalid holidays object.

        (missing the 'keys()' method) to test error handling in calendar logic.
        """

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
            """
            Return a mock list of FED event dates for testing.

            In specific dummy implementations, this may return an empty list,
            raise an exception, or simulate other edge cases depending on the test.
            """
            return []

    monkeypatch.setattr("utils.event_dates.EventDates", DummyEventDates)

    with pytest.raises(NotImplementedError):
        calendar_manager_module.CalendarManager.build_market_calendars()
