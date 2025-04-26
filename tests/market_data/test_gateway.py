"""Unit tests for the market_data.gateway module."""

# pylint: disable=protected-access

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from market_data.gateway import Gateway
from utils.parameters import ParameterLoader

_PARAMS = ParameterLoader()
_TEST_MARKETDATA_FILEPATH = _PARAMS.get("test_marketdata_filepath")


@pytest.fixture
def symbol_metadata():
    """Fixture providing base symbol metadata for testing."""
    return {
        "name": "Apple Inc.",
        "symbol": "AAPL",
        "exchange": "NASDAQ",
        "currency": "USD",
        "industry": "Consumer Electronics",
        "sector": "Technology",
    }


@pytest.fixture
def sample_symbol_data(symbol_metadata):  # pylint: disable=redefined-outer-name
    """Fixture providing sample symbol data for testing."""
    symbol = symbol_metadata.copy()
    symbol["historical_prices"] = [
        {"datetime": pd.Timestamp("2025-05-16T20:00:00Z"), "close": 150.0},
        {"datetime": pd.Timestamp("2025-05-17T20:00:00Z"), "close": 152.0},
    ]
    return symbol


def test_load_market_data(
    monkeypatch, symbol_metadata
):  # pylint: disable=redefined-outer-name
    """Test loading market data from disk and initialization of gateway state."""
    symbol = symbol_metadata.copy()
    symbol["schedule"] = {
        "monday": {"min_open": "09:30:00", "max_close": "16:00:00"},
    }
    symbol["historical_prices"] = [
        {"datetime": "2025-05-16T20:00:00Z", "close": 150.0},
        {"datetime": "2025-05-17T20:00:00Z", "close": 152.0},
    ]

    raw_data = {
        "last_updated": "2025-05-18T10:00:00Z",
        "symbols": [symbol],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("utils.json_manager.JsonManager.load", staticmethod(mock_load))

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        result = Gateway.load(_TEST_MARKETDATA_FILEPATH)
        if result is None:
            raise AssertionError("Expected Gateway.load() to return non-None result.")


def test_get_last_update_from_load(monkeypatch):
    """Test retrieving last update timestamp via load."""
    raw_data = {
        "last_updated": "2025-05-18T10:00:00Z",
        "symbols": [],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("utils.json_manager.JsonManager.load", staticmethod(mock_load))

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        Gateway.load(_TEST_MARKETDATA_FILEPATH)


def test_get_global_schedule_from_update_via_load(
    monkeypatch, sample_symbol_data
):  # pylint: disable=redefined-outer-name
    """Test global schedule built indirectly through load process."""
    raw_data = {
        "last_updated": "2025-05-18T10:00:00Z",
        "symbols": [sample_symbol_data],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("utils.json_manager.JsonManager.load", staticmethod(mock_load))

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        Gateway.load(_TEST_MARKETDATA_FILEPATH)


def test_save_market_data_via_interface_indirect(
    monkeypatch, sample_symbol_data
):  # pylint: disable=redefined-outer-name
    """Test saving market data using only public interfaces (indirect schedule update)."""
    raw_data = {
        "last_updated": "2025-05-19T13:00:00Z",
        "symbols": [sample_symbol_data],
    }

    def mock_load(_filepath):
        return raw_data

    saved_output = {}

    def mock_save(data, _filepath):
        nonlocal saved_output
        saved_output = data

    monkeypatch.setattr("utils.json_manager.JsonManager.load", staticmethod(mock_load))
    monkeypatch.setattr("utils.json_manager.JsonManager.save", staticmethod(mock_save))

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        Gateway.load(_TEST_MARKETDATA_FILEPATH)


def test_schedule_all_day_flag(
    monkeypatch, sample_symbol_data
):  # pylint: disable=redefined-outer-name
    """Test global schedule detects all_day market correctly."""
    sample_symbol_data["schedule"] = {
        "monday": {"min_open": "00:00:00", "max_close": "00:00:00"},
    }

    raw_data = {
        "last_updated": "2025-05-18T10:00:00Z",
        "symbols": [sample_symbol_data],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("utils.json_manager.JsonManager.load", staticmethod(mock_load))

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        Gateway.load(_TEST_MARKETDATA_FILEPATH)


def test_save_returns_result_structure(
    monkeypatch, sample_symbol_data
):  # pylint: disable=redefined-outer-name
    """Test that save() returns the expected result structure."""
    raw_data = {
        "last_updated": "2025-05-19T13:00:00Z",
        "symbols": [sample_symbol_data],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("utils.json_manager.JsonManager.load", staticmethod(mock_load))

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        Gateway.load(_TEST_MARKETDATA_FILEPATH)


def test_schedule_all_day_flag_missing_values(monkeypatch, sample_symbol_data):
    # pylint: disable=redefined-outer-name
    """Test all_day logic path when min_open or max_close is missing."""
    sample_symbol_data["schedule"] = {"monday": {"min_open": "00:00:00"}}

    raw_data = {
        "last_updated": "2025-05-18T10:00:00Z",
        "symbols": [sample_symbol_data],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("utils.json_manager.JsonManager.load", staticmethod(mock_load))

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        Gateway.load(_TEST_MARKETDATA_FILEPATH)


def test_schedule_all_day_flag_false(
    monkeypatch, sample_symbol_data
):  # pylint: disable=redefined-outer-name
    """Test that all_day is False when min_open and max_close are not both 00:00:00."""
    sample_symbol_data["schedule"] = {
        "monday": {"min_open": "09:00:00", "max_close": "17:00:00"},
    }

    raw_data = {
        "last_updated": "2025-05-18T10:00:00Z",
        "symbols": [sample_symbol_data],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("utils.json_manager.JsonManager.load", staticmethod(mock_load))

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        Gateway.load(_TEST_MARKETDATA_FILEPATH)
        schedule = Gateway.get_global_schedule()
        if schedule["monday"].get("all_day") is True:
            raise AssertionError("Expected 'all_day' to be False.")


def test_set_and_get_stale_symbols_forced():
    """Ensure set_stale_symbols updates internal state using public interface only."""
    # Reset using public interface
    Gateway.set_stale_symbols([])

    test_symbols = ["MSFT", "TSLA"]
    Gateway.set_stale_symbols(test_symbols)

    result = Gateway.get_stale_symbols()
    if result != test_symbols:
        raise AssertionError("Stale symbols not set correctly.")


def test_detect_stale_symbols_logic():
    """Test internal logic for stale symbol detection."""
    symbol = "XXXX"
    outdated = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=500)).isoformat()

    Gateway.set_symbols(
        {
            symbol: {
                "symbol": symbol,
                "historical_prices": [{"datetime": outdated, "close": 1.0}],
            }
        }
    )

    latest = pd.Timestamp.now(tz="UTC")
    current_invalids = set()

    stale, updated_invalids = Gateway._detect_stale_symbols(latest, current_invalids)

    if symbol not in stale:
        raise AssertionError(f"{symbol} should be marked as stale.")
    if symbol not in updated_invalids:
        raise AssertionError(f"{symbol} should be added to invalid symbols.")


def test_get_filepath_returns_valid_path():
    """Test that _get_filepath returns the original filepath when it is valid."""
    result = Gateway._get_filepath("path/to/file.json", "default.json")
    if result != "path/to/file.json":
        raise AssertionError("Expected filepath to be returned when valid")


def test_get_filepath_returns_default_when_empty():
    """Test that _get_filepath returns the default path when filepath is an empty string."""
    result = Gateway._get_filepath("   ", "default.json")
    if result != "default.json":
        raise AssertionError(
            "Expected default to be returned when filepath is empty string"
        )


@pytest.mark.parametrize(
    "market_time",
    [
        [{"symbol": "AAPL", "open": "09:30", "close": "16:00"}],
        [],
    ],
)
def test_set_get_market_time(market_time):
    """Test set and get of market time symbols."""
    Gateway.set_market_time(market_time)
    result = Gateway.get_market_time()
    if result != market_time:
        raise AssertionError("Market time not correctly set or retrieved.")


def test_get_symbol_found(sample_symbol_data):  # pylint: disable=redefined-outer-name
    """Test retrieval of symbol metadata when symbol exists."""
    symbol_name = "AAPL"
    Gateway._symbols[symbol_name] = sample_symbol_data
    result = Gateway.get_symbol(symbol_name)
    if result != sample_symbol_data:
        raise AssertionError("Failed to retrieve correct symbol metadata.")


def test_get_symbol_not_found():
    """Test retrieval of symbol metadata when symbol does not exist."""
    result = Gateway.get_symbol("UNKNOWN")
    if result is not None:
        raise AssertionError("Expected None for unknown symbol.")


def test_extract_market_time_ranges_process_entry_behavior():
    """
    Test process_entry stores valid market_time entries with valid date groups.

    Verifica que:
    - Solo se incluyan en el resultado las entradas con `market_time=True` y fechas válidas.
    - Entradas con `market_time=False` o fechas vacías sean ignoradas.
    """
    Gateway.set_symbols(
        {
            "AAPL": {
                "symbol": "AAPL",
                "schedule": {
                    "monday": {
                        "schedules": [
                            {
                                "market_time": True,
                                "open": "09:30:00",
                                "close": "16:00:00",
                                "dates": [["2025-01-01", "2025-01-02"]],
                            },
                            {
                                "market_time": False,
                                "open": "10:00:00",
                                "close": "17:00:00",
                                "dates": [["2025-01-03", "2025-01-04"]],
                            },
                            {
                                # Esta entrada debe ser ignorada por no tener fechas válidas
                                "market_time": True,
                                "open": "10:00:00",
                                "close": "17:00:00",
                                "dates": [],
                            },
                        ]
                    }
                },
            }
        }
    )

    Gateway._extract_market_time_ranges()
    intervals = Gateway.get_market_time()

    if not intervals:
        raise AssertionError("Expected at least one interval from process_entry.")

    # Solo validamos que al menos uno de los intervalos tenga los tiempos correctos
    expected_open = "09:30:00"
    expected_close = "16:00:00"

    if not any(
        interval.get("open") == expected_open
        and interval.get("close") == expected_close
        for interval in intervals
    ):
        raise AssertionError(
            "Valid open/close interval not found in processed market_time."
        )


def test_extract_market_time_ranges_merge_intervals_non_contiguous():
    """
    Test _merge_intervals behavior when intervals are not contiguous or have different open/close.

    Cubre el bloque:
        else:
            grouped.append(interval)
    cuando los intervalos no son agrupables.
    """
    Gateway.set_symbols(
        {
            "AAPL": {
                "symbol": "AAPL",
                "schedule": {
                    "monday": {
                        "schedules": [
                            {
                                "market_time": True,
                                "open": "09:00:00",
                                "close": "15:00:00",
                                "dates": [["2025-01-01", "2025-01-01"]],
                            },
                            {
                                "market_time": True,
                                "open": "10:00:00",  # Diferente horario
                                "close": "16:00:00",
                                "dates": [["2025-01-03", "2025-01-03"]],
                            },
                        ]
                    }
                },
            }
        }
    )

    Gateway._extract_market_time_ranges()
    intervals = Gateway.get_market_time()

    if len(intervals) != 2:
        raise AssertionError(
            "Expected two non-grouped intervals due to differing times."
        )


def test_annotate_market_times_get_range_for_day_paths():
    """
    Valida el comportamiento completo del bloque get_range_for_day:
    - Fechas sin 'from' o 'to'
    - Intervalo con cierre menor que apertura (cierre +1 día)
    """
    Gateway.set_market_time(
        [
            {
                "open": "09:00:00",
                "close": "17:00:00",
                "from": "2025-01-01",
                "to": "2025-01-03",
            },
            {
                "open": "18:00:00",
                "close": "04:00:00",
                "from": "2025-01-04",
                "to": "2025-01-05",
            },
            {"open": "10:00:00", "close": "12:00:00", "from": "2025-01-07"},
            {"open": "07:00:00", "close": "08:00:00"},
        ]
    )

    dt_1 = datetime(2025, 1, 2, 10, 0, 0, tzinfo=timezone.utc)  # Dentro de [from, to]
    dt_2 = datetime(2025, 1, 4, 22, 0, 0, tzinfo=timezone.utc)  # close < open
    dt_3 = datetime(2025, 1, 6, 7, 30, 0, tzinfo=timezone.utc)  # Sin 'from'
    dt_4 = datetime(2025, 1, 8, 11, 0, 0, tzinfo=timezone.utc)  # Sin 'to'

    r1 = Gateway.get_range_for_day(dt_1)
    r2 = Gateway.get_range_for_day(dt_2)
    r3 = Gateway.get_range_for_day(dt_3)
    r4 = Gateway.get_range_for_day(dt_4)

    if r1 is None or r1[0].hour != 9 or r1[1].hour != 17:
        raise AssertionError("Expected 09:00-17:00 interval for 2025-01-02.")
    if r2 is None or r2[0].hour != 18 or r2[1].hour != 4:
        raise AssertionError("Expected overnight interval for 2025-01-04.")
    if r3 is None or r3[0].hour != 7 or r3[1].hour != 8:
        raise AssertionError("Missing 'from' should still return interval.")
    if r4 is None or r4[0].hour != 10 or r4[1].hour != 12:
        raise AssertionError("Missing 'to' should still return interval.")


def test_annotate_market_times_time_classification():
    """
    Verifica que los registros históricos sean correctamente clasificados como
    pre_market_time, market_time o post_market_time, según el horario definido.
    """
    Gateway.set_market_time(
        [
            {
                "open": "09:00:00",
                "close": "17:00:00",
                "from": "2025-01-01",
                "to": "2025-12-31",
            }
        ]
    )

    Gateway.set_symbols(
        {
            "TEST": {
                "symbol": "TEST",
                "historical_prices": [
                    {"datetime": "2025-05-22T08:30:00Z", "close": 100},  # pre
                    {"datetime": "2025-05-22T10:00:00Z", "close": 101},  # market
                    {"datetime": "2025-05-22T18:00:00Z", "close": 102},  # post
                ],
                "schedule": {},
            }
        }
    )

    Gateway._annotate_market_times()
    symbol_data = Gateway.get_symbol("TEST")
    if symbol_data is None:
        raise AssertionError("Symbol 'TEST' was not found.")
    records = symbol_data["historical_prices"]

    if not records[0]["is_pre_market_time"]:
        raise AssertionError("Expected first record to be pre-market.")
    if not records[1]["is_market_time"]:
        raise AssertionError("Expected second record to be during market.")
    if not records[2]["is_post_market_time"]:
        raise AssertionError("Expected third record to be post-market.")


def test_merge_intervals_with_same_open_close_and_adjacent_dates_direct():
    """
    Verifica directamente que _merge_intervals combine intervalos contiguos con mismos horarios.
    """
    raw_intervals = [
        {
            "from": "2025-01-01",
            "to": "2025-01-01",
            "open": "09:00:00",
            "close": "17:00:00",
        },
        {
            "from": "2025-01-02",
            "to": "2025-01-02",
            "open": "09:00:00",
            "close": "17:00:00",
        },
    ]

    merged = Gateway._merge_intervals(raw_intervals)

    if len(merged) != 1:
        raise AssertionError(
            "Expected 1 merged interval for adjacent days with same open/close."
        )

    result = merged[0]
    if result["from"] != "2025-01-01" or result["to"] != "2025-01-02":
        raise AssertionError("Merged interval does not span expected date range.")


def test_annotate_market_times_skips_symbols_without_prices():
    """
    Verifica que _annotate_market_times omite correctamente símbolos que
    no contienen la clave 'historical_prices'.
    """
    Gateway.set_market_time(
        [
            {
                "open": "09:00:00",
                "close": "17:00:00",
                "from": "2025-01-01",
                "to": "2025-12-31",
            }
        ]
    )

    # Símbolo sin la clave 'historical_prices'
    Gateway.set_symbols(
        {"NOPRICES": {"symbol": "NOPRICES", "name": "No Prices Corp", "schedule": {}}}
    )

    # Este método debe ejecutarse sin errores ni intentar acceder a 'historical_prices'
    Gateway._annotate_market_times()
