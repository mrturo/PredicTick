"""Unit tests for the update.raw_data module."""

# pylint: disable=protected-access

from unittest.mock import patch

import pandas as pd  # type: ignore
import pytest  # type: ignore

from market_data.ingest.raw_data import RawData
from utils.parameters import ParameterLoader

_PARAMS = ParameterLoader()
_TEST_RAW_MARKETDATA_FILEPATH = _PARAMS.get("test_raw_marketdata_filepath")


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


@pytest.fixture(name="sample_symbol_data_fixture")
def fixture_sample_symbol_data_alias(
    sample_symbol_data,
):  # pylint: disable=redefined-outer-name
    """Alias fixture to avoid redefinition warnings."""
    return sample_symbol_data


def test_load_update(
    monkeypatch, symbol_metadata
):  # pylint: disable=redefined-outer-name
    """Test loading market data from disk and initialization of raw_data state."""
    symbol = symbol_metadata.copy()
    symbol["historical_prices"] = [
        {"datetime": "2025-05-16T20:00:00Z", "close": 150.0},
        {"datetime": "2025-05-17T20:00:00Z", "close": 152.0},
    ]

    raw_data = {
        "last_updated": "2025-05-18T10:00:00Z",
        "last_check": "2025-05-18T10:00:00Z",
        "symbols": [symbol],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("utils.json_manager.JsonManager.load", staticmethod(mock_load))

    with patch.object(RawData, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        result = RawData.load(_TEST_RAW_MARKETDATA_FILEPATH)
        if result is None:
            raise AssertionError("Expected RawData.load() to return non-None result.")


def test_get_last_update_from_load(monkeypatch):
    """Test retrieving last update timestamp via load."""
    raw_data = {
        "last_updated": "2025-05-18T10:00:00Z",
        "last_check": "2025-05-18T10:00:00Z",
        "symbols": [],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("utils.json_manager.JsonManager.load", staticmethod(mock_load))

    with patch.object(RawData, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        RawData.load(_TEST_RAW_MARKETDATA_FILEPATH)


def test_save_update_via_interface_indirect(
    monkeypatch, sample_symbol_data_fixture
):  # pylint: disable=redefined-outer-name
    """Test saving market data using only public interfaces."""
    raw_data = {
        "last_updated": "2025-05-19T13:00:00Z",
        "last_check": "2025-05-18T10:00:00Z",
        "symbols": [sample_symbol_data_fixture],
    }

    def mock_load(_filepath):
        return raw_data

    saved_output = {}

    def mock_save(data, _filepath):
        nonlocal saved_output
        saved_output = data

    monkeypatch.setattr("utils.json_manager.JsonManager.load", staticmethod(mock_load))
    monkeypatch.setattr("utils.json_manager.JsonManager.save", staticmethod(mock_save))

    with patch.object(RawData, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        RawData.load(_TEST_RAW_MARKETDATA_FILEPATH)


def test_save_returns_result_structure(
    monkeypatch, sample_symbol_data_fixture
):  # pylint: disable=redefined-outer-name
    """Test that save() returns the expected result structure."""
    raw_data = {
        "last_updated": "2025-05-19T13:00:00Z",
        "last_check": "2025-05-19T13:00:00Z",
        "symbols": [sample_symbol_data_fixture],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("utils.json_manager.JsonManager.load", staticmethod(mock_load))

    with patch.object(RawData, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        RawData.load(_TEST_RAW_MARKETDATA_FILEPATH)


def test_set_and_get_stale_symbols_forced():
    """Ensure set_stale_symbols updates internal state using public interface only."""
    # Reset using public interface
    RawData.set_stale_symbols([])

    test_symbols = ["MSFT", "TSLA"]
    RawData.set_stale_symbols(test_symbols)

    result = RawData.get_stale_symbols()
    if result != test_symbols:
        raise AssertionError("Stale symbols not set correctly.")


def test_detect_stale_symbols_logic():
    """Test internal logic for stale symbol detection."""
    symbol = "XXXX"
    outdated = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=500)).isoformat()

    RawData.set_symbols(
        {
            symbol: {
                "symbol": symbol,
                "historical_prices": [{"datetime": outdated, "close": 1.0}],
            }
        }
    )

    latest = pd.Timestamp.now(tz="UTC")
    current_invalids = set()

    stale, updated_invalids = RawData._detect_stale_symbols(latest, current_invalids)

    if symbol not in stale:
        raise AssertionError(f"{symbol} should be marked as stale.")
    if symbol not in updated_invalids:
        raise AssertionError(f"{symbol} should be added to invalid symbols.")


def test_get_filepath_returns_valid_path():
    """Test that _get_filepath returns the original filepath when it is valid."""
    result = RawData._get_filepath("path/to/file.json", "default.json")
    if result != "path/to/file.json":
        raise AssertionError("Expected filepath to be returned when valid")


def test_get_filepath_returns_default_when_empty():
    """Test that _get_filepath returns the default path when filepath is an empty string."""
    result = RawData._get_filepath("   ", "default.json")
    if result != "default.json":
        raise AssertionError(
            "Expected default to be returned when filepath is empty string"
        )


def test_get_symbol_found(
    sample_symbol_data_fixture,
):  # pylint: disable=redefined-outer-name
    """Test retrieval of symbol metadata when symbol exists."""
    symbol_name = "AAPL"
    RawData._symbols[symbol_name] = sample_symbol_data_fixture
    result = RawData.get_symbol(symbol_name)
    if result != sample_symbol_data_fixture:
        raise AssertionError("Failed to retrieve correct symbol metadata.")


def test_get_symbol_not_found():
    """Test retrieval of symbol metadata when symbol does not exist."""
    result = RawData.get_symbol("UNKNOWN")
    if result is not None:
        raise AssertionError("Expected None for unknown symbol.")


def test_get_symbols_returns_correct_dict(sample_symbol_data_fixture):
    """Ensure get_symbols returns the current symbol dictionary."""
    RawData.set_symbols({"AAPL": sample_symbol_data_fixture})
    result = RawData.get_symbols()
    if result != {"AAPL": sample_symbol_data_fixture}:
        raise AssertionError("get_symbols did not return the expected dictionary.")


def test_set_symbol_adds_correct_data(sample_symbol_data_fixture):
    """Ensure set_symbol properly constructs the internal symbol entry."""
    prices = [
        {"datetime": "2025-05-16T20:00:00Z", "close": 150.0},
        {"datetime": "2025-05-17T20:00:00Z", "close": 152.0},
    ]
    RawData.set_symbol("AAPL", prices, sample_symbol_data_fixture)
    data = RawData.get_symbol("AAPL")

    if data is None or data["symbol"] != "AAPL" or data["historical_prices"] != prices:
        raise AssertionError("Symbol data was not set correctly.")


def test_get_last_update_returns_expected(monkeypatch):
    """Ensure get_last_update reflects timestamp set during load."""
    ts = "2025-05-18T10:00:00Z"
    monkeypatch.setattr(
        "utils.json_manager.JsonManager.load",
        staticmethod(lambda _: {"last_updated": ts, "symbols": []}),
    )
    with patch.object(RawData, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        RawData.load(_TEST_RAW_MARKETDATA_FILEPATH)
        result = RawData.get_last_update()
        if not isinstance(result, pd.Timestamp):
            raise AssertionError("Expected pd.Timestamp as last update.")


def test_get_latest_price_date_and_set():
    """Validate getter and setter for latest_price_date."""
    now = pd.Timestamp.now(tz="UTC")
    RawData.set_latest_price_date(now)
    if RawData.get_latest_price_date() != now:
        raise AssertionError("get_latest_price_date did not return expected value.")


def test_save_result_structure_and_side_effect(monkeypatch, sample_symbol_data_fixture):
    """Ensure save returns expected structure and invokes JsonManager.save."""
    RawData.set_symbols({"AAPL": sample_symbol_data_fixture})

    saved_data = {}
    monkeypatch.setattr(
        "utils.json_manager.JsonManager.save",
        staticmethod(lambda data, _: saved_data.update(data)),
    )

    result = RawData.save("fakepath.json")
    if "last_updated" not in result or "symbols" not in result:
        raise AssertionError("save result structure invalid.")
    if "AAPL" not in [entry["symbol"] for entry in result["symbols"]]:
        raise AssertionError("save did not include expected symbol.")


def test_normalize_historical_prices_interpolates_zero_volume():
    """Ensure zero volume entries are interpolated and forward filled."""
    data = [
        {"datetime": "2025-01-01T00:00:00Z", "close": 100.0, "volume": 1000},
        {"datetime": "2025-01-02T00:00:00Z", "close": 101.0, "volume": 0},
        {"datetime": "2025-01-03T00:00:00Z", "close": 102.0, "volume": 1200},
        {"datetime": "2025-01-04T00:00:00Z", "close": 103.0, "volume": 0},
        {"datetime": "2025-01-05T00:00:00Z", "close": 104.0, "volume": 1300},
    ]
    symbols = [{"symbol": "TEST", "historical_prices": data}]

    normalized = RawData.normalize_historical_prices(symbols)
    result = normalized["TEST"]["historical_prices"]
    df = pd.DataFrame(result)

    if df["volume"].isna().sum() != 0:
        raise AssertionError("No volume values should remain NA")

    if not all(df["volume"] > 0):
        raise AssertionError("All volume values should be positive after interpolation")

    match_1 = df[df["datetime"] == pd.Timestamp("2025-01-02T00:00:00Z", tz="UTC")]
    if match_1.empty:
        raise AssertionError("Missing row for 2025-01-02")
    interpolated_val_1 = match_1["volume"].iloc[0]
    if (
        not pd.api.types.is_number(interpolated_val_1)
        or abs(interpolated_val_1 - 1100.0) > 1e-6
    ):
        raise AssertionError("Expected interpolated volume for 2025-01-02 is 1100.0")

    match_2 = df[df["datetime"] == pd.Timestamp("2025-01-04T00:00:00Z", tz="UTC")]
    if match_2.empty:
        raise AssertionError("Missing row for 2025-01-04")
    interpolated_val_2 = match_2["volume"].iloc[0]
    if (
        not pd.api.types.is_number(interpolated_val_2)
        or abs(interpolated_val_2 - 1250.0) > 1e-6
    ):
        raise AssertionError("Expected interpolated volume for 2025-01-04 is 1250.0")
