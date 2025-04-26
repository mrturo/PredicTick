"""
Unit tests for the `Validator` class, which ensures the structural and semantic integrity.

of historical market data entries for financial symbols.

These tests validate core functionalities including:
- Detection of missing or invalid data columns.
- Price and volume sanity checks.
- Chronological order of records.
- Return column consistency and outlier removal.
- Configuration-driven validation logic using mocks.
"""

# pylint: disable=protected-access


import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pytest  # type: ignore

from market_data.utils.validator import Validator
from utils.logger import Logger
from utils.parameters import ParameterLoader

_PARAMS = ParameterLoader()
_REQUIRED_MARKET_RAW_COLUMNS: list[str] = _PARAMS.get("required_market_raw_columns")


@pytest.fixture
def minimal_valid(minimal_valid_entry):
    """Alias fixture for compatibility with tests."""
    return minimal_valid_entry


def test_missing_columns():
    """Test that missing required columns are properly detected."""
    df = pd.DataFrame({"open": [1], "close": [1], "volume": [1]})
    result = Validator._has_missing_columns(df, _REQUIRED_MARKET_RAW_COLUMNS)

    if not result or "Missing column" not in result:
        raise AssertionError("Expected missing column error")


def test_invalid_prices():
    """Test that invalid price configurations (e.g., low > high) are flagged."""
    df = pd.DataFrame(
        {"low": [101], "high": [100], "open": [100], "close": [100], "adj_close": [0]}
    )
    result = Validator._has_invalid_prices(df)
    if result is None:
        raise AssertionError("Expected invalid price error")


def test_check_volume_and_time_negative_volume():
    """Test that negative volume values are flagged as invalid."""
    df = pd.DataFrame(
        {"volume": [-100, 100], "datetime": pd.date_range("2023-01-01", periods=2)}
    )
    error = Validator._check_volume_and_time("AAPL", df)
    if error is None or "Negative volume" not in error:
        raise AssertionError("Expected negative volume error")


def test_check_volume_and_time_irregular_intervals():
    """Test that time intervals with high standard deviation are flagged."""
    df = pd.DataFrame(
        {
            "volume": [100] * 5,
            "datetime": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-10", "2023-01-11", "2023-01-12"]
            ),
        }
    )
    error = Validator._check_volume_and_time("AAPL", df)
    if not error or "Inconsistent time intervals" not in error:
        raise AssertionError("Expected time interval consistency error")


def test_basic_checks_missing_datetime(monkeypatch):
    """Test that the datetime column is required and properly validated."""
    df = pd.DataFrame({"open": [1], "close": [1], "volume": [1]})
    monkeypatch.setattr(
        Validator,
        "_REQUIRED_MARKET_RAW_COLUMNS",
        ["open", "close", "volume", "datetime"],
    )
    monkeypatch.setattr(Validator._PARAMS, "get", lambda key, default=None: 1)
    error = Validator._basic_checks("AAPL", df, ["open", "close", "volume", "datetime"])
    if error is None or "Missing column" not in error:
        raise AssertionError("Expected missing datetime column error")


def test_basic_checks_datetime_unsorted(monkeypatch):
    """Test that unsorted datetime entries are detected."""
    df = pd.DataFrame(
        {
            "open": [1, 2],
            "close": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "adj_close": [1, 2],
            "volume": [1, 2],
            "datetime": pd.to_datetime(["2023-01-02", "2023-01-01"]),
        }
    )
    monkeypatch.setattr(Validator, "_REQUIRED_MARKET_RAW_COLUMNS", df.columns.tolist())
    monkeypatch.setattr(Validator._PARAMS, "get", lambda key, default=None: 2)

    error = Validator._basic_checks("AAPL", df, _REQUIRED_MARKET_RAW_COLUMNS)
    if error != "datetime column is not sorted":
        raise AssertionError("Expected datetime not sorted error")


def test_invalid_prices_non_positive_column():
    """Test that non-positive price entries (e.g., open=0) are flagged."""
    df = pd.DataFrame(
        {"low": [0], "high": [2], "open": [1], "close": [2], "adj_close": [2]}
    )
    result = Validator._has_invalid_prices(df)
    expected_message: str = "Non-positive 'low' price found"
    if result != expected_message:
        raise AssertionError(f"Expected: '{expected_message}', result: '{result}'")


def test_check_volume_and_time_zero_volume(monkeypatch):
    """Test that zero-volume entries log warnings but don't raise validation errors."""

    def _capture_warning(msg):
        setattr(Logger, "_last_warning", msg)

    monkeypatch.setattr("utils.logger.Logger.warning", _capture_warning)

    df = pd.DataFrame(
        {"volume": [0, 100], "datetime": pd.date_range("2023-01-01", periods=2)}
    )
    error = Validator._check_volume_and_time("AAPL", df)
    if error is not None:
        raise AssertionError("Expected no error for zero volume warning case")
    if getattr(Logger, "_last_warning", "") != "AAPL contains zero volume entries":
        raise AssertionError("Expected warning for zero volume entries")


def test_basic_checks_datetime_not_sorted(monkeypatch):
    """Test detection of unsorted datetime values prior to forced sorting."""
    df = pd.DataFrame(
        {
            "open": [100, 101],
            "close": [100, 101],
            "high": [101, 102],
            "low": [99, 100],
            "adj_close": [100, 101],
            "volume": [1000, 1000],
            "datetime": pd.to_datetime(["2023-01-02", "2023-01-01"]),
        }
    )
    monkeypatch.setattr(Validator, "_REQUIRED_MARKET_RAW_COLUMNS", df.columns.tolist())
    monkeypatch.setattr(Validator._PARAMS, "get", lambda key, default=None: 2)
    # Orden invertido y luego se fuerza a verificar antes de sort
    error = Validator._basic_checks("AAPL", df, _REQUIRED_MARKET_RAW_COLUMNS)
    if error != "datetime column is not sorted":
        raise AssertionError("Expected datetime not sorted error")


def test_basic_checks_invalid_price(monkeypatch):
    """Test that prices with zero or negative values are invalidated correctly."""
    df = pd.DataFrame(
        {
            "low": [100, 0],
            "open": [100, 0],
            "close": [100, 0],
            "adj_close": [100, 0],
            "high": [100, 0],
            "volume": [1000, 0],
            "datetime": pd.date_range("2023-01-01", periods=2),
        }
    )
    monkeypatch.setattr(Validator, "_REQUIRED_MARKET_RAW_COLUMNS", df.columns.tolist())
    monkeypatch.setattr(Validator._PARAMS, "get", lambda key, default=None: 1)
    error = Validator._basic_checks("AAPL", df, _REQUIRED_MARKET_RAW_COLUMNS)
    expected_message: str = "Non-positive 'open' price found"
    if error != expected_message:
        raise AssertionError(f"Expected: '{expected_message}', result: '{error}'")


def test_basic_checks_empty_dataframe():
    """Test that an empty DataFrame is rejected with an appropriate error."""
    df = pd.DataFrame()
    error = Validator._basic_checks("AAPL", df, _REQUIRED_MARKET_RAW_COLUMNS)
    if error != "Historical prices are empty":
        raise AssertionError("Expected error for empty historical prices")


def test_basic_checks_insufficient_data(monkeypatch):
    """Test that a DataFrame with fewer rows than required is flagged as insufficient."""
    df = pd.DataFrame(
        {
            "open": [1, 2],
            "close": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "adj_close": [1, 2],
            "volume": [1000, 1000],
            "datetime": pd.date_range("2023-01-01", periods=2),
        }
    )
    monkeypatch.setattr(Validator, "_REQUIRED_MARKET_RAW_COLUMNS", df.columns.tolist())
    monkeypatch.setattr(Validator._PARAMS, "get", lambda key, default=None: 10)

    error = Validator._basic_checks("AAPL", df, _REQUIRED_MARKET_RAW_COLUMNS)
    if error != "Insufficient historical data points":
        raise AssertionError("Expected insufficient data points error")


def test_basic_checks_pass_all(monkeypatch):
    """Test that _basic_checks returns None when all validations pass."""
    df = pd.DataFrame(
        {
            "open": [100, 101, 102],
            "close": [100.5, 101.5, 102.5],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "adj_close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1050],
            "datetime": pd.date_range("2023-01-01", periods=3),
        }
    )
    monkeypatch.setattr(Validator, "_REQUIRED_MARKET_RAW_COLUMNS", df.columns.tolist())
    monkeypatch.setattr(Validator._PARAMS, "get", lambda key, default=None: 2)
    error = Validator._basic_checks("AAPL", df, _REQUIRED_MARKET_RAW_COLUMNS)
    if error is not None:
        raise AssertionError(f"Expected no error, but got: {error}")


def test_validate_symbol_entry_empty_symbol():
    """Test that an empty symbol returns the correct error."""
    result = Validator._validate_symbol_entry({"symbol": "  "}, True)
    if result != ("", None, False, "Symbol is empty"):
        raise AssertionError("Expected 'Symbol is empty' error")


def test_validate_symbol_entry_symbol_not_listed(monkeypatch):
    """Test that a symbol not in _ALL_SYMBOLS returns the correct error."""
    monkeypatch.setattr(Validator, "_ALL_SYMBOLS", {"AAPL", "GOOG"})
    result = Validator._validate_symbol_entry({"symbol": "MSFT"}, True)
    if result != ("MSFT", None, False, "Symbol is not listed in symbol repository"):
        raise AssertionError("Expected error for symbol not in list")


def test_validate_symbol_entry_fails_basic_checks(monkeypatch):
    """Test that failure in _basic_checks is returned correctly."""
    monkeypatch.setattr(Validator, "_ALL_SYMBOLS", {"AAPL"})
    result = Validator._validate_symbol_entry(
        {
            "symbol": "AAPL",
            "historical_prices": [],  # Provoca "Historical prices are empty"
        },
        True,
    )
    if result != ("AAPL", None, False, "Historical prices are empty"):
        raise AssertionError("Expected error from _basic_checks")


def test_validate_data_invalid_format(monkeypatch):
    """Test case where RawData.load returns a non-list structure."""

    monkeypatch.setattr(
        "market_data.raw_data.RawData.load",
        lambda: {"symbols": {"AAPL": {"symbol": "AAPL"}}},
    )

    def _return_empty_set():
        return set()

    monkeypatch.setattr(
        Validator._PARAMS.symbol_repo, "get_invalid_symbols", _return_empty_set
    )

    result = Validator.validate_data(True)
    if result:
        raise AssertionError("Expected validation to fail due to invalid data format")


def test_update_clean_symbols(monkeypatch, symbol_metadata):
    """Test that cleaned symbols are updated correctly in the RawData."""
    updated_symbols = {}
    saved_called = {"called": False}

    # Mock RawData.get_symbol to return dummy metadata for AAPL only
    def mock_get_symbol(symbol):
        return symbol_metadata if symbol == "AAPL" else None

    # Capture set_symbol calls
    def mock_set_symbol(symbol, records, metadata):
        updated_symbols[symbol] = (records, metadata)

    # Capture save call
    def mock_save():
        saved_called["called"] = True

    # Apply monkeypatches
    monkeypatch.setattr("market_data.raw_data.RawData.get_symbol", mock_get_symbol)
    monkeypatch.setattr("market_data.raw_data.RawData.set_symbol", mock_set_symbol)
    monkeypatch.setattr("market_data.raw_data.RawData.save", mock_save)

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=2),
            "open": [100, 101],
            "close": [100.5, 101.5],
            "high": [101, 102],
            "low": [99, 100],
            "adj_close": [100.5, 101.5],
            "volume": [1000, 1100],
            "return": [0.0, 0.01],
        }
    )

    clean_dataframes = {"AAPL": df, "MSFT": df}  # Only AAPL has valid metadata
    Validator._update_clean_symbols(clean_dataframes)

    if "AAPL" not in updated_symbols:
        raise AssertionError("Expected AAPL to be updated in RawData")

    if "MSFT" in updated_symbols:
        raise AssertionError(
            "Did not expect MSFT to be updated due to missing metadata"
        )

    if not saved_called["called"]:
        raise AssertionError("Expected RawData.save to be called once")


def test_validate_data_symbols_not_dict(monkeypatch):
    """Test that validation fails if 'symbols' is not a dict of entries."""
    monkeypatch.setattr(
        "market_data.raw_data.RawData.load",
        lambda: {"symbols": ["AAPL", "MSFT"]},  # estructura inválida
    )

    def dummy_get_invalid_symbols():
        return set()

    monkeypatch.setattr(
        Validator._PARAMS.symbol_repo,
        "get_invalid_symbols",
        dummy_get_invalid_symbols,
    )

    last_error = {}

    def capture_error(msg):
        last_error["msg"] = msg

    monkeypatch.setattr("utils.logger.Logger.error", capture_error)

    result = Validator.validate_data(True)

    if result is True:
        raise AssertionError("Expected False due to invalid data format")
    if (
        "Invalid data format for validation. Expected dict of symbol entries."
        not in last_error.get("msg", "")
    ):
        raise AssertionError("Expected specific error message for invalid format")


def test_validate_symbols_success_and_changed(monkeypatch):
    """Test that valid symbol entries are processed and marked as changed when appropriate."""

    monkeypatch.setattr(Validator, "_ALL_SYMBOLS", {"AAPL"})
    monkeypatch.setattr(
        Validator, "_REQUIRED_MARKET_RAW_COLUMNS", _REQUIRED_MARKET_RAW_COLUMNS
    )
    monkeypatch.setattr(Validator._PARAMS, "get", lambda key, default=None: 2)

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=3),
            "low": [99, 100, 101],
            "open": [100, 101, 102],
            "close": [100.0, 110.0, 121.0],
            "high": [101, 111, 122],
            "adj_close": [100.0, 110.0, 121.0],
            "volume": [1000, 1000, 1000],
        }
    )

    entry = {"symbol": "AAPL", "historical_prices": df.to_dict(orient="records")}
    success, fail, _issues, clean_df = Validator._validate_symbols([entry], True)

    if success != 1:
        raise AssertionError("Expected one successful symbol")
    if fail != 0:
        raise AssertionError("Expected zero failures")
    if clean_df is None or "AAPL" not in clean_df:
        raise AssertionError("Expected AAPL in clean dataframes")


def test_validate_data_with_invalid_symbols(monkeypatch):
    """Ensure that symbols marked as invalid reduce valid_expected_symbols count."""
    symbols = {
        "AAPL": {"symbol": "AAPL", "historical_prices": []},
        "XYZ": {"symbol": "XYZ", "historical_prices": []},
    }

    monkeypatch.setattr(
        "market_data.raw_data.RawData.load", lambda: {"symbols": symbols}
    )
    monkeypatch.setattr(
        Validator._PARAMS.symbol_repo, "get_invalid_symbols", lambda: {"XYZ"}
    )

    captured = {"msg": ""}

    def capture_warning(msg):
        if "Expected valid symbols" in msg:
            captured["msg"] = msg

    monkeypatch.setattr("utils.logger.Logger.warning", capture_warning)

    Validator.validate_data(True)

    if "Expected valid symbols: 1" not in captured["msg"]:
        raise AssertionError("Expected warning for invalid symbols")


def test_validate_data_with_clean_data(monkeypatch):
    """Ensure that clean symbols are updated when clean_dataframes is not None."""

    symbols = {
        "AAPL": {
            "symbol": "AAPL",
            "historical_prices": [
                {
                    "datetime": "2023-01-01",
                    "open": 100,
                    "close": 110,
                    "high": 111,
                    "low": 99,
                    "adj_close": 110,
                    "volume": 1000,
                    "return": 0.0,
                },
                {
                    "datetime": "2023-01-02",
                    "open": 110,
                    "close": 121,
                    "high": 122,
                    "low": 109,
                    "adj_close": 121,
                    "volume": 1000,
                    "return": 0.02,
                },
            ],
        }
    }

    monkeypatch.setattr(
        "market_data.raw_data.RawData.load", lambda: {"symbols": symbols}
    )
    monkeypatch.setattr(Validator._PARAMS.symbol_repo, "get_invalid_symbols", set)
    monkeypatch.setattr(Validator, "_ALL_SYMBOLS", {"AAPL"})
    monkeypatch.setattr(
        Validator,
        "_REQUIRED_MARKET_RAW_COLUMNS",
        ["datetime", "open", "close", "high", "low", "adj_close", "volume", "return"],
    )
    monkeypatch.setattr(Validator._PARAMS, "get", lambda key, default=None: 2)

    updated = {"called": False}

    def mock_update(_data):
        updated["called"] = True

    monkeypatch.setattr(
        "market_data.validator.Validator._update_clean_symbols", mock_update
    )

    Validator.validate_data(True)

    if not updated["called"]:
        raise AssertionError("Expected _update_clean_symbols to be called")


def test_validate_data_excluded_non_dict_entries(monkeypatch):
    """Test that non-dict entries in symbol data are counted and logged as excluded."""
    symbols = {
        "AAPL": {"symbol": "AAPL", "historical_prices": []},
        "INVALID": "not_a_dict",
    }

    monkeypatch.setattr(
        "market_data.raw_data.RawData.load", lambda: {"symbols": symbols}
    )
    monkeypatch.setattr(Validator._PARAMS.symbol_repo, "get_invalid_symbols", set)

    warnings = []

    def capture_warning(msg):
        warnings.append(msg)

    monkeypatch.setattr("utils.logger.Logger.warning", capture_warning)

    Validator.validate_data(True)

    if not any("Skipped 1 non-dict entries in symbol data" in msg for msg in warnings):
        raise AssertionError("Expected warning for skipped non-dict entries")


def test_set_nan_if_not_empty_behavior():
    """Test _set_nan_if_not_empty returns expected result based on index content."""
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

    # Caso con índice vacío: no debería modificar nada
    df_unmodified = df.copy()
    df_result, changed = Validator._set_nan_if_not_empty(
        df.copy(), pd.Index([]), "value"
    )
    if changed:
        raise AssertionError("Expected no change when index is empty")
    if not df_result.equals(df_unmodified):
        raise AssertionError("DataFrame should remain unchanged when index is empty")

    # Caso con índice no vacío: debería modificar el DataFrame
    idx = pd.Index([1])
    df_expected = df.copy()
    df_expected.loc[idx, "value"] = np.nan
    df_result, changed = Validator._set_nan_if_not_empty(df.copy(), idx, "value")
    if not changed:
        raise AssertionError("Expected change when index is not empty")
    if not df_result.equals(df_expected):
        raise AssertionError("Expected DataFrame with NaN at index 1")


def test_validate_symbols_adds_changed_symbol(monkeypatch):
    """Ensure that changed symbol is added to changed_symbols set."""
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=3),
            "low": [99, 100, 101],
            "open": [100, 101, 102],
            "close": [100.0, 110.0, 121.0],
            "high": [101, 111, 122],
            "adj_close": [100.0, 110.0, 121.0],
            "volume": [1000, 1000, 1000],
            "return": [0.0, 0.01, 0.03],
        }
    )

    monkeypatch.setattr(Validator, "_ALL_SYMBOLS", {"AAPL"})
    monkeypatch.setattr(
        Validator, "_REQUIRED_MARKET_ENRICHED_COLUMNS", df.columns.tolist()
    )
    monkeypatch.setattr(Validator._PARAMS, "get", lambda key, default=None: 2)

    entry = {"symbol": "AAPL", "historical_prices": df.to_dict(orient="records")}
    _, _, _, clean_df = Validator._validate_symbols([entry], raw_flow=False)

    if not clean_df or "AAPL" not in clean_df:
        raise AssertionError("Expected AAPL to be part of clean dataframes")


def test_validate_symbols_sets_changed_flag(monkeypatch):
    """Test that _validate_symbols includes symbol if marked as changed=True."""

    def mock_validate_symbol_entry(_entry, _raw_flow):
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2023-01-01", periods=3),
                "open": [100, 101, 102],
                "close": [100.0, 110.0, 121.0],
                "high": [101, 111, 122],
                "low": [99, 109, 120],
                "adj_close": [100.0, 110.0, 121.0],
                "volume": [1000, 1000, 1000],
            }
        )
        return "AAPL", df, True, None

    monkeypatch.setattr(
        "market_data.validator.Validator._validate_symbol_entry",
        mock_validate_symbol_entry,
    )

    success, fail, _, clean = Validator._validate_symbols(
        [{"symbol": "AAPL", "historical_prices": []}], raw_flow=True
    )

    if success != 1 or fail != 0:
        raise AssertionError("Expected one success and zero failures")
    if not clean or "AAPL" not in clean:
        raise AssertionError("Expected AAPL to be included in clean_dataframes")


def test_invalid_prices_low_gt_open():
    """Test that low > close is flagged as invalid."""
    df = pd.DataFrame(
        {
            "low": [100],
            "open": [99],
            "close": [102],
            "high": [103],
            "adj_close": [102],
        }
    )
    result = Validator._has_invalid_prices(df)
    expected_message: str = "Invalid price range: low > open"
    if result != expected_message:
        raise AssertionError(f"Expected: '{expected_message}', result: '{result}'")


def test_invalid_prices_low_gt_close():
    """Test that low > close is flagged as invalid."""
    df = pd.DataFrame(
        {
            "low": [100],
            "open": [101],
            "close": [99],
            "high": [103],
            "adj_close": [102],
        }
    )
    result = Validator._has_invalid_prices(df)
    expected_message: str = "Invalid price range: low > close"
    if result != expected_message:
        raise AssertionError(f"Expected: '{expected_message}', result: '{result}'")


def test_invalid_prices_low_gt_adj_close():
    """Test that low > adj_close is flagged as invalid."""
    df = pd.DataFrame(
        {
            "low": [100],
            "open": [100],
            "close": [105],
            "high": [110],
            "adj_close": [99],
        }
    )
    result = Validator._has_invalid_prices(df)
    expected_message: str = "Invalid price range: low > adj_close"
    if result != expected_message:
        raise AssertionError(f"Expected: '{expected_message}', result: '{result}'")


def test_invalid_prices_high_lt_open():
    """Test that high < open is flagged as invalid."""
    df = pd.DataFrame(
        {
            "low": [95],
            "open": [106],
            "close": [100],
            "high": [105],
            "adj_close": [100],
        }
    )
    result = Validator._has_invalid_prices(df)
    expected_message: str = "Invalid price range: high < open"
    if result != expected_message:
        raise AssertionError(f"Expected: '{expected_message}', result: '{result}'")


def test_invalid_prices_high_lt_close():
    """Test that high < close is flagged as invalid."""
    df = pd.DataFrame(
        {
            "low": [95],
            "open": [100],
            "close": [106],
            "high": [105],
            "adj_close": [106],
        }
    )
    result = Validator._has_invalid_prices(df)
    expected_message: str = "Invalid price range: high < close"
    if result != expected_message:
        raise AssertionError(f"Expected: '{expected_message}', result: '{result}'")


def test_invalid_prices_high_lt_adj_close():
    """Test that high < adj_close is flagged as invalid."""
    df = pd.DataFrame(
        {
            "low": [95],
            "open": [100],
            "close": [100],
            "high": [104],
            "adj_close": [106],
        }
    )
    result = Validator._has_invalid_prices(df)
    expected_message: str = "Invalid price range: high < adj_close"
    if result != expected_message:
        raise AssertionError(f"Expected: '{expected_message}', result: '{result}'")


def test_validate_symbol_entry_symbol_not_str():
    """Test that a non-string symbol raises a TypeError."""
    with pytest.raises(TypeError) as exc_info:
        Validator._validate_symbol_entry({"symbol": 12345}, True)
    if "'symbol' value must be of type str" not in str(exc_info.value):
        raise AssertionError("Expected TypeError for non-string symbol")
