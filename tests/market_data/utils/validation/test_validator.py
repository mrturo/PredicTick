"""Unit tests for the Validator class in the market data validation pipeline.

This test suite verifies the integrity, structure, and consistency of the
`Validator` class used to validate historical OHLCV financial market data.

Covered features:
    - Required column presence
    - Price validation via range and sign checks
    - Volume validity and timestamp interval regularity
    - DataFrame structural checks (sorting, monotonicity, emptiness)
    - Symbol membership and metadata enforcement
    - Internal utility functions such as `_set_nan_if_not_empty` and `_validate_time_deltas`
    - Full validation pipeline entry via `validate_data`

Mocking is used for configuration and external dependencies to isolate logic.

All tests follow the `unittest` framework and ensure the Validator operates
reliably under both expected and edge-case conditions."""

# pylint: disable=protected-access,too-many-public-methods

import unittest
from unittest.mock import patch

import pandas as pd  # type: ignore

from market_data.utils.validation.validator import Validator
from utils.parameters import ParameterLoader


class TestValidator(unittest.TestCase):
    """Unit tests for the Validator class integrity and structure checks."""

    def setUp(self):
        self.params = ParameterLoader()
        self.required_cols = self.params.get("required_market_raw_columns")
        self.symbol = "AAPL"
        self.valid_data = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=5, freq="D"),
                "open": [150, 152, 154, 156, 158],
                "high": [151, 153, 155, 157, 159],
                "low": [149, 151, 153, 155, 157],
                "close": [150.5, 152.5, 154.5, 156.5, 158.5],
                "adj_close": [150.4, 152.4, 154.4, 156.4, 158.4],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

    def test_missing_required_column(self):
        """Should return an error if any required column is missing."""
        for col in self.required_cols:
            df_missing = self.valid_data.drop(columns=[col])
            error = Validator._has_missing_columns(df_missing, self.required_cols)
            self.assertEqual(error, f"Missing column: {col}")

    def test_invalid_prices_returns_error(self):
        """Should detect invalid prices via price validators."""
        df = self.valid_data.copy()
        df.loc[0, "high"] = df.loc[0, "low"] - 1
        with patch(
            "market_data.utils.validation.price_validator.PriceValidator.check_price_ranges"
        ) as mock_range:
            mock_range.return_value = "Range error"
            result = Validator.has_invalid_prices(df, raw_flow=True)
            self.assertEqual(result, "Range error")

    def test_invalid_sign_returns_error(self):
        """Should detect sign-related price anomalies."""
        df = self.valid_data.copy()
        with patch(
            "market_data.utils.validation.price_validator.PriceValidator.check_price_ranges"
        ) as mock_range, patch(
            "market_data.utils.validation.price_validator.PriceValidator.check_nonpositive_prices"
        ) as mock_sign:
            mock_range.return_value = None
            mock_sign.return_value = "Non-positive price error"
            result = Validator.has_invalid_prices(df, raw_flow=True)
            self.assertEqual(result, "Non-positive price error")

    def test_valid_prices_pass_validation(self):
        """Should return None if both range and sign validations pass."""
        df = self.valid_data.copy()
        with patch(
            "market_data.utils.validation.price_validator.PriceValidator.check_price_ranges"
        ) as mock_range, patch(
            "market_data.utils.validation.price_validator.PriceValidator.check_nonpositive_prices"
        ) as mock_sign:
            mock_range.return_value = None
            mock_sign.return_value = None
            result = Validator.has_invalid_prices(df, raw_flow=True)
            self.assertIsNone(result)

    def test_negative_volume_detected(self):
        """Should return error on negative volume."""
        df = self.valid_data.copy()
        df.loc[2, "volume"] = -100
        error = Validator._check_volume_and_time(self.symbol, df)
        self.assertEqual(error, "Negative volume found")

    def test_inconsistent_time_intervals1(self):
        """Should detect non-uniform datetime deltas."""
        df = self.valid_data.copy()
        df.loc[2, "datetime"] = df.loc[2, "datetime"] + pd.Timedelta(hours=12)
        df.sort_values("datetime", inplace=True)
        checked = Validator._check_volume_and_time(self.symbol, df)
        self.assertEqual(checked, None)

    def test_insufficient_history(self):
        """Should return error if data length is under minimum threshold."""
        df = self.valid_data.iloc[:2].copy()
        with patch.object(Validator._PARAMS, "get", return_value=5):
            result = Validator._basic_checks(self.symbol, df, self.required_cols, True)
            self.assertEqual(result, "Insufficient historical data points")

    def test_unsorted_datetime_returns_error(self):
        """Should return error if 'datetime' is not monotonic increasing."""
        df_unsorted = self.valid_data.iloc[::-1].copy()
        error = Validator._basic_checks(
            self.symbol, df_unsorted, self.required_cols, raw_flow=True  # type: ignore
        )
        self.assertEqual(error, "datetime column is not sorted")

    def test_empty_dataframe(self):
        """Should return error if historical DataFrame is empty."""
        df = pd.DataFrame()
        result = Validator._basic_checks(self.symbol, df, self.required_cols, True)
        self.assertEqual(result, "Historical prices are empty")

    def test_inconsistent_time_intervals2(self):
        """Should detect non-uniform datetime deltas."""
        df = self.valid_data.copy()
        df["datetime"] = [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-04"),
            pd.Timestamp("2024-01-05"),
            pd.Timestamp("2024-01-06"),
        ]
        error = Validator._check_volume_and_time(self.symbol, df)
        self.assertEqual(error, None)

    def test_validate_symbol_entry_with_invalid_symbol_type(self):
        """Should raise TypeError if symbol is not a string."""
        entry = {"symbol": 123, "historical_prices": self.valid_data.to_dict("records")}
        with self.assertRaises(TypeError):
            Validator._validate_symbol_entry(entry, raw_flow=True)

    def test_validate_symbol_entry_with_empty_symbol(self):
        """Should return error if symbol is empty string."""
        entry = {"symbol": "", "historical_prices": self.valid_data.to_dict("records")}
        _symbol, df, changed, error = Validator._validate_symbol_entry(
            entry, raw_flow=True
        )
        self.assertEqual(error, "Symbol is empty")
        self.assertIsNone(df)
        self.assertFalse(changed)

    def test_validate_symbol_entry_not_in_repo(self):
        """Should return error if symbol is not in repository."""
        entry = {
            "symbol": "FAKE",
            "historical_prices": self.valid_data.to_dict("records"),
        }
        symbol, _df, _changed, error = Validator._validate_symbol_entry(
            entry, raw_flow=True
        )
        self.assertEqual(symbol, "FAKE")
        self.assertEqual(error, "Symbol is not listed in symbol repository")

    def test_validate_symbol_entry_with_valid_data(self):
        """Should return DataFrame and no error for valid entry."""
        entry = {
            "symbol": self.symbol,
            "historical_prices": self.valid_data.to_dict("records"),
        }
        with patch.object(Validator._PARAMS, "get", return_value=5):
            symbol, df, changed, error = Validator._validate_symbol_entry(
                entry, raw_flow=True
            )
            self.assertIsNone(error)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(symbol, self.symbol)
            self.assertFalse(changed)

    def test_validate_symbols_with_mixed_data(self):
        """Should separate valid and invalid symbol entries."""
        entries = [
            {"symbol": "FAKE", "historical_prices": self.valid_data.to_dict("records")},
            {
                "symbol": self.symbol,
                "historical_prices": self.valid_data.to_dict("records"),
            },
        ]
        with patch.object(Validator._PARAMS, "get", return_value=5):
            success, fail, issues, clean = Validator._validate_symbols(
                entries, raw_flow=True
            )
            self.assertEqual(success, 1)
            self.assertEqual(fail, 1)
            self.assertIn("FAKE", issues)
            self.assertIn(self.symbol, clean)

    def test_validate_data_with_non_dict_input_true(self):
        """Should log and return False if input is not a dictionary."""
        with patch("utils.logger.Logger.error") as mock_log:
            result = Validator.validate_data(["not", "a", "dict"], raw_flow=True)
            self.assertFalse(result)
            self.assertIn("Invalid data format", mock_log.call_args[0][0])

    def test_validate_data_with_non_dict_input_false(self):
        """Should log and return False if input is not a dictionary."""
        with patch("utils.logger.Logger.error") as mock_log:
            result = Validator.validate_data(["not", "a", "dict"], raw_flow=False)
            self.assertFalse(result)
            self.assertIn("Invalid data format", mock_log.call_args[0][0])

    def test_validate_data_with_excluded_symbols_and_warnings(self):
        """Should handle dict with valid and non-dict entries, log and validate clean ones."""
        symbols_data = {
            "AAPL": {
                "symbol": self.symbol,
                "historical_prices": self.valid_data.to_dict("records"),
            },
            "BAD": "non-dict",
        }
        with patch.object(Validator._PARAMS, "get", return_value=5), patch.object(
            Validator._PARAMS.symbol_repo, "get_invalid_symbols", return_value=set()
        ), patch("utils.logger.Logger.warning") as mock_warning:
            result = Validator.validate_data(symbols_data, raw_flow=True)
            self.assertTrue(result)
            self.assertTrue(mock_warning.called)

    def test_validate_time_deltas_with_inconsistent_intervals(self):
        """Should return error for inconsistent time intervals (std > 86400)."""
        times = pd.to_datetime(
            [
                "2024-01-01 00:00:00",
                "2024-01-02 00:00:00",
                "2024-01-05 12:00:00",
            ]
        )
        deltas = times.to_series().diff().dropna()
        result = Validator._validate_time_deltas(deltas)
        self.assertEqual(result, "Inconsistent time intervals detected")

    def test_validate_time_deltas_with_consistent_intervals(self):
        """Should return None for consistent time intervals (std <= 86400)."""
        times = pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ]
        )
        deltas = times.to_series().diff().dropna()
        result = Validator._validate_time_deltas(deltas)
        self.assertIsNone(result)

    def test_set_nan_if_not_empty_with_non_empty_index(self):
        """Should set NaN at specified index and return True flag."""
        df = self.valid_data.copy()
        idx = df[df["volume"] > 1200].index
        updated_df, changed = Validator._set_nan_if_not_empty(df, idx, "volume")
        self.assertTrue(changed)
        self.assertTrue(updated_df.loc[idx, "volume"].isna().all())

    def test_set_nan_if_not_empty_with_empty_index(self):
        """Should return unchanged DataFrame and False flag if index is empty."""
        df = self.valid_data.copy()
        empty_idx = pd.Index([])
        updated_df, changed = Validator._set_nan_if_not_empty(df, empty_idx, "volume")
        self.assertFalse(changed)
        pd.testing.assert_frame_equal(updated_df, df)

    def test_validate_symbols_detects_changed_symbols(self):
        """Should correctly track symbols marked as changed during validation."""
        entry = {
            "symbol": self.symbol,
            "historical_prices": self.valid_data.to_dict("records"),
        }
        mock_return = (self.symbol, self.valid_data.copy(), True, None)
        with patch.object(
            Validator, "_validate_symbol_entry", return_value=mock_return
        ):
            success, fail, _issues, clean = Validator._validate_symbols(
                [entry], raw_flow=True
            )
            self.assertEqual(success, 1)
            self.assertEqual(fail, 0)
            self.assertIn(self.symbol, clean)

    def test_update_clean_symbols_skips_missing_metadata(self):
        """Should skip symbol if metadata is missing in original dataset."""
        clean_df = self.valid_data.copy()
        clean_dataframes = {"AAPL": clean_df}
        symbols_data = {"AAPL": None}
        with patch("utils.logger.Logger.warning") as mock_warn:
            updated = Validator._update_clean_symbols(symbols_data, clean_dataframes)
            self.assertEqual(updated, symbols_data)
            self.assertIn("AAPL", updated)
            self.assertIsNone(updated["AAPL"])
            self.assertTrue(mock_warn.called)
            self.assertIn(
                "Symbol AAPL not found in metadata", mock_warn.call_args[0][0]
            )

    def test_validate_data_logs_expected_symbols_and_issues(self):
        """Should log warning when expected symbols differ and errors are found."""
        bad_symbol = "FAKE"
        good_symbol = self.symbol
        symbols_data = {
            bad_symbol: {"symbol": bad_symbol, "historical_prices": []},
            good_symbol: {
                "symbol": good_symbol,
                "historical_prices": self.valid_data.to_dict("records"),
            },
        }
        with patch.object(Validator._PARAMS, "get", return_value=5), patch.object(
            Validator._PARAMS.symbol_repo,
            "get_invalid_symbols",
            return_value={bad_symbol},
        ), patch("utils.logger.Logger.warning") as mock_warn, patch(
            "utils.logger.Logger.debug"
        ) as mock_debug, patch(
            "utils.logger.Logger.error"
        ) as mock_error:
            result = Validator.validate_data(symbols_data, raw_flow=True)
            self.assertFalse(result)
            expected_warns = [args[0] for args, _ in mock_warn.call_args_list]
            self.assertTrue(any("Expected valid symbols" in w for w in expected_warns))
            self.assertTrue(any("Issues encountered" in w for w in expected_warns))
            expected_debugs = [args[0] for args, _ in mock_debug.call_args_list]
            self.assertTrue(any("Symbols failed" in d for d in expected_debugs))
            expected_errors = [args[0] for args, _ in mock_error.call_args_list]
            self.assertTrue(
                any(f"- {bad_symbol}:" in e or bad_symbol in e for e in expected_errors)
            )
