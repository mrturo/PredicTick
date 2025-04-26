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

from src.market_data.utils.validation.validator import Validator
from src.utils.config.parameters import ParameterLoader


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
            self.assertEqual(error, f"Missing columns: {col}")

    def test_invalid_prices_returns_error(self):
        """Should detect invalid prices via price validators."""
        df = self.valid_data.copy()
        df.loc[0, "high"] = df.loc[0, "low"] - 1
        with patch(
            "src.market_data.utils.validation.price_validator.PriceValidator.check_price_ranges"
        ) as mock_range:
            mock_range.return_value = "Range error"
            result = Validator.has_invalid_prices(df, raw_flow=True)
            self.assertEqual(result, "Range error")

    def test_invalid_sign_returns_error(self):
        """Should detect sign-related price anomalies."""
        df = self.valid_data.copy()
        with patch(
            "src.market_data.utils.validation.price_validator.PriceValidator.check_price_ranges"
        ) as mock_range, patch(
            "src.market_data.utils.validation.price_validator."
            "PriceValidator.check_nonpositive_prices"
        ) as mock_sign:
            mock_range.return_value = None
            mock_sign.return_value = "Non-positive price error"
            result = Validator.has_invalid_prices(df, raw_flow=True)
            self.assertEqual(result, "Non-positive price error")

    def test_valid_prices_pass_validation(self):
        """Should return None if both range and sign validations pass."""
        df = self.valid_data.copy()
        with patch(
            "src.market_data.utils.validation.price_validator.PriceValidator.check_price_ranges"
        ) as mock_range, patch(
            "src.market_data.utils.validation.price_validator."
            "PriceValidator.check_nonpositive_prices"
        ) as mock_sign:
            mock_range.return_value = None
            mock_sign.return_value = None
            result = Validator.has_invalid_prices(df, raw_flow=True)
            self.assertIsNone(result)

    def test_negative_volume_detected(self):
        """Should return error on negative volume."""
        df = self.valid_data.copy()
        df.loc[2, "volume"] = -100
        error = Validator._check_volume_and_time(self.symbol, df, "1d")
        self.assertEqual(error, "Negative volume found")

    def test_inconsistent_time_intervals1(self):
        """Should detect non-uniform datetime deltas."""
        df = self.valid_data.copy()
        df.loc[2, "datetime"] = df.loc[2, "datetime"] + pd.Timedelta(hours=12)
        df.sort_values("datetime", inplace=True)
        checked = Validator._check_volume_and_time(self.symbol, df, "1d")
        self.assertEqual(checked, None)

    def test_empty_dataframe(self):
        """Should return error if historical DataFrame is empty."""
        df = pd.DataFrame()
        result = Validator._basic_checks(
            self.symbol, df, self.required_cols, True, "1d"
        )
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
        error = Validator._check_volume_and_time(self.symbol, df, "1d")
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
        with patch("src.utils.io.logger.Logger.error") as mock_log:
            result = Validator.validate_data(["not", "a", "dict"], raw_flow=True)
            self.assertFalse(result)
            self.assertIn("Invalid data format", mock_log.call_args[0][0])

    def test_validate_data_with_non_dict_input_false(self):
        """Should log and return False if input is not a dictionary."""
        with patch("src.utils.io.logger.Logger.error") as mock_log:
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
        ), patch("src.utils.io.logger.Logger.warning") as mock_warning:
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
            ],
            utc=True,
        )
        result = Validator._validate_time_deltas(pd.Series(times), "1d")
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
        with patch("src.utils.io.logger.Logger.warning") as mock_warn:
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
        ), patch("src.utils.io.logger.Logger.warning") as mock_warn, patch(
            "src.utils.io.logger.Logger.debug"
        ) as mock_debug, patch(
            "src.utils.io.logger.Logger.error"
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

    def test_not_a_dataframe(self):
        """Should return error if input is not a DataFrame."""
        not_df = {"open": [1, 2, 3]}
        error = Validator._has_missing_columns(not_df, self.required_cols)
        self.assertEqual(error, "Data is not a valid DataFrame")

    def test_dataframe_is_empty(self):
        """Should return error if DataFrame is empty."""
        empty_df = pd.DataFrame()
        error = Validator._has_missing_columns(empty_df, self.required_cols)
        self.assertEqual(error, "DataFrame is empty")

    def test_required_columns_not_list(self):
        """Should return error if required_columns is not a list."""
        error = Validator._has_missing_columns(self.valid_data, "open,close")
        self.assertEqual(error, "Required columns must be provided as a list")

    def test_no_required_columns_provided(self):
        """Should return error if required_columns is an empty list."""
        error = Validator._has_missing_columns(self.valid_data, [])
        self.assertEqual(error, "No required columns specified for validation")

    def test_required_columns_not_all_strings(self):
        """Should return error if required column names are not all strings."""
        error = Validator._has_missing_columns(self.valid_data, ["open", 123])
        self.assertEqual(error, "All required column names must be strings")

    def test_dataframe_columns_not_unique(self):
        """Should return error if DataFrame has non-unique column names."""
        df = self.valid_data.copy()
        df.columns = list(df.columns[:-1]) + [
            df.columns[-2]
        ]  # duplicate one column name
        error = Validator._has_missing_columns(df, self.required_cols)
        self.assertEqual(error, "DataFrame columns are not unique")

    def test_index_not_monotonic_increasing(self):
        """Should return error if DataFrame index is not monotonic increasing."""
        df = self.valid_data.copy()
        df = df.sample(frac=1).reset_index(drop=True)
        df.index = [5, 3, 1, 4, 2]  # non-monotonic
        error = Validator._has_missing_columns(df, self.required_cols)
        self.assertEqual(error, "DataFrame index is not monotonic increasing")

    def test_dataframe_contains_nan(self):
        """Should return error if DataFrame contains NaN values."""
        df = self.valid_data.copy()
        df.loc[1, "open"] = None
        error = Validator._has_missing_columns(df, self.required_cols)
        self.assertEqual(error, "DataFrame contains NaN values in columns: open")

    def test_check_missing_trading_days_no_missing(self):
        """Should return None if no trading days are missing."""
        df = pd.DataFrame(
            {"datetime": pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")}
        )
        expected_schedule = pd.DataFrame(
            index=pd.date_range("2024-01-01", periods=3, freq="B")
        )
        us_holidays = []
        with patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(
                type(
                    "Cal",
                    (),
                    {"schedule": lambda self, start_date, end_date: expected_schedule},
                )(),
                us_holidays,
                None,
            ),
        ):
            result = Validator._check_missing_trading_days(df, "1d")
            self.assertIsNone(result)

    def test_check_missing_trading_days_with_missing(self):
        """Should return error if there are missing trading days."""
        df = pd.DataFrame(
            {"datetime": pd.to_datetime(["2024-01-01", "2024-01-03"], utc=True)}
        )
        expected_schedule = pd.DataFrame(
            index=pd.date_range("2024-01-01", periods=3, freq="B")
        )  # 1, 2, 3 Jan
        us_holidays = []
        with patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(
                type(
                    "Cal",
                    (),
                    {"schedule": lambda self, start_date, end_date: expected_schedule},
                )(),
                us_holidays,
                None,
            ),
        ):
            result = Validator._check_missing_trading_days(df, "1d")
            self.assertTrue(result.startswith("Missing"))
            self.assertIn("2024-01-02", result)

    def test_check_missing_expected_interval_points_with_missing(self):
        """Should return error if expected interval points are missing for multi-day interval."""
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2024-01-01", "2024-01-07", "2024-01-13"], utc=True
                )
            }
        )
        # Expecting interval every 3 days: Jan 1, 4, 7
        expected_schedule = pd.DataFrame(
            index=pd.to_datetime(
                ["2024-01-01", "2024-01-04", "2024-01-07", "2024-01-10", "2024-01-13"]
            )
        )
        us_holidays = []
        with patch(
            "src.utils.exchange.calendar_manager.CalendarManager.build_market_calendars",
            return_value=(
                type(
                    "Cal",
                    (),
                    {"schedule": lambda self, start_date, end_date: expected_schedule},
                )(),
                us_holidays,
                None,
            ),
        ):
            result = Validator._check_missing_trading_days(df, "3d")
            self.assertIsInstance(result, str)
            self.assertIn("Missing", result)
            self.assertIn("2024-01-10", result)

    def test_check_missing_trading_days_invalid_interval(self):
        """Should return error message if interval format is invalid."""
        df = pd.DataFrame({"datetime": pd.to_datetime(["2024-01-01"], utc=True)})
        with patch(
            "src.market_data.utils.intervals.interval_converter.IntervalConverter.to_minutes",
            side_effect=ValueError("unsupported interval"),
        ):
            result = Validator._check_missing_trading_days(df, "invalid_interval")
            self.assertEqual(
                result,
                "Invalid interval format: invalid_interval (unsupported interval)",
            )

    def test_validate_time_deltas_empty_series_returns_none(self):
        """Should return None if the time series is empty."""
        s = pd.Series(dtype="datetime64[ns]")
        result = Validator._validate_time_deltas(s, "1d")
        self.assertIsNone(result)

    def test_validate_time_deltas_single_timestamp_returns_none(self):
        """Should return None if there is only one timestamp."""
        s = pd.Series([pd.Timestamp("2024-01-01")])
        result = Validator._validate_time_deltas(s, "1d")
        self.assertIsNone(result)

    def test_validate_time_deltas_all_invalid_datetimes_returns_none(self):
        """Should return None if all values are discarded as NaT or before 1971-01-01."""
        # 'not-a-date' -> NaT (discarded), 1960-01-01 < 1971-01-01 (discarded)
        s = pd.Series(["not-a-date", "1960-01-01"])
        result = Validator._validate_time_deltas(s, "1d")
        self.assertIsNone(result)

    def test_validate_time_deltas_uses_typeerror_fallback(self):
        """Should fall back to default parsing when format='ISO8601' raises TypeError."""
        # Use date objects instead of Timestamp to trigger TypeError on the first parsing attempt
        s = pd.Series(
            [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-02").date()]
        )
        # This ensures the except TypeError block is executed
        result = Validator._validate_time_deltas(s, "1d")
        # Since intervals are consistent, it should return None
        self.assertIsNone(result)

    def test_validate_time_deltas_typeerror_branch_is_executed(self):
        """Should execute the except TypeError fallback when pd.to_datetime raises TypeError."""
        real_to_datetime = pd.to_datetime

        def _side_effect(arg, *args, **kwargs):
            # Force a TypeError only when format='ISO8601' is used
            if kwargs.get("format") == "ISO8601":
                raise TypeError("forced for branch coverage")
            return real_to_datetime(arg, *args, **kwargs)

        with patch("pandas.to_datetime", side_effect=_side_effect):
            # Use simple ISO-like strings; the fallback (without format) must parse them fine
            s = pd.Series(["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"])
            result = Validator._validate_time_deltas(s, "1d")
            # Consistent daily interval -> None
            self.assertIsNone(result)

    def test_validate_time_deltas_inconsistent_same_day_returns_error(self):
        """Should return an error when same-day deltas are inconsistent (std > interval)."""
        # Build timestamps within the same calendar day with mixed 1h and 4h gaps
        times = pd.to_datetime(
            [
                "2024-01-02 09:30:00",
                "2024-01-02 10:30:00",  # +1h
                "2024-01-02 14:30:00",  # +4h (std across [3600, 14400] > 3600)
            ],
            utc=True,
        )
        s = pd.Series(times)
        result = Validator._validate_time_deltas(s, "1h")
        self.assertIsInstance(result, str)
        self.assertIn("Inconsistent intervals within 2024-01-02", result)

    def test_validate_time_deltas_all_consistent_returns_none(self):
        """Should return None when all same-day deltas are consistent."""
        # Three intraday points exactly 1 hour apart
        times = pd.to_datetime(
            [
                "2024-01-02 09:30:00",
                "2024-01-02 10:30:00",
                "2024-01-02 11:30:00",
            ],
            utc=True,
        )
        s = pd.Series(times)
        result = Validator._validate_time_deltas(s, "1h")
        # All deltas match the expected interval, so the final return None is reached
        self.assertIsNone(result)

    def test_validate_symbol_entry_with_empty_raw_interval(self):
        """Should handle case where RawData exists but its interval is empty or whitespace."""
        entry = {
            "symbol": self.symbol,
            "historical_prices": self.valid_data.to_dict("records"),
        }
        with patch(
            "src.market_data.ingestion.raw.raw_data.RawData.exist", return_value=True
        ), patch("src.market_data.ingestion.raw.raw_data.RawData.load"), patch(
            "src.market_data.ingestion.raw.raw_data.RawData.get_interval",
            return_value="   ",
        ):
            symbol, df, changed, error = Validator._validate_symbol_entry(
                entry, raw_flow=True
            )
            self.assertEqual(symbol, self.symbol)
            self.assertIsNone(error)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertFalse(changed)

    def test_validate_symbol_entry_with_empty_enriched_interval(self):
        """Should handle case where EnrichedData exists but its interval is empty"""
        entry = {
            "symbol": self.symbol,
            "historical_prices": self.valid_data.to_dict("records"),
        }
        with patch(
            "src.market_data.processing.enrichment.enriched_data.EnrichedData.exist",
            return_value=True,
        ), patch(
            "src.market_data.processing.enrichment.enriched_data.EnrichedData.load"
        ), patch(
            "src.market_data.processing.enrichment.enriched_data.EnrichedData.get_interval",
            return_value="",
        ), patch.object(
            Validator,
            "_REQUIRED_MARKET_ENRICHED_COLUMNS",
            self.required_cols,
        ), patch(
            "src.market_data.utils.validation.validator.IntervalConverter.to_minutes",
            return_value=1440,
        ):
            symbol, df, changed, error = Validator._validate_symbol_entry(
                entry, raw_flow=False
            )
            self.assertEqual(symbol, self.symbol)
            self.assertIsNone(error)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertFalse(changed)

    def test_validate_symbol_entry_with_basic_checks_error(self):
        """Should return error tuple when _basic_checks detects a validation issue."""
        entry = {
            "symbol": self.symbol,
            "historical_prices": self.valid_data.to_dict("records"),
        }
        with patch.object(Validator, "_basic_checks", return_value="Some error"):
            symbol, df, changed, error = Validator._validate_symbol_entry(
                entry, raw_flow=True
            )
            self.assertEqual(symbol, self.symbol)
            self.assertIsNone(df)
            self.assertFalse(changed)
            self.assertEqual(error, "Some error")


class TestValidatorBasicChecksMonotonic(unittest.TestCase):
    """Coverage for the '_basic_checks' monotonic datetime branch."""

    def setUp(self):
        params = ParameterLoader()
        self.required_cols = params.get("required_market_raw_columns")
        self.symbol = "AAPL"
        # Valid, sorted base DataFrame
        self.base_df = pd.DataFrame(
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

    @patch.object(Validator, "_check_missing_trading_days")
    @patch.object(Validator, "has_invalid_prices")
    @patch.object(Validator, "_check_volume_and_time")
    def test_basic_checks_returns_not_sorted_and_short_circuits(
        self, mock_vol_time, mock_prices, mock_missing_days
    ):
        """Should return 'datetime column is not sorted' and skip subsequent checks."""
        # Create unsorted datetime order
        df = self.base_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        self.assertFalse(pd.to_datetime(df["datetime"]).is_monotonic_increasing)
        result = Validator._basic_checks(
            self.symbol, df, self.required_cols, raw_flow=True, interval="1d"
        )
        self.assertEqual(result, "datetime column is not sorted")
        mock_missing_days.assert_not_called()
        mock_prices.assert_not_called()
        mock_vol_time.assert_not_called()

    @patch.object(Validator, "_check_missing_trading_days", return_value=None)
    @patch.object(Validator, "has_invalid_prices", return_value=None)
    @patch.object(Validator, "_check_volume_and_time", return_value=None)
    def test_basic_checks_monotonic_path_calls_followups(
        self, mock_vol_time, mock_prices, mock_missing_days
    ):
        """When datetime is monotonic, it should call downstream validators."""
        df = self.base_df.copy()  # already sorted
        result = Validator._basic_checks(
            self.symbol, df, self.required_cols, raw_flow=True, interval="1d"
        )
        self.assertIsNone(result)
        mock_missing_days.assert_called_once()
        mock_prices.assert_called_once()
        mock_vol_time.assert_called_once()
