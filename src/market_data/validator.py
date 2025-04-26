"""
This module provides functionality to validate the structure and integrity.

of historical market data for financial symbols. It ensures the presence of
required columns, correct datetime ordering, and valid symbol configurations.
"""

from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd

from market_data.gateway import Gateway
from utils.logger import Logger
from utils.parameters import ParameterLoader


# pylint: disable=too-few-public-methods
class Validator:
    """Validates the structure and integrity of market data."""

    _PARAMS = ParameterLoader()
    _REQUIRED_MARKET_COLUMNS = _PARAMS.get("required_market_columns")
    _ALL_SYMBOLS = _PARAMS.get("all_symbols")

    @staticmethod
    def _has_missing_columns(df: pd.DataFrame) -> Union[str, None]:
        for col in Validator._REQUIRED_MARKET_COLUMNS:
            if col not in df.columns:
                return f"Missing column: {col}"
        return None

    @staticmethod
    def _has_invalid_prices(df: pd.DataFrame) -> Union[str, None]:
        result = None
        if (df["low"] > df["high"]).any():
            result = "Invalid price range: low > high"

        for col in ["open", "close", "high", "low", "adj_close"]:
            if (df[col] <= 0).any():
                result = f"Non-positive '{col}' price found"
        return result

    @staticmethod
    def _check_volume_and_time(symbol: str, df: pd.DataFrame) -> Union[str, None]:
        if (df["volume"] < 0).any():
            return "Negative volume found"
        if (df["volume"] == 0).any():
            Logger.warning(f"{symbol} contains zero volume entries")

        time_deltas = df["datetime"].diff().dropna()
        if len(time_deltas) > 1 and time_deltas.dt.total_seconds().std() > 86400:
            return "Inconsistent time intervals detected"
        return None

    @staticmethod
    def _basic_checks(symbol: str, df: pd.DataFrame) -> Union[str, None]:
        """Performs basic structure and semantic checks on the DataFrame."""
        if df.empty:
            return "Historical prices are empty"

        missing_col_error = Validator._has_missing_columns(df)
        if missing_col_error:
            return missing_col_error

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        if not df["datetime"].is_monotonic_increasing:
            return "datetime column is not sorted"

        df.sort_values("datetime", inplace=True)

        if df.shape[0] < Validator._PARAMS.get("min_history_length", 250):
            return "Insufficient historical data points"

        price_error = Validator._has_invalid_prices(df)
        if price_error:
            return price_error

        volume_time_error = Validator._check_volume_and_time(symbol, df)
        return volume_time_error

    @staticmethod
    def _set_nan_if_not_empty(df: pd.DataFrame, idx: pd.Index, column: str) -> Any:
        """Set NaN in specified column at given index if index is not empty."""
        if not idx.empty:
            df.loc[idx, column] = np.nan
            return df, True
        return df, False

    @staticmethod
    def _clean_extreme_z_scores(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """Detect and remove extreme z-score outliers from the return column."""
        changed = False
        returns_clean = (
            pd.to_numeric(df["return"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(returns_clean) > 5:
            z_scores = (returns_clean - returns_clean.mean()) / returns_clean.std()
            extreme_z = z_scores[z_scores.abs() > 10].iloc[:-5]
            df, changed = Validator._set_nan_if_not_empty(df, extreme_z.index, "return")

        return df, changed

    @staticmethod
    def _apply_cleaning(symbol: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        changed = False

        df, changed_z = Validator._clean_extreme_z_scores(df)
        changed |= changed_z

        expected_return = df["close"].pct_change().fillna(0)
        if not expected_return.empty and not df["return"].empty:
            if not (expected_return - df["return"]).abs().lt(1e-4).all():
                Logger.warning(
                    f"    {symbol} return column adjusted to match expected values."
                )
                df["return"] = expected_return
                changed = True

        return df, changed

    @staticmethod
    def _validate_symbol_entry(
        entry: dict,
    ) -> Tuple[str, Union[pd.DataFrame, None], bool, Union[str, None]]:
        symbol = entry.get("symbol", "").strip().upper()
        if not symbol:
            return symbol, None, False, "Symbol is empty"

        if symbol not in Validator._ALL_SYMBOLS:
            return symbol, None, False, "Symbol is not listed in symbol repository"

        df = pd.DataFrame(entry.get("historical_prices", []))
        error = Validator._basic_checks(symbol, df)
        if error:
            return symbol, None, False, error

        df, changed = Validator._apply_cleaning(symbol, df)
        return symbol, df, changed, None

    @staticmethod
    def _validate_symbols(data: List[dict]) -> Tuple[int, int, dict, Union[dict, None]]:
        """Validates all symbols for structure, columns, and chronological order."""
        success_count = 0
        fail_count = 0
        issues = {}
        clean_dataframes = {}
        changed_symbols = set()

        for entry in data:
            symbol, df, changed, error = Validator._validate_symbol_entry(entry)
            if error:
                fail_count += 1
                issues[symbol] = error
                continue
            clean_dataframes[symbol] = df
            if changed:
                changed_symbols.add(symbol)
            success_count += 1

        if not changed_symbols:
            clean_dataframes = None

        return success_count, fail_count, issues, clean_dataframes

    @staticmethod
    def _update_clean_symbols(clean_dataframes: dict) -> None:
        """Updates the Gateway with cleaned DataFrames and metadata."""
        for symbol, df in clean_dataframes.items():
            existing_metadata = Gateway.get_symbol(symbol)
            if not existing_metadata:
                Logger.warning(f"Symbol {symbol} not found in metadata. Skipping.")
                continue

            metadata = {
                key: existing_metadata.get(key)
                for key in [
                    "name",
                    "type",
                    "sector",
                    "industry",
                    "currency",
                    "exchange",
                    "schedule",
                ]
            }

            Gateway.set_symbol(symbol, df.to_dict(orient="records"), metadata)

        Gateway.save()

    @staticmethod
    def validate_market_data() -> bool:
        """Validates integrity of the market data JSON file."""
        Logger.debug("Validation Summary")

        symbols_data = Gateway.load().get("symbols")
        if not isinstance(symbols_data, dict):
            Logger.error(
                "Invalid data format for validation. Expected dict of symbol entries."
            )
            return False

        raw_data = [entry for entry in symbols_data.values() if isinstance(entry, dict)]
        excluded = len(symbols_data) - len(raw_data)
        if excluded > 0:
            Logger.warning(f"  * Skipped {excluded} non-dict entries in symbol data.")

        total_symbols = len(raw_data)
        all_symbols_in_file = {
            entry.get("symbol") for entry in raw_data if "symbol" in entry
        }

        invalid_symbols_in_file = all_symbols_in_file.intersection(
            Validator._PARAMS.symbol_repo.get_invalid_symbols()
        )
        valid_expected_symbols = total_symbols - len(invalid_symbols_in_file)

        Logger.debug(f"  * Total symbols in file: {total_symbols}")
        if valid_expected_symbols != total_symbols:
            Logger.warning(f"  * Expected valid symbols: {valid_expected_symbols}")

        success_symbols, failed_symbols, issues, clean_dataframes = (
            Validator._validate_symbols(raw_data)
        )

        if clean_dataframes:
            Validator._update_clean_symbols(clean_dataframes)

        Logger.debug(f"  * Symbols passed: {success_symbols}")
        if failed_symbols > 0:
            Logger.debug(f"  * Symbols failed: {failed_symbols}")

        if issues:
            Logger.warning("Issues encountered:")
            for symbol, error in issues.items():
                Logger.error(f"   - {symbol}: {error}")

        Logger.separator()
        return len(issues.items()) == 0
