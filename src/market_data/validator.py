"""
This module provides functionality to validate the structure and integrity.

of historical market data for financial symbols. It ensures the presence of
required columns, correct datetime ordering, and valid symbol configurations.
"""

from typing import Any, List, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from market_data.raw_data import RawData
from utils.logger import Logger
from utils.parameters import ParameterLoader


# pylint: disable=too-few-public-methods
class Validator:
    """Validates the structure and integrity of market data."""

    _PARAMS = ParameterLoader()
    _REQUIRED_MARKET_RAW_COLUMNS: list[str] = _PARAMS.get("required_market_raw_columns")
    _REQUIRED_MARKET_ENRICHED_COLUMNS: list[str] = _PARAMS.get(
        "required_market_enriched_columns"
    )
    _ALL_SYMBOLS = _PARAMS.get("all_symbols")

    @staticmethod
    def _has_missing_columns(
        df: pd.DataFrame, required_columns: list[str]
    ) -> Union[str, None]:
        for col in required_columns:
            if col not in df.columns:
                return f"Missing column: {col}"
        return None

    @staticmethod
    def _has_invalid_prices(df: pd.DataFrame) -> Union[str, None]:
        result: Union[str, None] = None
        if (df["low"] > df["high"]).any():
            result = "Invalid price range: low > high"
        elif (df["low"] > df["open"]).any():
            result = "Invalid price range: low > open"
        elif (df["low"] > df["close"]).any():
            result = "Invalid price range: low > close"
        elif (df["low"] > df["adj_close"]).any():
            result = "Invalid price range: low > adj_close"
        elif (df["high"] < df["open"]).any():
            result = "Invalid price range: high < open"
        elif (df["high"] < df["close"]).any():
            result = "Invalid price range: high < close"
        elif (df["high"] < df["adj_close"]).any():
            result = "Invalid price range: high < adj_close"

        for col in ["open", "low", "high", "close", "adj_close"]:
            if result is None and (df[col] <= 0).any():
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
    def _basic_checks(
        symbol: str, df: pd.DataFrame, required_columns: list[str]
    ) -> Union[str, None]:
        """Performs basic structure and semantic checks on the DataFrame."""
        if df.empty:
            return "Historical prices are empty"

        missing_col_error = Validator._has_missing_columns(df, required_columns)
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
    def _validate_symbol_entry(
        entry: dict, raw_flow: bool
    ) -> Tuple[str, Union[pd.DataFrame, None], bool, Union[str, None]]:
        symbol = entry.get("symbol", "").strip().upper()
        if not symbol:
            return symbol, None, False, "Symbol is empty"

        if symbol not in Validator._ALL_SYMBOLS:
            return symbol, None, False, "Symbol is not listed in symbol repository"

        df = pd.DataFrame(entry.get("historical_prices", []))
        error = Validator._basic_checks(
            symbol,
            df,
            (
                Validator._REQUIRED_MARKET_RAW_COLUMNS
                if raw_flow is True
                else Validator._REQUIRED_MARKET_ENRICHED_COLUMNS
            ),
        )
        if error:
            return symbol, None, False, error
        return symbol, df, False, None

    @staticmethod
    def _validate_symbols(
        data: List[dict], raw_flow: bool
    ) -> Tuple[int, int, dict, Union[dict, None]]:
        """Validates all symbols for structure, columns, and chronological order."""
        success_count = 0
        fail_count = 0
        issues = {}
        clean_dataframes = {}
        changed_symbols = set()

        for entry in data:
            symbol, df, changed, error = Validator._validate_symbol_entry(
                entry, raw_flow
            )
            if error:
                fail_count += 1
                issues[symbol] = error
            else:
                clean_dataframes[symbol] = df
                if changed:
                    changed_symbols.add(symbol)
                success_count += 1

        return success_count, fail_count, issues, clean_dataframes

    @staticmethod
    def _update_clean_symbols(clean_dataframes: dict) -> None:
        """Updates the RawData with cleaned DataFrames and metadata."""
        for symbol, df in clean_dataframes.items():
            existing_metadata = RawData.get_symbol(symbol)
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
                ]
            }

            RawData.set_symbol(symbol, df.to_dict(orient="records"), metadata)

        RawData.save()

    @staticmethod
    def validate_market_data(raw_flow: bool) -> bool:
        """Validates integrity of the market data JSON file."""
        Logger.debug("Validation Summary")

        symbols_data = RawData.load().get("symbols")
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
            Validator._validate_symbols(raw_data, raw_flow)
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
