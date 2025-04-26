"""Validator module for market data integrity and structure validation.

This module defines the `Validator` class, which provides a comprehensive
validation pipeline for historical OHLCV market data. It ensures structural
correctness, column presence, chronological consistency, price validity,
volume sanity, and symbol membership based on configurable parameters.

The validation process is composed of multiple modular steps:
    - Required columns and structure check
    - Chronological validation (`datetime` monotonicity and sorting)
    - Minimum data length requirement
    - OHLC/Adj Close price range validation
    - Price positivity/negativity checks
    - Volume sanity and timestamp interval regularity
    - Symbol repository membership and metadata consistency

This module is critical in ensuring only clean and structurally valid datasets
enter further processing pipelines like enrichment, modeling, or storage.
"""

from typing import Any, List, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from market_data.utils.validation.price_validator import PriceValidator
from utils.logger import Logger
from utils.parameters import ParameterLoader


class Validator:
    """Validates the structure and integrity of market data."""

    _PARAMS = ParameterLoader()
    _ALL_SYMBOLS = _PARAMS.get("all_symbols")
    _REQUIRED_MARKET_RAW_COLUMNS: list[str] = _PARAMS.get("required_market_raw_columns")
    _REQUIRED_MARKET_ENRICHED_COLUMNS: list[str] = list(
        dict.fromkeys(
            (_PARAMS.get("required_market_raw_columns") or [])
            + (_PARAMS.get("required_market_enriched_columns") or [])
        )
    )

    @staticmethod
    def _has_missing_columns(
        df: pd.DataFrame, required_columns: list[str]
    ) -> Union[str, None]:
        for col in required_columns:
            if col not in df.columns:
                return f"Missing column: {col}"
        return None

    @staticmethod
    def has_invalid_prices(df: pd.DataFrame, raw_flow: bool) -> Union[str, None]:
        """Run a full OHLC/Adj Close validation pipeline."""
        range_error = PriceValidator.check_price_ranges(df)
        if range_error is not None:  # pragma: no branch
            return range_error
        sign_error = PriceValidator.check_nonpositive_prices(df, raw_flow)
        if sign_error is not None:  # pragma: no branch
            return sign_error
        return None

    @staticmethod
    def _validate_time_deltas(time_deltas: pd.Series) -> Union[str, None]:
        if len(time_deltas) > 1 and time_deltas.dt.total_seconds().std() > 86400:
            return "Inconsistent time intervals detected"
        return None

    @staticmethod
    def _check_volume_and_time(symbol: str, df: pd.DataFrame) -> Union[str, None]:
        if (df["volume"] < 0).any():
            return "Negative volume found"
        if (df["volume"] == 0).any():
            Logger.warning(f"     - {symbol} contains zero volume entries")
        time_deltas = df["datetime"].diff().dropna()
        return Validator._validate_time_deltas(time_deltas)

    @staticmethod
    def _basic_checks(
        symbol: str, df: pd.DataFrame, required_columns: list[str], raw_flow: bool
    ) -> Union[str, None]:
        """Performs basic structure and semantic checks on the DataFrame."""
        if df.empty:
            return "Historical prices are empty"
        response = Validator._has_missing_columns(df, required_columns)
        if response is None or len(response.strip()) == 0:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            if not df["datetime"].is_monotonic_increasing:
                return "datetime column is not sorted"
            df.sort_values("datetime", inplace=True)
            if df.shape[0] < Validator._PARAMS.get("min_history_length", 250):
                return "Insufficient historical data points"
            response = Validator.has_invalid_prices(df, raw_flow)
        if response is None or len(response.strip()) == 0:
            response = Validator._check_volume_and_time(symbol, df)
        return response

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
        symbol = entry.get("symbol", "") or ""
        if not isinstance(symbol, str):
            raise TypeError("'symbol' value must be of type str")
        symbol = symbol.strip().upper()
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
            raw_flow,
        )
        return (symbol, None, False, error) if error else (symbol, df, False, None)

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
    def _update_clean_symbols(symbols_data: Any, clean_dataframes: dict) -> Any:
        """Updates the data with cleaned DataFrames and metadata."""
        for symbol, df in clean_dataframes.items():
            existing_metadata = symbols_data[symbol]
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
            symbols_data[symbol] = {
                "symbol": symbol,
                "name": metadata.get("name"),
                "type": metadata.get("type"),
                "sector": metadata.get("sector"),
                "industry": metadata.get("industry"),
                "currency": metadata.get("currency"),
                "exchange": metadata.get("exchange"),
                "historical_prices": df.to_dict(orient="records"),
            }
        return symbols_data

    @staticmethod
    def validate_data(symbols_data: Any, raw_flow: bool) -> bool:
        """Validates integrity of the market data JSON file."""
        if raw_flow:
            Logger.debug("Validation Raw Data Summary")
        else:
            Logger.debug("Validation Enriched Data Summary")
        if not isinstance(symbols_data, dict):
            Logger.error(
                "Invalid data format for validation. Expected dict of symbol entries."
            )
            return False
        data = [entry for entry in symbols_data.values() if isinstance(entry, dict)]
        excluded = len(symbols_data) - len(data)
        if excluded > 0:
            Logger.warning(f"  * Skipped {excluded} non-dict entries in symbol data.")
        total_symbols = len(data)
        all_symbols_in_file = {
            entry.get("symbol") for entry in data if "symbol" in entry
        }
        invalid_symbols_in_file = all_symbols_in_file.intersection(
            Validator._PARAMS.symbol_repo.get_invalid_symbols()
        )
        valid_expected_symbols = total_symbols - len(invalid_symbols_in_file)
        Logger.debug(f"  * Total symbols in file: {total_symbols}")
        if valid_expected_symbols != total_symbols:
            Logger.warning(f"  * Expected valid symbols: {valid_expected_symbols}")
        success_symbols, failed_symbols, issues, clean_dataframes = (
            Validator._validate_symbols(data, raw_flow)
        )
        if clean_dataframes and raw_flow is True:
            symbols_data = Validator._update_clean_symbols(
                symbols_data, clean_dataframes
            )
        Logger.debug(f"  * Symbols passed: {success_symbols}")
        if failed_symbols > 0:
            Logger.debug(f"  * Symbols failed: {failed_symbols}")
        if issues:
            Logger.warning("Issues encountered:")
            for symbol, error in issues.items():
                Logger.error(f"   - {symbol}: {error}")
        Logger.separator()
        return len(issues.items()) == 0
