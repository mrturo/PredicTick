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

import math
from typing import Any, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.market_data.ingestion.raw.raw_data import RawData
from src.market_data.processing.enrichment.enriched_data import EnrichedData
from src.market_data.utils.intervals.interval_converter import \
    IntervalConverter
from src.market_data.utils.validation.price_validator import PriceValidator
from src.utils.config.parameters import ParameterLoader
from src.utils.exchange.calendar_manager import CalendarManager
from src.utils.io.logger import Logger


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
        """Check for missing required columns in the DataFrame."""
        result: Union[str, None] = None
        if not isinstance(df, pd.DataFrame):
            result = "Data is not a valid DataFrame"
        elif df.empty:
            result = "DataFrame is empty"
        elif not isinstance(required_columns, list):
            result = "Required columns must be provided as a list"
        elif not required_columns:
            result = "No required columns specified for validation"
        elif not all(isinstance(col, str) for col in required_columns):
            result = "All required column names must be strings"
        elif not df.columns.is_unique:
            result = "DataFrame columns are not unique"
        elif not df.index.is_monotonic_increasing:
            result = "DataFrame index is not monotonic increasing"
        elif not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            result = f"Missing columns: {', '.join(missing_cols)}"
        elif df.isnull().values.any():
            null_columns = df.columns[df.isnull().any()].tolist()
            result = (
                f"DataFrame contains NaN values in columns: {', '.join(null_columns)}"
            )
        return (
            result.strip() if result is not None and len(result.strip()) > 0 else None
        )

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
    def _validate_time_deltas(
        time_series: pd.Series, interval: str
    ) -> Union[str, None]:
        interval_seconds = IntervalConverter.to_minutes(interval) * 60
        try:
            parsed = pd.to_datetime(
                time_series, utc=True, errors="coerce", format="ISO8601"
            )
        except TypeError:
            parsed = pd.to_datetime(time_series, utc=True, errors="coerce")
        time_series = parsed.dropna()
        time_series = time_series[time_series > pd.Timestamp("1971-01-01", tz="UTC")]
        if time_series.empty or len(time_series) < 2:
            return None
        time_deltas = time_series.diff().dropna()
        df_deltas = pd.DataFrame(
            {
                "prev_datetime": time_series[:-1].values,
                "next_datetime": time_series[1:].values,
                "delta": time_deltas.dt.total_seconds().values,
            }
        )
        df_deltas["prev_date"] = pd.to_datetime(df_deltas["prev_datetime"]).dt.date
        df_deltas["next_date"] = pd.to_datetime(df_deltas["next_datetime"]).dt.date
        df_deltas = df_deltas[df_deltas["prev_date"] == df_deltas["next_date"]]
        if df_deltas.empty:
            return None
        for date, group in df_deltas.groupby("next_date"):
            std = np.std(group["delta"])
            if std > interval_seconds:
                Logger.error(f"Inconsistent intervals within {date}")
                Logger.error(f"Expected delta (s): {interval_seconds}")
                Logger.error("Datetime pairs with unexpected deltas:")
                for _, row in group.iterrows():
                    actual_delta = row["delta"]
                    if not math.isclose(actual_delta, interval_seconds, rel_tol=0.01):
                        Logger.error(
                            f"  - {row['prev_datetime']} -> {row['next_datetime']} = "
                            f"{actual_delta:.0f}s"
                        )
                return f"Inconsistent intervals within {date}"
        return None

    @staticmethod
    def _check_volume_and_time(
        symbol: str, df: pd.DataFrame, interval: str
    ) -> Union[str, None]:
        if (df["volume"] < 0).any():
            return "Negative volume found"
        if (df["volume"] == 0).any():
            Logger.warning(f"     - {symbol} contains zero volume entries")
        return Validator._validate_time_deltas(df["datetime"], interval)

    @staticmethod
    def _check_missing_trading_days(
        df: pd.DataFrame, interval: str
    ) -> Union[str, None]:
        """Check for missing trading days using market calendar and holiday exclusions."""
        try:
            minutes = IntervalConverter.to_minutes(interval)
        except ValueError as e:
            return f"Invalid interval format: {interval} ({e})"
        calendar, us_holidays, _ = CalendarManager.build_market_calendars()
        start_date = df["datetime"].min().date()
        end_date = df["datetime"].max().date()
        expected_schedule = calendar.schedule(start_date=start_date, end_date=end_date)
        business_days = expected_schedule.index.tz_localize(None).normalize()
        holiday_dates = pd.to_datetime(us_holidays)
        business_days = business_days.difference(holiday_dates)
        actual_dates = df["datetime"].dt.tz_convert(None).dt.normalize().unique()
        actual_dates = pd.to_datetime(sorted(actual_dates))
        if minutes <= 1440:
            missing_dates = business_days.difference(actual_dates)
            if not missing_dates.empty:
                return (
                    f"Missing {len(missing_dates)} trading days: "
                    f"{[d.strftime('%Y-%m-%d') for d in missing_dates]}"
                )
        else:
            current_date = start_date
            missing_expected = []
            while current_date <= end_date:
                if (
                    pd.Timestamp(current_date) in business_days
                    and pd.Timestamp(current_date) not in actual_dates
                ):
                    missing_expected.append(current_date)
                current_date += pd.Timedelta(days=minutes // 1440)
            if missing_expected:
                return (
                    f"Missing {len(missing_expected)} expected interval points: "
                    f"{[d.strftime('%Y-%m-%d') for d in missing_expected]}"
                )
        return None

    @staticmethod
    def _basic_checks(
        symbol: str,
        df: pd.DataFrame,
        required_columns: list[str],
        raw_flow: bool,
        interval: str,
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
            response = Validator._check_missing_trading_days(df, interval)
        if response is None or len(response.strip()) == 0:
            response = Validator.has_invalid_prices(df, raw_flow)
        if response is None or len(response.strip()) == 0:
            response = Validator._check_volume_and_time(symbol, df, interval)
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
        _required_columns = (
            Validator._REQUIRED_MARKET_RAW_COLUMNS
            if raw_flow is True
            else Validator._REQUIRED_MARKET_ENRICHED_COLUMNS
        )
        raw_interval: Optional[str] = None
        if RawData.exist() is True:
            RawData.load()
            raw_interval = RawData.get_interval()
            raw_interval = (
                raw_interval.strip()
                if raw_interval is not None and len(raw_interval.strip()) > 0
                else None
            )
        enriched_interval: Optional[str] = None
        if EnrichedData.exist() is True:
            EnrichedData.load()
            enriched_interval = EnrichedData.get_interval()
            enriched_interval = (
                enriched_interval.strip()
                if enriched_interval is not None and len(enriched_interval.strip()) > 0
                else None
            )
        _interval = raw_interval if raw_flow is True else enriched_interval
        error = Validator._basic_checks(
            symbol,
            df,
            _required_columns,
            raw_flow,
            _interval,
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
    def _check_symbols_data_format(
        symbols_data: Any, logs: bool
    ) -> Union[List[dict], None]:
        """Verifica que symbols_data sea un dict válido y filtra solo entradas tipo dict."""
        if not isinstance(symbols_data, dict):
            if logs:
                Logger.error(
                    "Invalid data format for validation. Expected dict of symbol entries."
                )
            return None
        data = [entry for entry in symbols_data.values() if isinstance(entry, dict)]
        excluded = len(symbols_data) - len(data)
        if excluded > 0 and logs:
            Logger.warning(f"  * Skipped {excluded} non-dict entries in symbol data.")
        return data

    @staticmethod
    def _log_symbol_stats(data: List[dict], logs: bool) -> None:
        """Registra estadísticas sobre los símbolos del archivo."""
        total_symbols = len(data)
        all_symbols_in_file = {
            entry.get("symbol") for entry in data if "symbol" in entry
        }
        invalid_symbols_in_file = all_symbols_in_file.intersection(
            Validator._PARAMS.symbol_repo.get_invalid_symbols()
        )
        valid_expected_symbols = total_symbols - len(invalid_symbols_in_file)
        if logs:
            Logger.debug(f"  * Total symbols in file: {total_symbols}")
            if valid_expected_symbols != total_symbols:
                Logger.warning(f"  * Expected valid symbols: {valid_expected_symbols}")

    @staticmethod
    def validate_data(symbols_data: Any, raw_flow: bool, logs: bool = True) -> bool:
        """Validates integrity of the market data JSON file."""
        if logs:
            Logger.debug(
                "Validation Raw Data Summary"
                if raw_flow
                else "Validation Enriched Data Summary"
            )
        data = Validator._check_symbols_data_format(symbols_data, logs)
        if data is None:
            return False
        Validator._log_symbol_stats(data, logs)
        success_symbols, failed_symbols, issues, clean_dataframes = (
            Validator._validate_symbols(data, raw_flow)
        )
        if clean_dataframes and raw_flow:
            symbols_data = Validator._update_clean_symbols(
                symbols_data, clean_dataframes
            )
        if logs:
            Logger.debug(f"  * Symbols passed: {success_symbols}")
            if failed_symbols > 0:
                Logger.debug(f"  * Symbols failed: {failed_symbols}")
            if issues:
                Logger.warning("Issues encountered:")
                for symbol, error in issues.items():
                    Logger.error(f"   - {symbol}: {error}")
        return len(issues.items()) == 0
