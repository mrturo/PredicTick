"""
Module for generating, managing, and enriching historical financial symbol data.

This module defines the `EnrichedData` class, responsible for:
- Loading, saving, and processing historical price data.
- Computing and scaling features such as prices, volume, and a wide range of technical indicators.
- Normalizing timestamps into relative time dimensions (day, week, month, year).
- Standardizing the data structure for use in ML models and visualizations.

The calculated indicators include: RSI, MACD, ATR, Bollinger Bands, OBV, momentum,
intraday and overnight returns, single and multi-candle patterns across multiple timeframes,
among others. All hyperparameters used are centrally managed via `ParameterLoader`.

Classes:
    - EnrichedData: A centralized access point for managing enriched symbol data,
      with static methods for per-symbol processing, disk persistence, and global range config.

Dependencies:
    - numpy
    - pandas
    - market_data.raw_data.RawData
    - market_data.technical_indicators.TechnicalIndicators
    - utils.json_manager.JsonManager
    - utils.parameters.ParameterLoader

Note:
    This module is optimized for backtesting and training pipelines that require
    temporal alignment across multiple symbols and feature-rich technical data.
"""

from typing import Any, Dict, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from market_data.raw_data import RawData
from market_data.technical_indicators import TechnicalIndicators
from market_data.time_resampler import TimeResampler
from utils.interval import Interval, IntervalConverter
from utils.json_manager import JsonManager
from utils.logger import Logger
from utils.parameters import ParameterLoader


class EnrichedData:
    """Centralized access point for managing symbol enriched data."""

    _PARAMS = ParameterLoader()
    _ATR_WINDOW = _PARAMS.get("atr_window")
    _BOLLINGER_BAND_METHOD = _PARAMS.get("bollinger_band_method")
    _BOLLINGER_WINDOW = _PARAMS.get("bollinger_window")
    _RAW_DATA_INTERVAL = Interval.market_raw_data()
    _ENRICHED_MARKETDATA_FILEPATH = _PARAMS.get("enriched_marketdata_filepath")
    _MACD_FAST = _PARAMS.get("macd_fast")
    _MACD_SIGNAL = _PARAMS.get("macd_signal")
    _MACD_SLOW = _PARAMS.get("macd_slow")
    _OBV_FILL_METHOD = _PARAMS.get("obv_fill_method")
    _RSI_WINDOW = _PARAMS.get("rsi_window_backtest")
    _STOCH_RSI_MIN_PERIODS = _PARAMS.get("stoch_rsi_min_periods")
    _STOCH_RSI_WINDOW = _PARAMS.get("stoch_rsi_window")
    _VOLUME_WINDOW = _PARAMS.get("volume_window")
    _WILLIAMS_R_WINDOW = _PARAMS.get("williams_r_window")

    _id: Optional[str] = None
    _interval: Optional[str] = None
    _last_updated: Optional[pd.Timestamp] = None
    _ranges: Any = {}
    _symbols: Dict[str, dict] = {}

    @staticmethod
    def _get_filepath(filepath: Optional[str], default=None) -> Optional[str]:
        """Resolve a filepath, falling back to a default if necessary."""
        if filepath is None or len(filepath.strip()) == 0:
            return default
        return filepath

    @staticmethod
    def get_id() -> Optional[str]:
        """Return the file id value."""
        return EnrichedData._id

    @staticmethod
    def set_id(file_id: Optional[str]) -> None:
        """Set the file id value."""
        EnrichedData._id = file_id

    @staticmethod
    def get_interval() -> Optional[str]:
        """Retrieve the currently set interval value for enriched data."""
        return EnrichedData._interval

    @staticmethod
    def set_interval(interval: Optional[str]) -> None:
        """Set the interval value to be used for enriched data operations."""
        EnrichedData._interval = interval

    @staticmethod
    def get_last_updated() -> Optional[pd.Timestamp]:
        """Return the last update timestamp."""
        return EnrichedData._last_updated

    @staticmethod
    def set_last_updated(last_updated: Optional[pd.Timestamp]) -> None:
        """Set the last update timestamp."""
        EnrichedData._last_updated = last_updated

    @staticmethod
    def get_ranges() -> Any:
        """Return the object of ranges."""
        return EnrichedData._ranges

    @staticmethod
    def set_ranges(ranges: Any) -> None:
        """Set the object of ranges."""
        EnrichedData._ranges = ranges

    @staticmethod
    def get_symbol(symbol: str):
        """Retrieve metadata for a specific symbol."""
        return EnrichedData._symbols.get(symbol)

    @staticmethod
    def get_symbols() -> Dict[str, dict]:
        """Return all loaded symbols."""
        return EnrichedData._symbols

    @staticmethod
    def set_symbols(symbols: Dict[str, dict]) -> None:
        """Set the dictionary of symbols."""
        EnrichedData._symbols = symbols

    @staticmethod
    def compute_time_fractions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes normalized time fractions and appends them to the input DataFrame.

        The generated columns are:
        - time_of_day: Fraction of the day (0.0 = 00:00, 1.0 = 00:00 next day)
        - time_of_week: Fraction of the week (0.0 = Monday 00:00, 1.0 = next Monday 00:00)
        - time_of_month: Fraction of the month (0.0 = first day 00:00, 1.0 = first day next month)
        - time_of_year: Fraction of the year (0.0 = Jan 1st, 1.0 = Jan 1st next year)

        Parameters:
            df (pd.DataFrame): DataFrame containing a 'datetime' column of type datetime64[ns].

        Returns:
            pd.DataFrame: The same DataFrame with added time fraction columns.
        """
        datetimes = df["datetime"]
        dt = datetimes.dt

        # Day fraction
        day_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        df["scaled_time_of_day"] = (day_seconds / 86400.0).astype(np.float32)

        # Week fraction
        weekday_seconds = dt.weekday * 86400 + day_seconds
        df["scaled_time_of_week"] = (weekday_seconds / (7 * 86400)).astype(np.float32)

        # Month fraction
        month_start = datetimes.dt.tz_localize(None).dt.to_period("M").dt.start_time
        next_month_start = month_start + pd.offsets.MonthBegin(1)
        month_start = month_start.dt.tz_localize(datetimes.dt.tz)  # restore original tz
        next_month_start = next_month_start.dt.tz_localize(datetimes.dt.tz)
        month_elapsed = (datetimes - month_start).dt.total_seconds()
        month_total = (next_month_start - month_start).dt.total_seconds()
        df["scaled_time_of_month"] = (month_elapsed / month_total).astype(np.float32)

        # Year fraction
        year_start = datetimes.dt.tz_localize(None).dt.to_period("Y").dt.start_time
        next_year_start = year_start + pd.offsets.YearBegin(1)
        year_start = year_start.dt.tz_localize(datetimes.dt.tz)
        next_year_start = next_year_start.dt.tz_localize(datetimes.dt.tz)
        year_elapsed = (datetimes - year_start).dt.total_seconds()
        year_total = (next_year_start - year_start).dt.total_seconds()
        df["scaled_time_of_year"] = (year_elapsed / year_total).astype(np.float32)

        return df

    @staticmethod
    def filter_prices_from_global_min(
        symbols: Dict[str, dict],
    ) -> Optional[Dict[str, dict]]:
        """
        Filter each symbol's historical_prices from the maximum of the minimum datetimes.

        across all symbols. This ensures all series start from the latest common beginning.
        """

        min_datetimes = [
            min(pd.to_datetime([row["datetime"] for row in data["historical_prices"]]))
            for data in symbols.values()
            if data.get("historical_prices")
        ]
        if not min_datetimes:
            return None
        global_start = max(min_datetimes)

        for _symbol, data in symbols.items():
            if "historical_prices" not in data:
                continue

            filtered = [
                row
                for row in data["historical_prices"]
                if pd.to_datetime(row["datetime"]) >= global_start
            ]
            data["historical_prices"] = filtered
        return symbols

    @staticmethod
    def load(filepath: Optional[str] = None) -> Dict:
        """Load market enriched data from disk and initialize internal structures."""
        local_filepath: Optional[str] = EnrichedData._get_filepath(
            filepath, EnrichedData._ENRICHED_MARKETDATA_FILEPATH
        )
        enriched_data = JsonManager.load(local_filepath)
        file_id: Optional[str] = None
        interval: Optional[str] = None
        last_updated: Optional[pd.Timestamp] = None
        ranges: Any = {}
        symbols: list = []
        if enriched_data is not None:
            file_id = enriched_data["id"]
            interval = enriched_data["interval"]
            last_updated = enriched_data["last_updated"]
            ranges = enriched_data["ranges"]
            symbols = enriched_data["symbols"]
        enriched = RawData.normalize_historical_prices(symbols)
        EnrichedData._id = file_id
        EnrichedData._interval = interval
        EnrichedData._last_updated = last_updated
        EnrichedData._ranges = ranges
        EnrichedData._symbols = dict(enriched)
        return {
            "id": EnrichedData.get_id(),
            "last_updated": EnrichedData.get_last_updated(),
            "interval": EnrichedData.get_interval(),
            "ranges": EnrichedData.get_ranges(),
            "symbols": EnrichedData.get_symbols(),
        }

    @staticmethod
    def save(filepath: Optional[str] = None) -> dict[str, Any]:
        """Persist current symbol data to disk."""
        last_updated = (
            pd.Timestamp.now(tz="UTC")
            if EnrichedData.get_last_updated() is None
            else EnrichedData.get_last_updated()
        )
        result = {
            "id": EnrichedData.get_id(),
            "last_updated": last_updated,
            "interval": EnrichedData.get_interval(),
            "ranges": EnrichedData.get_ranges(),
            "symbols": list(EnrichedData._symbols.values()),
        }
        local_filepath = EnrichedData._get_filepath(
            filepath, EnrichedData._ENRICHED_MARKETDATA_FILEPATH
        )
        JsonManager.save(result, local_filepath)
        return result

    @staticmethod
    def _convert_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert numerical values in records to native float/int Python types."""
        return [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.floating, np.float32, np.float64))
                    else (
                        int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else v
                    )
                )
                for k, v in row.items()
            }
            for row in records
        ]

    @staticmethod
    def _scale_column(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
        """
        Normalize a numeric Series to the range [0, 1] using min-max scaling.

        Falls back to a constant 0.0 Series if scaling is not possible
        (e.g. min_val == max_val or either is None).

        Args:
            series (pd.Series): Input numeric series to scale.
            min_val (float): Minimum value for scaling.
            max_val (float): Maximum value for scaling.

        Returns:
            pd.Series: Scaled series in float32 within [0, 1], or 0.0 if invalid bounds.
        """
        return (
            (series - min_val) / (max_val - min_val)
            if min_val is not None and max_val not in (None, min_val)
            else pd.Series(0.0, index=series.index, dtype="float32")
        )

    @staticmethod
    def _add_indicators(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """
        Enrich a DataFrame with technical indicators using `TechnicalIndicators`.

        This function appends multiple features including trend, volatility,
        momentum, volume, and pattern-based indicators. Optional prefixing
        allows reuse in multi-symbol settings or feature namespaces.

        Args:
            df (pd.DataFrame): Enriched candle data with standard OHLCV columns.
            raw_df (pd.DataFrame): Original high-frequency data for resampling.
            prefix (str, optional): Column prefix to apply to each added indicator.

        Returns:
            pd.DataFrame: Input `df` augmented with computed technical indicators.
        """
        prefix = prefix.strip()

        def prefixed(col: str) -> str:
            return f"{prefix}{col}" if prefix else col

        df[prefixed("adx_14d")] = TechnicalIndicators.compute_adx_14d(
            df["datetime"],
            df[prefixed("high")],
            df[prefixed("low")],
            df[prefixed("close")],
        )
        df[prefixed("atr")] = TechnicalIndicators.compute_atr(
            df[prefixed("high")],
            df[prefixed("low")],
            df[prefixed("close")],
            EnrichedData._ATR_WINDOW,
        )
        df[prefixed("atr_14d")] = TechnicalIndicators.compute_atr_14d(
            df["datetime"],
            df[prefixed("high")],
            df[prefixed("low")],
            df[prefixed("close")],
        )
        df[prefixed("average_price")] = TechnicalIndicators.compute_average_price(
            df[prefixed("high")], df[prefixed("low")]
        )
        df[prefixed("bollinger_pct_b")] = TechnicalIndicators.compute_bollinger_pct_b(
            df[prefixed("close")]
        )
        if EnrichedData._BOLLINGER_BAND_METHOD == "max-min":
            df[prefixed("bb_width")] = TechnicalIndicators.compute_bb_width(
                df[prefixed("close")], EnrichedData._BOLLINGER_WINDOW
            )
        df[prefixed("intraday_return")] = TechnicalIndicators.compute_intraday_return(
            df[prefixed("close")], df[prefixed("open")]
        )
        df[prefixed("macd")] = TechnicalIndicators.compute_macd(
            df[prefixed("close")],
            EnrichedData._MACD_FAST,
            EnrichedData._MACD_SLOW,
            EnrichedData._MACD_SIGNAL,
        )["histogram"]
        if "volume" in df.columns:
            df[prefixed("obv")] = TechnicalIndicators.compute_obv(
                df[prefixed("close")], df[prefixed("volume")]
            )
        df[prefixed("overnight_return")] = TechnicalIndicators.compute_overnight_return(
            df[prefixed("open")]
        )
        df[prefixed("price_change")] = TechnicalIndicators.compute_price_change(
            df[prefixed("close")], df[prefixed("open")]
        )
        df[prefixed("price_derivative")] = TechnicalIndicators.compute_price_derivative(
            df[prefixed("close")]
        )
        df[prefixed("range")] = TechnicalIndicators.compute_range(
            df[prefixed("high")], df[prefixed("low")]
        )
        if "volume" in df.columns:
            df[prefixed("relative_volume")] = (
                TechnicalIndicators.compute_relative_volume(
                    df[prefixed("volume")], EnrichedData._VOLUME_WINDOW
                )
            )
        df[prefixed("return")] = TechnicalIndicators.compute_return(
            df[prefixed("close")]
        )
        df[prefixed("rsi")] = TechnicalIndicators.compute_rsi(
            df[prefixed("close")], EnrichedData._RSI_WINDOW
        )
        df[prefixed("smoothed_derivative")] = (
            TechnicalIndicators.compute_smoothed_derivative(df[prefixed("close")])
        )
        df[prefixed("stoch_rsi")] = TechnicalIndicators.compute_stoch_rsi(
            df[prefixed("close")], EnrichedData._STOCH_RSI_WINDOW
        )
        df[prefixed("typical_price")] = TechnicalIndicators.compute_typical_price(
            df[prefixed("high")], df[prefixed("low")], df[prefixed("close")]
        )
        df[prefixed("volatility")] = TechnicalIndicators.compute_volatility(
            df[prefixed("high")], df[prefixed("low")], df[prefixed("open")]
        )
        if "volume" in df.columns:
            df[prefixed("volume_change")] = TechnicalIndicators.compute_volume_change(
                df[prefixed("volume")]
            )
        if "volume" in df.columns:
            df[prefixed("volume_rvol_20d")] = (
                TechnicalIndicators.compute_volume_rvol_20d(
                    df["datetime"], df[prefixed("volume")]
                )
            )
        df[prefixed("williams_r")] = TechnicalIndicators.compute_williams_r(
            df[prefixed("high")],
            df[prefixed("low")],
            df[prefixed("close")],
            EnrichedData._WILLIAMS_R_WINDOW,
        )
        df[prefixed("candle_pattern")] = TechnicalIndicators.compute_candle_pattern(
            df[prefixed("open")],
            df[prefixed("high")],
            df[prefixed("low")],
            df[prefixed("close")],
            len(prefix) == 0,
        )
        df[prefixed("multi_candle_pattern")] = (
            TechnicalIndicators.compute_multi_candle_pattern(
                df[prefixed("open")],
                df[prefixed("high")],
                df[prefixed("low")],
                df[prefixed("close")],
                len(prefix) == 0,
            )
        )
        return df

    @staticmethod
    def _compute_features(
        df: pd.DataFrame,
        ranges: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Generate scaled price/volume features and technical indicators.

        Steps
        -----
        1. **Scale** raw OHLCV prices/volume into `[0, 1]` using global
           `ranges` (pre-computed per asset universe).
        2. **Add indicators** (raw + scaled variants).
        3. Identify the last row where *any* relevant column contains
           `NaN` and drop all rows **up to** esa posición (inclusive).
        4. Eliminate remanentes `NaN` y devuelve `DataFrame` limpio.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data indexed por fecha (ya alineado con `raw_df`).
        raw_df : pd.DataFrame
            Serie original (sin escalar) necesaria para indicadores que
            dependan de precios “verdaderos”.
        ranges : dict[str, float]
            Diccionario con claves:
                - ``min_price``, ``max_price``
                - ``min_volume``, ``max_volume``

        Returns
        -------
        pd.DataFrame
            Mismo índice que `df`, con columnas adicionales:
            * scaled_X
            * indicadores derivados.
        """
        df = df.copy()
        price_cols = ["open", "low", "high", "close", "adj_close"]
        for col in price_cols:
            df[f"scaled_{col}"] = EnrichedData._scale_column(
                df[col], ranges["min_price"], ranges["max_price"]
            )
        df["scaled_volume"] = EnrichedData._scale_column(
            df["volume"], ranges["min_volume"], ranges["max_volume"]
        )

        df = EnrichedData._add_indicators(df)
        df = EnrichedData._add_indicators(df, prefix="scaled_")

        always_keep = {
            "volume",
            "obv",
            "relative_volume",
            "volume_change",
            "volume_rvol_20d",
        }
        columns_to_check = [c for c in df.columns if c not in always_keep]
        columns_with_data = [c for c in columns_to_check if not df[c].isna().all()]
        if not columns_with_data:
            return df.iloc[0:0]

        last_nan_idx = (
            df[columns_with_data].isna().any(axis=1).pipe(lambda s: s[s].index.max())
        )
        if pd.notna(last_nan_idx):
            df = df.loc[last_nan_idx + 1 :]

        df = df.dropna(subset=columns_with_data)

        return df

    @staticmethod
    def _format_symbol_output(key: str, value: Any, df: pd.DataFrame) -> dict:
        """Format the enriched and scaled DataFrame into the final output structure."""
        # pylint: disable=duplicate-code
        base_cols = [
            "datetime",
            "open",
            "low",
            "high",
            "close",
            "adj_close",
            "volume",
            "adx_14d",
            "atr",
            "atr_14d",
            "average_price",
            "bb_width",
            "bollinger_pct_b",
            "intraday_return",
            "macd",
            "obv",
            "overnight_return",
            "price_change",
            "price_derivative",
            "range",
            "relative_volume",
            "return",
            "rsi",
            "smoothed_derivative",
            "stoch_rsi",
            "typical_price",
            "volatility",
            "volume_change",
            "volume_rvol_20d",
            "williams_r",
            "candle_pattern",
            "multi_candle_pattern",
        ]
        raw_cols = [
            col
            for col in df.columns
            if col in base_cols or col in value.get("features", [])
        ]
        raw_no_datetime = [col for col in raw_cols if col != "datetime"]
        raw_records = EnrichedData._convert_records(
            df[raw_no_datetime].to_dict(orient="records")  # type: ignore
        )
        df["raw"] = raw_records
        df = EnrichedData.compute_time_fractions(df)

        formatted = pd.DataFrame(
            {
                "datetime": pd.to_datetime(df["datetime"]).astype(str),  # type: ignore
                **{
                    (
                        col.replace("scaled_", "") if col.startswith("scaled_") else col
                    ): df[col]
                    for col in ["raw"]
                    + list(df.columns[df.columns.str.startswith("scaled_")])
                },
            }
        )

        return {
            "symbol": value.get("symbol", key),
            "name": value.get("name", ""),
            "type": value.get("type", ""),
            "sector": value.get("sector", ""),
            "industry": value.get("industry", ""),
            "currency": value.get("currency", ""),
            "exchange": value.get("exchange", ""),
            "historical_prices": EnrichedData._convert_records(
                formatted.to_dict(orient="records")  # type: ignore
            ),
        }

    @staticmethod
    def _floor_datetimes(
        df: pd.DataFrame,
        interval: str,
        *,
        column: str = "datetime",
    ) -> pd.DataFrame:
        """
        Floor the *column* in *df* to the beginning of its time bucket.

        The granularity is inferred from *interval*:

        * ``Xm`` (minutes) → seconds become ``00``.
        * ``Xh`` (hours)   → minutes **and** seconds become ``00``.
        * ``Xd``/``1wk``/``1mo``/``1y`` → time of day becomes ``00:00:00``.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame that contains the datetime column.  Time-zone aware
            values are fully supported.
        interval : str
            Interval string such as ``'15m'``, ``'1h'``, ``'1d'``, ``'1wk'``.
        column : str, default ``'datetime'``
            Name of the column to process.

        Returns
        -------
        pandas.DataFrame
            The *same* object passed in, with *column* modified in-place.
        """
        freq = IntervalConverter.to_pandas_floor_freq(interval)
        df[column] = pd.to_datetime(df[column]).dt.floor(freq)
        return df

    @staticmethod
    def process_symbol(key: str, value: Any, ranges: Any, interval: Any) -> dict:
        """Process a single symbol by computing features and formatting output."""
        raw_df = pd.DataFrame(value["historical_prices"])
        interval_raw_data = interval["raw_data"]
        interval_enriched_data = interval["enriched_data"]
        resampled_ratio_df: pd.DataFrame = pd.DataFrame()
        raw_df = EnrichedData._floor_datetimes(raw_df, interval_enriched_data)
        if interval_raw_data != interval_enriched_data:
            ratio = IntervalConverter.get_ratio(
                interval_raw_data, interval_enriched_data
            )
            label = ratio["label"]
            Logger.debug(
                f"Time intervals between raw_data ({interval_raw_data}) and enriched_data "
                f"({interval_enriched_data}) are different, with a ratio: {label}"
            )
            resampled_ratio_df: pd.DataFrame = TimeResampler.by_ratio(
                raw_df, interval_raw_data, interval_enriched_data
            )
        else:
            resampled_ratio_df = raw_df.copy()
        df = EnrichedData._compute_features(resampled_ratio_df, ranges)
        return EnrichedData._format_symbol_output(key, value, df)

    @staticmethod
    def generate(filepath: Optional[str] = None) -> dict[str, Any]:
        """Generate enriched symbol data with normalized prices and volumes, saving the result."""
        local_filepath = EnrichedData._get_filepath(
            filepath, EnrichedData._ENRICHED_MARKETDATA_FILEPATH
        )
        if local_filepath and JsonManager.exists(local_filepath):
            JsonManager.delete(local_filepath)

        interval = Interval.market_enriched_data()
        RawData.load()
        symbols_data = RawData.get_symbols()

        df_all = pd.concat(
            [pd.DataFrame(s["historical_prices"]) for s in symbols_data.values()],
            ignore_index=True,
        )

        min_price = float(df_all["low"].min())
        max_price = float(df_all["high"].max())
        min_volume = float(df_all["volume"].min())
        max_volume = float(df_all["volume"].max())

        EnrichedData._symbols = {}
        EnrichedData.set_id(RawData.get_id())
        EnrichedData.set_interval(interval)
        EnrichedData.set_last_updated(RawData.get_last_update())
        for symbol_key, symbol_value in symbols_data.items():
            EnrichedData._symbols[symbol_key] = EnrichedData.process_symbol(
                symbol_key,
                symbol_value,
                {
                    "min_price": min_price,
                    "max_price": max_price,
                    "min_volume": min_volume,
                    "max_volume": max_volume,
                },
                {
                    "raw_data": RawData.get_interval(),
                    "enriched_data": EnrichedData.get_interval(),
                },
            )
        EnrichedData.set_ranges(
            {
                "price": {
                    "min": min_price,
                    "max": max_price,
                },
                "volume": {
                    "min": min_volume,
                    "max": max_volume,
                },
            }
        )

        filtered_symbols = EnrichedData.filter_prices_from_global_min(
            EnrichedData.get_symbols()
        )
        if filtered_symbols:
            EnrichedData.set_symbols(filtered_symbols)
        return EnrichedData.save(filepath)

    @staticmethod
    def get_indicator_parameters() -> dict:
        """
        Return a dictionary of all indicator parameters used in the calculation.

        This function allows easy inspection of the configured hyperparameters
        for technical analysis, useful for logging or reproducibility.

        Returns:
            dict: Dictionary of indicator configuration values.
        """
        return {
            "rsi_window": EnrichedData._RSI_WINDOW,
            "macd_fast": EnrichedData._MACD_FAST,
            "macd_slow": EnrichedData._MACD_SLOW,
            "macd_signal": EnrichedData._MACD_SIGNAL,
            "bollinger_window": EnrichedData._BOLLINGER_WINDOW,
            "bollinger_band_method": EnrichedData._BOLLINGER_BAND_METHOD,
            "stoch_rsi_window": EnrichedData._STOCH_RSI_WINDOW,
            "stoch_rsi_min_periods": EnrichedData._STOCH_RSI_MIN_PERIODS,
            "obv_fill_method": EnrichedData._OBV_FILL_METHOD,
            "atr_window": EnrichedData._ATR_WINDOW,
            "williams_r_window": EnrichedData._WILLIAMS_R_WINDOW,
        }
