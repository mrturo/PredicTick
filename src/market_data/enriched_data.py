"""Market enriched data for managing symbol metadata and historical prices."""

from typing import Any, Dict, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import ta  # type: ignore

from market_data.raw_data import RawData
from utils.json_manager import JsonManager
from utils.parameters import ParameterLoader


# pylint: disable=too-many-public-methods
class TechnicalIndicators:
    """Static methods for computing common technical indicators."""

    @staticmethod
    def compute_return(close: pd.Series) -> pd.Series:
        """Compute percentage return based on 'close' prices."""
        return close.pct_change(fill_method=None)

    @staticmethod
    def compute_volatility(
        high: pd.Series, low: pd.Series, open_: pd.Series
    ) -> pd.Series:
        """Compute volatility as (high - low) / open, avoiding division by zero."""
        return (high - low) / open_.replace(0, pd.NA)

    @staticmethod
    def compute_price_change(close: pd.Series, open_: pd.Series) -> pd.Series:
        """Compute daily price change as close - open."""
        return close - open_

    @staticmethod
    def compute_volume_change(volume: pd.Series) -> pd.Series:
        """Compute percentage change in volume."""
        return volume.pct_change(fill_method=None)

    @staticmethod
    def compute_typical_price(
        high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """Compute the typical price as average of high, low and close."""
        return (high + low + close) / 3

    @staticmethod
    def compute_average_price(high: pd.Series, low: pd.Series) -> pd.Series:
        """Compute average of high and low prices."""
        return (high + low) / 2

    @staticmethod
    def compute_range(high: pd.Series, low: pd.Series) -> pd.Series:
        """Compute price range as high - low."""
        return high - low

    @staticmethod
    def compute_relative_volume(volume: pd.Series, window: int) -> pd.Series:
        """
        Compute relative volume using rolling mean.

        Args:
            volume (pd.Series): Volume series.
            window (int): Rolling window size.

        Returns:
        pd.Series: Relative volume.
        """
        return volume / volume.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def compute_atr_14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Compute 14-period Average True Range (ATR)."""
        return ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()

    @staticmethod
    def compute_overnight_return(open_: pd.Series) -> pd.Series:
        """Compute overnight return as percentage change in open prices."""
        return open_.pct_change(fill_method=None).fillna(0)

    @staticmethod
    def compute_intraday_return(close: pd.Series, open_: pd.Series) -> pd.Series:
        """Compute intraday return as (close / open) - 1."""
        return (close / open_) - 1

    @staticmethod
    def compute_volume_rvol(volume: pd.Series, window: int = 20) -> pd.Series:
        """Compute relative volume using rolling average."""
        return volume / volume.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Compute On-Balance Volume (OBV) indicator."""
        return ta.volume.OnBalanceVolumeIndicator(
            close=close, volume=volume
        ).on_balance_volume()

    @staticmethod
    def compute_adx_14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Compute 14-period Average Directional Index (ADX)."""
        return ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()

    @staticmethod
    def compute_bollinger_pct_b(
        close: pd.Series, window: int = 20, window_dev: float = 2
    ) -> pd.Series:
        """Compute Bollinger Band %B indicator."""
        return ta.volatility.BollingerBands(
            close=close, window=window, window_dev=window_dev
        ).bollinger_pband()

    @staticmethod
    def compute_macd(
        series: pd.Series, fast: int, slow: int, signal: int
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence) indicators.

        Args:
            series (pd.Series): Price series.
            fast (int): Fast EMA period.
            slow (int): Slow EMA period.
            signal (int): Signal EMA period.

        Returns:
            pd.DataFrame: DataFrame with MACD, signal line, and histogram.
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame(
            {"macd": macd_line, "signal": signal_line, "histogram": histogram}
        )

    @staticmethod
    def compute_return_1h(close: pd.Series) -> pd.Series:
        """Compute hourly return as percent change in close."""
        return close.pct_change().astype("float32")

    @staticmethod
    def compute_volatility_3h(return_1h: pd.Series) -> pd.Series:
        """Compute 3-hour rolling volatility (std dev) of return."""
        return return_1h.rolling(window=3).std().astype("float32")

    @staticmethod
    def compute_momentum_3h(close: pd.Series) -> pd.Series:
        """Compute 3-hour momentum."""
        return (close - close.shift(3)).astype("float32")

    @staticmethod
    def compute_rsi(close: pd.Series, window: int) -> pd.Series:
        """Compute RSI over a given window."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.astype("float32")

    @staticmethod
    def compute_bb_width(close: pd.Series, window: int) -> pd.Series:
        """Compute Bollinger Band width as max-min over window."""
        return (
            close.rolling(window)
            .apply(lambda x: x.max() - x.min(), raw=True)
            .astype("float32")
        )

    @staticmethod
    def compute_stoch_rsi(close: pd.Series, window: int) -> pd.Series:
        """Compute Stochastic RSI."""
        rsi = TechnicalIndicators.compute_rsi(close, window)
        min_rsi = rsi.rolling(window=window, min_periods=1).min()
        max_rsi = rsi.rolling(window=window, min_periods=1).max()
        return ((rsi - min_rsi) / (max_rsi - min_rsi)).astype("float32")

    @staticmethod
    def compute_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int
    ) -> pd.Series:
        """Compute Average True Range (ATR) over a given window."""
        tr = pd.concat(
            [
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        )
        true_range = tr.max(axis=1)
        return true_range.rolling(window=window).mean().astype("float32")

    @staticmethod
    def compute_williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int
    ) -> pd.Series:
        """Compute Williams %R indicator."""
        high_max = high.rolling(window=window).max()
        low_min = low.rolling(window=window).min()
        williams_r = -100 * ((high_max - close) / (high_max - low_min))
        return williams_r.astype("float32")


class EnrichedData:
    """Centralized access point for managing symbol enriched data."""

    _PARAMS = ParameterLoader()
    _ENRICHED_MARKETDATA_FILEPATH = _PARAMS.get("enriched_marketdata_filepath")
    _VOLUME_WINDOW = _PARAMS.get("volume_window")
    _RSI_WINDOW = _PARAMS.get("rsi_window_backtest")
    _MACD_FAST = _PARAMS.get("macd_fast")
    _MACD_SLOW = _PARAMS.get("macd_slow")
    _MACD_SIGNAL = _PARAMS.get("macd_signal")
    _BOLLINGER_WINDOW = _PARAMS.get("bollinger_window")
    _BOLLINGER_BAND_METHOD = _PARAMS.get("bollinger_band_method")
    _STOCH_RSI_WINDOW = _PARAMS.get("stoch_rsi_window")
    _STOCH_RSI_MIN_PERIODS = _PARAMS.get("stoch_rsi_min_periods")
    _OBV_FILL_METHOD = _PARAMS.get("obv_fill_method")
    _ATR_WINDOW = _PARAMS.get("atr_window")
    _WILLIAMS_R_WINDOW = _PARAMS.get("williams_r_window")

    _symbols: Dict[str, dict] = {}
    _ranges: Any = {}

    @staticmethod
    def _get_filepath(filepath: Optional[str], default=None) -> Optional[str]:
        """Resolve a filepath, falling back to a default if necessary."""
        if filepath is None or len(filepath.strip()) == 0:
            return default
        return filepath

    @staticmethod
    def get_symbols() -> Dict[str, dict]:
        """Return all loaded symbols."""
        return EnrichedData._symbols

    @staticmethod
    def set_symbols(symbols: Dict[str, dict]) -> None:
        """Set the dictionary of symbols."""
        EnrichedData._symbols = symbols

    @staticmethod
    def get_symbol(symbol: str):
        """Retrieve metadata for a specific symbol."""
        return EnrichedData._symbols.get(symbol)

    @staticmethod
    def get_ranges() -> Any:
        """Return the object of ranges."""
        return EnrichedData._ranges

    @staticmethod
    def set_ranges(ranges: Any) -> None:
        """Set the object of ranges."""
        EnrichedData._ranges = ranges

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

        # Obtener el mínimo datetime por símbolo
        min_datetimes = [
            min(pd.to_datetime([row["datetime"] for row in data["historical_prices"]]))
            for data in symbols.values()
            if data.get("historical_prices")
        ]

        if not min_datetimes:
            return None

        # Calcular el máximo entre los mínimos
        global_start = max(min_datetimes)

        # Filtrar historical_prices para cada símbolo
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
        symbols: list = []

        if enriched_data is not None:
            symbols = enriched_data["symbols"]

        enriched = RawData.normalize_historical_prices(symbols)

        EnrichedData._symbols = dict(enriched)
        return {
            "symbols": EnrichedData._symbols,
        }

    @staticmethod
    def save(filepath: Optional[str] = None) -> dict[str, Any]:
        """Persist current symbol data to disk."""
        result = {
            "ranges": EnrichedData._ranges,
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
    def _compute_features(df: pd.DataFrame, ranges: dict) -> pd.DataFrame:
        """Compute raw and scaled features for a symbol DataFrame."""
        df = df.copy()

        def scale_column(
            series: pd.Series, min_val: float, max_val: float
        ) -> pd.Series:
            return (
                (series - min_val) / (max_val - min_val)
                if min_val is not None and max_val not in (None, min_val)
                else pd.Series(0.0, index=series.index, dtype="float32")
            )

        def add_indicators(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
            def prefixed(col: str) -> str:
                return f"{prefix}{col}" if prefix else col

            df[prefixed("adx_14")] = TechnicalIndicators.compute_adx_14(
                df[prefixed("high")], df[prefixed("low")], df[prefixed("close")]
            )
            df[prefixed("atr")] = TechnicalIndicators.compute_atr(
                df[prefixed("high")],
                df[prefixed("low")],
                df[prefixed("close")],
                EnrichedData._ATR_WINDOW,
            )
            df[prefixed("atr_14")] = TechnicalIndicators.compute_atr_14(
                df[prefixed("high")], df[prefixed("low")], df[prefixed("close")]
            )
            df[prefixed("average_price")] = TechnicalIndicators.compute_average_price(
                df[prefixed("high")], df[prefixed("low")]
            )
            if EnrichedData._BOLLINGER_BAND_METHOD == "max-min":
                df[prefixed("bb_width")] = TechnicalIndicators.compute_bb_width(
                    df[prefixed("close")], EnrichedData._BOLLINGER_WINDOW
                )
            df[prefixed("bollinger_pct_b")] = (
                TechnicalIndicators.compute_bollinger_pct_b(df[prefixed("close")])
            )
            df[prefixed("intraday_return")] = (
                TechnicalIndicators.compute_intraday_return(
                    df[prefixed("close")], df[prefixed("open")]
                )
            )
            df[prefixed("macd")] = TechnicalIndicators.compute_macd(
                df[prefixed("close")],
                EnrichedData._MACD_FAST,
                EnrichedData._MACD_SLOW,
                EnrichedData._MACD_SIGNAL,
            )["histogram"]
            df[prefixed("momentum_3h")] = TechnicalIndicators.compute_momentum_3h(
                df[prefixed("close")]
            )

            # Conditional: only if volume is available
            if "volume" in df.columns:
                df[prefixed("obv")] = TechnicalIndicators.compute_obv(
                    df[prefixed("close")], df[prefixed("volume")]
                )
                df[prefixed("relative_volume")] = (
                    TechnicalIndicators.compute_relative_volume(
                        df[prefixed("volume")], EnrichedData._VOLUME_WINDOW
                    )
                )
                df[prefixed("volume_change")] = (
                    TechnicalIndicators.compute_volume_change(df[prefixed("volume")])
                )
                df[prefixed("volume_rvol_20d")] = (
                    TechnicalIndicators.compute_volume_rvol(df[prefixed("volume")])
                )

            df[prefixed("overnight_return")] = (
                TechnicalIndicators.compute_overnight_return(df[prefixed("open")])
            )
            df[prefixed("price_change")] = TechnicalIndicators.compute_price_change(
                df[prefixed("close")], df[prefixed("open")]
            )
            df[prefixed("range")] = TechnicalIndicators.compute_range(
                df[prefixed("high")], df[prefixed("low")]
            )
            df[prefixed("return")] = TechnicalIndicators.compute_return(
                df[prefixed("close")]
            )
            df[prefixed("return_1h")] = TechnicalIndicators.compute_return_1h(
                df[prefixed("close")]
            )
            df[prefixed("rsi")] = TechnicalIndicators.compute_rsi(
                df[prefixed("close")], EnrichedData._RSI_WINDOW
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
            df[prefixed("volatility_3h")] = TechnicalIndicators.compute_volatility_3h(
                df[prefixed("return_1h")]
            )
            df[prefixed("williams_r")] = TechnicalIndicators.compute_williams_r(
                df[prefixed("high")],
                df[prefixed("low")],
                df[prefixed("close")],
                EnrichedData._WILLIAMS_R_WINDOW,
            )
            return df

        # Escalar precios
        for col in ["open", "low", "high", "close", "adj_close"]:
            df[f"scaled_{col}"] = scale_column(
                df[col], ranges["min_price"], ranges["max_price"]
            )
        df["scaled_volume"] = scale_column(
            df["volume"], ranges["min_volume"], ranges["max_volume"]
        )

        # Calcular indicadores
        df = add_indicators(df)
        df = add_indicators(df, prefix="scaled_")

        # Determine which columns to consider for NaN filtering
        columns_to_check = df.columns.difference(
            ["volume", "obv", "relative_volume", "volume_change", "volume_rvol_20d"]
        )

        # Exclude columns that are fully missing (like 'volume' for FX pairs)
        columns_with_data = [
            col for col in columns_to_check if not df[col].isna().all()
        ]

        # Identify the last row index with NaN in relevant columns
        last_nan_index = (
            df[columns_with_data].isna().any(axis=1).pipe(lambda s: s[s].index.max())
        )

        # Drop rows up to and including the last NaN index
        if last_nan_index is not None:
            df = df.loc[last_nan_index + 1 :]

        # Drop any remaining NaNs in those same relevant columns
        df = df.dropna(subset=columns_with_data)

        return df

    @staticmethod
    def _format_symbol_output(key: str, value: Any, df: pd.DataFrame) -> dict:
        """Format the enriched and scaled DataFrame into the final output structure."""
        base_cols = [
            "datetime",
            "open",
            "low",
            "high",
            "close",
            "adj_close",
            "volume",
            "adx_14",
            "atr",
            "atr_14",
            "average_price",
            "bb_width",
            "bollinger_pct_b",
            "intraday_return",
            "macd",
            "momentum_3h",
            "obv",
            "overnight_return",
            "price_change",
            "range",
            "relative_volume",
            "return",
            "return_1h",
            "rsi",
            "stoch_rsi",
            "typical_price",
            "volatility",
            "volatility_3h",
            "volume_change",
            "volume_rvol_20d",
            "williams_r",
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
    def process_symbol(key: str, value: Any, ranges: Any) -> dict:
        """Process a single symbol by computing features and formatting output."""
        df = pd.DataFrame(value["historical_prices"])
        df = EnrichedData._compute_features(df, ranges)
        df = EnrichedData._format_symbol_output(key, value, df)
        return df

    @staticmethod
    def generate(filepath: Optional[str] = None) -> dict[str, Any]:
        """Generate enriched symbol data with normalized prices and volumes, saving the result."""
        local_filepath = EnrichedData._get_filepath(
            filepath, EnrichedData._ENRICHED_MARKETDATA_FILEPATH
        )
        if local_filepath:
            JsonManager.delete(local_filepath)

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
