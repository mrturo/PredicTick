"""
Provides a collection of static methods to compute a wide range of technical indicators.

used in quantitative analysis and algorithmic trading strategies.

This module includes momentum, volatility, volume, trend, and pattern-based indicators,
with both fixed-period and dynamic rolling window variants. Many methods are adapted
to intraday granularities and leverage centralized configuration for time intervals.

Examples of included indicators:
- ADX, ATR, RSI, MACD, OBV
- Bollinger Bands, %B, BB width
- Intraday/overnight return, price/volume change, volatility
- Candle and multi-candle pattern detection with raw/signal output modes

All computations are optimized for memory efficiency using float32 and are robust to
missing or insufficient data.

Dependencies:
- pandas, numpy, ta, utils.parameters, utils.interval, market_data.candle

Typical usage:
    from technical_indicators import TechnicalIndicators
    rsi = TechnicalIndicators.compute_rsi(close_series, window=14)
"""

from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import ta  # type: ignore

from market_data.candle import Candle, MultiCandlePattern
from utils.interval import Interval
from utils.parameters import ParameterLoader


# pylint: disable=too-many-public-methods
class TechnicalIndicators:
    """Static methods for computing common technical indicators."""

    _PARAMS = ParameterLoader()
    _RAW_DATA_INTERVAL = Interval.market_raw_data()
    _INTERVAL_TO_MINUTES = _PARAMS.get("interval_to_minutes")

    @staticmethod
    def _estimate_records_per_day(df: pd.DataFrame) -> int:
        """Estimate number of records per day from a datetime-indexed DataFrame."""
        df = df.copy()
        df["date"] = (
            df.index.date if df.index.name else pd.to_datetime(df["datetime"]).dt.date
        )
        counts = df.groupby("date").size()
        return int(counts.median())

    @staticmethod
    def compute_adx_14d(
        datetime: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> Optional[pd.Series]:
        """Compute ADX over the last 14 days."""
        try:
            df = pd.DataFrame(
                {"datetime": datetime, "high": high, "low": low, "close": close}
            )
            records_per_day = TechnicalIndicators._estimate_records_per_day(df)
            window = 14 * records_per_day
            if window <= 0:
                return pd.Series(np.nan, index=high.index, dtype="float32")
            if len(df) < window:
                return pd.Series(np.nan, index=high.index, dtype="float32")
            return (
                ta.trend.ADXIndicator(high=high, low=low, close=close, window=window)
                .adx()
                .astype("float32")
            )
        except (KeyError, ValueError, ZeroDivisionError, TypeError):
            return pd.Series(np.nan, index=high.index, dtype="float32")

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
    def compute_atr_14d(
        datetime: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> Optional[pd.Series]:
        """Compute ATR over the last 14 days."""
        try:
            df = pd.DataFrame(
                {"datetime": datetime, "high": high, "low": low, "close": close}
            )
            records_per_day = TechnicalIndicators._estimate_records_per_day(df)
            window = 14 * records_per_day
            if window <= 0:
                return pd.Series(np.nan, index=high.index, dtype="float32")
            if len(df) < window:
                return pd.Series(np.nan, index=high.index, dtype="float32")
            return (
                ta.volatility.AverageTrueRange(
                    high=high, low=low, close=close, window=window
                )
                .average_true_range()
                .astype("float32")
            )
        except (KeyError, ValueError, ZeroDivisionError, TypeError):
            return pd.Series(np.nan, index=high.index, dtype="float32")

    @staticmethod
    def compute_average_price(high: pd.Series, low: pd.Series) -> pd.Series:
        """Compute average of high and low prices."""
        return (high + low) / 2

    @staticmethod
    def compute_bb_width(close: pd.Series, window: int) -> pd.Series:
        """Compute Bollinger Band width as max-min over window."""
        return (
            close.rolling(window)
            .apply(lambda x: x.max() - x.min(), raw=True)
            .astype("float32")
        )

    @staticmethod
    def compute_bollinger_pct_b(
        close: pd.Series, window: int = 20, window_dev: float = 2
    ) -> pd.Series:
        """Compute Bollinger Band %B indicator."""
        return ta.volatility.BollingerBands(
            close=close, window=window, window_dev=window_dev
        ).bollinger_pband()

    @staticmethod
    def compute_intraday_return(close: pd.Series, open_: pd.Series) -> pd.Series:
        """Compute intraday return as (close / open) - 1."""
        return (close / open_) - 1

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
    def compute_momentum_3h(close: pd.Series) -> pd.Series:
        """Compute 3-hour momentum."""
        step: int = TechnicalIndicators._INTERVAL_TO_MINUTES[
            TechnicalIndicators._RAW_DATA_INTERVAL
        ]
        periods = 180 // step
        return (close - close.shift(periods)).astype("float32")

    @staticmethod
    def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Compute On-Balance Volume (OBV) indicator."""
        return ta.volume.OnBalanceVolumeIndicator(
            close=close, volume=volume
        ).on_balance_volume()

    @staticmethod
    def compute_overnight_return(open_: pd.Series) -> pd.Series:
        """Compute overnight return as percentage change in open prices."""
        return open_.pct_change(fill_method=None).fillna(0)

    @staticmethod
    def compute_price_change(close: pd.Series, open_: pd.Series) -> pd.Series:
        """Compute daily price change as close - open."""
        return close - open_

    @staticmethod
    def compute_price_derivative(close: pd.Series) -> pd.Series:
        """Compute discrete price derivative (approximation)."""
        return close.diff().astype("float32")

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
    def compute_return(close: pd.Series) -> pd.Series:
        """Compute percentage return based on 'close' prices."""
        return close.pct_change(fill_method=None)

    @staticmethod
    def compute_return_1h(close: pd.Series) -> pd.Series:
        """Compute 1-hour return as percent change from 60 minutes ago."""
        step: int = TechnicalIndicators._INTERVAL_TO_MINUTES[
            TechnicalIndicators._RAW_DATA_INTERVAL
        ]
        periods = 60 // step
        return close.pct_change(periods=periods).astype("float32")

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
    def compute_smoothed_derivative(close: pd.Series, window: int = 5) -> pd.Series:
        """Compute smoothed price derivative using rolling mean."""
        return close.diff().rolling(window=window).mean().astype("float32")

    @staticmethod
    def compute_stoch_rsi(close: pd.Series, window: int) -> pd.Series:
        """Compute Stochastic RSI."""
        rsi = TechnicalIndicators.compute_rsi(close, window)
        min_rsi = rsi.rolling(window=window, min_periods=1).min()
        max_rsi = rsi.rolling(window=window, min_periods=1).max()
        return ((rsi - min_rsi) / (max_rsi - min_rsi)).astype("float32")

    @staticmethod
    def compute_typical_price(
        high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """Compute the typical price as average of high, low and close."""
        return (high + low + close) / 3

    @staticmethod
    def compute_volatility(
        high: pd.Series, low: pd.Series, open_: pd.Series
    ) -> pd.Series:
        """Compute volatility as (high - low) / open, avoiding division by zero."""
        return (high - low) / open_.replace(0, pd.NA)

    @staticmethod
    def compute_volatility_3h(return_1h: pd.Series) -> pd.Series:
        """Compute 3-hour rolling volatility (std dev) of return."""
        step: int = TechnicalIndicators._INTERVAL_TO_MINUTES[
            TechnicalIndicators._RAW_DATA_INTERVAL
        ]
        window = 180 // step
        return return_1h.rolling(window=window).std().astype("float32")

    @staticmethod
    def compute_volume_change(volume: pd.Series) -> pd.Series:
        """Compute percentage change in volume."""
        return volume.pct_change(fill_method=None)

    @staticmethod
    def compute_volume_rvol_20d(
        datetime: pd.Series, volume: pd.Series
    ) -> Optional[pd.Series]:
        """Compute relative volume (RVOL) over the last 20 days."""
        try:
            df = pd.DataFrame({"datetime": datetime, "volume": volume})
            records_per_day = TechnicalIndicators._estimate_records_per_day(df)
            window = 20 * records_per_day
            if window <= 0:
                return pd.Series(np.nan, index=volume.index, dtype="float32")
            if len(df) < window:
                return pd.Series(np.nan, index=volume.index, dtype="float32")
            if len(volume) < window:
                return pd.Series(np.nan, index=volume.index, dtype="float32")
            result = volume / volume.rolling(window=window, min_periods=window).mean()
            return result.astype("float32")
        except (KeyError, ValueError, ZeroDivisionError, TypeError):
            return pd.Series(np.nan, index=volume.index, dtype="float32")

    @staticmethod
    def compute_williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int
    ) -> pd.Series:
        """Compute Williams %R indicator."""
        high_max = high.rolling(window=window).max()
        low_min = low.rolling(window=window).min()
        williams_r = -100 * ((high_max - close) / (high_max - low_min))
        return williams_r.astype("float32")

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    @staticmethod
    def compute_candle_pattern(
        datetime: pd.Series,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        raw: bool,
        candle_freq: Optional[str] = None,
    ) -> Optional[pd.Series]:
        """
        Compute single-candle patterns.

        Args:
            datetime (pd.Series): Datetime prices.
            open_ (pd.Series): Opening prices.
            high (pd.Series): High prices.
            low (pd.Series): Low prices.
            close (pd.Series): Closing prices.
            raw (bool): If True, returns pattern name; if False, returns bullishness score.
            candle_freq: (str): Timeframe of each candle used to adjust pattern sensitivity.

        Returns:
            pd.Series: Pattern name or score per row.
        """
        candle_freq = candle_freq or TechnicalIndicators._RAW_DATA_INTERVAL
        source_minutes = TechnicalIndicators._INTERVAL_TO_MINUTES[
            TechnicalIndicators._RAW_DATA_INTERVAL
        ]
        target_minutes = TechnicalIndicators._INTERVAL_TO_MINUTES[candle_freq]

        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(datetime),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
            }
        ).set_index("datetime")

        if target_minutes != source_minutes:
            rule = f"{target_minutes}min"
            resampled = (
                df.resample(rule)
                .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
                .dropna()
            )

            candles = [
                Candle(o, h, l, c)
                for o, h, l, c in zip(
                    resampled["open"],
                    resampled["high"],
                    resampled["low"],
                    resampled["close"],
                )
            ]
            values = [c.detect_pattern() if raw else c.score() for c in candles]
            results = pd.DataFrame({"datetime": resampled.index, "pattern": values})

            merged = pd.merge_asof(
                df.reset_index(),
                results,
                on="datetime",
                direction="backward",
                tolerance=pd.Timedelta(f"{target_minutes}min"),
            ).set_index("datetime")

            return merged["pattern"].astype("object" if raw else "float32")

        candles = [Candle(o, h, l, c) for o, h, l, c in zip(open_, high, low, close)]
        result = [c.detect_pattern() if raw else c.score() for c in candles]
        return pd.Series(result, index=open_.index).astype(
            "object" if raw else "float32"
        )

    @staticmethod
    def compute_multi_candle_pattern(
        datetime: pd.Series,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        raw: bool,
        candle_freq: Optional[str] = None,
    ) -> Optional[pd.Series]:
        """
        Compute multi-candle patterns over rolling windows of 3 candles.

        Args:
            datetime (pd.Series): Datetime prices.
            open_ (pd.Series): Opening prices.
            high (pd.Series): High prices.
            low (pd.Series): Low prices.
            close (pd.Series): Closing prices.
            raw (bool): If True, returns pattern name; if False, returns bullishness score.
            candle_freq: (str): Timeframe of each candle used to adjust pattern sensitivity.

        Returns:
            pd.Series: Pattern name or score per row.
        """
        candle_freq = candle_freq or TechnicalIndicators._RAW_DATA_INTERVAL
        source_minutes = TechnicalIndicators._INTERVAL_TO_MINUTES[
            TechnicalIndicators._RAW_DATA_INTERVAL
        ]
        target_minutes = TechnicalIndicators._INTERVAL_TO_MINUTES[candle_freq]

        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(datetime),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
            }
        ).set_index("datetime")

        if target_minutes != source_minutes:
            rule = f"{target_minutes}min"
            resampled = (
                df.resample(rule)
                .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
                .dropna()
            )

            candles = [
                Candle(o, h, l, c)
                for o, h, l, c in zip(
                    resampled["open"],
                    resampled["high"],
                    resampled["low"],
                    resampled["close"],
                )
            ]
            values = [
                (
                    None
                    if i < 2
                    else (
                        MultiCandlePattern.detect_pattern(candles[i - 2 : i + 1])
                        if raw
                        else MultiCandlePattern.score(candles[i - 2 : i + 1])
                    )
                )
                for i in range(len(candles))
            ]
            results = pd.DataFrame({"datetime": resampled.index, "pattern": values})

            merged = pd.merge_asof(
                df.reset_index(),
                results,
                on="datetime",
                direction="backward",
                tolerance=pd.Timedelta(f"{target_minutes}min"),
            ).set_index("datetime")

            return merged["pattern"].astype("object" if raw else "float32")

        candles = [Candle(o, h, l, c) for o, h, l, c in zip(open_, high, low, close)]
        values = [
            (
                None
                if i < 2
                else (
                    MultiCandlePattern.detect_pattern(candles[i - 2 : i + 1])
                    if raw
                    else MultiCandlePattern.score(candles[i - 2 : i + 1])
                )
            )
            for i in range(len(candles))
        ]
        return pd.Series(values, index=open_.index).astype(
            "object" if raw else "float32"
        )
