"""
Collection of static helpers to compute commonly‑used technical indicators for.

quantitative analysis and algorithmic trading.

All calculations are **data‑driven**: window lengths are inferred from the actual
sampling rate and the median trading‑session span in the input data.  This makes
every method robust to symbols that trade 8 h, 6.5 h, 24 h, or any other
schedule, without relying on hard‑coded parameters.

Functions return *float32* results for memory efficiency and gracefully fall
back to *NaN* when the available history is insufficient.

Dependencies (Python ≥ 3.10)
---------------------------
• pandas  • numpy  • ta  • utils.parameters  • utils.interval
• market_data.candle  • market_data.time_resampler
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import ta  # type: ignore

from market_data.candle import Candle, MultiCandlePattern
from market_data.time_resampler import TimeResampler
from utils.interval import Interval
from utils.logger import Logger
from utils.parameters import ParameterLoader


class TechnicalIndicators:  # pylint: disable=too-many-public-methods
    """Utility class exposing static indicator functions."""

    _PARAMS = ParameterLoader()
    _RAW_DATA_INTERVAL = Interval.market_raw_data()
    _ENRICHED_DATA_INTERVAL = Interval.market_enriched_data()
    _INTERVAL_TO_MINUTES = _PARAMS.get("interval_to_minutes")

    @staticmethod
    def _estimate_records_per_day(df: pd.DataFrame) -> int:
        """Median number of records per *calendar* day in *df*."""
        tmp = df.copy()
        tmp["date"] = (
            tmp.index.date
            if tmp.index.name
            else pd.to_datetime(tmp["datetime"]).dt.date
        )
        return int(tmp.groupby("date").size().median())

    @staticmethod
    def _infer_bar_seconds(idx: pd.Index) -> int:  # type: ignore[override]
        """Return the modal bar length in *seconds* from a :class:`DatetimeIndex`."""
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError("Index must be a DatetimeIndex to infer bar length")

        diffs = idx.to_series().diff().dropna()
        if diffs.empty:
            raise ValueError("Need at least two timestamps to infer bar size")

        bar_delta = diffs.mode().iloc[0]
        if not hasattr(bar_delta, "total_seconds"):
            raise TypeError(
                "Time differences are not timedelta‑like; cannot infer bar size"
            )

        return int(bar_delta.total_seconds())

    @staticmethod
    def _bar_seconds_or_default(idx: pd.Index) -> int:
        """
        Safe wrapper around :py:meth:`_infer_bar_seconds`.

        Falls back to the configured *raw-data* interval when *idx* is not
        datetime-based – useful in unit-tests or pipelines that lost their index.
        """
        try:
            return TechnicalIndicators._infer_bar_seconds(idx)
        except Exception as exc:  # pylint: disable=broad-except
            step_min = TechnicalIndicators._INTERVAL_TO_MINUTES.get(
                TechnicalIndicators._RAW_DATA_INTERVAL, 1
            )
            Logger.debug(
                f"_bar_seconds_or_default | fallback to {step_min} min per bar ({exc})"
            )
            return step_min * 60

    @staticmethod
    def _records_per_session(idx: pd.DatetimeIndex) -> int:
        """Number of rows that cover one full trading session."""
        sec_per_bar = TechnicalIndicators._infer_bar_seconds(idx)
        daily_span_sec = (
            pd.Series(idx)
            .groupby(idx.date)
            .agg(lambda s: (s.max() - s.min()).total_seconds() + sec_per_bar)
        )
        return int(math.ceil(daily_span_sec.median() / sec_per_bar))

    @staticmethod
    def compute_adx_14d(
        datetime: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """14-session Average Directional Index (ADX).

        The window length is derived from the median number of bars per trading
        session observed in *datetime*, making the calculation agnostic to the
        instrument's trading schedule.  When fewer than 14 completed sessions
        are available, the function returns a *float32* Series filled with NaNs.
        """
        df = (
            pd.DataFrame(
                {"datetime": datetime, "high": high, "low": low, "close": close}
            )
            .set_index("datetime")
            .sort_index()
        )
        try:
            if df.index.tz is None:
                df = df.tz_localize("UTC")
            if (df.index[-1] - df.index[0]).days < 14:
                Logger.debug("ADX-14d | insufficient history (<14 days)")
                return pd.Series(np.nan, index=high.index, dtype="float32")
            bars_day = TechnicalIndicators._records_per_session(df.index)
            window = 14 * bars_day
            Logger.debug(f"ADX-14d | bars_day={bars_day} window={window}")
            adx = ta.trend.ADXIndicator(
                high=df["high"], low=df["low"], close=df["close"], window=window
            ).adx()
            adx = adx.astype("float32")
            adx.index = high.index
            return adx
        except Exception as exc:  # pylint: disable=broad-except
            Logger.debug(f"ADX-14d | failure: {exc}")
            return pd.Series(np.nan, index=high.index, dtype="float32")

    @staticmethod
    def compute_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int
    ) -> pd.Series:
        """Compute Average True Range (ATR) over a given window."""
        tr = pd.concat(
            [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        )
        return tr.max(axis=1).rolling(window).mean().astype("float32")

    @staticmethod
    def compute_atr_14d(
        datetime: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """14‑session Average True Range (ATR)."""
        df = (
            pd.DataFrame(
                {"datetime": datetime, "high": high, "low": low, "close": close}
            )
            .set_index("datetime")
            .sort_index()
        )
        try:
            if df.index.tz is None:
                df = df.tz_localize("UTC")
            if (df.index[-1] - df.index[0]).days < 14:
                Logger.debug("ATR‑14d | insufficient history (<14 days)")
                return pd.Series(np.nan, index=high.index, dtype="float32")

            bars_day = TechnicalIndicators._records_per_session(df.index)
            window = 14 * bars_day
            Logger.debug(f"ATR‑14d | bars_day={bars_day} window={window}")

            atr = ta.volatility.AverageTrueRange(
                high=df["high"], low=df["low"], close=df["close"], window=window
            ).average_true_range()
            atr = atr.astype("float32")
            atr.index = high.index
            return atr
        except Exception as exc:  # pylint: disable=broad-except
            Logger.debug(f"ATR‑14d | failure: {exc}")
            return pd.Series(np.nan, index=high.index, dtype="float32")

    @staticmethod
    def compute_momentum_3h(close: pd.Series) -> pd.Series:
        """3‑hour momentum (price difference)."""
        try:
            bar_sec = TechnicalIndicators._bar_seconds_or_default(close.index)
            periods = math.ceil(3 * 3600 / bar_sec)
            Logger.debug(f"Momentum‑3h | bar_sec={bar_sec} periods={periods}")
            return (close - close.shift(periods)).astype("float32")
        except Exception as exc:  # pylint: disable=broad-except
            Logger.debug(f"Momentum‑3h | failure: {exc}")
            return pd.Series(np.nan, index=close.index, dtype="float32")

    @staticmethod
    def compute_return_1h(close: pd.Series) -> pd.Series:
        """1‑hour percentage return."""
        try:
            bar_sec = TechnicalIndicators._bar_seconds_or_default(close.index)
            periods = math.ceil(3600 / bar_sec)
            Logger.debug(f"Return‑1h | bar_sec={bar_sec} periods={periods}")
            return close.pct_change(periods=periods, fill_method=None).astype("float32")
        except Exception as exc:  # pylint: disable=broad-except
            Logger.debug(f"Return‑1h | failure: {exc}")
            return pd.Series(np.nan, index=close.index, dtype="float32")

    @staticmethod
    def compute_volatility_3h(return_1h: pd.Series) -> pd.Series:
        """3‑hour rolling volatility (standard deviation of 1‑h returns)."""
        try:
            bar_sec = TechnicalIndicators._bar_seconds_or_default(return_1h.index)
            window = math.ceil(3 * 3600 / bar_sec)
            Logger.debug(f"Volatility‑3h | bar_sec={bar_sec} window={window}")
            return return_1h.rolling(window).std().astype("float32")
        except Exception as exc:  # pylint: disable=broad-except
            Logger.debug(f"Volatility‑3h | failure: {exc}")
            return pd.Series(np.nan, index=return_1h.index, dtype="float32")

    @staticmethod
    def compute_volume_rvol_20d(datetime: pd.Series, volume: pd.Series) -> pd.Series:
        """20‑session Relative Volume (RVOL)."""
        df = pd.DataFrame({"datetime": datetime, "volume": volume}).set_index(
            "datetime"
        )
        try:
            if df.index.tz is None:
                df = df.tz_localize("UTC")
            if (df.index[-1] - df.index[0]).days < 20:
                Logger.debug("RVOL‑20d | insufficient history (<20 days)")
                return pd.Series(np.nan, index=volume.index, dtype="float32")

            bars_day = TechnicalIndicators._records_per_session(df.index)
            window = 20 * bars_day
            Logger.debug(f"RVOL‑20d | bars_day={bars_day} window={window}")

            rvol = volume / volume.rolling(window, min_periods=window).mean()
            return rvol.astype("float32")
        except Exception as exc:  # pylint: disable=broad-except
            Logger.debug(f"RVOL‑20d | failure: {exc}")
            return pd.Series(np.nan, index=volume.index, dtype="float32")

    @staticmethod
    def compute_intraday_return(close: pd.Series, open_: pd.Series) -> pd.Series:
        """Compute intraday return as percentage change from open to close.

        Args:
            close (pd.Series): Series of close prices.
            open_ (pd.Series): Series of open prices.

        Returns:
            pd.Series: Intraday returns.
        """
        return (close / open_) - 1

    @staticmethod
    def compute_price_change(close: pd.Series, open_: pd.Series) -> pd.Series:
        """Compute raw price change between open and close.

        Args:
            close (pd.Series): Series of close prices.
            open_ (pd.Series): Series of open prices.

        Returns:
            pd.Series: Absolute price change.
        """
        return close - open_

    @staticmethod
    def compute_price_derivative(close: pd.Series) -> pd.Series:
        """Compute first-order price derivative (discrete difference).

        Args:
            close (pd.Series): Series of close prices.

        Returns:
            pd.Series: Difference between consecutive prices.
        """
        return close.diff().astype("float32")

    @staticmethod
    def compute_range(high: pd.Series, low: pd.Series) -> pd.Series:
        """Compute high-low range for each bar.

        Args:
            high (pd.Series): Series of high prices.
            low (pd.Series): Series of low prices.

        Returns:
            pd.Series: Price range per bar.
        """
        return high - low

    @staticmethod
    def compute_relative_volume(volume: pd.Series, window: int) -> pd.Series:
        """Compute relative volume against rolling average.

        Args:
            volume (pd.Series): Series of volume values.
            window (int): Rolling window length.

        Returns:
            pd.Series: Relative volume ratio.
        """
        return volume / volume.rolling(window, min_periods=1).mean()

    @staticmethod
    def compute_return(close: pd.Series) -> pd.Series:
        """Compute simple percentage returns.

        Args:
            close (pd.Series): Series of close prices.

        Returns:
            pd.Series: Percentage change of close prices.
        """
        return close.pct_change(fill_method=None)

    @staticmethod
    def compute_rsi(close: pd.Series, window: int) -> pd.Series:
        """Compute Relative Strength Index (RSI).

        Args:
            close (pd.Series): Series of close prices.
            window (int): Number of periods for RSI calculation.

        Returns:
            pd.Series: RSI values.
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        return (100 - 100 / (1 + rs)).astype("float32")

    @staticmethod
    def compute_smoothed_derivative(close: pd.Series, window: int = 5) -> pd.Series:
        """Compute smoothed first-order derivative using a rolling mean.

        Args:
            close (pd.Series): Series of close prices.
            window (int, optional): Window size for smoothing. Defaults to 5.

        Returns:
            pd.Series: Smoothed derivative.
        """
        return close.diff().rolling(window).mean().astype("float32")

    @staticmethod
    def compute_stoch_rsi(close: pd.Series, window: int) -> pd.Series:
        """Compute Stochastic RSI.

        Args:
            close (pd.Series): Series of close prices.
            window (int): Window size for RSI and normalization.

        Returns:
            pd.Series: Stochastic RSI values.
        """
        rsi = TechnicalIndicators.compute_rsi(close, window)
        min_rsi = rsi.rolling(window, min_periods=1).min()
        max_rsi = rsi.rolling(window, min_periods=1).max()
        return ((rsi - min_rsi) / (max_rsi - min_rsi)).astype("float32")

    @staticmethod
    def compute_typical_price(
        high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """Compute typical price as mean of high, low and close.

        Args:
            high (pd.Series): Series of high prices.
            low (pd.Series): Series of low prices.
            close (pd.Series): Series of close prices.

        Returns:
            pd.Series: Typical price.
        """
        return (high + low + close) / 3

    @staticmethod
    def compute_volatility(
        high: pd.Series, low: pd.Series, open_: pd.Series
    ) -> pd.Series:
        """Compute simple volatility as (high - low) / open.

        Args:
            high (pd.Series): Series of high prices.
            low (pd.Series): Series of low prices.
            open_ (pd.Series): Series of open prices.

        Returns:
            pd.Series: Volatility ratio.
        """
        return (high - low) / open_.replace(0, pd.NA)

    @staticmethod
    def compute_volume_change(volume: pd.Series) -> pd.Series:
        """Compute percentage change in volume.

        Args:
            volume (pd.Series): Series of volume values.

        Returns:
            pd.Series: Percentage change in volume.
        """
        return volume.pct_change(fill_method=None)

    @staticmethod
    def compute_macd(
        series: pd.Series, fast: int, slow: int, signal: int
    ) -> pd.DataFrame:
        """Moving Average Convergence Divergence (MACD)."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return pd.DataFrame(
            {"macd": macd_line, "signal": signal_line, "histogram": hist}
        )

    @staticmethod
    def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On‑Balance Volume (OBV)."""
        return ta.volume.OnBalanceVolumeIndicator(
            close=close, volume=volume
        ).on_balance_volume()

    @staticmethod
    def compute_candle_pattern(
        raw_df: pd.DataFrame,
        raw_interval: str,
        datetime: pd.Series,
        output_as_name: bool,
        candle_interval: str,
    ) -> Optional[pd.Series]:
        """Detect single‑candle patterns."""
        try:
            candles_df = TimeResampler.by_ratio(
                raw_df, raw_interval, candle_interval, datetime
            )
            out = []
            for _, row in candles_df.iterrows():
                candle = Candle(
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                )
                out.append(
                    candle.detect_pattern() if output_as_name else candle.score()
                )
            result = pd.Series(out, index=candles_df.index)
            return result
        except Exception as exc:  # pylint: disable=broad-except
            Logger.debug(f"candle_pattern | failure: {exc}")
            return None

    @staticmethod
    def compute_multi_candle_pattern(
        raw_df: pd.DataFrame,
        raw_interval: str,
        datetime: pd.Series,
        output_as_name: bool,
        candle_interval: str,
    ) -> Optional[pd.Series]:
        """Detect 3‑bar multi‑candle patterns."""
        try:
            candles_df = TimeResampler.by_ratio(
                raw_df, raw_interval, candle_interval, datetime
            )
            results: list[str | float] = []
            for i in range(len(candles_df)):
                sub = candles_df.iloc[max(0, i - 2) : i + 1]
                candles = [
                    Candle(float(r.open), float(r.high), float(r.low), float(r.close))
                    for r in sub.itertuples(index=False)
                ]
                if len(candles) < 2:
                    results.append(np.nan)  # type: ignore[arg-type]
                else:
                    if output_as_name:
                        results.append(MultiCandlePattern.detect_pattern(candles) or "")
                    else:
                        results.append(MultiCandlePattern.score(candles))
            return pd.Series(results, index=candles_df.index)
        except Exception as exc:  # pylint: disable=broad-except
            Logger.debug(f"multi_candle_pattern | failure: {exc}")
            return None

    @staticmethod
    def compute_average_price(high: pd.Series, low: pd.Series) -> pd.Series:
        """Compute average of high and low prices."""
        return (high + low) / 2

    @staticmethod
    def compute_bollinger_pct_b(
        close: pd.Series, window: int = 20, window_dev: float = 2
    ) -> pd.Series:
        """Compute Bollinger Band %B indicator."""
        return ta.volatility.BollingerBands(
            close=close, window=window, window_dev=window_dev
        ).bollinger_pband()

    @staticmethod
    def compute_bb_width(close: pd.Series, window: int) -> pd.Series:
        """Compute Bollinger Band width as max-min over window."""
        return (
            close.rolling(window)
            .apply(lambda x: x.max() - x.min(), raw=True)
            .astype("float32")
        )

    @staticmethod
    def compute_overnight_return(open_: pd.Series) -> pd.Series:
        """Compute overnight return as percentage change in open prices."""
        return open_.pct_change(fill_method=None).fillna(0)

    @staticmethod
    def compute_williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int
    ) -> pd.Series:
        """Compute Williams %R indicator."""
        high_max = high.rolling(window=window).max()
        low_min = low.rolling(window=window).min()
        williams_r = -100 * ((high_max - close) / (high_max - low_min))
        return williams_r.astype("float32")
