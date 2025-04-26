"""
Collection of static helpers to compute commonly‑used technical indicators for.

quantitative analysis and algorithmic trading.

All calculations are *data‑driven*: window lengths are inferred from the actual
sampling rate and the median trading‑session span in the input data. This makes
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

import datetime
import math
from typing import Any, List, Optional, Set, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import ta  # type: ignore
from pandas.api.types import is_numeric_dtype

from market_data.transform.candle import Candle, MultiCandlePattern
from market_data.utils.interval import Interval, IntervalValidator
from utils.logger import Logger
from utils.parameters import ParameterLoader


class TechnicalIndicators:  # pylint: disable=too-many-public-methods
    """Utility class exposing static indicator functions.

    The class groups a wide range of **pure** functions that work on *pandas*
    Series/DataFrames. All methods are declared as *@staticmethod* so they can
    be imported individually or used via *TechnicalIndicators.* without
    instantiation.
    """

    _PARAMS = ParameterLoader()
    _RAW_DATA_INTERVAL = Interval.market_raw_data()
    _ENRICHED_DATA_INTERVAL = Interval.market_enriched_data()
    _INTERVAL_TO_MINUTES = _PARAMS.get("interval_to_minutes")

    # ------------------------------------------------------------------
    # Private helpers (docstrings included as requested)
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_records_per_day(df: pd.DataFrame) -> int:
        """Estimate the median number of bars per *calendar* day in *df*.

        Args:
            df (pd.DataFrame): DataFrame whose index is a ``DatetimeIndex`` or
                that contains a ``'datetime'`` column.
        Returns:
            int: Median count of rows observed per day. When the input index is
                not datetime‑based the function falls back to grouping by the
                ``'datetime'`` column.
        """
        tmp = df.copy()
        tmp["date"] = (
            tmp.index.date
            if tmp.index.name
            else pd.to_datetime(tmp["datetime"]).dt.date
        )
        return int(tmp.groupby("date").size().median())

    @staticmethod
    def _infer_bar_seconds(idx: pd.Index) -> int:  # type: ignore[override]
        """Infer the modal bar length *in seconds* from a ``DatetimeIndex``.

        Args:
            idx (pd.Index): Index to analyse. Must be a ``pd.DatetimeIndex``.
        Returns:
            int: Modal number of seconds between consecutive bars.
        Raises:
            TypeError: If *idx* is not a ``DatetimeIndex`` or contains
                non‑timedelta differences.
            ValueError: If fewer than two bars are available.
        """
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
        """Return the inferred bar length or the configured default.

        Args:
            idx (pd.Index): Index used to infer the bar length.
        Returns:
            int: Bar length in seconds. On failure the function returns ``step``
                where ``step`` equals the default raw‑data interval expressed
                in seconds.
        """
        try:
            return TechnicalIndicators._infer_bar_seconds(idx)
        except Exception as exc:  # pylint: disable=broad-except
            step_min = TechnicalIndicators._INTERVAL_TO_MINUTES.get(
                TechnicalIndicators._RAW_DATA_INTERVAL, 1
            )
            Logger.warning(
                f"     [_bar_seconds_or_default] fallback to {step_min} min per bar ({exc})"
            )
            return int(step_min * 60)

    @staticmethod
    def _records_per_session(idx: pd.DatetimeIndex) -> int:
        """Compute how many rows cover *one* trading session.

        Args:
            idx (pd.DatetimeIndex): Index containing intra‑day timestamps.
        Returns:
            int: Number of rows needed to span a full session. The function
                works by computing the median session length in seconds and
                dividing by the modal bar length.
        """
        sec_per_bar = TechnicalIndicators._infer_bar_seconds(idx)
        daily_span_sec = (
            pd.Series(idx)
            .groupby(idx.date)
            .agg(lambda s: (s.max() - s.min()).total_seconds() + sec_per_bar)
        )
        return int(math.ceil(daily_span_sec.median() / sec_per_bar))

    # ------------------------------------------------------------------
    # Public indicators
    # ------------------------------------------------------------------
    @staticmethod
    def compute_adx_14d(
        datetime: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """14‑session Average Directional Index (ADX).

        Args:
            datetime (pd.Series): Datetime series (tz aware recommended).
            high (pd.Series): High‑price series.
            low (pd.Series): Low‑price series.
            close (pd.Series): Close‑price series.
        Returns:
            pd.Series: ADX values as *float32*. NaNs are returned when history
                is < 14 complete sessions or an internal failure occurs.
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
                raise ValueError("requires at least 14 complete trading sessions")
            bars_day = TechnicalIndicators._records_per_session(df.index)
            window = 14 * bars_day
            adx = ta.trend.ADXIndicator(
                high=df["high"], low=df["low"], close=df["close"], window=window
            ).adx()
            adx = adx.astype("float32")
            adx.index = high.index
            return adx
        except Exception as exc:  # pylint: disable=broad-except
            Logger.warning(f"     [ADX-14d] failure: {exc}")
            return pd.Series(np.nan, index=high.index, dtype="float32")

    @staticmethod
    def compute_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int
    ) -> pd.Series:
        """Average True Range (ATR).

        Args:
            high (pd.Series): High‑price series.
            low (pd.Series): Low‑price series.
            close (pd.Series): Close‑price series.
            window (int): Rolling window size in bars.
        Returns:
            pd.Series: ATR values (*float32*).
        """
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
        """14‑session Average True Range (ATR).

        Args:
            datetime (pd.Series): Datetime series.
            high (pd.Series): High‑price series.
            low (pd.Series): Low‑price series.
            close (pd.Series): Close‑price series.
        Returns:
            pd.Series: ATR values (*float32*) or NaNs on failure.
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
                raise ValueError("requires at least 14 complete trading sessions")
            bars_day = TechnicalIndicators._records_per_session(df.index)
            window = 14 * bars_day
            atr = ta.volatility.AverageTrueRange(
                high=df["high"], low=df["low"], close=df["close"], window=window
            ).average_true_range()
            atr = atr.astype("float32")
            atr.index = high.index
            return atr
        except Exception as exc:  # pylint: disable=broad-except
            Logger.warning(f"     [ATR‑14d] failure: {exc}")
            return pd.Series(np.nan, index=high.index, dtype="float32")

    @staticmethod
    def compute_volume_rvol_20d(datetime: pd.Series, volume: pd.Series) -> pd.Series:
        """20‑session Relative Volume (RVOL).

        Args:
            datetime (pd.Series): Datetime series.
            volume (pd.Series): Raw volume series.
        Returns:
            pd.Series: RVOL ratio (*float32*).
        """
        df = pd.DataFrame({"datetime": datetime, "volume": volume}).set_index(
            "datetime"
        )
        try:
            if df.index.tz is None:
                df = df.tz_localize("UTC")
            if (df.index[-1] - df.index[0]).days < 20:
                raise ValueError("insufficient history (<20 days)")
            bars_day = TechnicalIndicators._records_per_session(df.index)
            window = 20 * bars_day
            rvol = volume / volume.rolling(window, min_periods=window).mean()
            return rvol.astype("float32")
        except Exception as exc:  # pylint: disable=broad-except
            Logger.warning(f"     [RVOL‑20d] failure: {exc}")
            return pd.Series(np.nan, index=volume.index, dtype="float32")

    @staticmethod
    def compute_intraday_return(close: pd.Series, open_: pd.Series) -> pd.Series:
        """Intraday return as *percentage* change from open to close.

        Args:
            close (pd.Series): Close‑price series.
            open_ (pd.Series): Open‑price series.
        Returns:
            pd.Series: Intraday returns (*float32*).
        """
        return ((close / open_) - 1).astype("float32")

    @staticmethod
    def compute_price_change(close: pd.Series, open_: pd.Series) -> pd.Series:
        """Raw price change between open and close.

        Args:
            close (pd.Series): Close‑price series.
            open_ (pd.Series): Open‑price series.
        Returns:
            pd.Series: Difference close – open (*float32*).
        """
        return (close - open_).astype("float32")

    @staticmethod
    def compute_price_derivative(close: pd.Series) -> pd.Series:
        """First‑order discrete derivative of close price.

        Args:
            close (pd.Series): Close‑price series.
        Returns:
            pd.Series: ``close.diff()`` cast to *float32*.
        """
        return close.diff().astype("float32")

    @staticmethod
    def compute_range(high: pd.Series, low: pd.Series) -> pd.Series:
        """High‑low range per bar.

        Args:
            high (pd.Series): High‑price series.
            low (pd.Series): Low‑price series.
        Returns:
            pd.Series: ``high - low`` (*float32*).
        """
        return (high - low).astype("float32")

    @staticmethod
    def compute_relative_volume(volume: pd.Series, window: int) -> pd.Series:
        """Relative volume against its rolling mean.

        Args:
            volume (pd.Series): Raw volume series.
            window (int): Rolling window length in bars.
        Returns:
            pd.Series: ``volume / mean(volume)`` (*float32*).
        """
        return (volume / volume.rolling(window, min_periods=1).mean()).astype("float32")

    @staticmethod
    def compute_return(close: pd.Series) -> pd.Series:
        """Simple percentage returns of *close* prices.

        Args:
            close (pd.Series): Close‑price series.
        Returns:
            pd.Series: ``close.pct_change()`` (*float32*).
        """
        return close.pct_change(fill_method=None).astype("float32")

    @staticmethod
    def compute_rsi(close: pd.Series, window: int) -> pd.Series:
        """Relative Strength Index (RSI).

        Args:
            close (pd.Series): Close‑price series.
            window (int): Window length in bars.
        Returns:
            pd.Series: RSI values *float32* ranging 0–100.
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
        """Smoothed first‑order derivative of *close*.

        Args:
            close (pd.Series): Close‑price series.
            window (int, optional): Rolling window for mean smoothing. Defaults
                to ``5``.
        Returns:
            pd.Series: Smoothed derivative (*float32*).
        """
        return close.diff().rolling(window).mean().astype("float32")

    @staticmethod
    def compute_stoch_rsi(close: pd.Series, window: int) -> pd.Series:
        """Stochastic RSI (0–1‑scaled RSI).

        Args:
            close (pd.Series): Close‑price series.
            window (int): Window length for RSI and normalization.
        Returns:
            pd.Series: Stochastic RSI (*float32*).
        """
        rsi = TechnicalIndicators.compute_rsi(close, window)
        min_rsi = rsi.rolling(window, min_periods=1).min()
        max_rsi = rsi.rolling(window, min_periods=1).max()
        return ((rsi - min_rsi) / (max_rsi - min_rsi)).astype("float32")

    @staticmethod
    def compute_typical_price(
        high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """Typical price: mean of high, low, close.

        Args:
            high (pd.Series): High‑price series.
            low (pd.Series): Low‑price series.
            close (pd.Series): Close‑price series.
        Returns:
            pd.Series: Typical price (*float32*).
        """
        return ((high + low + close) / 3).astype("float32")

    @staticmethod
    def compute_volatility(
        high: pd.Series, low: pd.Series, open_: pd.Series
    ) -> pd.Series:
        """Simple intrabar volatility defined as ``(high - low) / open``.

        Args:
            high (pd.Series): High‑price series.
            low (pd.Series): Low‑price series.
            open_ (pd.Series): Open‑price series.
        Returns:
            pd.Series: Volatility ratio (*float32*). Invalid divisions return
                NaN.
        """
        return ((high - low) / open_.replace(0, pd.NA)).astype("float32")

    @staticmethod
    def compute_volume_change(volume: pd.Series) -> pd.Series:
        """Percentage change in volume.

        Args:
            volume (pd.Series): Raw volume series.
        Returns:
            pd.Series: ``volume.pct_change()`` (*float32*).
        """
        return volume.pct_change(fill_method=None).astype("float32")

    @staticmethod
    def compute_macd(
        series: pd.Series, fast: int, slow: int, signal: int
    ) -> pd.DataFrame:
        """Moving Average Convergence/Divergence (MACD).

        Args:
            series (pd.Series): Price series to analyse (e.g. close).
            fast (int): Fast EMA span.
            slow (int): Slow EMA span.
            signal (int): Signal EMA span applied to the MACD line.
        Returns:
            pd.DataFrame: Columns ``'macd'``, ``'signal'`` and ``'histogram'``
                (*float32*).
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return pd.DataFrame(
            {
                "macd": macd_line.astype("float32"),
                "signal": signal_line.astype("float32"),
                "histogram": hist.astype("float32"),
            }
        )

    @staticmethod
    def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On‑Balance Volume (OBV).

        Args:
            close (pd.Series): Close‑price series.
            volume (pd.Series): Raw volume series.
        Returns:
            pd.Series: OBV cumulative series (*float32*).
        """
        return (
            ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
            .on_balance_volume()
            .astype("float32")
        )

    @staticmethod
    def compute_candle_pattern(
        _open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        output_as_name: bool,
    ) -> Optional[Union[pd.Series, None]]:
        """Detect single‑candle patterns.

        Args:
            _open (pd.Series): Open‑price series.
            high (pd.Series): High‑price series.
            low (pd.Series): Low‑price series.
            close (pd.Series): Close‑price series.
            output_as_name (bool): When ``True`` the pattern name is returned
                (dtype ``category``); otherwise a numeric *score* (*float32*).
        Returns:
            Optional[pd.Series]: Series of pattern names/scores or ``None`` on
                unexpected failure.
        """
        try:
            ohlc = pd.DataFrame(
                {
                    "open": _open.astype("float32"),
                    "high": high.astype("float32"),
                    "low": low.astype("float32"),
                    "close": close.astype("float32"),
                }
            )
            if output_as_name:
                series = ohlc.apply(
                    lambda r: Candle(r.open, r.high, r.low, r.close).detect_pattern(),
                    axis=1,
                ).astype("category")
            else:
                series = ohlc.apply(
                    lambda r: Candle(r.open, r.high, r.low, r.close).score(), axis=1
                ).astype("float32")
            return series
        except Exception as exc:  # pylint: disable=broad-except
            Logger.warning(f"     [candle_pattern] failure: {exc}")
            if output_as_name:
                return pd.Series(pd.NA, index=_open.index, dtype="category")
            return pd.Series(np.nan, index=_open.index, dtype="float32")

    @staticmethod
    def compute_multi_candle_pattern(
        _open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        output_as_name: bool,
    ) -> Optional[pd.Series]:
        """Detect 3‑bar multi‑candle patterns.

        Args:
            _open (pd.Series): Open‑price series.
            high (pd.Series): High‑price series.
            low (pd.Series): Low‑price series.
            close (pd.Series): Close‑price series.
            output_as_name (bool): If ``True`` return pattern names; else return
                numeric pattern scores.
        Returns:
            Optional[pd.Series]: Category or float32 series. NaNs/<NA> fill the
                first two rows since three candles are required.
        """
        try:
            ohlc = pd.DataFrame(
                {"open": _open, "high": high, "low": low, "close": close}
            )
            for col in ohlc.columns:
                if not is_numeric_dtype(ohlc[col]):
                    ohlc[col] = pd.to_numeric(ohlc[col], errors="coerce")
                ohlc[col] = ohlc[col].astype("float32", copy=False)
            n_rows = len(ohlc)
            if n_rows < 3:
                raise ValueError("insufficient history (<3 bars)")
            results: List[Any] = []
            for i in range(n_rows):
                if i < 2:
                    results.append(pd.NA if output_as_name else np.nan)
                    continue
                window = ohlc.iloc[i - 2 : i + 1]
                candles = [
                    Candle(r.open, r.high, r.low, r.close)
                    for r in window.itertuples(index=False)
                ]
                if output_as_name:
                    pattern = MultiCandlePattern.detect_pattern(candles)
                    results.append(pattern if pattern is not None else pd.NA)
                else:
                    score = MultiCandlePattern.score(candles)
                    results.append(np.float32(score))
            dtype = "category" if output_as_name else "float32"
            return pd.Series(results, index=_open.index, dtype=dtype)
        except Exception as exc:  # pylint: disable=broad-except
            Logger.warning(f"     [multi_candle_pattern] failure: {exc}")
            if output_as_name:
                return pd.Series(pd.NA, index=_open.index, dtype="category")
            return pd.Series(np.nan, index=_open.index, dtype="float32")

    @staticmethod
    def compute_internal_multi_candle_pattern(
        _raw_df: pd.DataFrame,
        _enriched_df: pd.DataFrame,
        output_as_name: bool,
    ) -> Optional[pd.Series]:
        """Compute_internal_multi_candle_pattern."""
        try:
            if output_as_name:
                return pd.Series(pd.NA, dtype="category")
            return pd.Series(np.nan, dtype="float32")
        except Exception as exc:  # pylint: disable=broad-except
            Logger.warning(
                f"     [compute_internal_multi_candle_pattern] failure: {exc}"
            )
            if output_as_name:
                return pd.Series(pd.NA, dtype="category")
            return pd.Series(np.nan, dtype="float32")

    @staticmethod
    def compute_average_price(high: pd.Series, low: pd.Series) -> pd.Series:
        """Average of *high* and *low* prices.

        Args:
            high (pd.Series): High‑price series.
            low (pd.Series): Low‑price series.
        Returns:
            pd.Series: Average price (*float32*).
        """
        return ((high + low) / 2).astype("float32")

    @staticmethod
    def compute_bollinger_pct_b(
        close: pd.Series, window: int = 20, window_dev: float = 2
    ) -> pd.Series:
        """Bollinger Band %B indicator.

        Args:
            close (pd.Series): Close‑price series.
            window (int, optional): Rolling window length. Defaults to ``20``.
            window_dev (float, optional): Standard deviations. Defaults to ``2``.
        Returns:
            pd.Series: %B values (*float32*).
        """
        return (
            ta.volatility.BollingerBands(
                close=close, window=window, window_dev=window_dev
            )
            .bollinger_pband()
            .astype("float32")
        )

    @staticmethod
    def compute_bb_width(close: pd.Series, window: int) -> pd.Series:
        """Bollinger Band *width* as max–min spread.

        Args:
            close (pd.Series): Close‑price series.
            window (int): Rolling window length.
        Returns:
            pd.Series: Width values (*float32*).
        """
        return (
            close.rolling(window)
            .apply(lambda x: x.max() - x.min(), raw=True)
            .astype("float32")
        )

    @staticmethod
    def compute_overnight_return(open_: pd.Series) -> pd.Series:
        """Over‑night percentage return of *open* prices.

        Args:
            open_ (pd.Series): Open‑price series.
        Returns:
            pd.Series: Overnight returns (*float32*). The first value is 0.
        """
        return open_.pct_change(fill_method=None).fillna(0).astype("float32")

    @staticmethod
    def compute_williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int
    ) -> pd.Series:
        """Williams %R oscillator.

        Args:
            high (pd.Series): High‑price series.
            low (pd.Series): Low‑price series.
            close (pd.Series): Close‑price series.
            window (int): Look‑back window length.
        Returns:
            pd.Series: Williams %R (*float32*) ranging –100 to 0.
        """
        high_max = high.rolling(window=window).max()
        low_min = low.rolling(window=window).min()
        williams_r = -100 * ((high_max - close) / (high_max - low_min))
        return williams_r.astype("float32")

    @staticmethod
    def compute_temporal_event_feature(
        df: pd.DataFrame,
        event_dates: Set[datetime.date],
        is_raw: bool,
    ) -> Any:
        """
        Add decaying proximity features for temporal events (e.g., holidays, Fed meetings).

        The function appends three columns:
            * is        1 if the observation date matches an event date, 0 otherwise.
            * is_pre    linear decay weight if the date precedes an event within `window`.
            * is_post   linear decay weight if the date follows an event within `window`.

        Decay weights follow `(window - k + 1) / window`, where *k* is the day distance to
        the nearest event inside the window.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing a 'datetime' column (timezone-aware `Timestamp`).
        event_dates : set[datetime.date]
            Set of all relevant event dates.

        Returns
        -------
        pd.DataFrame
            Same DataFrame with the new proximity columns added.
        """
        window: int = 5

        # Work with pure `date` objects for hashability and fast set membership checks
        dates = df["datetime"].dt.date

        # Exact match: 1 if the date is an event, else 0
        decay = dates.isin(event_dates).astype("float32")

        # Pre- and post-event decay initialised to zero
        pre_decay = pd.Series(0.0, index=df.index, dtype="float32")
        post_decay = pd.Series(0.0, index=df.index, dtype="float32")

        # Iterate over symmetric window; vectorised `.isin` keeps this fast
        for offset in range(1, window + 1):
            weight: float = (window - offset + 1) / window

            # Date `offset` days ahead belongs to an event → pre-event decay today
            pre_mask = (dates + datetime.timedelta(offset)).isin(event_dates)
            pre_decay = pre_decay.where(~pre_mask, weight)

            # Date `offset` days behind belongs to an event → post-event decay today
            post_mask = (dates - datetime.timedelta(offset)).isin(event_dates)
            post_decay = post_decay.where(~post_mask, weight)

        return {
            "is_pre": (
                pre_decay.astype("float32") if is_raw is False else pre_decay * 100
            ),
            "is": decay.astype("float32") if is_raw is False else decay == 1,
            "is_post": (
                post_decay.astype("float32") if is_raw is False else post_decay * 100
            ),
        }

    @staticmethod
    def compute_weekday(datetimes: pd.Series, is_raw: bool) -> pd.Series:
        try:
            dt = pd.to_datetime(datetimes)
            is_weekday = dt.dt.weekday < 5
            if is_raw:
                return is_weekday.astype("boolean")  # type: ignore
            return is_weekday.astype("float32")  # type: ignore
        except Exception as exc:  # pylint: disable=broad-except
            Logger.warning(f"     [compute_weekday] failure: {exc}")
            if is_raw:
                return pd.Series(pd.NA, dtype="category")
            return pd.Series(np.nan, dtype="float32")

    @staticmethod
    def compute_weekend(datetimes: pd.Series, is_raw: bool) -> pd.Series:
        try:
            dt = pd.to_datetime(datetimes)
            is_weekday = dt.dt.weekday > 4
            if is_raw:
                return is_weekday.astype("boolean")  # type: ignore
            return is_weekday.astype("float32")  # type: ignore
        except Exception as exc:  # pylint: disable=broad-except
            Logger.warning(f"     [compute_weekend] failure: {exc}")
            if is_raw:
                return pd.Series(pd.NA, dtype="category")
            return pd.Series(np.nan, dtype="float32")

    @staticmethod
    def compute_workday(
        in_is_weekend: pd.Series, in_is_holiday: pd.Series, is_raw: bool
    ) -> pd.Series:
        try:
            if is_raw:
                is_weekend = in_is_weekend.astype("boolean")
                is_holiday = in_is_holiday.astype("boolean")
            else:
                is_weekend = in_is_weekend.astype("float32") == 1
                is_holiday = in_is_holiday.astype("float32") == 1
            is_workday = (~is_weekend) & (~is_holiday)
            return (
                is_workday.astype("boolean") if is_raw else is_workday.astype("float32")
            )
        except Exception as exc:  # pylint: disable=broad-except
            Logger.warning(f"     [compute_workday] failure: {exc}")
            if is_raw:
                return pd.Series(pd.NA, dtype="boolean")
            return pd.Series(np.nan, dtype="float32")

    @staticmethod
    def compute_time_fractions(df: pd.DataFrame, is_raw: bool) -> Any:
        """
        Computes normalized time fractions and appends them to the input DataFrame.

        The generated columns are:
        - time_of_day: Fraction of the day (0.0 = 00:00, 1.0 = 00:00 next day)
        - time_of_week: Fraction of the week (0.0 = Monday 00:00, 1.0 = next Monday 00:00)
        - time_of_month: Fraction of the month (0.0 = first day 00:00, 1.0 = first day next month)
        - time_of_year: Fraction of the year (0.0 = Jan 1st, 1.0 = Jan 1st next year)

        Args:
            df (pd.DataFrame): DataFrame with a ``'datetime'`` column (tz aware).
        Returns:
            pd.DataFrame: Same object with four additional ``scaled_time_*`` columns.
        """
        datetimes = df["datetime"]
        dt = datetimes.dt

        # Day fraction
        day_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        time_of_day = day_seconds / 86400.0

        # Week fraction
        weekday_seconds = dt.weekday * 86400 + day_seconds
        time_of_week = weekday_seconds / (7 * 86400)

        # Month fraction
        month_start = datetimes.dt.tz_localize(None).dt.to_period("M").dt.start_time
        next_month_start = month_start + pd.offsets.MonthBegin(1)
        month_start = month_start.dt.tz_localize(datetimes.dt.tz)
        next_month_start = next_month_start.dt.tz_localize(datetimes.dt.tz)
        month_elapsed = (datetimes - month_start).dt.total_seconds()
        month_total = next_month_start - month_start
        month_total_seconds = month_total.dt.total_seconds()
        time_of_month = month_elapsed / month_total_seconds

        # Year fraction
        year_start = datetimes.dt.tz_localize(None).dt.to_period("Y").dt.start_time
        next_year_start = year_start + pd.offsets.YearBegin(1)
        year_start = year_start.dt.tz_localize(datetimes.dt.tz)
        next_year_start = next_year_start.dt.tz_localize(datetimes.dt.tz)
        year_elapsed = (datetimes - year_start).dt.total_seconds()
        year_total = next_year_start - year_start
        year_total_seconds = year_total.dt.total_seconds()
        time_of_year = year_elapsed / year_total_seconds

        return {
            "time_of_day": (time_of_day * (1 if not is_raw else 24)).astype(np.float32),
            "time_of_week": (
                (time_of_week * (1 if not is_raw else 7)).astype(np.float32)
            )
            + (1 if is_raw else 0),
            "time_of_month": (
                (
                    time_of_month
                    * (1 if not is_raw else month_total.dt.days.astype(float))
                ).astype(np.float32)
            )
            + (1 if is_raw else 0),
            "time_of_year": (
                (
                    time_of_year
                    * (1 if not is_raw else year_total.dt.days.astype(float))
                ).astype(np.float32)
            )
            + (1 if is_raw else 0),
        }

    @staticmethod
    def get_schedule_by_date(
        df: pd.DataFrame,
        market_time: Any,
        interval: str,
        *,
        column_name: str = "schedule",
        inplace: bool = False,
    ) -> pd.DataFrame:
        try:
            tgt = df if inplace else df.copy()
            suffix = IntervalValidator.PATTERN.fullmatch(interval.strip()).group(1)
            intraday = suffix in ("min", "m", "hour", "h")
            floor_times = (not intraday) or suffix in ("hour", "h")

            def _floor(time_str: str) -> str:
                """Return the same time rounded down to the previous hour."""
                if not isinstance(time_str, str) or ":" not in time_str:
                    return time_str
                hour = int(time_str.split(":")[0])
                return f"{hour:02d}:00"

            sched_df = pd.DataFrame(market_time).assign(
                date_from=lambda x: pd.to_datetime(x["date_from"]).dt.date,
                date_to=lambda x: pd.to_datetime(x["date_to"]).dt.date,
                time_from=lambda x: x["time_from"].apply(
                    _floor if floor_times else (lambda s: s)
                ),
                time_to=lambda x: x["time_to"].apply(
                    _floor if floor_times else (lambda s: s)
                ),
            )
            dates = tgt["datetime"].dt.date
            sched_col = pd.Series(pd.NA, index=tgt.index, dtype="object")
            for _, row in sched_df.iterrows():
                mask = (dates >= row.date_from) & (dates <= row.date_to)
                if mask.any():
                    sched_dict = {"time_from": row.time_from, "time_to": row.time_to}
                    sched_col.loc[mask] = [sched_dict] * int(mask.sum())
            tgt[column_name] = sched_col
            return tgt
        except Exception as exc:  # pylint: disable=broad-except
            Logger.warning(f"     [get_schedule_by_date] failure: {exc}")
            return df if inplace else df.copy()

    @staticmethod
    def _is_intraday_interval(interval: str) -> bool:
        match = IntervalValidator.PATTERN.fullmatch(interval.strip())
        if not match:
            raise ValueError(f"Invalid interval: {interval}")
        return match.group(1) in ("min", "m", "hour", "h")

    @staticmethod
    def compute_market_time(
        df: pd.DataFrame,
        market_time: Any,
        interval: str,
        is_raw: bool,
        is_workday: pd.Series[bool],
    ) -> Any:
        try:
            df_sched = TechnicalIndicators.get_schedule_by_date(
                df, market_time, interval
            )
            idx = df_sched.index
            intraday = TechnicalIndicators._is_intraday_interval(interval)
            if not intraday:
                is_market_time = pd.Series(True, index=idx)
                is_pre_market_time = pd.Series(False, index=idx)
                is_post_market_time = pd.Series(False, index=idx)
            else:
                sched = df_sched["schedule"]

                def _parse_time(s: pd.Series, key: str) -> pd.Series:
                    return s.apply(
                        lambda v: (
                            datetime.datetime.strptime(v[key], "%H:%M").time()
                            if isinstance(v, dict)
                            else None
                        )
                    )

                t_from = _parse_time(sched, "time_from")
                t_to = _parse_time(sched, "time_to")
                current_t = df_sched["datetime"].dt.time
                is_market_time = (current_t >= t_from) & (current_t <= t_to)
                is_pre_market_time = (current_t < t_from) & t_from.notna()
                is_post_market_time = (current_t > t_to) & t_to.notna()
            cast = (
                (lambda s: s.astype("boolean"))
                if is_raw
                else (lambda s: s.astype("float32"))
            )
            is_market_day = is_pre_market_time | is_market_time | is_post_market_time
            is_market_day = is_market_day & is_workday
            return {
                "is_market_day": cast(is_market_day),
                "is_pre_market_time": cast(is_pre_market_time),
                "is_market_time": cast(is_market_time),
                "is_post_market_time": cast(is_post_market_time),
            }
        except Exception as exc:  # pylint: disable=broad-except
            Logger.warning(f"     [compute_market_time] failure: {exc}")
            nan_bool = pd.Series(pd.NA, index=df.index, dtype="boolean")
            nan_float = pd.Series(np.nan, index=df.index, dtype="float32")
            return (
                {
                    "is_market_day": nan_bool,
                    "is_pre_market_time": nan_bool,
                    "is_market_time": nan_bool,
                    "is_post_market_time": nan_bool,
                }
                if is_raw
                else {
                    "is_market_day": nan_float,
                    "is_pre_market_time": nan_float,
                    "is_market_time": nan_float,
                    "is_post_market_time": nan_float,
                }
            )
