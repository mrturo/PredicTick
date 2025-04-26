"""Provides technical indicator computation utilities used in prediction and backtesting."""

import numpy as np
import pandas as pd

from utils.parameters import ParameterLoader


class IndicatorCalculator:
    """Computes technical indicators for backtesting."""

    # Load parameters
    _PARAMS = ParameterLoader()
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

    @staticmethod
    def compute_rsi(series: pd.Series, window: int) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI) of a given time series.

        Args:
            series (pd.Series): Time series data.
            window (int): Lookback period for RSI.

        Returns:
            pd.Series: RSI values.
        """
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return pd.Series(rsi, index=series.index)

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
    def compute_stoch_rsi(series: pd.Series, window: int) -> pd.Series:
        """
        Calculate the Stochastic RSI from a given time series.

        Args:
            series (pd.Series): RSI time series.
            window (int): Lookback window for min/max normalization.

        Returns:
            pd.Series: Stochastic RSI values scaled to [0, 100].
        """
        min_val = series.rolling(
            window=window, min_periods=IndicatorCalculator._STOCH_RSI_MIN_PERIODS
        ).min()
        max_val = series.rolling(
            window=window, min_periods=IndicatorCalculator._STOCH_RSI_MIN_PERIODS
        ).max()
        return ((series - min_val) / (max_val - min_val) * 100).astype("float32")

    @staticmethod
    def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a suite of technical indicators to a historical OHLCV DataFrame.

        Indicators computed:
        - return_1h
        - volatility_3h
        - momentum_3h
        - RSI
        - MACD histogram
        - Bollinger Bands width
        - Stochastic RSI
        - OBV (On-Balance Volume)
        - ATR (Average True Range)
        - Williams %R

        Args:
            df (pd.DataFrame): OHLCV data with columns: ['open', 'high', 'low', 'close', 'volume'].

        Returns:
            pd.DataFrame: Original DataFrame with additional indicator columns.
        """
        close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

        df["return_1h"] = close.pct_change().astype("float32")
        df["volatility_3h"] = df["return_1h"].rolling(window=3).std().astype("float32")
        df["momentum_3h"] = (close - close.shift(3)).astype("float32")
        df["rsi"] = IndicatorCalculator.compute_rsi(
            close, IndicatorCalculator._RSI_WINDOW
        )
        df["macd"] = IndicatorCalculator.compute_macd(
            close,
            IndicatorCalculator._MACD_FAST,
            IndicatorCalculator._MACD_SLOW,
            IndicatorCalculator._MACD_SIGNAL,
        )["histogram"]
        df["volume"] = volume.astype("float32")

        if IndicatorCalculator._BOLLINGER_BAND_METHOD == "max-min":
            df["bb_width"] = (
                close.rolling(IndicatorCalculator._BOLLINGER_WINDOW)
                .apply(lambda x: x.max() - x.min(), raw=True)
                .astype("float32")
            )

        df["stoch_rsi"] = IndicatorCalculator.compute_stoch_rsi(
            close, IndicatorCalculator._STOCH_RSI_WINDOW
        )
        df["obv"] = (
            (np.sign(close.diff()) * volume)
            .fillna(IndicatorCalculator._OBV_FILL_METHOD)
            .cumsum()
            .astype("float32")
        )

        tr_components = pd.concat(
            [
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        )

        df["atr"] = (
            tr_components.max(axis=1)
            .rolling(window=IndicatorCalculator._ATR_WINDOW)
            .mean()
            .astype("float32")
        )

        high_max = high.rolling(IndicatorCalculator._WILLIAMS_R_WINDOW).max()
        low_min = low.rolling(IndicatorCalculator._WILLIAMS_R_WINDOW).min()
        df["williams_r"] = (-100 * ((high_max - close) / (high_max - low_min))).astype(
            "float32"
        )

        return df

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
            "rsi_window": IndicatorCalculator._RSI_WINDOW,
            "macd_fast": IndicatorCalculator._MACD_FAST,
            "macd_slow": IndicatorCalculator._MACD_SLOW,
            "macd_signal": IndicatorCalculator._MACD_SIGNAL,
            "bollinger_window": IndicatorCalculator._BOLLINGER_WINDOW,
            "bollinger_band_method": IndicatorCalculator._BOLLINGER_BAND_METHOD,
            "stoch_rsi_window": IndicatorCalculator._STOCH_RSI_WINDOW,
            "stoch_rsi_min_periods": IndicatorCalculator._STOCH_RSI_MIN_PERIODS,
            "obv_fill_method": IndicatorCalculator._OBV_FILL_METHOD,
            "atr_window": IndicatorCalculator._ATR_WINDOW,
            "williams_r_window": IndicatorCalculator._WILLIAMS_R_WINDOW,
        }
