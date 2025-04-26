"""Provides technical indicator computation utilities used in prediction and backtesting."""

import numpy as np
import pandas as pd

from utils.parameters import ParameterLoader

# Load parameters
parameters = ParameterLoader()

rsi_window = parameters["rsi_window_backtest"]
macd_fast = parameters["macd_fast"]
macd_slow = parameters["macd_slow"]
macd_signal = parameters["macd_signal"]
bollinger_window = parameters["bollinger_window"]
bollinger_band_method = parameters["bollinger_band_method"]
stoch_rsi_window = parameters["stoch_rsi_window"]
stoch_rsi_min_periods = parameters["stoch_rsi_min_periods"]
obv_fill_method = parameters["obv_fill_method"]
atr_window = parameters["atr_window"]
williams_r_window = parameters["williams_r_window"]


class IndicatorCalculator:
    """Computes technical indicators for backtesting."""

    @staticmethod
    def compute_rsi(series: pd.Series, window: int) -> pd.Series:
        """Calculate the Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_macd_diff(
        series: pd.Series, fast: int, slow: int, signal: int
    ) -> pd.Series:
        """Calculate MACD line minus signal line."""
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    @staticmethod
    def compute_stoch_rsi(series: pd.Series, window: int) -> pd.Series:
        """Calculate stochastic RSI."""
        min_val = series.rolling(window=window, min_periods=stoch_rsi_min_periods).min()
        max_val = series.rolling(window=window, min_periods=stoch_rsi_min_periods).max()
        return 100 * (series - min_val) / (max_val - min_val)

    @staticmethod
    def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Apply technical indicator calculations to OHLCV dataframe."""
        close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

        df["return_1h"] = close.pct_change()
        df["volatility_3h"] = df["return_1h"].rolling(window=3).std()
        df["momentum_3h"] = close - close.shift(3)
        df["rsi"] = IndicatorCalculator.compute_rsi(close, rsi_window)
        df["macd"] = IndicatorCalculator.compute_macd_diff(
            close, macd_fast, macd_slow, macd_signal
        )
        df["volume"] = volume

        if bollinger_band_method == "max-min":
            df["bb_width"] = close.rolling(bollinger_window).apply(
                lambda x: x.max() - x.min(), raw=True
            )

        df["stoch_rsi"] = IndicatorCalculator.compute_stoch_rsi(close, stoch_rsi_window)
        df["obv"] = (np.sign(close.diff()) * volume).fillna(obv_fill_method).cumsum()

        df["atr"] = (
            pd.concat(
                [
                    (high - low),
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs(),
                ],
                axis=1,
            )
            .max(axis=1)
            .rolling(window=atr_window)
            .mean()
        )

        df["williams_r"] = -100 * (
            (high.rolling(williams_r_window).max() - close)
            / (
                high.rolling(williams_r_window).max()
                - low.rolling(williams_r_window).min()
            )
        )

        return df

    @staticmethod
    def get_indicator_parameters() -> dict:
        """Returns a dictionary with all technical indicator parameters."""
        return {
            "rsi_window": parameters["rsi_window_backtest"],
            "macd_fast": parameters["macd_fast"],
            "macd_slow": parameters["macd_slow"],
            "macd_signal": parameters["macd_signal"],
            "bollinger_window": parameters["bollinger_window"],
            "bollinger_band_method": parameters["bollinger_band_method"],
            "stoch_rsi_window": parameters["stoch_rsi_window"],
            "stoch_rsi_min_periods": parameters["stoch_rsi_min_periods"],
            "obv_fill_method": parameters["obv_fill_method"],
            "atr_window": parameters["atr_window"],
            "williams_r_window": parameters["williams_r_window"],
        }
