"""
Module for building and enriching a market DataFrame with technical and contextual indicators.

This module centralizes the computation of indicators related to price, volume,
economic events, temporal features, and market session context.
All configuration parameters are loaded from the centralized `ParameterLoader`.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd  # type: ignore

from src.market_data.processing.enrichment.market_context import MarketContext
from src.market_data.processing.indicators.patterns import (
    compute_candle_pattern, compute_multi_candle_pattern)
from src.market_data.processing.indicators.price import (
    compute_average_price, compute_bb_width, compute_intraday_return,
    compute_overnight_return, compute_price_change, compute_price_derivative,
    compute_range, compute_return, compute_smoothed_derivative,
    compute_typical_price, compute_volatility)
from src.market_data.processing.indicators.schedule import compute_market_time
from src.market_data.processing.indicators.temporal import (
    compute_temporal_event_feature, compute_time_fractions, compute_weekday,
    compute_weekend, compute_workday)
from src.market_data.processing.indicators.trend import (
    compute_adx_14d, compute_atr, compute_atr_14d, compute_bollinger_pct_b,
    compute_macd, compute_open_close_result, compute_rsi, compute_stoch_rsi,
    compute_williams_r)
from src.market_data.processing.indicators.volume import (
    compute_obv, compute_relative_volume, compute_volume_change,
    compute_volume_rvol_20d)
from src.market_data.utils.intervals.interval import Interval
from src.utils.config.parameters import ParameterLoader


class IndicatorBuilder:  # pylint: disable=too-few-public-methods
    """Constructs and appends technical, volume-based, temporal, and contextual indicators.

    to a previously enriched market DataFrame.

    This class applies a comprehensive set of transformations on a market DataFrame
    to support downstream tasks such as modeling or visualization.
    All methods are static and require no instantiation.
    """

    _PARAMS = ParameterLoader()
    _WILLIAMS_R_WINDOW = _PARAMS.get("williams_r_window")
    _VOLUME_WINDOW = _PARAMS.get("volume_window")
    _STOCH_RSI_WINDOW = _PARAMS.get("stoch_rsi_window")
    _RSI_WINDOW = _PARAMS.get("rsi_window_backtest")
    _REQUIRED_MARKET_ENRICHED_COLUMNS: list[str] = list(
        dict.fromkeys(
            (_PARAMS.get("required_market_raw_columns") or [])
            + (_PARAMS.get("required_market_enriched_columns") or [])
        )
    )
    _MACD_SLOW = _PARAMS.get("macd_slow")
    _MACD_SIGNAL = _PARAMS.get("macd_signal")
    _MACD_FAST = _PARAMS.get("macd_fast")
    _BOLLINGER_WINDOW = _PARAMS.get("bollinger_window")
    _BOLLINGER_BAND_METHOD = _PARAMS.get("bollinger_band_method")
    _ATR_WINDOW = _PARAMS.get("atr_window")
    _ENRICHED_DATA_INTERVAL = Interval.market_enriched_data()

    @staticmethod
    def add_indicators(
        enriched_df: pd.DataFrame,
        market_context: MarketContext,
        prefix: str = "",
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Appends all configured indicators to the enriched DataFrame.

        Sequentially applies technical, volume, temporal, economic event,
        and market-time context indicators.
        """
        prefix = prefix.strip()
        is_raw: bool = len(prefix) == 0

        def prefixed(col: str) -> str:
            return f"{prefix}{col}" if prefix else col

        enriched_df = IndicatorBuilder._add_technical_indicators(
            enriched_df, prefixed, is_raw
        )
        enriched_df = IndicatorBuilder._add_volume_indicators(enriched_df, prefixed)
        enriched_df = IndicatorBuilder._add_temporal_indicators(
            enriched_df, prefixed, is_raw
        )
        enriched_df = IndicatorBuilder._add_event_indicators(
            enriched_df, market_context, prefixed, is_raw
        )
        enriched_df, market_time = IndicatorBuilder._add_market_time_indicators(
            enriched_df, market_context, prefixed, is_raw
        )
        return enriched_df, market_time

    @staticmethod
    def _add_technical_indicators(
        enriched_df: pd.DataFrame, prefixed, is_raw: bool
    ) -> pd.DataFrame:
        """Adds standard technical indicators based on price data.

        Includes indicators such as RSI, MACD, ADX, ATR, returns, and derived prices.
        """
        enriched_df[prefixed("adx_14d")] = compute_adx_14d(
            enriched_df["datetime"],
            enriched_df[prefixed("high")],
            enriched_df[prefixed("low")],
            enriched_df[prefixed("close")],
        )
        enriched_df[prefixed("atr")] = compute_atr(
            enriched_df[prefixed("high")],
            enriched_df[prefixed("low")],
            enriched_df[prefixed("close")],
            IndicatorBuilder._ATR_WINDOW,
        )
        enriched_df[prefixed("atr_14d")] = compute_atr_14d(
            enriched_df["datetime"],
            enriched_df[prefixed("high")],
            enriched_df[prefixed("low")],
            enriched_df[prefixed("close")],
        )
        enriched_df[prefixed("average_price")] = compute_average_price(
            enriched_df[prefixed("high")], enriched_df[prefixed("low")]
        )
        enriched_df[prefixed("bollinger_pct_b")] = compute_bollinger_pct_b(
            enriched_df[prefixed("close")]
        )
        if IndicatorBuilder._BOLLINGER_BAND_METHOD == "max-min":
            enriched_df[prefixed("bb_width")] = compute_bb_width(
                enriched_df[prefixed("close")], IndicatorBuilder._BOLLINGER_WINDOW
            )
        enriched_df[prefixed("intraday_return")] = compute_intraday_return(
            enriched_df[prefixed("close")], enriched_df[prefixed("open")]
        )
        enriched_df[prefixed("macd")] = compute_macd(
            enriched_df[prefixed("close")],
            IndicatorBuilder._MACD_FAST,
            IndicatorBuilder._MACD_SLOW,
            IndicatorBuilder._MACD_SIGNAL,
        )["histogram"]
        enriched_df[prefixed("overnight_return")] = compute_overnight_return(
            enriched_df[prefixed("open")]
        )
        enriched_df[prefixed("price_change")] = compute_price_change(
            enriched_df[prefixed("close")], enriched_df[prefixed("open")]
        )
        enriched_df[prefixed("price_derivative")] = compute_price_derivative(
            enriched_df[prefixed("close")]
        )
        enriched_df[prefixed("range")] = compute_range(
            enriched_df[prefixed("high")], enriched_df[prefixed("low")]
        )
        enriched_df[prefixed("return")] = compute_return(enriched_df[prefixed("close")])
        enriched_df[prefixed("rsi")] = compute_rsi(
            enriched_df[prefixed("close")], IndicatorBuilder._RSI_WINDOW
        )
        enriched_df[prefixed("smoothed_derivative")] = compute_smoothed_derivative(
            enriched_df[prefixed("close")]
        )
        enriched_df[prefixed("stoch_rsi")] = compute_stoch_rsi(
            enriched_df[prefixed("close")], IndicatorBuilder._STOCH_RSI_WINDOW
        )
        enriched_df[prefixed("typical_price")] = compute_typical_price(
            enriched_df[prefixed("high")],
            enriched_df[prefixed("low")],
            enriched_df[prefixed("close")],
        )
        enriched_df[prefixed("volatility")] = compute_volatility(
            enriched_df[prefixed("high")],
            enriched_df[prefixed("low")],
            enriched_df[prefixed("open")],
        )
        enriched_df[prefixed("williams_r")] = compute_williams_r(
            enriched_df[prefixed("high")],
            enriched_df[prefixed("low")],
            enriched_df[prefixed("close")],
            IndicatorBuilder._WILLIAMS_R_WINDOW,
        )
        enriched_df[prefixed("open_close_result")] = compute_open_close_result(
            enriched_df[prefixed("open")], enriched_df[prefixed("close")], is_raw
        )
        return enriched_df

    @staticmethod
    def _add_volume_indicators(enriched_df: pd.DataFrame, prefixed) -> pd.DataFrame:
        """Adds volume-based indicators to the DataFrame."""
        if "volume" in enriched_df.columns:
            enriched_df[prefixed("obv")] = compute_obv(
                enriched_df[prefixed("close")], enriched_df[prefixed("volume")]
            )
            enriched_df[prefixed("relative_volume")] = compute_relative_volume(
                enriched_df[prefixed("volume")], IndicatorBuilder._VOLUME_WINDOW
            )
            enriched_df[prefixed("volume_change")] = compute_volume_change(
                enriched_df[prefixed("volume")]
            )
            enriched_df[prefixed("volume_rvol_20d")] = compute_volume_rvol_20d(
                enriched_df["datetime"], enriched_df[prefixed("volume")]
            )
        return enriched_df

    @staticmethod
    def _add_temporal_indicators(
        enriched_df: pd.DataFrame, prefixed, is_raw: bool
    ) -> pd.DataFrame:
        """Adds candlestick and multi-candle pattern features."""
        enriched_df[prefixed("candle_pattern")] = compute_candle_pattern(
            enriched_df[prefixed("open")],
            enriched_df[prefixed("high")],
            enriched_df[prefixed("low")],
            enriched_df[prefixed("close")],
            is_raw,
        )
        enriched_df[prefixed("multi_candle_pattern")] = compute_multi_candle_pattern(
            enriched_df[prefixed("open")],
            enriched_df[prefixed("high")],
            enriched_df[prefixed("low")],
            enriched_df[prefixed("close")],
            is_raw,
        )
        return enriched_df

    @staticmethod
    def _add_event_indicators(
        enriched_df: pd.DataFrame,
        market_context: MarketContext,
        prefixed,
        is_raw: bool,
    ) -> pd.DataFrame:
        """Adds features related to economic calendar events."""
        enriched_df[prefixed("is_market_day")] = False
        enriched_df[prefixed("is_pre_market_time")] = False
        enriched_df[prefixed("is_market_time")] = False
        enriched_df[prefixed("is_post_market_time")] = False
        features_fed_event = compute_temporal_event_feature(
            df=enriched_df,
            event_dates=set(market_context.fed_events),
            is_raw=is_raw,
        )
        enriched_df[prefixed("is_pre_fed_event")] = features_fed_event["is_pre"]
        enriched_df[prefixed("is_fed_event")] = features_fed_event["is"]
        enriched_df[prefixed("is_post_fed_event")] = features_fed_event["is_post"]
        features_holiday = compute_temporal_event_feature(
            df=enriched_df,
            event_dates=set(market_context.us_holidays),
            is_raw=is_raw,
        )
        enriched_df[prefixed("is_pre_holiday")] = features_holiday["is_pre"]
        enriched_df[prefixed("is_holiday")] = features_holiday["is"]
        enriched_df[prefixed("is_post_holiday")] = features_holiday["is_post"]
        enriched_df[prefixed("is_weekday")] = compute_weekday(
            enriched_df["datetime"], is_raw
        )
        enriched_df[prefixed("is_weekend")] = compute_weekend(
            enriched_df["datetime"], is_raw
        )
        enriched_df[prefixed("is_workday")] = compute_workday(
            enriched_df[prefixed("is_weekend")],
            enriched_df[prefixed("is_holiday")],
            is_raw,
        )
        features_time_fractions = compute_time_fractions(df=enriched_df, is_raw=is_raw)
        enriched_df[prefixed("time_of_day")] = features_time_fractions["time_of_day"]
        enriched_df[prefixed("time_of_week")] = features_time_fractions["time_of_week"]
        enriched_df[prefixed("time_of_month")] = features_time_fractions[
            "time_of_month"
        ]
        enriched_df[prefixed("time_of_year")] = features_time_fractions["time_of_year"]
        return enriched_df

    @staticmethod
    def _add_market_time_indicators(
        enriched_df: pd.DataFrame,
        market_context: MarketContext,
        prefixed,
        is_raw: bool,
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Adds columns indicating the market time context of each row."""
        features_market_time, market_time = compute_market_time(
            df=enriched_df,
            market_time=market_context.market_time,
            interval=IndicatorBuilder._ENRICHED_DATA_INTERVAL or "",
            is_raw=is_raw,
            is_workday=enriched_df[prefixed("is_workday")],
        )
        enriched_df[prefixed("is_market_day")] = features_market_time["is_market_day"]
        enriched_df[prefixed("is_pre_market_time")] = features_market_time[
            "is_pre_market_time"
        ]
        enriched_df[prefixed("is_market_time")] = features_market_time["is_market_time"]
        enriched_df[prefixed("is_post_market_time")] = features_market_time[
            "is_post_market_time"
        ]
        return enriched_df, market_time
