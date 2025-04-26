"""This module defines the `MultiCandlePattern` class, which identifies and scores.

multi-candle reversal and continuation patterns in financial market data. It leverages
single-candle classifications (from the `Candle` class) to detect complex structures
such as engulfings, stars, piercing lines, tweezers, and three-soldier/crow patterns.

Patterns are recognized using explicit logical rules and are scored according to
a configurable scale provided by `ParameterLoader`.
"""

from typing import List, Optional

from src.market_data.processing.candles.candle import Candle
from src.utils.config.parameters import ParameterLoader


class MultiCandlePattern:
    """Detects multi-candle reversal patterns using detailed single-candle classifications.

    Supports common formations like engulfings, stars, piercing lines, tweezers,
    and soldier/crow patterns. Provides scoring based on pattern strength or fallback
    to individual candle scores.
    """

    _PARAMS = ParameterLoader()
    _SCORE = _PARAMS.get("candle_multiple_score")

    @staticmethod
    def _is_bullish_engulfing(c1: Candle, c2: Candle) -> bool:
        """Determines if a bullish engulfing pattern is formed by two candles."""
        return (
            c1.detect_pattern()
            in {"bearish", "marubozu_bearish", "spinning_top_bearish"}
            and c2.detect_pattern() in {"bullish", "marubozu_bullish"}
            and c2.open < c1.close
            and c2.close > c1.open
        )

    @staticmethod
    def _is_bearish_engulfing(c1: Candle, c2: Candle) -> bool:
        """Determines if a bearish engulfing pattern is formed by two candles."""
        return (
            c1.detect_pattern()
            in {"bullish", "marubozu_bullish", "spinning_top_bullish"}
            and c2.detect_pattern() in {"bearish", "marubozu_bearish"}
            and c2.open > c1.close
            and c2.close < c1.open
        )

    @staticmethod
    def _is_morning_star(c1: Candle, c2: Candle, c3: Candle) -> bool:
        """Detects a morning star pattern across three candles."""
        return (
            c1.detect_pattern() in {"bearish", "marubozu_bearish"}
            and c2.is_indecisive()
            and c3.detect_pattern() in {"bullish", "marubozu_bullish"}
            and c3.close > (c1.open + c1.close) / 2
        )

    @staticmethod
    def _is_evening_star(c1: Candle, c2: Candle, c3: Candle) -> bool:
        """Detects an evening star pattern across three candles."""
        return (
            c1.detect_pattern() in {"bullish", "marubozu_bullish"}
            and c2.is_indecisive()
            and c3.detect_pattern() in {"bearish", "marubozu_bearish"}
            and c3.close < (c1.open + c1.close) / 2
        )

    @staticmethod
    def _is_piercing_line(c1: Candle, c2: Candle) -> bool:
        """Identifies a bullish piercing line pattern."""
        return (
            c1.is_bearish()
            and c2.is_bullish()
            and c2.open < c1.low
            and c2.close > (c1.open + c1.close) / 2
            and c2.close < c1.open
        )

    @staticmethod
    def _is_dark_cloud_cover(c1: Candle, c2: Candle) -> bool:
        """Identifies a bearish dark cloud cover pattern."""
        return (
            c1.is_bullish()
            and c2.is_bearish()
            and c2.open > c1.high
            and c2.close < (c1.open + c1.close) / 2
            and c2.close > c1.open
        )

    @staticmethod
    def _is_tweezer_bottom(c1: Candle, c2: Candle) -> bool:
        """Identifies a tweezer bottom reversal pattern."""
        return c1.is_bearish() and c2.is_bullish() and abs(c1.low - c2.low) < 1e-3

    @staticmethod
    def _is_tweezer_top(c1: Candle, c2: Candle) -> bool:
        return c1.is_bullish() and c2.is_bearish() and abs(c1.high - c2.high) < 1e-3

    @staticmethod
    def _is_three_white_soldiers(c1: Candle, c2: Candle, c3: Candle) -> bool:
        """Identifies a bullish continuation pattern: three white soldiers."""
        return (
            all(c.is_bullish() for c in [c1, c2, c3])
            and c2.open > c1.open
            and c2.close > c1.close
            and c3.open > c2.open
            and c3.close > c2.close
            and all(c.upper_shadow() < c.body() * 0.5 for c in [c1, c2, c3])
        )

    @staticmethod
    def _is_three_black_crows(c1: Candle, c2: Candle, c3: Candle) -> bool:
        """Identifies a bearish continuation pattern: three black crows."""
        return (
            all(c.is_bearish() for c in [c1, c2, c3])
            and c2.open < c1.open
            and c2.close < c1.close
            and c3.open < c2.open
            and c3.close < c2.close
            and all(c.lower_shadow() < c.body() * 0.5 for c in [c1, c2, c3])
        )

    @staticmethod
    def detect_pattern(candles: List[Candle]) -> Optional[str]:
        """Identifies the most recent multi-candle pattern if present."""
        if len(candles) < 3:
            return None
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        conditions = [
            (MultiCandlePattern._is_bullish_engulfing(c2, c3), "bullish_engulfing"),
            (MultiCandlePattern._is_bearish_engulfing(c2, c3), "bearish_engulfing"),
            (MultiCandlePattern._is_morning_star(c1, c2, c3), "morning_star"),
            (MultiCandlePattern._is_evening_star(c1, c2, c3), "evening_star"),
            (MultiCandlePattern._is_piercing_line(c2, c3), "piercing_line"),
            (MultiCandlePattern._is_dark_cloud_cover(c2, c3), "dark_cloud_cover"),
            (MultiCandlePattern._is_tweezer_bottom(c2, c3), "tweezer_bottom"),
            (MultiCandlePattern._is_tweezer_top(c2, c3), "tweezer_top"),
            (
                MultiCandlePattern._is_three_white_soldiers(c1, c2, c3),
                "three_white_soldiers",
            ),
            (MultiCandlePattern._is_three_black_crows(c1, c2, c3), "three_black_crows"),
        ]
        for condition, label in conditions:
            if condition:
                return label
        return ""

    @staticmethod
    def score(candles: List[Candle]) -> float:
        """Computes a bullishness score based on recent candle formations.

        If a known multi-candle pattern is detected, its score is returned.
        Otherwise, the average score of the last up to 3 individual candles is used,
        scaled to remain strictly between the minimum and maximum multi-pattern scores,
        and rounded to 3 decimal places to avoid overlap and ensure clarity.
        """
        last_candles = candles[-3:]
        pattern = MultiCandlePattern.detect_pattern(last_candles)
        if pattern:
            return MultiCandlePattern._SCORE[pattern]
        min_score = min(MultiCandlePattern._SCORE.values())
        max_score = max(MultiCandlePattern._SCORE.values())
        raw_score = sum(c.score() for c in last_candles) / len(last_candles)
        epsilon = 1e-6
        safe_min = min_score + epsilon
        safe_range = (max_score - min_score) - 2 * epsilon
        adjusted_score = safe_min + raw_score * safe_range
        return round(adjusted_score, 3)
