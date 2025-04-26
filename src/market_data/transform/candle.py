"""
Module that defines Candle and MultiCandlePattern classes.

The Candle class is for analyzing individual Japanese candlesticks, and
the MultiCandlePattern class is for detecting multi-candle reversal patterns.

Provides methods to calculate candlestick attributes such as body, shadows,
and to classify patterns like doji, hammer, shooting star, marubozu,
spinning top, as well as multi-candle patterns like engulfing and star formations.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np  # type: ignore

from utils.parameters import ParameterLoader


@dataclass
# pylint: disable=too-many-public-methods
class Candle:
    """
    Represents a Japanese candlestick and provides technical analysis methods.

    Attributes:
        open (float): Opening price.
        high (float): Highest price.
        low (float): Lowest price.
        close (float): Closing price.
    """

    _PARAMS = ParameterLoader()
    _METRICS = _PARAMS.get("candle_metrics")
    _SCORE = _PARAMS.get("candle_simple_score")

    open: Union[float, np.floating]
    high: Union[float, np.floating]
    low: Union[float, np.floating]
    close: Union[float, np.floating]

    def body(self) -> float:
        """Returns the size of the candlestick body."""
        return abs(self.close - self.open)

    def upper_shadow(self) -> float:
        """Calculates the length of the upper shadow."""
        return self.high - max(self.open, self.close)

    def lower_shadow(self) -> float:
        """Calculates the length of the lower shadow."""
        return min(self.open, self.close) - self.low

    def is_bullish(self) -> bool:
        """Indicates whether the candlestick is bullish."""
        return self.close > self.open

    def is_bearish(self) -> bool:
        """Indicates whether the candlestick is bearish."""
        return self.close < self.open

    def is_doji(self) -> bool:
        """Detects if the candlestick is a doji based on the configured body threshold."""
        body_pct = self.body() / (self.high - self.low + 1e-9)
        return body_pct < Candle._METRICS["doji_body_threshold"]

    def is_hammer_bullish(self) -> bool:
        """Detects a bullish hammer pattern based on shadow ratios and body size."""
        return (
            self.is_bullish()
            and self.lower_shadow()
            > self.body() * Candle._METRICS["hammer_shadow_ratio"]
            and self.upper_shadow()
            < self.body() * Candle._METRICS["upper_shadow_max_ratio"]
            and self.body() / (self.high - self.low + 1e-9)
            < Candle._METRICS["hammer_body_threshold"]
        )

    def is_hammer_bearish(self) -> bool:
        """Detects a bearish hammer pattern based on shadow ratios and body size."""
        return (
            self.is_bearish()
            and self.lower_shadow()
            > self.body() * Candle._METRICS["hammer_shadow_ratio"]
            and self.upper_shadow()
            < self.body() * Candle._METRICS["upper_shadow_max_ratio"]
            and self.body() / (self.high - self.low + 1e-9)
            < Candle._METRICS["hammer_body_threshold"]
        )

    def is_hammer(self) -> bool:
        """
        Determines if the candlestick meets the criteria for a hammer pattern.

        Returns:
            bool: True if the candle is a bullish or bearish hammer.
        """
        return self.is_hammer_bullish() or self.is_hammer_bearish()

    def is_shooting_star_bullish(self) -> bool:
        """Detects a bullish shooting star pattern."""
        return (
            self.is_bullish()
            and self.upper_shadow()
            > self.body() * Candle._METRICS["shooting_star_shadow_ratio"]
            and self.lower_shadow()
            < self.body() * Candle._METRICS["lower_shadow_max_ratio"]
            and self.body() / (self.high - self.low + 1e-9)
            < Candle._METRICS["shooting_star_body_threshold"]
        )

    def is_shooting_star_bearish(self) -> bool:
        """Detects a bearish shooting star pattern."""
        return (
            self.is_bearish()
            and self.upper_shadow()
            > self.body() * Candle._METRICS["shooting_star_shadow_ratio"]
            and self.lower_shadow()
            < self.body() * Candle._METRICS["lower_shadow_max_ratio"]
            and self.body() / (self.high - self.low + 1e-9)
            < Candle._METRICS["shooting_star_body_threshold"]
        )

    def is_shooting_star(self) -> bool:
        """Detects a shooting star pattern regardless of direction."""
        return self.is_shooting_star_bullish() or self.is_shooting_star_bearish()

    def is_marubozu_bullish(self) -> bool:
        """Detects a bullish marubozu candle with minimal shadows."""
        return (
            self.is_bullish()
            and self.upper_shadow()
            < self.body() * Candle._METRICS["upper_shadow_max_ratio"]
            and self.lower_shadow()
            < self.body() * Candle._METRICS["lower_shadow_max_ratio"]
        )

    def is_marubozu_bearish(self) -> bool:
        """Detects a bearish marubozu candle with minimal shadows."""
        return (
            self.is_bearish()
            and self.upper_shadow()
            < self.body() * Candle._METRICS["upper_shadow_max_ratio"]
            and self.lower_shadow()
            < self.body() * Candle._METRICS["lower_shadow_max_ratio"]
        )

    def is_marubozu(self) -> bool:
        """Detects a marubozu candle: no (or very small) upper and lower shadows."""
        return self.is_marubozu_bullish() or self.is_marubozu_bearish()

    def is_spinning_top_up(self) -> bool:
        """Detects a bullish spinning top: small body with symmetrical shadows."""
        body_pct = self.body() / (self.high - self.low + 1e-9)
        shadow_ratio = min(self.upper_shadow(), self.lower_shadow()) / (
            self.body() + 1e-9
        )
        return (
            self.is_bullish()
            and Candle._METRICS["doji_body_threshold"]
            < body_pct
            < Candle._METRICS["hammer_body_threshold"]
            and shadow_ratio > 1.0
        )

    def is_spinning_top_down(self) -> bool:
        """Detects a bearish spinning top: small body with symmetrical shadows."""
        body_pct = self.body() / (self.high - self.low + 1e-9)
        shadow_ratio = min(self.upper_shadow(), self.lower_shadow()) / (
            self.body() + 1e-9
        )
        return (
            self.is_bearish()
            and Candle._METRICS["doji_body_threshold"]
            < body_pct
            < Candle._METRICS["hammer_body_threshold"]
            and shadow_ratio > 1.0
        )

    def is_spinning_top(self) -> bool:
        """Detects a spinning top: small body with long and symmetric shadows."""
        return self.is_spinning_top_up() or self.is_spinning_top_down()

    def is_indecisive(self) -> bool:
        """
        Determines if the candle reflects price indecision.

        A candle is considered indecisive if it is either a doji or a spinning top.
        These patterns typically indicate a balance between buying and selling pressure.

        Returns:
            bool: True if the candle is a doji or a spinning top.
        """
        return self.is_doji() or self.is_spinning_top()

    def is_dragonfly_doji(self) -> bool:
        """
        Detecta un Dragonfly Doji: cuerpo muy pequeño, sombra inferior larga y sin sombra superior.

        Señal potencial de reversión alcista.
        """
        return (
            self.is_doji()
            and self.lower_shadow()
            > self.body() * Candle._METRICS["hammer_shadow_ratio"]
            and self.upper_shadow() < Candle._METRICS["doji_max_upper_shadow"]
        )

    def is_gravestone_doji(self) -> bool:
        """
        Detecta un Gravestone Doji: cuerpo muy pequeño, sombra superior larga y sin sombra inferior.

        Señal potencial de reversión bajista.
        """
        return (
            self.is_doji()
            and self.upper_shadow()
            > self.body() * Candle._METRICS["shooting_star_shadow_ratio"]
            and self.lower_shadow() < Candle._METRICS["doji_max_lower_shadow"]
        )

    def is_inverted_hammer(self) -> bool:
        """
        Detecta un Inverted Hammer: patrón tras tendencia bajista con sombra superior larga.

        Potencial señal de reversión alcista.
        """
        return (
            self.is_bullish()
            and self.upper_shadow()
            > self.body() * Candle._METRICS["shooting_star_shadow_ratio"]
            and self.lower_shadow()
            < self.body() * Candle._METRICS["lower_shadow_max_ratio"]
            and self.body() / (self.high - self.low + 1e-9)
            < Candle._METRICS["hammer_body_threshold"]
        )

    def is_hanging_man(self) -> bool:
        """
        Detecta un Hanging Man: igual a un hammer pero aparece en tendencia alcista.

        Potencial señal de agotamiento de tendencia alcista.
        """
        return (
            self.is_bearish()
            and self.lower_shadow()
            > self.body() * Candle._METRICS["hammer_shadow_ratio"]
            and self.upper_shadow()
            < self.body() * Candle._METRICS["upper_shadow_max_ratio"]
            and self.body() / (self.high - self.low + 1e-9)
            < Candle._METRICS["hammer_body_threshold"]
        )

    def has_long_upper_shadow(self) -> bool:
        """Checks if the upper shadow is significantly long."""
        return (
            self.upper_shadow()
            > self.body() * Candle._METRICS["shooting_star_shadow_ratio"]
        )

    def has_long_lower_shadow(self) -> bool:
        """Checks if the lower shadow is significantly long."""
        return (
            self.lower_shadow() > self.body() * Candle._METRICS["hammer_shadow_ratio"]
        )

    def close_position(self) -> float:
        """Returns the relative position of the close price within the candle's range."""
        return (self.close - self.low) / (self.high - self.low + 1e-9)

    def detect_pattern(self) -> str:
        """
        Classifies the candlestick into one of the defined patterns or basic direction.

        Returns:
            str: Name of the detected pattern or the candlestick's trend.
        """
        conditions = [
            (self.is_dragonfly_doji(), "dragonfly_doji"),
            (self.is_gravestone_doji(), "gravestone_doji"),
            (self.is_inverted_hammer(), "inverted_hammer"),
            (self.is_hanging_man(), "hanging_man"),
            (self.is_hammer_bullish(), "hammer_bullish"),
            (self.is_hammer_bearish(), "hammer_bearish"),
            (self.is_shooting_star_bullish(), "shooting_star_bullish"),
            (self.is_shooting_star_bearish(), "shooting_star_bearish"),
            (self.is_marubozu_bullish(), "marubozu_bullish"),
            (self.is_marubozu_bearish(), "marubozu_bearish"),
            (self.is_spinning_top_up(), "spinning_top_bullish"),
            (self.is_spinning_top_down(), "spinning_top_bearish"),
            (self.is_doji(), "doji"),
            (self.has_long_upper_shadow(), "long_upper_shadow"),
            (self.has_long_lower_shadow(), "long_lower_shadow"),
        ]
        for condition, label in conditions:
            if condition:
                return label
        return "bullish" if self.is_bullish() else "bearish"

    def score(self) -> float:
        """
        Returns a bullishness score from 0 to 1 based on the candlestick type.

        0 = strongly bearish, 1 = strongly bullish.

        Returns:
            float: Score between 0.0 and 1.0
        """
        description = self.detect_pattern()
        if description:
            return Candle._SCORE[description]
        return 0.5


class MultiCandlePattern:
    """
    Detects multi-candle reversal patterns using detailed single-candle classifications.

    Supports common formations like engulfings, stars, piercing lines, tweezers,
    and soldier/crow patterns. Provides scoring based on pattern strength or fallback
    to individual candle scores.
    """

    _PARAMS = ParameterLoader()
    _SCORE = _PARAMS.get("candle_multiple_score")

    @staticmethod
    def _is_bullish_engulfing(c1: Candle, c2: Candle) -> bool:
        """
        Determines if a bullish engulfing pattern is formed by two candles.

        Args:
            c1 (Candle): The first candle, expected to be bearish.
            c2 (Candle): The second candle, expected to be bullish and engulf the first.

        Returns:
            bool: True if the pattern is bullish engulfing.
        """
        return (
            c1.detect_pattern()
            in {"bearish", "marubozu_bearish", "spinning_top_bearish"}
            and c2.detect_pattern() in {"bullish", "marubozu_bullish"}
            and c2.open < c1.close
            and c2.close > c1.open
        )

    @staticmethod
    def _is_bearish_engulfing(c1: Candle, c2: Candle) -> bool:
        """
        Determines if a bearish engulfing pattern is formed by two candles.

        Args:
            c1 (Candle): The first candle, expected to be bullish.
            c2 (Candle): The second candle, expected to be bearish and engulf the first.

        Returns:
            bool: True if the pattern is bearish engulfing.
        """
        return (
            c1.detect_pattern()
            in {"bullish", "marubozu_bullish", "spinning_top_bullish"}
            and c2.detect_pattern() in {"bearish", "marubozu_bearish"}
            and c2.open > c1.close
            and c2.close < c1.open
        )

    @staticmethod
    def _is_morning_star(c1: Candle, c2: Candle, c3: Candle) -> bool:
        """
        Detects a morning star pattern across three candles.

        Args:
            c1 (Candle): Bearish candle.
            c2 (Candle): Indecisive candle (e.g., doji or spinning top).
            c3 (Candle): Bullish candle confirming reversal.

        Returns:
            bool: True if morning star pattern is detected.
        """
        return (
            c1.detect_pattern() in {"bearish", "marubozu_bearish"}
            and c2.is_indecisive()
            and c3.detect_pattern() in {"bullish", "marubozu_bullish"}
            and c3.close > (c1.open + c1.close) / 2
        )

    @staticmethod
    def _is_evening_star(c1: Candle, c2: Candle, c3: Candle) -> bool:
        """
        Detects an evening star pattern across three candles.

        Args:
            c1 (Candle): Bullish candle.
            c2 (Candle): Indecisive candle (e.g., doji or spinning top).
            c3 (Candle): Bearish candle confirming reversal.

        Returns:
            bool: True if evening star pattern is detected.
        """
        return (
            c1.detect_pattern() in {"bullish", "marubozu_bullish"}
            and c2.is_indecisive()
            and c3.detect_pattern() in {"bearish", "marubozu_bearish"}
            and c3.close < (c1.open + c1.close) / 2
        )

    @staticmethod
    def _is_piercing_line(c1: Candle, c2: Candle) -> bool:
        """
        Identifies a bullish piercing line pattern.

        Args:
            c1 (Candle): The first candle, which should be bearish.
            c2 (Candle): Bullish candle opening below c1 and closing above its midpoint.

        Returns:
            bool: True if the pattern is a valid piercing line.
        """
        return (
            c1.is_bearish()
            and c2.is_bullish()
            and c2.open < c1.low
            and c2.close > (c1.open + c1.close) / 2
            and c2.close < c1.open
        )

    @staticmethod
    def _is_dark_cloud_cover(c1: Candle, c2: Candle) -> bool:
        """
        Identifies a bearish dark cloud cover pattern.

        Args:
            c1 (Candle): The first candle, which should be bullish.
            c2 (Candle): The second candle, a bearish that opens above and closes below midpoint.

        Returns:
            bool: True if the pattern is a valid dark cloud cover.
        """
        return (
            c1.is_bullish()
            and c2.is_bearish()
            and c2.open > c1.high
            and c2.close < (c1.open + c1.close) / 2
            and c2.close > c1.open
        )

    @staticmethod
    def _is_tweezer_bottom(c1: Candle, c2: Candle) -> bool:
        """
        Identifies a tweezer bottom reversal pattern.

        Args:
            c1 (Candle): The first bearish candle.
            c2 (Candle): The second bullish candle with the same low as the first.

        Returns:
            bool: True if tweezer bottom is detected.
        """
        return c1.is_bearish() and c2.is_bullish() and abs(c1.low - c2.low) < 1e-3

    @staticmethod
    def _is_tweezer_top(c1: Candle, c2: Candle) -> bool:
        return c1.is_bullish() and c2.is_bearish() and abs(c1.high - c2.high) < 1e-3

    @staticmethod
    def _is_three_white_soldiers(c1: Candle, c2: Candle, c3: Candle) -> bool:
        """
        Identifies a bullish continuation pattern: three white soldiers.

        Args:
            c1, c2, c3 (Candle): Three consecutive bullish candles with increasing closes
                                and small upper shadows.

        Returns:
            bool: True if the pattern matches the criteria.
        """
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
        """
        Identifies a bearish continuation pattern: three black crows.

        Args:
            c1, c2, c3 (Candle): Three consecutive bearish candles with decreasing closes
                                and small lower shadows.

        Returns:
            bool: True if the pattern matches the three black crows criteria.
        """
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
        """
        Identifies the most recent multi-candle pattern if present.

        Args:
            candles (List[Candle]): List of recent candles (minimum 3).

        Returns:
            Optional[str]: Name of the pattern if detected, else empty string.
        """
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
        """
        Computes a bullishness score based on recent candle formations.

        If a known multi-candle pattern is detected, its score is returned.
        Otherwise, the average score of the last up to 3 individual candles is used,
        scaled to remain strictly between the minimum and maximum multi-pattern scores,
        and rounded to 3 decimal places to avoid overlap and ensure clarity.

        Args:
            candles (List[Candle]): List of recent candles (up to 3 considered).

        Returns:
            float: Score in range [0.0, 1.0], avoiding overlap with multi-pattern scores.
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
