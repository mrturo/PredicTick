"""This module defines the `Candle` dataclass, a technical abstraction representing a.

Japanese candlestick, commonly used in financial market analysis. It includes numerous
methods for detecting classic candlestick patterns and computing shape-related metrics.

The class supports pattern recognition for doji, hammer, marubozu, spinning top,
shooting star, dragonfly, gravestone, inverted hammer, and hanging man. These methods
rely on configurable thresholds provided by `ParameterLoader`.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np  # type: ignore

from src.utils.config.parameters import ParameterLoader


@dataclass
# pylint: disable=too-many-public-methods
class Candle:
    """Represents a Japanese candlestick and provides technical analysis methods."""

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
        """Determines if the candlestick meets the criteria for a hammer pattern."""
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
        """Determines if the candle reflects price indecision.

        A candle is considered indecisive if it is either a doji or a spinning top.
        These patterns typically indicate a balance between buying and selling pressure.
        """
        return self.is_doji() or self.is_spinning_top()

    def is_dragonfly_doji(self) -> bool:
        """Detects a Dragonfly Doji.

        Very small body with a long lower shadow and no upper shadow, a potential bullish
        reversal signal.
        """
        return (
            self.is_doji()
            and self.lower_shadow()
            > self.body() * Candle._METRICS["hammer_shadow_ratio"]
            and self.upper_shadow() < Candle._METRICS["doji_max_upper_shadow"]
        )

    def is_gravestone_doji(self) -> bool:
        """Detects a Gravestone Doji.

        Very small body with a long upper shadow and no lower shadow, a potential bearish
        reversal.
        """
        return (
            self.is_doji()
            and self.upper_shadow()
            > self.body() * Candle._METRICS["shooting_star_shadow_ratio"]
            and self.lower_shadow() < Candle._METRICS["doji_max_lower_shadow"]
        )

    def is_inverted_hammer(self) -> bool:
        """Detects an Inverted Hammer: pattern after a downtrend with a long upper shadow.

        Potential bullish reversal signal.
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
        """Detects a Hanging Man: similar to a hammer but appearing in an uptrend.

        Potential sign of bullish trend exhaustion.
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
        """Classifies the candlestick into one of the defined patterns or basic direction."""
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
        """Returns a bullishness score from 0 to 1 based on the candlestick type.

        0 = strongly bearish, 1 = strongly bullish.
        """
        description = self.detect_pattern()
        if description:
            return Candle._SCORE[description]
        return 0.5
