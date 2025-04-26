"""Defines the MarketContext dataclass used to encapsulate market-specific context."""

import datetime
from dataclasses import dataclass
from typing import Any, List


@dataclass(frozen=True)
class MarketContext:
    """Immutable container for contextual market inputs."""

    us_holidays: List[datetime.date]
    fed_events: List[datetime.date]
    market_time: Any
