"""Utilities for structuring market data version information."""

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd  # type: ignore


@dataclass
class MarketDataVersion:
    """Represents a versioned market data object."""

    data: Any
    timestamp: Optional[pd.Timestamp]
    file_id: Optional[str]
