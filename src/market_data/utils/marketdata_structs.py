"""Utilities for structuring market data version information."""

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass
class MarketDataVersion:
    """
    Represents a versioned market data object, including the data content, its timestamp,.

    and the associated file identifier.
    """

    data: Any
    timestamp: Optional[pd.Timestamp]
    file_id: Optional[str]
