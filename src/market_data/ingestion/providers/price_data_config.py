"""Provides an interface to fetch historical price data and metadata using Yahoo Finance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


# pylint: disable=too-many-instance-attributes
@dataclass
class PriceDataConfig:
    """Configuration object for retrieving historical market price data from provider."""

    symbols: Union[str, list[str]]
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = "1d"
    group_by: str = "ticker"
    auto_adjust: bool = True
    prepost: bool = False
    threads: bool = True
    proxy: Optional[str] = None
    progress: bool = True
