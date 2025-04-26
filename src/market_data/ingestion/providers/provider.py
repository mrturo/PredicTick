"""This module defines a generic interface for accessing market data using Yahoo Finance.

as the backend provider. It includes functionality to retrieve ticker metadata and
download historical price data for one or more financial instruments.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd  # type: ignore
import yfinance as yf  # type: ignore

from src.market_data.ingestion.providers.price_data_config import \
    PriceDataConfig
from src.market_data.ingestion.providers.ticker_metadata import TickerMetadata
from src.utils.io.output_suppressor import OutputSuppressor


class Provider:
    """Generic market data provider interface using Yahoo Finance as backend.

    Provides methods to retrieve ticker metadata and historical price data.
    """

    def get_metadata(self, symbol: str) -> TickerMetadata:
        """Retrieves metadata for a specific symbol."""
        ticker = yf.Ticker(symbol)
        return TickerMetadata.from_dict(ticker.info)

    def get_price_data(self, config: PriceDataConfig) -> Optional[pd.DataFrame]:
        """Downloads historical price data for one or more symbols."""
        result: Optional[pd.DataFrame] = None
        if config.proxy:
            os.environ["HTTP_PROXY"] = config.proxy
            os.environ["HTTPS_PROXY"] = config.proxy
        group_by_value = (
            "column" if isinstance(config.symbols, str) else config.group_by
        )
        with OutputSuppressor.suppress(capture=True) as (_out_buf, err_buf):
            result = yf.download(
                tickers=config.symbols,
                start=config.start,
                end=config.end,
                interval=config.interval,
                group_by=group_by_value,
                auto_adjust=config.auto_adjust,
                prepost=config.prepost,
                threads=config.threads,
                progress=config.progress,
            )
        stderr_text = err_buf.getvalue() if err_buf is not None else None
        stderr_text = (
            stderr_text.strip()
            if stderr_text is not None and len(stderr_text.strip()) > 0
            else None
        )
        if stderr_text is not None:
            raise ValueError(stderr_text)
        return result
