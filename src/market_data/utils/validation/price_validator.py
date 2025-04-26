"""PriceValidator module for basic OHLCV data integrity checks.

This module provides a set of static validation methods to identify
common inconsistencies in price data, particularly within OHLCV structures
(open, high, low, close, adjusted close).

Key validation routines include:
    - `check_price_ranges`: Detects logical contradictions in price hierarchy,
      such as 'low' being greater than 'high' or 'open' exceeding 'high'.
    - `check_nonpositive_prices`: Validates the sign of price columns, optionally
      disallowing zero or negative values depending on whether the data is raw
      (pre-cleaned) or already adjusted.

These checks are typically used as part of broader data pipelines to
prevent invalid financial data from entering enrichment or modeling stages.

Usage Example:
    >>> PriceValidator.check_price_ranges(df)
    >>> PriceValidator.check_nonpositive_prices(df, raw_flow=True)

All methods return either `None` (if the data passes validation) or a
descriptive error message indicating the first issue encountered.
"""

from typing import List, Union

import pandas as pd  # type: ignore


class PriceValidator:
    """Collection of utilities to validate OHLCV columns."""

    @staticmethod
    def check_price_ranges(df: pd.DataFrame) -> Union[str, None]:
        """Checks for basic inconsistencies between OHLC/Adj Close values."""
        mismatches = [
            ("low", "high", (df["low"] > df["high"]).any()),
            ("low", "open", (df["low"] > df["open"]).any()),
            ("low", "close", (df["low"] > df["close"]).any()),
            ("low", "adj_close", (df["low"] > df["adj_close"]).any()),
            ("high", "open", (df["high"] < df["open"]).any()),
            ("high", "close", (df["high"] < df["close"]).any()),
            ("high", "adj_close", (df["high"] < df["adj_close"]).any()),
        ]
        for lower_col, upper_col, condition in mismatches:
            if condition:  # pragma: no branch
                message = (
                    f"Invalid price range: {lower_col} > {upper_col}"
                    if lower_col == "low"
                    else f"Invalid price range: {lower_col} < {upper_col}"
                )
                return message
        return None

    @staticmethod
    def check_nonpositive_prices(
        df: pd.DataFrame,
        raw_flow: bool,
    ) -> Union[str, None]:
        """Checks for the presence of non-positive or negative prices.

        If ``raw_flow`` is ``True``, zero and negative values are not allowed;
        if ``False``, only negative values are disallowed (zeros are accepted).
        """
        columns = ["open", "low", "high", "close", "adj_close"]
        invalid_cols: List[str] = []
        cmp_op = (lambda s: (s <= 0).any()) if raw_flow else (lambda s: (s < 0).any())
        for col in columns:
            if cmp_op(df[col]):  # pragma: no cover
                invalid_cols.append(col)
        if invalid_cols:
            descriptor = "Non-positive" if raw_flow else "Negative"
            message = f"{descriptor} value price found. Columns: {invalid_cols}"
            return message
        return None
