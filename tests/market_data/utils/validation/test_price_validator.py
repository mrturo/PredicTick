"""Unit tests for the PriceValidator utility."""

import pandas as pd  # type: ignore
import pytest  # type: ignore

from src.market_data.utils.validation.price_validator import PriceValidator


@pytest.mark.parametrize(
    "df,expected",
    [
        (
            pd.DataFrame(
                {
                    "open": [100, 102],
                    "high": [105, 108],
                    "low": [95, 101],
                    "close": [102, 104],
                    "adj_close": [102, 104],
                }
            ),
            None,
        ),
        (
            pd.DataFrame(
                {
                    "open": [100],
                    "high": [105],
                    "low": [106],  # Invalid
                    "close": [102],
                    "adj_close": [102],
                }
            ),
            "Invalid price range: low > high",
        ),
        (
            pd.DataFrame(
                {
                    "open": [106],  # Invalid
                    "high": [105],
                    "low": [100],
                    "close": [104],
                    "adj_close": [104],
                }
            ),
            "Invalid price range: high < open",
        ),
    ],
)
def test_check_price_ranges(df, expected):
    """Test check_price_ranges returns expected messages for valid and invalid inputs."""
    result = PriceValidator.check_price_ranges(df)
    if result != expected:
        pytest.fail(f"Unexpected result: {result}. Expected: {expected}")


@pytest.mark.parametrize(
    "raw_flow,df,expected",
    [
        (
            True,
            pd.DataFrame(
                {
                    "open": [100, 0],
                    "high": [105, 108],
                    "low": [95, 0],
                    "close": [102, 104],
                    "adj_close": [102, 104],
                }
            ),
            "Non-positive value price found. Columns: ['open', 'low']",
        ),
        (
            False,
            pd.DataFrame(
                {
                    "open": [100],
                    "high": [105],
                    "low": [95],
                    "close": [102],
                    "adj_close": [102],
                }
            ),
            None,
        ),
        (
            True,
            pd.DataFrame(
                {
                    "open": [100],
                    "high": [105],
                    "low": [-1],
                    "close": [102],
                    "adj_close": [102],
                }
            ),
            "Non-positive value price found. Columns: ['low']",
        ),
        (
            False,
            pd.DataFrame(
                {
                    "open": [0],
                    "high": [105],
                    "low": [95],
                    "close": [102],
                    "adj_close": [102],
                }
            ),
            None,
        ),
        (
            False,
            pd.DataFrame(
                {
                    "open": [-1],
                    "high": [105],
                    "low": [95],
                    "close": [102],
                    "adj_close": [102],
                }
            ),
            "Negative value price found. Columns: ['open']",
        ),
    ],
)
def test_check_nonpositive_prices(raw_flow, df, expected):
    """Test check_nonpositive_prices under raw vs adjusted logic."""
    result = PriceValidator.check_nonpositive_prices(df, raw_flow)
    if result != expected:
        pytest.fail(f"Unexpected result: {result}. Expected: {expected}")
