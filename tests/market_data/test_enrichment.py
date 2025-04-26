"""Unit tests for the Enrichment class.

These tests validate feature engineering logic including basic price-derived metrics,
technical indicators from TA-Lib, zero-volume interpolation, and temporal features
based on calendar indexing. Coverage includes edge cases like missing columns
and insufficient data.
"""

# pylint: disable=protected-access

from unittest.mock import patch

import numpy as np
import pandas as pd

from market_data.enrichment import Enrichment


def test_enrich_historical_prices_minimal():
    """Test enrich_historical_prices with minimal valid input."""
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=3, freq="D"),
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1000, 1100, 1200],
        }
    )

    enriched = Enrichment.enrich_historical_prices(df)

    required_columns = [
        "return",
        "volatility",
        "price_change",
        "volume_change",
        "typical_price",
        "average_price",
        "candle",
        "range",
        "relative_volume",
    ]

    for col in required_columns:
        if col not in enriched.columns:
            raise AssertionError(f"Missing column: {col}")


def test_enrich_historical_prices_insufficient_data():
    """Test enrich_historical_prices skips indicators when data is insufficient."""
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=10, freq="D"),
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000] * 10,
        }
    )

    enriched = Enrichment.enrich_historical_prices(df)

    if "atr_14" in enriched.columns:
        raise AssertionError("Expected technical indicators to be skipped")


def test_enrich_historical_prices_missing_columns():
    """Test enrich_historical_prices handles missing essential columns."""
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=3, freq="D"),
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 98.0, 97.0],
        }
    )

    enriched = Enrichment.enrich_historical_prices(df)

    if not enriched.equals(df):
        raise AssertionError(
            "Expected unchanged DataFrame when essential columns are missing"
        )


def test_enrich_historical_prices_zero_volume_interpolation():
    """Test enrich_historical_prices handles zero volume with interpolation."""
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=3, freq="D"),
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [0, 1100, 1200],
        }
    )

    enriched = Enrichment.enrich_historical_prices(df)

    if enriched.loc[0, "volume"] == 0 or pd.isna(enriched.loc[0, "volume"]):
        raise AssertionError("Volume interpolation failed for zero volume entry")


def test_enrich_historical_prices_with_indicators():
    """Test enrich_historical_prices includes all indicators when data is sufficient."""
    periods = 260  # Ensure > 252
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=periods, freq="D"),
            "open": np.linspace(100, 200, periods),
            "high": np.linspace(101, 201, periods),
            "low": np.linspace(99, 199, periods),
            "close": np.linspace(100.5, 200.5, periods),
            "volume": np.linspace(1000, 2000, periods),
        }
    )

    enriched = Enrichment.enrich_historical_prices(df)

    expected_cols = [
        "atr_14",
        "overnight_return",
        "intraday_return",
        "volume_rvol_20d",
        "obv",
        "adx_14",
        "bollinger_pct_b",
        "pattern_bullish_engulfing",
        "pattern_bearish_engulfing",
        "pct_from_52w_high",
    ]
    for col in expected_cols:
        if col not in enriched.columns:
            raise AssertionError(f"Missing technical feature: {col}")


@patch("market_data.schedule_builder.ScheduleBuilder.build_day_index_maps")
@patch("market_data.schedule_builder.ScheduleBuilder.merge_and_estimate_totals")
def test_add_month_and_year_days(mock_merge, mock_build):
    """Test add_month_and_year_days with mocked schedule logic."""
    df = pd.DataFrame({"datetime": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"]})

    dummy_dates = pd.DataFrame(
        {
            "local_year": [2024, 2024],
            "local_month": [1, 1],
            "local_date": [
                pd.to_datetime("2024-01-01").date(),
                pd.to_datetime("2024-01-02").date(),
            ],
        }
    )
    mock_build.return_value = (
        {(2024, 1): {pd.to_datetime("2024-01-01").date(): 1}},
        {2024: {pd.to_datetime("2024-01-01").date(): 1}},
    )

    mock_merged = dummy_dates.copy()
    mock_merged["local_month_days_current"] = [1, 2]
    mock_merged["local_year_days_current"] = [1, 2]
    mock_merged["local_month_days_total"] = [31, 31]
    mock_merged["local_year_days_total"] = [365, 365]
    mock_merge.return_value = mock_merged

    enriched = Enrichment.add_month_and_year_days(df, calendar_type=7)
    expected_cols = [
        "month_days_current",
        "month_days_total",
        "year_days_current",
        "year_days_total",
    ]
    for col in expected_cols:
        if col not in enriched.columns:
            raise AssertionError(f"Missing expected column: {col}")
