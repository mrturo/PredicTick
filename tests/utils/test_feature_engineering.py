"""Unit tests for the feature_engineering module functions."""

import pandas as pd
import pytest

from utils.feature_engineering import FeatureEngineering


def check_eq(actual, expected, message=""):
    """Helper to compare values and raise a detailed error if they differ."""
    if actual != expected:
        raise AssertionError(message or f"Expected {expected}, got {actual}")


def test_check_eq_function():
    """Check that check_eq passes for equal values."""
    check_eq(1, 1)


def test_check_eq_fail():
    """Check that check_eq raises AssertionError on unequal values."""
    with pytest.raises(AssertionError, match="Expected 2, got 1"):
        check_eq(1, 2)


def test_enrich_with_common_features():
    """Verify enrich_with_common_features adds expected fields correctly."""
    index = pd.date_range("2023-01-03 09:30", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "datetime": index,
            "Open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
            "volume": [10, 20, 30],
        }
    )
    enriched = FeatureEngineering.enrich_with_common_features(df.copy(), symbol="AAPL")
    expected_columns = [
        "symbol",
        "hour",
        "is_pre_market",
        "is_post_market",
        "day_of_week",
        "is_fed_day",
        "target",
    ]
    missing = [col for col in expected_columns if col not in enriched.columns]
    check_eq(missing, [])
    check_eq(enriched["symbol"].iloc[0], "AAPL")
    check_eq(enriched["target"].dtype.name, "Int8")


@pytest.mark.parametrize(
    "close_values,expected",
    [
        ([100, 100, 102], 2),  # UP
        ([100, 100, 98], 0),  # DOWN
        ([100, 100, 100], 1),  # NEUTRAL
    ],
)
def test_classify_target_label(close_values, expected):
    """Test that classify_target correctly labels UP, DOWN, and NEUTRAL cases."""
    index = pd.date_range("2023-01-03 09:30", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": [0] * 3,
            "high": [0] * 3,
            "low": [0] * 3,
            "close": close_values,
            "volume": [0] * 3,
        },
        index=index,
    )

    result = FeatureEngineering.classify_target(df, threshold=0.01, periods_ahead=2)
    check_eq(result.iloc[0], expected)


def test_prepare_raw_dataframe(monkeypatch):
    """Ensure prepare_raw_dataframe integrates symbol and target correctly."""
    df = pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2023-01-03 09:30", periods=1, freq="1min", tz="UTC"
            ),
            "Open": [1],
            "high": [2],
            "low": [0],
            "close": [1.5],
            "volume": [100],
        }
    )

    enriched = df.copy()
    enriched["symbol"] = "AAPL"
    enriched["target"] = 2

    monkeypatch.setattr(
        FeatureEngineering, "enrich_with_common_features", lambda df, symbol: enriched
    )

    result = FeatureEngineering.prepare_raw_dataframe(df.copy(), "AAPL")
    check_eq("symbol" in result.columns, True)
    check_eq(result["symbol"].iloc[0], "AAPL")


def test_classify_target_defaults():
    """Test classify_target with default threshold and periods."""
    df = pd.DataFrame(
        {
            "close": [100, 100, 101],
        },
        index=pd.date_range("2023-01-03 09:30", periods=3, freq="1min", tz="UTC"),
    )
    result = FeatureEngineering.classify_target(df, periods_ahead=1)
    check_eq(result.dropna().iloc[1], 2)
