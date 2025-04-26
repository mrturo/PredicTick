"""Data preparation utilities for financial model training."""

# pylint: disable=import-error,no-name-in-module

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.market_data._old.gateway import Gateway  # type: ignore
from src.utils.config.parameters import ParameterLoader
from src.utils.io.logger import Logger
from training.data_filters import apply_cutoff_filters

PARAMS = ParameterLoader(Gateway.get_last_updated())
SYMBOL_COLUMN = "symbol"
DATETIME_COLUMN = "datetime"
TARGET_COLUMN = "target"


def load_combined_dataset() -> pd.DataFrame:
    """Load symbol data with feature engineering and cutoff filtering."""
    combined = []
    cutoff_date = pd.to_datetime(PARAMS.get("cutoff_date"))
    if pd.isna(cutoff_date):
        cutoff_date = None
    elif cutoff_date.tzinfo is None:
        cutoff_date = cutoff_date.tz_localize(PARAMS.get("market_tz"))
    market_data = Gateway.load()["symbols"]
    for symbol, entry in market_data.items():
        df = pd.DataFrame(entry["historical_prices"])
        if df.empty or not {"close", "high", "low", "volume"}.issubset(df.columns):
            Logger.warning(f"  Skipping {symbol}: insufficient data.")
            continue
        try:
            df = apply_cutoff_filters(
                df,
                cutoff_from=None,
                cutoff_to=cutoff_date,
                timezone=PARAMS.get("market_tz"),
                cutoff_minutes=PARAMS.get("cutoff_minutes"),
            )
        except (KeyError, TypeError, ValueError) as e:
            Logger.warning(f"  Skipping {symbol}: error during cutoff filtering: {e}")
            continue
        if df.empty or not {"close", "high", "low", "volume"}.issubset(df.columns):
            Logger.warning(f"  Skipping {symbol}: data missing after cutoff filtering.")
            continue
        try:
            df = None  # FeatureEngineering.prepare_raw_dataframe(df, symbol)
        except (KeyError, TypeError, ValueError) as e:
            Logger.warning(
                f"  Skipping {symbol}: error during feature engineering: {e}"
            )
            continue
        combined.append(df)
    if not combined:
        raise ValueError("No symbol had valid data after applying cutoff filters.")
    return pd.concat(combined).reset_index()


def engineer_cross_features(df: pd.DataFrame, base_symbols: list) -> pd.DataFrame:
    """Adds spreads and correlations between correlated assets."""
    local_df = df.copy()
    local_df.set_index(DATETIME_COLUMN, inplace=True)
    for base in base_symbols:
        base_df = local_df[local_df[SYMBOL_COLUMN] == base]
        if base_df.empty or "return_1h" not in base_df.columns:
            continue
        spread_col = f"spread_vs_{base}".lower()
        corr_col = f"corr_5d_{base}".lower()
        return_col = f"{base}_return".lower()
        base_returns = base_df["return_1h"].rename(return_col)
        local_df = local_df.join(base_returns, on=DATETIME_COLUMN)
        if local_df[return_col].isna().all():
            Logger.warning(
                f"⚠️ All values are NaN for {base} — skipping {spread_col} and {corr_col}."
            )
            local_df.drop(columns=[return_col], inplace=True)
            continue
        local_df[spread_col] = local_df["return_1h"] - local_df[return_col]
        local_df[corr_col] = (
            local_df["return_1h"]
            .rolling(window=5)
            .corr(local_df[return_col])
            .astype(np.float32)
        )
        local_df.drop(columns=[return_col], inplace=True)
    local_df.reset_index(inplace=True)
    return local_df


def get_valid_cross_features(df: pd.DataFrame, base_symbols: list) -> list:
    """Filter cross features that are not all NaN."""
    features = []
    for base in base_symbols:
        spread = f"spread_vs_{base}".lower()
        corr = f"corr_5d_{base}".lower()
        if spread in df.columns and not df[spread].isna().all():
            features.append(spread)
        else:
            Logger.warning(f"Skipping {spread} — missing or all NaN")
        if corr in df.columns and not df[corr].isna().all():
            features.append(corr)
        else:
            Logger.warning(f"Skipping {corr} — missing or all NaN")
    return features


def filter_valid_cross_features(
    df: pd.DataFrame, features: list, min_rows_required: int = 500
) -> list:
    """Filter features with too few valid values."""
    valid = []
    for feat in features:
        count = df[feat].notna().sum()
        if count >= min_rows_required:
            valid.append(feat)
        else:
            Logger.warning(
                f"⚠️ Dropping {feat} — only {count} valid rows"
                f" (min required: {min_rows_required})"
            )
    return valid


def clean_features(x: pd.DataFrame) -> pd.DataFrame:
    """Cleans the feature DataFrame by removing columns that are:

      - Completely empty (all NaN)
      - Constant (same value across all rows)

    Raises an exception if any column remains entirely empty after cleaning.
    """
    cleaned_df = x.dropna(axis=1, how="all")
    nunique = cleaned_df.nunique()
    cleaned_df = cleaned_df.loc[:, nunique > 1]
    # Additional safety validations
    if cleaned_df.shape[1] == 0:
        raise ValueError("No useful columns remain after cleaning features.")
    if cleaned_df.isnull().all().any():
        raise ValueError(
            "Some columns remain completely empty after cleaning features."
        )
    return cleaned_df
