"""Trainer — Abstract base class for financial model training pipelines.

Includes shared logic for pipeline construction, final model training,
artifact saving, and common utilities. Hyperparameter optimization and data
preparation are delegated to subclasses or external modules.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from market_data.gateway import Gateway
from utils.logger import Logger
from utils.parameters import ParameterLoader

# Constantes compartidas
SYMBOL_COLUMN = "symbol"
TARGET_COLUMN = "target"


# pylint: disable=too-few-public-methods
class TrainParameters:
    """
    TrainParameters — Centralized static container for training configuration.

    Fetches and exposes model training parameters from the ParameterLoader at runtime.
    This includes paths for artifact persistence, feature definitions, Optuna setup,
    cross-validation strategy, cutoff filtering parameters, and market timezone settings.

    All parameters are accessed as class-level constants, intended for read-only use
    throughout the training pipeline.

    Attributes:
        FEATURES (list[str]): Feature columns used for training.
        N_SPLITS (int): Number of splits for GroupKFold or TimeSeriesSplit.
        N_TRIALS (int): Number of trials for Optuna hyperparameter optimization.
        MODEL_FILEPATH (str): Path to persist the trained model.
        SCALER_FILEPATH (str): Path to persist the fitted scaler.
        OPTUNA_FILEPATH (str): Path to persist the Optuna study.
        CUTOFF_DATE (str | None): Latest allowed date for input data, in ISO format.
        CUTOFF_MINUTES (int): Minimum time gap from the present to include a row.
        MARKET_TZ (str): Timezone of the market data.
        CORRELATIVE_SYMBOLS (list[str]): Symbols used for generating cross-features.
    """

    _PARAMS = ParameterLoader(Gateway.get_last_update())
    FEATURES = _PARAMS.get("features")
    N_SPLITS = _PARAMS.get("n_splits")
    N_TRIALS = _PARAMS.get("n_trials")
    MODEL_FILEPATH = _PARAMS.get("model_filepath")
    SCALER_FILEPATH = _PARAMS.get("scaler_filepath")
    OPTUNA_FILEPATH = _PARAMS.get("optuna_filepath")
    CUTOFF_DATE = _PARAMS.get("cutoff_date")
    CUTOFF_MINUTES = _PARAMS.get("cutoff_minutes")
    MARKET_TZ = _PARAMS.get("market_tz")
    CORRELATIVE_SYMBOLS = _PARAMS.get("correlative_symbols")
    XGB_PARAMS = _PARAMS.get("xgb_params")


def build_xgb_params(extra_params):
    """
    Builds the parameter dictionary for XGBClassifier by combining the extra parameters.

    from the JSON file (TrainParameters.XGB_PARAMS) with the extra parameters found (extra_params).
    Values from extra_params override those from the JSON in case of conflicts.
    """
    return {**TrainParameters.XGB_PARAMS, **extra_params}


class Trainer(ABC):
    """Abstract base class containing shared logic for training financial models."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.study = None
        self.cross_features = []

    def save_with_timestamp(
        self, obj: Any, filepath: str, label: str = "", flow: str = ""
    ) -> None:
        """Save object to both original and timestamped file."""
        base, ext = os.path.splitext(filepath)
        flow = "" if flow == "" else f"_{flow}"
        timestamped_simple = f"{base}{flow}{ext}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(obj, timestamped_simple)
        Logger.success(f"Saved {label}: {timestamped_simple}")

    def build_pipeline(
        self, xgb_params: dict, numerical_features: list, categorical_features: list
    ) -> Pipeline:
        """Build preprocessing and modeling pipeline."""
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", XGBClassifier(**xgb_params)),
            ]
        )

        return pipeline

    def train_final_model(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        best_params: dict,
        categorical_features: Optional[list] = None,
    ) -> Tuple[Pipeline, StandardScaler]:
        """Train final model on full dataset using best parameters."""
        resampled = RandomUnderSampler().fit_resample(x, y)
        if len(resampled) == 2:
            x_res, y_res = resampled
        else:
            x_res, y_res, _ = resampled

        numerical_features = TrainParameters.FEATURES.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        categorical_features = (
            (categorical_features or [])
            .select_dtypes(include=["category", "object"])
            .columns.tolist()
        )

        pipeline = self.build_pipeline(
            build_xgb_params(best_params),
            numerical_features=numerical_features,
            categorical_features=categorical_features or [],
        )
        pipeline.fit(x_res, y_res)

        scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"]

        return pipeline, scaler

    def save_artifacts(self) -> None:
        """Save model, scaler, and Optuna study with timestamps."""
        self.save_with_timestamp(self.model, TrainParameters.MODEL_FILEPATH, "model")
        self.save_with_timestamp(self.scaler, TrainParameters.SCALER_FILEPATH, "scaler")
        self.save_with_timestamp(
            self.study, TrainParameters.OPTUNA_FILEPATH, "optuna_study"
        )

    @abstractmethod
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
        """Loads, processes, and prepares the dataset for model training."""

    @abstractmethod
    def run(self) -> None:
        """Execute full training flow."""
