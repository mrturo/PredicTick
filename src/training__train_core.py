"""
CoreTrainer — Abstract base class for financial model training pipelines.

Includes shared logic for pipeline construction, hyperparameter optimization with Optuna,
data preprocessing, class balancing, and artifact persistence.
"""

import datetime
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from updater.data_updater import FileManager
from utils.logger import Logger
from utils.parameters import ParameterLoader


class CoreTrainer(ABC):
    """Abstract base class containing shared logic for training financial models."""

    def __init__(self):
        self.parameters = ParameterLoader(FileManager.last_update())
        self.model = None
        self.scaler = None
        self.study = None

    def save_with_timestamp(
        self, obj: Any, filepath: str, label: str = "", flow: str = ""
    ) -> None:
        """Save object to both original and timestamped file."""
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        base, ext = os.path.splitext(filepath)
        flow = "" if flow == "" else f"_{flow}"
        timestamped_filepath = f"{base}{flow}_{now}{ext}"
        timestamped_simple = f"{base}{flow}{ext}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(obj, timestamped_simple)
        joblib.dump(obj, timestamped_filepath)
        Logger.success(f"Saved {label}:")
        Logger.simple(f" * {timestamped_simple}")
        Logger.simple(f" * {timestamped_filepath}")

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

    def optimize_hyperparameters(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        categorical_features: Optional[list] = None,
    ) -> optuna.Study:
        """Run Optuna hyperparameter optimization."""

        def objective(trial: optuna.Trial) -> float:
            xgb_params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "use_label_encoder": False,
                "tree_method": "hist",
                "verbosity": 0,
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }

            pipeline = self.build_pipeline(
                xgb_params,
                numerical_features=self.parameters["features"],
                categorical_features=categorical_features or [],
            )

            if groups is not None:
                cv = GroupKFold(n_splits=self.parameters["n_splits"])
                scores = cross_val_score(
                    pipeline, x, y, groups=groups, cv=cv, scoring="accuracy"
                )
            else:
                cv = TimeSeriesSplit(n_splits=self.parameters["n_splits"])
                scores = []
                for train_idx, val_idx in cv.split(x):
                    x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
                    x_val, y_val = x.iloc[val_idx], y.iloc[val_idx]

                    x_res, y_res = RandomUnderSampler().fit_resample(x_train, y_train)
                    pipeline.fit(x_res, y_res)
                    score = pipeline.score(x_val, y_val)
                    scores.append(score)

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.parameters["n_trials"])
        return study

    def train_final_model(
        self,
        x: np.ndarray,
        y: np.ndarray,
        best_params: dict,
        categorical_features: Optional[list] = None,
    ) -> Tuple[Pipeline, StandardScaler]:
        """Train final model on full dataset using best parameters."""
        xgb_params = {
            **best_params,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "tree_method": "hist",
            "verbosity": 0,
        }

        x_res, y_res = RandomUnderSampler().fit_resample(x, y)

        pipeline = self.build_pipeline(
            xgb_params,
            numerical_features=self.parameters["features"],
            categorical_features=categorical_features or [],
        )
        pipeline.fit(x_res, y_res)

        scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"]

        return pipeline, scaler

    def save_artifacts(self) -> None:
        """Save model, scaler, and Optuna study with timestamps."""
        self.save_with_timestamp(self.model, self.parameters["model_filepath"], "model")
        self.save_with_timestamp(
            self.scaler, self.parameters["scaler_filepath"], "scaler"
        )
        self.save_with_timestamp(
            self.study, self.parameters["optuna_filepath"], "optuna_study"
        )

    @abstractmethod
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
        """Implemented in child class: load and prepare X, y, groups if applicable."""
        raise NotImplementedError

    def run(self) -> None:
        """Execute full training flow."""
        Logger.info("Preparing data...")
        x, y, groups = self.prepare_data()

        Logger.info("Optimizing hyperparameters...")
        self.study = self.optimize_hyperparameters(x, y, groups=groups)

        Logger.success(f"Best trial found: {self.study.best_trial.number}")
        for k, v in self.study.best_trial.params.items():
            Logger.simple(f"  {k}: {v}")

        Logger.info("Training final model...")
        self.model, self.scaler = self.train_final_model(
            x, y, self.study.best_trial.params
        )

        Logger.success("Saving artifacts...")
        self.save_artifacts()
        Logger.success("Training pipeline completed.")


# pylint: disable=too-many-arguments, too-many-positional-arguments
def apply_cutoff_filters(
    df: pd.DataFrame,
    cutoff_from: Optional[pd.Timestamp],
    cutoff_to: Optional[pd.Timestamp],
    timezone: str,
    cutoff_minutes: int,
) -> pd.DataFrame:
    """Apply date/time filtering to a DataFrame based on cutoff constraints."""

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df.set_index("Datetime", inplace=True)
    df = df.tz_convert(timezone).sort_index()
    df["date"] = df.index.date

    if cutoff_from is not None:
        df = df[df.index > cutoff_from]
    if cutoff_to is not None:
        df = df[df.index <= cutoff_to]

    now = pd.Timestamp.now(tz=df.index.tz)
    df = df[
        df.index.to_series().apply(
            lambda idx: (now - idx).total_seconds() > cutoff_minutes * 60
        )
    ]

    df.reset_index(inplace=True)

    return df
