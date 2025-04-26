"""Hyperparameter optimization using Optuna for financial model pipelines."""

from typing import Optional

import numpy as np
import optuna
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score

from market_data.gateway import Gateway
from training.base_trainer import Trainer, build_xgb_params
from utils.parameters import ParameterLoader

_PARAMS = ParameterLoader(Gateway.get_last_update())
_FEATURES = _PARAMS.get("features")
_N_SPLITS = _PARAMS.get("n_splits")
_N_TRIALS = _PARAMS.get("n_trials")


def optimize_hyperparameters(
    trainer: Trainer,
    x: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
    categorical_features: Optional[list] = None,
) -> optuna.Study:
    """Run Optuna hyperparameter optimization for the model pipeline."""

    def objective(trial: optuna.Trial) -> float:
        pipeline = trainer.build_pipeline(
            build_xgb_params(
                {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.6, 1.0
                    ),
                }
            ),
            numerical_features=_FEATURES,
            categorical_features=categorical_features or [],
        )

        if groups is not None:
            cv = GroupKFold(n_splits=_N_SPLITS)
            scores = cross_val_score(
                pipeline, x, y, groups=groups, cv=cv, scoring="accuracy"
            )
        else:
            cv = TimeSeriesSplit(n_splits=_N_SPLITS)
            scores = []
            for train_idx, val_idx in cv.split(x):
                x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
                x_val, y_val = x.iloc[val_idx], y.iloc[val_idx]

                resampled = RandomUnderSampler().fit_resample(x_train, y_train)
                x_res, y_res = resampled[:2]
                pipeline.fit(x_res, y_res)
                score = pipeline.score(x_val, y_val)
                scores.append(score)

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=_N_TRIALS)
    return study
