"""CustomTrainer — Implementation of the ``Trainer`` abstract class.

using modular components.
"""

from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.utils.io.logger import Logger
from training import data_preparation as dp
from training.base_trainer import Trainer, TrainParameters
from training.hyperparameter_optimization import optimize_hyperparameters


class CustomTrainer(Trainer):
    """Concrete ``Trainer`` implementation for financial models.

    that rely on engineered features.
    """

    _SYMBOL_COLUMN = "symbol"
    _DATETIME_COLUMN = "datetime"
    _TARGET_COLUMN = "target"

    def prepare_data(self) -> tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
        """Load, process, and prepare the dataset for model training."""
        Logger.info("🔄 Loading and processing multi-asset dataset...")
        df = dp.load_combined_dataset()
        df = dp.engineer_cross_features(df, TrainParameters.CORRELATIVE_SYMBOLS)
        self.cross_features = dp.get_valid_cross_features(
            df, TrainParameters.CORRELATIVE_SYMBOLS
        )
        Logger.info(f"🧩 Initial cross features: {self.cross_features}")
        self.cross_features = dp.filter_valid_cross_features(df, self.cross_features)
        Logger.info(f"✅ Final cross features: {self.cross_features}")
        feature_cols = (
            TrainParameters.FEATURES
            + self.cross_features
            + [CustomTrainer._TARGET_COLUMN]
        )
        df = df[
            [
                col
                for col in feature_cols + [CustomTrainer._SYMBOL_COLUMN]
                if col in df.columns
            ]
        ]
        min_valid_cols = int(len(feature_cols) * 0.8)
        df = df[df[feature_cols].notna().sum(axis=1) >= min_valid_cols]
        if df.empty:
            raise ValueError(
                "No data left after filtering out rows that violate the NaN threshold."
            )
        df = df[df[CustomTrainer._TARGET_COLUMN].notna()]
        if df.empty:
            raise ValueError(
                "No data left after dropping rows with a NaN target value."
            )
        symbol_counts = df[CustomTrainer._SYMBOL_COLUMN].value_counts()
        valid_symbols = symbol_counts[symbol_counts >= TrainParameters.N_SPLITS].index
        df = df[df[CustomTrainer._SYMBOL_COLUMN].isin(valid_symbols)]
        if len(valid_symbols) < TrainParameters.N_SPLITS:
            raise ValueError(
                f"Found only {len(valid_symbols)} symbols with at least "
                f"{TrainParameters.N_SPLITS} samples — not enough for GroupKFold. "
                "Reduce ``n_splits`` or provide more data."
            )
        if df.empty:
            raise ValueError("No symbol contains enough data to support GroupKFold.")
        x = df[
            TrainParameters.FEATURES
            + self.cross_features
            + [CustomTrainer._SYMBOL_COLUMN]
        ]
        y = df[CustomTrainer._TARGET_COLUMN].astype(int)
        groups = df[CustomTrainer._SYMBOL_COLUMN]
        return x, y, groups

    def run(self) -> None:
        """Execute the full training pipeline."""
        Logger.info("Preparing data...")
        x, y, groups = self.prepare_data()
        x = dp.clean_features(x)
        if x.isnull().all().any():
            raise ValueError(
                "At least one column in ``X`` is completely empty (all NaN)."
            )
        if (x.nunique() <= 1).any():
            raise ValueError("At least one column in ``X`` is constant.")
        if groups is not None and not isinstance(groups, pd.Series):
            groups = pd.Series(groups, index=x.index, name=CustomTrainer._SYMBOL_COLUMN)
        Logger.info("Optimizing hyper-parameters...")
        self.study = optimize_hyperparameters(self, x, y, groups=groups)
        Logger.success(f"Best trial found: {self.study.best_trial.number}")
        for key, val in self.study.best_trial.params.items():
            Logger.debug(f"  {key}: {val}")
        Logger.info("Training final model...")
        self.model, self.scaler = self.train_final_model(
            x, y, self.study.best_trial.params
        )
        Logger.success("Saving artifacts...")
        self.save_artifacts()
        Logger.success("Training pipeline completed.")


if __name__ == "__main__":
    trainer = CustomTrainer()
    trainer.run()
