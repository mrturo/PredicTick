"""CustomTrainer — Implementation of the Trainer abstract class using modular components."""

from typing import Optional

import numpy as np
import pandas as pd

from training import data_preparation as dp
from training.base_trainer import Trainer, TrainParameters
from training.hyperparameter_optimization import optimize_hyperparameters
from utils.logger import Logger


class CustomTrainer(Trainer):
    """Concrete Trainer implementation for financial models using engineered features."""

    _SYMBOL_COLUMN = "symbol"
    _DATETIME_COLUMN = "datetime"
    _TARGET_COLUMN = "target"

    def prepare_data(self) -> tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
        """Loads, processes, and prepares the dataset for model training."""
        Logger.info("🔄 Loading and processing multiactive dataset...")
        df = dp.load_combined_dataset()
        df = dp.engineer_cross_features(df, TrainParameters.CORRELATIVE_SYMBOLS)

        # Selección de features de cruce válidos
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

        # Validación por cantidad de NaNs por fila
        min_valid_cols = int(len(feature_cols) * 0.8)
        df = df[df[feature_cols].notna().sum(axis=1) >= min_valid_cols]
        if df.empty:
            raise ValueError("No data left after filtering NaNs by row threshold.")

        df = df[df[CustomTrainer._TARGET_COLUMN].notna()]
        if df.empty:
            raise ValueError("No data left after dropping rows with NaN target.")

        # Eliminar símbolos con pocas muestras
        symbol_counts = df[CustomTrainer._SYMBOL_COLUMN].value_counts()
        valid_symbols = symbol_counts[symbol_counts >= TrainParameters.N_SPLITS].index
        df = df[df[CustomTrainer._SYMBOL_COLUMN].isin(valid_symbols)]
        if len(valid_symbols) < TrainParameters.N_SPLITS:
            raise ValueError(
                f"Not enough symbols with at least {TrainParameters.N_SPLITS} samples for "
                f"GroupKFold (found {len(valid_symbols)}). Reduce n_splits or provide more data."
            )
        if df.empty:
            raise ValueError("No symbols with enough data to support GroupKFold.")

        x = df[
            TrainParameters.FEATURES
            + self.cross_features
            + [CustomTrainer._SYMBOL_COLUMN]
        ]
        y = df[CustomTrainer._TARGET_COLUMN].astype(int)
        groups = df[CustomTrainer._SYMBOL_COLUMN]

        return x, y, groups

    def run(self) -> None:
        """Execute full training flow."""
        Logger.info("Preparing data...")
        x, y, groups = self.prepare_data()
        x = dp.clean_features(x)
        if x.isnull().all().any():
            raise ValueError(
                "Al menos una columna de X está completamente vacía (todo NaN)."
            )
        if (x.nunique() <= 1).any():
            raise ValueError("Al menos una columna de X es constante.")

        if groups is not None and not isinstance(groups, pd.Series):
            groups = pd.Series(groups, index=x.index, name=CustomTrainer._SYMBOL_COLUMN)

        Logger.info("Optimizing hyperparameters...")
        self.study = optimize_hyperparameters(self, x, y, groups=groups)

        Logger.success(f"Best trial found: {self.study.best_trial.number}")
        for k, v in self.study.best_trial.params.items():
            Logger.debug(f"  {k}: {v}")

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
