"""
SymbolRepository — Utility class for loading symbol lists used in training and analysis pipelines.

Supports loading symbols for training, XTB, Mercado Pago, and correlative models from JSON files.
"""

import json
import os

from utils.logger import Logger


class SymbolRepository:
    """Loads symbol lists from configured file paths."""

    def __init__(
        self,
        mercado_pago_path: str,
        training_path: str,
        xtb_path: str,
        correlative_path: str,
    ):
        self.training_path = training_path
        self.mercado_pago_path = mercado_pago_path
        self.xtb_path = xtb_path
        self.correlative_path = correlative_path

    @staticmethod
    def _load_json(filepath: str):
        """Load a JSON file and return its contents."""
        if not os.path.exists(filepath):
            Logger.warning(f"Symbol file not found: {filepath}")
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            Logger.error(f"Error loading symbol file {filepath}: {e}")
            return []

    def load_training_symbols(self):
        """Load training symbols from configured file."""
        return self._load_json(self.training_path)

    def load_mercado_pago_symbols(self):
        """Load Mercado Pago symbols from configured file."""
        return self._load_json(self.mercado_pago_path)

    def load_xtb_symbols(self):
        """Load XTB symbols from configured file."""
        return self._load_json(self.xtb_path)

    def load_correlative_symbols(self):
        """Load correlative symbols from configured file."""
        return self._load_json(self.correlative_path)
