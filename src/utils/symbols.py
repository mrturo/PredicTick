"""
Symbols module for managing symbol lists in training and analysis pipelines.

Provides repository classes and utilities to load, update, and filter symbol lists
stored in JSON files, including management of invalid symbols.

Classes:
    Symbols: Loads and stores a unique, sorted list of financial symbols from a JSON file.
    SymbolRepository: Manages categorized symbol lists, filtering out invalid symbols.
"""

from dataclasses import dataclass
from typing import List, Set

from utils.json_manager import JsonManager


@dataclass
class Symbols:
    """
    Load and store a unique sorted list of financial symbols from a JSON file.

    Attributes:
        filepath (str): Path to the JSON file containing the symbol list.
        list (List[str]): Sorted list of unique symbols loaded from the file.
    """

    filepath: str
    list: List[str]

    def __init__(self, filepath: str):
        """
        Initialize the Symbols object.

        Parameters:
            filepath (str): Path to the JSON file containing the symbol list.

        Raises:
            ValueError: If filepath is None, empty or contains only whitespace.
        """
        if not filepath or not isinstance(filepath, str) or filepath.strip() == "":
            raise ValueError(
                "Filepath for Symbols cannot be empty, None, or whitespace."
            )

        self.filepath = filepath
        if not JsonManager.exists(filepath):
            JsonManager.save([], filepath)
            self.list = []
        else:
            self.list = sorted(set(JsonManager.load(filepath)))


class SymbolRepository:
    """
    Repository and interface for accessing categorized symbol lists.

    Manages access to multiple categories of financial symbols while
    filtering out any symbols marked as invalid.
    """

    def __init__(self, filepath: str, invalid_filepath: str):
        """
        Initialize the SymbolRepository.

        Parameters:
            filepath (str): Path to the JSON file with symbol categories.
            invalid_filepath (str): Path to the JSON file with invalid symbols.
        """
        symbols = JsonManager().load(filepath) or {}
        self.correlative = symbols.get("correlative", [])
        self.training = symbols.get("training", [])
        self.prediction_groups = symbols.get("prediction_groups", [])
        self.invalid = Symbols(invalid_filepath)

    def _remove_invalid(self, symbols: list) -> List[str]:
        """
        Remove invalid symbols from a symbol list.

        Parameters:
            symbols (list): List of symbols to filter.

        Returns:
            List[str]: List of valid symbols.
        """
        return sorted(list(set(symbols) - set(self.invalid.list)))

    def get_all_symbols(self) -> List[str]:
        """
        Get all unique, valid symbols across all categories.

        Returns:
            List[str]: List of all valid symbols.
        """
        local_prediction_groups = []
        for prediction_group in self.prediction_groups:
            local_prediction_groups += prediction_group.get("symbols", [])
        all_symbols = list(
            set(self.training + local_prediction_groups + self.correlative)
        )
        return self._remove_invalid(all_symbols)

    def get_correlative_symbols(self) -> List[str]:
        """
        Get valid correlative symbols.

        Returns:
            List[str]: List of valid correlative symbols.
        """
        return self._remove_invalid(self.correlative)

    def get_invalid_symbols(self) -> List[str]:
        """
        Get list of invalid symbols.

        Returns:
            List[str]: List of invalid symbols.
        """
        return self.invalid.list

    def set_invalid_symbols(self, invalid_symbols: Set[str]) -> None:
        """
        Set and save invalid symbols.

        Parameters:
            invalid_symbols (Set[str]): Set of symbols to mark as invalid.

        Returns:
            List[str]: Sorted list of invalid symbols.
        """
        self.invalid.list = sorted(invalid_symbols)
        JsonManager.save(self.invalid.list, self.invalid.filepath)

    def get_training_symbols(self) -> List[str]:
        """
        Get valid training symbols.

        Returns:
            List[str]: List of valid training symbols.
        """
        return self._remove_invalid(self.training)

    def get_all_prediction_group_name(self) -> List[str]:
        """
        Get names of all prediction groups.

        Returns:
            List[str]: List of prediction group names.
        """
        prediction_groups: List[str] = []
        for prediction_group_entry in self.prediction_groups:
            prediction_groups.append(prediction_group_entry.get("name"))
        return prediction_groups

    def get_prediction_group_symbols(self, prediction_group: str) -> List[str]:
        """
        Get valid symbols of a prediction group by name.

        Parameters:
            prediction_group (str): Name of the prediction group.

        Returns:
            List[str]: List of valid symbols for the specified prediction group.
        """
        result: List[str] = []
        for prediction_group_entry in self.prediction_groups:
            if prediction_group_entry.get("name") == prediction_group:
                result = self._remove_invalid(prediction_group_entry.get("symbols", []))
        return result
