"""Unit tests for the Symbols and SymbolRepository utilities.

This module validates the loading, deduplication, categorization, and
filtering logic applied to symbol lists used in training, prediction,
and analysis workflows.
"""

from unittest.mock import MagicMock, patch

import pytest

from utils.symbols import SymbolRepository, Symbols


@pytest.fixture
def symbol_data():
    """Return a mock symbol dataset categorized by use-case and prediction_group source."""
    return {
        "correlative": ["AAPL", "TSLA"],
        "training": ["MSFT", "GOOG"],
        "prediction_groups": [
            {"name": "prediction_group-1", "symbols": ["MELI", "BMA"]},
            {"name": "prediction_group-2", "symbols": ["XOM", "CVX"]},
        ],
    }


@pytest.fixture
def invalid_symbols():
    """Return a mock list of symbols considered invalid for analysis or trading."""
    return ["TSLA", "BMA"]


@patch("utils.symbols.JsonManager")
def test_symbols_load_existing(mock_json_manager):
    """Verify correct loading and deduplication of symbols from an existing JSON file."""
    mock_json_manager.exists.return_value = True
    mock_json_manager.load.return_value = ["MSFT", "AAPL", "MSFT"]
    symbols = Symbols("dummy_path.json")
    if symbols.list != ["AAPL", "MSFT"]:
        raise AssertionError("Symbols list should be sorted and unique")


@patch("utils.symbols.JsonManager")
def test_symbols_initialize_empty(mock_json_manager):
    """Verify a new Symbols instance initializes an empty list if the file does not exist."""
    mock_json_manager.exists.return_value = False
    mock_json_manager.save = MagicMock()
    symbols = Symbols("new_file.json")
    if symbols.list != []:
        raise AssertionError(
            "New Symbols instance should be initialized with empty list"
        )


@patch("utils.symbols.JsonManager")
def test_repository_get_all_symbols(mock_json_manager, request):
    """Verify get_all_symbols filters out all invalid symbols correctly."""
    data = request.getfixturevalue("symbol_data")
    invalid = request.getfixturevalue("invalid_symbols")

    mock_json_manager.side_effect = lambda: MagicMock(load=MagicMock(return_value=data))
    mock_json_manager.exists.return_value = True
    mock_json_manager.load.return_value = invalid
    repo = SymbolRepository("symbols.json", "invalid.json")
    all_symbols = repo.get_all_symbols()
    expected = sorted(set(["AAPL", "MSFT", "GOOG", "MELI", "XOM", "CVX"]))
    if all_symbols != expected:
        raise AssertionError(f"Expected {expected}, got {all_symbols}")


@patch("utils.symbols.JsonManager")
def test_repository_get_by_category(mock_json_manager, request):
    """Verify accessors in SymbolRepository correctly exclude invalid symbols."""
    data = request.getfixturevalue("symbol_data")
    invalid = request.getfixturevalue("invalid_symbols")

    mock_json_manager.side_effect = lambda: MagicMock(load=MagicMock(return_value=data))
    mock_json_manager.exists.return_value = True
    mock_json_manager.load.return_value = invalid
    repo = SymbolRepository("symbols.json", "invalid.json")

    if repo.get_correlative_symbols() != ["AAPL"]:
        raise AssertionError("Correlative symbols filtering failed")
    if repo.get_training_symbols() != ["GOOG", "MSFT"]:
        raise AssertionError("Training symbols filtering failed")
    if repo.get_prediction_group_symbols("prediction_group-1") != ["MELI"]:
        raise AssertionError("Group-1 symbols filtering failed")
    if repo.get_prediction_group_symbols("prediction_group-2") != ["CVX", "XOM"]:
        raise AssertionError("Group-2 symbols filtering failed")


@patch("utils.symbols.JsonManager")
def test_repository_set_invalid(mock_json_manager):
    """Verify SymbolRepository correctly sets and persists the list of invalid symbols."""
    mock_json_manager.exists.return_value = True
    mock_json_manager.load.return_value = []
    mock_json_manager.save = MagicMock()
    repo = SymbolRepository("symbols.json", "invalid.json")
    repo.set_invalid_symbols(set(["AAPL", "TSLA"]))
    mock_json_manager.save.assert_called_once_with(["AAPL", "TSLA"], "invalid.json")


@pytest.fixture
def mock_symbol_repo_instance():
    """Fixture to simulate SymbolRepository with invalid symbols."""
    repo = SymbolRepository("dummy_path_valid.json", "dummy_path_invalid.json")
    repo.invalid = MagicMock()
    repo.invalid.list = ["SYM1", "SYM2", "SYM3"]
    return repo


# pylint: disable=redefined-outer-name
def test_get_invalid_symbols(mock_symbol_repo_instance):
    """Verify get_invalid_symbols returns the expected list of invalid symbols."""
    result = mock_symbol_repo_instance.get_invalid_symbols()
    if result != ["SYM1", "SYM2", "SYM3"]:
        raise AssertionError(f"Expected ['SYM1', 'SYM2', 'SYM3'], got {result}")


@pytest.mark.parametrize("invalid_path", [None, "", "   ", 123, [], {}])
def test_symbols_init_invalid_filepath(invalid_path):
    """
    Verify Symbols raises ValueError for None, empty, whitespace or non-string filepaths.
    """
    with pytest.raises(
        ValueError, match="Filepath for Symbols cannot be empty, None, or whitespace."
    ):
        Symbols(invalid_path)
