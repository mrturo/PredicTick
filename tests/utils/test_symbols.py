"""Unit tests for the SymbolRepository class and its JSON loading logic."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.symbols import SymbolRepository


def create_temp_repo(tmp_path):
    """Create a SymbolRepository instance with test JSON files."""
    training = tmp_path / "training.json"
    mercado = tmp_path / "mercado.json"
    xtb = tmp_path / "xtb.json"
    correlative = tmp_path / "correlative.json"

    for path, symbols in zip(
        [training, mercado, xtb, correlative],
        [["AAPL", "TSLA"], ["MP1", "MP2"], ["EURUSD"], ["BTC", "ETH"]],
    ):
        path.write_text(json.dumps(symbols))

    return SymbolRepository(
        mercado_pago_path=str(mercado),
        training_path=str(training),
        xtb_path=str(xtb),
        correlative_path=str(correlative),
    )


class TestSymbolRepository(unittest.TestCase):
    """Unit tests for methods in SymbolRepository."""

    def test_load_training_symbols(self):
        """Test loading training symbols from file."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            repo = create_temp_repo(Path(tmpdirname))
            self.assertListEqual(repo.load_training_symbols(), ["AAPL", "TSLA"])

    def test_load_mercado_pago_symbols(self):
        """Test loading mercado pago symbols from file."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            repo = create_temp_repo(Path(tmpdirname))
            self.assertListEqual(repo.load_mercado_pago_symbols(), ["MP1", "MP2"])

    def test_load_xtb_symbols(self):
        """Test loading XTB symbols from file."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            repo = create_temp_repo(Path(tmpdirname))
            self.assertListEqual(repo.load_xtb_symbols(), ["EURUSD"])

    def test_load_correlative_symbols(self):
        """Test loading correlative symbols from file."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            repo = create_temp_repo(Path(tmpdirname))
            self.assertListEqual(repo.load_correlative_symbols(), ["BTC", "ETH"])

    @patch("utils.symbols.Logger.warning")
    def test_load_json_file_not_found(self, mock_warning):
        """Test fallback to empty list if JSON file is missing."""
        repo = SymbolRepository("missing.json", "a.json", "b.json", "c.json")
        result = repo._load_json("nonexistent.json")  # pylint: disable=protected-access
        self.assertEqual(result, [])
        mock_warning.assert_called_once()

    @patch("utils.symbols.Logger.error")
    def test_load_json_decode_error(self, mock_error):
        """Test fallback to empty list if JSON is malformed."""
        with unittest.mock.patch(
            "builtins.open", unittest.mock.mock_open(read_data="{ invalid json }")
        ):
            with patch("os.path.exists", return_value=True):
                repo = SymbolRepository("a", "b", "c", "d")
                result = repo._load_json("bad.json")  # pylint: disable=protected-access
                self.assertEqual(result, [])
                mock_error.assert_called_once()
