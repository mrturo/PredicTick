"""Global test configuration for pytest."""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


@pytest.fixture
def symbol_metadata():
    """Fixture that returns static metadata for the AAPL symbol."""
    return {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "type": "Equity",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "currency": "USD",
        "exchange": "NASDAQ",
        "schedule": {},
    }
