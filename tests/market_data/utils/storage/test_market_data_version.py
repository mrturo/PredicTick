"""Unit tests for MarketDataVersion."""

# pylint: disable=protected-access

from typing import Any, Optional

import pandas as pd  # type: ignore
import pytest  # type: ignore

from src.market_data.utils.storage.market_data_version import MarketDataVersion


@pytest.mark.parametrize(
    "data_obj,timestamp,file_id",
    [
        ({"price": 42.0}, pd.Timestamp("2025-01-01 12:00:00"), "file_001"),
        ([1, 2, 3], pd.Timestamp("2024-07-15"), "abc123"),
        ("raw-bytes", None, None),  # optional fields omitted
    ],
)
def test_market_data_version_initialization(
    data_obj: Any, timestamp: Optional[pd.Timestamp], file_id: Optional[str]
):
    """Verify that all fields are stored exactly as provided."""
    version = MarketDataVersion(data=data_obj, timestamp=timestamp, file_id=file_id)
    if version.data is not data_obj:
        raise AssertionError("Field 'data' was not set correctly")
    if version.timestamp is not timestamp:
        raise AssertionError("Field 'timestamp' was not set correctly")
    if version.file_id != file_id:
        raise AssertionError("Field 'file_id' was not set correctly")


def test_market_data_version_equality():
    """Two instances with identical field values must compare equal."""
    left = MarketDataVersion(
        data={"a": 1}, timestamp=pd.Timestamp("2025-06-30"), file_id="xyz"
    )
    right = MarketDataVersion(
        data={"a": 1}, timestamp=pd.Timestamp("2025-06-30"), file_id="xyz"
    )
    if left != right:
        raise AssertionError("Instances with identical content should compare equal")


def test_market_data_version_repr_contains_fields():
    """__repr__ must include the class name and all field names."""
    obj = MarketDataVersion(data=[], timestamp=None, file_id="fid")
    text = repr(obj)
    for fragment in ("MarketDataVersion", "data", "timestamp", "file_id"):
        if fragment not in text:
            raise AssertionError(f"'{fragment}' not found in __repr__: {text}")
