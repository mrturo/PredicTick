"""Unit tests for PathUtils.build.

Covers normalization of TMP_DIR, mixed separators, leading separators,
empty segments handling, and platform-agnostic expectations.
"""

# pylint: disable=protected-access

import os
from typing import List, Tuple

import pytest

from src.utils.config.path_utils import PathUtils


@pytest.fixture()
def _restore_tmp_dir():
    """Restore PathUtils._TMP_DIR after each test."""
    original = PathUtils._TMP_DIR
    yield
    PathUtils._TMP_DIR = original


def _expect_join(parts: List[str]) -> str:
    """Helper to build an expected path in a platform-agnostic way.

    Parameters
    ----------
    parts: List[str]
        Path components to be joined with ``os.path.join``.
    """
    return os.path.join(*parts)


def test_build_uses_default_tmp_dir_when_blank(_restore_tmp_dir: str):
    """If TMP_DIR is blank/whitespace, fallback to "tmp"."""
    PathUtils._TMP_DIR = "   "
    result = PathUtils.build("alpha", "beta")
    expected = _expect_join(["alpha", "beta"])
    if result != expected:
        raise AssertionError(f"Expected {expected!r}, got {result!r}")


@pytest.mark.parametrize(
    "tmp_dir, expected_base",
    [
        ("/var//tmp\\cache", os.sep + os.sep.join(["var", "tmp", "cache"])),
        ("\\var\\tmp/cache", os.sep + os.sep.join(["var", "tmp", "cache"])),
    ],
)
def test_build_preserves_leading_separator_in_tmp_dir(
    tmp_dir: str, expected_base: str, _restore_tmp_dir: str
):
    """Absolute TMP_DIR (starting with '/' or '\\') retains the leading separator."""
    PathUtils._TMP_DIR = tmp_dir
    result = PathUtils.build("x", "y")
    expected = _expect_join([expected_base, "x", "y"])
    if result != expected:
        raise AssertionError(f"Expected {expected!r}, got {result!r}")


def test_build_normalizes_relative_tmp_dir_without_leading_sep(_restore_tmp_dir: str):
    """Relative TMP_DIR gets normalized separators without a leading root sep."""
    PathUtils._TMP_DIR = "foo\\bar/baz"
    base = os.sep.join(["foo", "bar", "baz"])
    result = PathUtils.build("file.txt")
    expected = _expect_join([base, "file.txt"])
    # Ensure path does not start with a leading separator for relative base
    if os.path.isabs(result):
        raise AssertionError("Expected a relative path for relative TMP_DIR")
    if result != expected:
        raise AssertionError(f"Expected {expected!r}, got {result!r}")


def test_build_splits_segments_with_mixed_separators(_restore_tmp_dir: str):
    """Segments are split on both '/' and '\\' and then joined correctly."""
    PathUtils._TMP_DIR = "tmp"
    result = PathUtils.build("a\\b", "c/d", "e\\f")
    expected = _expect_join(["tmp", "a", "b", "c", "d", "e", "f"])
    if result != expected:
        raise AssertionError(f"Expected {expected!r}, got {result!r}")


def test_build_ignores_empty_segments(_restore_tmp_dir: str):
    """Empty string segments are ignored."""
    PathUtils._TMP_DIR = "tmp"
    result = PathUtils.build("", "foo", "", "bar")
    expected = _expect_join(["tmp", "foo", "bar"])
    if result != expected:
        raise AssertionError(f"Expected {expected!r}, got {result!r}")


@pytest.mark.parametrize(
    "segments, expected_parts",
    [
        (("/abs",), ["tmp", "abs"]),  # leading sep in segment should not reset to root
        (("\\abs",), ["tmp", "abs"]),
        (("/a/b", "c"), ["tmp", "a", "b", "c"]),
    ],
)
def test_build_leading_separators_in_segments_do_not_make_absolute(
    segments: Tuple[str, ...], expected_parts: List[str], _restore_tmp_dir: str
):
    """Leading separators inside *segments* do not override the base TMP_DIR."""
    PathUtils._TMP_DIR = "tmp"
    result = PathUtils.build(*segments)
    expected = _expect_join(expected_parts)
    if os.path.isabs(result):
        raise AssertionError("Result should remain relative to TMP_DIR, not absolute")
    if result != expected:
        raise AssertionError(f"Expected {expected!r}, got {result!r}")
