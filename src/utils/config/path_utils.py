"""Path utilities module.

Provides helper functions for constructing normalized file system paths,
ensuring consistent handling of separators and temporary directory usage.
"""

import os
import re
from typing import List


class PathUtils:  # pylint: disable=too-few-public-methods
    """Utility class for building normalized file system paths."""

    _TMP_DIR: str = os.getenv("TMP_DIR", "")

    @staticmethod
    def build(*segments: str) -> str:
        """Build a normalized path under _TMP_DIR, splitting input segments."""

        base_path: str = ""
        tmp_dir: str = PathUtils._TMP_DIR.strip() or ""
        if len(tmp_dir) > 0:
            base_path: str = os.sep.join(
                [p for p in re.split(r"[\\/]", tmp_dir) if p.strip()]
            )
            if tmp_dir.startswith(("\\", "/")):
                base_path = os.sep + base_path

        parts: List[str] = []
        for segment in segments:
            if segment:
                parts.extend(re.split(r"[\\/]", segment))
        return os.path.join(base_path, *parts)
