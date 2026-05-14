"""Regex search."""

from __future__ import annotations

import re
from pathlib import Path

from .common import format_matches, search_lines


def grep_regex(repo_root: Path, pattern: str, *, subpath: str = ".", limit: int = 50) -> str:
    try:
        return format_matches(search_lines(repo_root, pattern, regex=True, subpath=subpath, limit=limit))
    except re.error as exc:
        return f"Error: invalid regex: {exc}"
