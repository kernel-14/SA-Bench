"""Exact string search."""

from __future__ import annotations

from pathlib import Path

from .common import format_matches, search_lines


def grep_exact(repo_root: Path, pattern: str, *, subpath: str = ".", limit: int = 50) -> str:
    return format_matches(search_lines(repo_root, pattern, regex=False, subpath=subpath, limit=limit))
