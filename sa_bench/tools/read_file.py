"""Repository-local file reader."""

from __future__ import annotations

from pathlib import Path

from .common import safe_path


def read_file(repo_root: Path, path: str, *, start: int = 1, end: int | None = None) -> str:
    target = safe_path(repo_root, path)
    if not target.exists():
        return f"Error: file not found: {path}"
    if not target.is_file():
        return f"Error: not a file: {path}"
    start = max(1, start)
    end = end if end is not None else start + 120
    if end < start:
        end = start
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = lines[start - 1 : end]
    return "\n".join(f"{idx}: {line}" for idx, line in enumerate(selected, start=start))
