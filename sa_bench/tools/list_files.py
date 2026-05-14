"""Repository-local directory lister."""

from __future__ import annotations

from pathlib import Path

from .common import SKIP_DIRS, safe_path


def list_files(repo_root: Path, directory: str = ".", *, limit: int = 200) -> str:
    target = safe_path(repo_root, directory)
    if not target.exists():
        return f"Error: path not found: {directory}"
    if target.is_file():
        return f"{directory} (file, {target.stat().st_size} bytes)"
    rows: list[str] = []
    for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if entry.name in SKIP_DIRS:
            continue
        suffix = "/" if entry.is_dir() else ""
        size = "" if entry.is_dir() else f" ({entry.stat().st_size} bytes)"
        rows.append(f"{entry.name}{suffix}{size}")
        if len(rows) >= limit:
            rows.append(f"... truncated at {limit} entries")
            break
    return "\n".join(rows) if rows else "(empty directory)"
