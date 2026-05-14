"""Config-oriented search for common Python/YAML/JSON/TOML/INI files."""

from __future__ import annotations

import re
from pathlib import Path

from .common import CONFIG_NAME_HINTS, CONFIG_SUFFIXES, Match, format_matches, iter_text_files, relpath


def search_config(repo_root: Path, key: str, *, limit: int = 50) -> str:
    if not key:
        return "(no matches)"
    key_parts = [part for part in re.split(r"[^A-Za-z0-9]+", key) if len(part) >= 2]
    if not key_parts:
        key_parts = [key]
    pattern = re.compile("|".join(re.escape(part) for part in key_parts), re.IGNORECASE)
    matches: list[Match] = []
    for path in iter_text_files(repo_root):
        lowered_name = path.name.lower()
        if path.suffix.lower() not in CONFIG_SUFFIXES:
            continue
        if not any(hint in lowered_name for hint in CONFIG_NAME_HINTS) and path.suffix.lower() == ".py":
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for idx, line in enumerate(lines, start=1):
            if pattern.search(line):
                matches.append(Match(relpath(repo_root, path), idx, line[:500]))
                if len(matches) >= limit:
                    return format_matches(matches)
    return format_matches(matches)
