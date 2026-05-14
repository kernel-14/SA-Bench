"""In-memory repository index for faster repeated claim searches."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .tools.common import CONFIG_NAME_HINTS, CONFIG_SUFFIXES, Match, iter_text_files, relpath, safe_path


@dataclass
class IndexedFile:
    path: Path
    rel: str
    lines: list[str]


class RepoIndex:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.files: list[IndexedFile] = []
        for path in iter_text_files(self.repo_root):
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            self.files.append(IndexedFile(path=path, rel=relpath(self.repo_root, path), lines=lines))

    def list_files(self, directory: str = ".", *, limit: int = 200) -> str:
        target = safe_path(self.repo_root, directory)
        rows: list[str] = []
        for item in self.files:
            try:
                rel = item.path.relative_to(target)
            except ValueError:
                continue
            if len(rel.parts) == 1:
                rows.append(item.rel)
            elif len(rel.parts) > 1:
                rows.append(rel.parts[0] + "/")
            if len(rows) >= limit:
                rows.append(f"... truncated at {limit} entries")
                break
        return "\n".join(sorted(set(rows))) if rows else "(empty directory)"

    def grep_exact(self, pattern: str, *, limit: int = 50) -> str:
        if not pattern:
            return "(no matches)"
        needle = pattern.lower()
        matches: list[Match] = []
        for item in self.files:
            for idx, line in enumerate(item.lines, start=1):
                if needle in line.lower():
                    matches.append(Match(item.rel, idx, line[:500]))
                    if len(matches) >= limit:
                        return _format_matches(matches)
        return _format_matches(matches)

    def grep_regex(self, pattern: str, *, limit: int = 50) -> str:
        if not pattern:
            return "(no matches)"
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return f"Error: invalid regex: {exc}"
        matches: list[Match] = []
        for item in self.files:
            for idx, line in enumerate(item.lines, start=1):
                if compiled.search(line):
                    matches.append(Match(item.rel, idx, line[:500]))
                    if len(matches) >= limit:
                        return _format_matches(matches)
        return _format_matches(matches)

    def search_config(self, key: str, *, limit: int = 50) -> str:
        if not key:
            return "(no matches)"
        parts = [part for part in re.split(r"[^A-Za-z0-9]+", key) if len(part) >= 2]
        if not parts:
            parts = [key]
        pattern = re.compile("|".join(re.escape(part) for part in parts), re.IGNORECASE)
        matches: list[Match] = []
        for item in self.files:
            lowered_name = Path(item.rel).name.lower()
            suffix = Path(item.rel).suffix.lower()
            if suffix not in CONFIG_SUFFIXES:
                continue
            if suffix == ".py" and not any(hint in lowered_name for hint in CONFIG_NAME_HINTS):
                continue
            for idx, line in enumerate(item.lines, start=1):
                if pattern.search(line):
                    matches.append(Match(item.rel, idx, line[:500]))
                    if len(matches) >= limit:
                        return _format_matches(matches)
        return _format_matches(matches)

    def read_file(self, path: str, *, start: int = 1, end: int | None = None) -> str:
        target = safe_path(self.repo_root, path)
        rel = relpath(self.repo_root, target)
        match = next((item for item in self.files if item.rel == rel), None)
        if match is None:
            return f"Error: file not found: {path}"
        start = max(1, start)
        end = end if end is not None else start + 120
        if end < start:
            end = start
        selected = match.lines[start - 1 : end]
        return "\n".join(f"{idx}: {line}" for idx, line in enumerate(selected, start=start))


def _format_matches(matches: list[Match]) -> str:
    if not matches:
        return "(no matches)"
    return "\n".join(f"{m.file}:{m.line}: {m.text}" for m in matches)
