"""Shared helpers for safe repository-local tools."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "result",
    "results",
    "outputs",
}
TEXT_SUFFIXES = {
    ".py",
    ".ipynb",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".txt",
    ".md",
    ".rst",
    ".sh",
    ".bash",
    ".zsh",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".cu",
    ".m",
    ".jl",
    ".r",
}
CONFIG_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
CONFIG_NAME_HINTS = ("config", "arg", "param", "hyper", "train", "setting")


@dataclass(frozen=True)
class Match:
    file: str
    line: int
    text: str


def safe_path(repo_root: Path, rel: str | os.PathLike[str] = ".") -> Path:
    root = repo_root.resolve()
    target = (root / Path(rel)).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes repository: {rel}")
    return target


def is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    try:
        with path.open("rb") as handle:
            chunk = handle.read(2048)
        return b"\x00" not in chunk
    except OSError:
        return False


def iter_text_files(repo_root: Path, subpath: str = ".") -> list[Path]:
    start = safe_path(repo_root, subpath)
    if start.is_file():
        return [start] if is_probably_text(start) else []
    files: list[Path] = []
    for root, dirs, names in os.walk(start):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".tox")]
        for name in names:
            path = Path(root) / name
            if is_probably_text(path):
                files.append(path)
    return files


def relpath(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def search_lines(
    repo_root: Path,
    pattern: str,
    *,
    regex: bool,
    subpath: str = ".",
    limit: int = 50,
    flags: int = re.IGNORECASE,
) -> list[Match]:
    if not pattern:
        return []
    compiled = re.compile(pattern, flags) if regex else None
    matches: list[Match] = []
    needle = pattern.lower()
    for path in iter_text_files(repo_root, subpath):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines, start=1):
            ok = bool(compiled.search(line)) if compiled else needle in line.lower()
            if ok:
                matches.append(Match(file=relpath(repo_root, path), line=idx, text=line[:500]))
                if len(matches) >= limit:
                    return matches
    return matches


def format_matches(matches: list[Match]) -> str:
    if not matches:
        return "(no matches)"
    return "\n".join(f"{m.file}:{m.line}: {m.text}" for m in matches)


def count_results(output: str) -> int:
    if not output or output == "(no matches)":
        return 0
    return len([line for line in output.splitlines() if line.strip()])
