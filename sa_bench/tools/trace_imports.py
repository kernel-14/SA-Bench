"""Lightweight import and dependent tracing for Python files."""

from __future__ import annotations

import ast
from pathlib import Path

from .common import iter_text_files, relpath, safe_path


def trace_imports(repo_root: Path, file: str) -> str:
    target = safe_path(repo_root, file)
    if not target.exists() or not target.is_file():
        return f"Error: file not found: {file}"
    try:
        tree = ast.parse(target.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        return f"Error: cannot parse imports: {exc}"

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            imports.append(prefix)

    stem = target.stem
    dependents: list[str] = []
    for path in iter_text_files(repo_root):
        if path == target or path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if f"import {stem}" in text or f"from {stem}" in text or f".{stem} import" in text:
            dependents.append(relpath(repo_root, path))

    lines = ["imports:"]
    lines.extend(f"  {item}" for item in sorted(set(imports))[:80])
    lines.append("dependents:")
    lines.extend(f"  {item}" for item in sorted(set(dependents))[:80])
    return "\n".join(lines)
