"""Evidence classification and lightweight guardrails."""

from __future__ import annotations

import re

from .types import EvidenceItem


IMPLEMENTATION_KINDS = {"code", "config"}
SUPPORTING_KINDS = {"docs", "test"}


def classify_path(path: str) -> str:
    lowered = path.lower()
    name = lowered.rsplit("/", 1)[-1]
    if lowered.endswith(("_test.py", ".test.py")) or "/tests/" in lowered or name.startswith("test"):
        return "test"
    if lowered.endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".cpp", ".cc", ".c", ".h", ".hpp", ".cu", ".jl")):
        return "code"
    if lowered.endswith((".yaml", ".yml", ".toml", ".ini", ".cfg")) or "config" in name or "args" in name:
        return "config"
    if lowered.endswith((".md", ".rst", ".txt")) or "readme" in name or "docs" in lowered:
        return "docs"
    return "other"


def is_implementation_kind(kind: str) -> bool:
    return kind in IMPLEMENTATION_KINDS


def only_docs_or_tests(kinds: list[str]) -> bool:
    return bool(kinds) and all(kind in SUPPORTING_KINDS for kind in kinds)


def has_direct_implementation(kinds: list[str]) -> bool:
    return any(kind in IMPLEMENTATION_KINDS for kind in kinds)


def direct_evidence_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    return [item for item in items if item.kind in IMPLEMENTATION_KINDS]


def rank_evidence_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    kind_rank = {"code": 0, "config": 1, "other": 2, "docs": 3, "test": 4}
    relevance_rank = {"direct": 0, "semantic": 1, "config": 1, "supporting": 2, "candidate": 3, "weak": 4}
    return sorted(
        items,
        key=lambda item: (
            kind_rank.get(item.kind, 5),
            relevance_rank.get(item.relevance, 5),
            item.file,
            item.lines,
        ),
    )


def is_stub_like_snippet(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "placeholder",
        "stub",
        "todo",
        "to do",
        "replace with",
        "not implemented",
        "skeleton",
        "dummy",
        "pass #",
        "np.zeros_like",
        "zeros_like",
        "return 0",
        "return none",
    )
    if any(marker in lowered for marker in markers):
        return True
    code_lines = _effective_code_lines(lowered)
    if len(code_lines) <= 1 and any(re.search(r"\b(def|class)\b", line) for line in code_lines):
        return True
    # Heuristic: short non-executable fragments that are mostly comments/docstrings are weak.
    if len(code_lines) <= 3 and sum(bool(re.search(r"\b(def|class|for|while|return|if)\b", line)) for line in code_lines) <= 1:
        return True
    return False


def is_stub_like_item(item: EvidenceItem) -> bool:
    return is_stub_like_snippet(item.snippet)


def _effective_code_lines(text: str) -> list[str]:
    lines: list[str] = []
    in_docstring = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        triple_count = line.count('"""') + line.count("'''")
        if triple_count:
            if triple_count % 2 == 1:
                in_docstring = not in_docstring
            if line.startswith(('"""', "'''")) or line.endswith(('"""', "'''")):
                continue
        if in_docstring:
            continue
        lines.append(line)
    return lines
