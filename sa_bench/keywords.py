"""Keyword generation for deterministic SAU evidence collection."""

from __future__ import annotations

import re

from .types import Dimension, SAUClaim


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "where",
    "which",
    "into",
    "during",
    "against",
    "between",
    "through",
    "across",
    "section",
    "appendix",
    "table",
    "figure",
    "claim",
    "model",
    "method",
    "using",
    "uses",
    "used",
    "must",
    "step",
}

DIMENSION_HINTS = {
    Dimension.D1: ["lr", "learning_rate", "batch", "epoch", "weight_decay", "beta", "config"],
    Dimension.D2: ["loss", "forward", "norm", "score", "sigma", "drift", "gradient"],
    Dimension.D3: ["dataset", "dataloader", "train", "eval", "metric", "epoch", "augmentation"],
    Dimension.D4: ["pipeline", "phase", "trainer", "runner", "forward", "step", "schedule"],
}


def claim_queries(claim: SAUClaim, *, max_queries: int = 14) -> list[str]:
    original_text = claim.claim
    text = _normalize_symbols(original_text)
    queries: list[str] = []

    queries.extend(_resolution_queries(original_text))

    for number in _numbers(text):
        queries.append(number)
        if "e-" in number.lower():
            try:
                queries.append(str(float(number)))
            except ValueError:
                pass

    for phrase in re.findall(r"[A-Z][A-Za-z0-9]*(?:[- ][A-Z0-9][A-Za-z0-9]*){1,4}", text):
        if len(phrase) <= 80:
            queries.append(phrase.strip())

    tokens = _tokens(text)
    for token in tokens[:10]:
        queries.append(token)

    if claim.dimension == Dimension.D2:
        queries.extend(_formula_queries(text))

    queries.extend(DIMENSION_HINTS[claim.dimension])
    return _dedupe(queries)[:max_queries]


def claim_regexes(claim: SAUClaim, *, max_queries: int = 8) -> list[str]:
    original_text = claim.claim
    tokens = _tokens(_normalize_symbols(original_text))
    regexes: list[str] = []
    for width, height in _resolution_pairs(original_text):
        regexes.append(rf"{width}\s*[x×,]\s*{height}|{width}.*{height}")
    if len(tokens) >= 2:
        for idx in range(min(4, len(tokens) - 1)):
            a, b = map(re.escape, tokens[idx : idx + 2])
            regexes.append(f"{a}[_\\- ]*{b}|{a}.*{b}")
    for number in _numbers(claim.claim):
        regexes.append(re.escape(number).replace("\\.", r"\."))
    if claim.dimension == Dimension.D1:
        regexes.append(r"(lr|learning[_-]?rate|batch[_-]?size|weight[_-]?decay|epoch|seed)")
    elif claim.dimension == Dimension.D2:
        regexes.append(r"(loss|norm|sigma|alpha|beta|score|gradient|drift|control)")
    elif claim.dimension == Dimension.D3:
        regexes.append(r"(dataset|dataloader|metric|evaluate|ablation|augmentation|epoch)")
    else:
        regexes.append(r"(pipeline|phase|trainer|runner|workflow|schedule|step)")
    return _dedupe(regexes)[:max_queries]


def _normalize_symbols(text: str) -> str:
    replacements = {
        "β": "beta",
        "α": "alpha",
        "σ": "sigma",
        "λ": "lambda",
        "ε": "epsilon",
        "θ": "theta",
        "∇": "grad",
        "×": "x",
        "−": "-",
        "–": "-",
        "—": "-",
        "²": "2",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _numbers(text: str) -> list[str]:
    raw = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![A-Za-z])", text, re.IGNORECASE)
    numbers: list[str] = []
    for item in raw:
        if len(item) == 1:
            continue
        numbers.append(item)
    return numbers


def _resolution_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for match in re.finditer(r"\b(\d{2,5})\s*[x×]\s*(\d{2,5})\b", text, re.IGNORECASE):
        pairs.append((match.group(1), match.group(2)))
    return pairs


def _resolution_queries(text: str) -> list[str]:
    queries: list[str] = []
    for width, height in _resolution_pairs(text):
        queries.extend([f"{width}x{height}", f"{width}×{height}", width, height])
    return queries


def _tokens(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", text)
    scored: list[tuple[int, str]] = []
    for token in tokens:
        lowered = token.lower().strip("_")
        if lowered in STOPWORDS or len(lowered) < 3:
            continue
        score = len(lowered)
        if any(ch.isupper() for ch in token):
            score += 4
        if "_" in token or "-" in token:
            score += 3
        scored.append((score, lowered))
    return [token for _, token in sorted(scored, reverse=True)]


def _formula_queries(text: str) -> list[str]:
    queries = []
    for symbol in ("sigma", "alpha", "beta", "epsilon", "lambda", "score", "norm", "grad"):
        if symbol in text.lower():
            queries.append(symbol)
    return queries


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
