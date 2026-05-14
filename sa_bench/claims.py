"""SAU claim loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import Dimension, SAUClaim


def sau_path_for_paper(data_dir: Path, paper_id: str) -> Path:
    direct = data_dir / paper_id / "sau.json"
    nested = data_dir / paper_id / "sau" / "sau.json"
    requirements = data_dir / paper_id / "sau" / "requirements.json"
    for path in (direct, nested, requirements):
        if path.exists():
            return path
    raise FileNotFoundError(f"SAU file not found for paper {paper_id!r} under {data_dir}")


def list_papers(data_dir: Path) -> list[str]:
    papers: list[str] = []
    if not data_dir.exists():
        return papers
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if any((child / rel).exists() for rel in ("sau.json", "sau/sau.json", "sau/requirements.json")):
            papers.append(child.name)
    return papers


def load_claims(path: Path) -> tuple[str | None, list[SAUClaim]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    paper_id = raw.get("paper_id") if isinstance(raw, dict) else None
    claims: list[SAUClaim] = []

    if isinstance(raw, dict) and isinstance(raw.get("requirements"), dict):
        for dim in Dimension:
            claims.extend(_claims_from_items(raw["requirements"].get(dim.value, []), dim))
    elif isinstance(raw, dict) and isinstance(raw.get("requirements"), list):
        for item in raw["requirements"]:
            claims.append(_claim_from_item(item, _dimension_from_item(item)))
    elif isinstance(raw, dict):
        for dim in Dimension:
            claims.extend(_claims_from_items(raw.get(dim.value, []), dim))
    elif isinstance(raw, list):
        for item in raw:
            claims.append(_claim_from_item(item, _dimension_from_item(item)))
    else:
        raise ValueError(f"Unsupported SAU JSON format: {path}")

    if not claims:
        raise ValueError(f"No SAU claims found in {path}")
    return paper_id, claims


def filter_claims(claims: list[SAUClaim], dimensions: list[Dimension]) -> list[SAUClaim]:
    wanted = set(dimensions)
    return [claim for claim in claims if claim.dimension in wanted]


def _claims_from_items(items: Any, dimension: Dimension) -> list[SAUClaim]:
    if not isinstance(items, list):
        raise ValueError(f"{dimension.value} claims must be a list")
    return [_claim_from_item(item, dimension) for item in items]


def _claim_from_item(item: Any, dimension: Dimension) -> SAUClaim:
    if not isinstance(item, dict):
        raise ValueError("SAU claim entry must be an object")
    payload = {
        "id": str(item.get("id") or item.get("claim_id") or ""),
        "claim": str(item.get("claim") or item.get("text") or ""),
        "source": str(item.get("source") or ""),
        "dimension": dimension,
    }
    if not payload["id"] or not payload["claim"]:
        raise ValueError(f"SAU claim missing id or claim text: {item}")
    return SAUClaim.model_validate(payload)


def _dimension_from_item(item: Any) -> Dimension:
    if not isinstance(item, dict):
        raise ValueError("SAU claim entry must be an object")
    raw = str(item.get("type") or item.get("dimension") or "").strip().upper()
    if not raw:
        raise ValueError(f"SAU claim missing dimension/type: {item}")
    return Dimension(raw)
