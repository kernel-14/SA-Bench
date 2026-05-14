"""Aggregation and report shaping."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .types import Dimension, DimensionResult, SAUScoreResult, ScoredClaim


def aggregate_result(
    *,
    paper_id: str,
    repo_path: Path,
    model: str,
    scored_claims: dict[Dimension, list[ScoredClaim]],
) -> SAUScoreResult:
    dimensions: dict[str, DimensionResult] = {}
    dim_scores: list[float] = []
    for dim in Dimension:
        claims = scored_claims.get(dim, [])
        if not claims:
            continue
        score = round(sum(claim.score for claim in claims) / len(claims), 4)
        dimensions[dim.value] = DimensionResult(score=score, total_claims=len(claims), claims=claims)
        dim_scores.append(score)
    overall = round(sum(dim_scores) / len(dim_scores), 4) if dim_scores else 0.0
    return SAUScoreResult(
        paper_id=paper_id,
        repo_path=str(repo_path),
        scored_at=datetime.now(timezone.utc),
        model=model,
        dimensions=dimensions,
        overall_score=overall,
    )
