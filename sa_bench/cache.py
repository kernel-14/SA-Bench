"""Small JSON cache for resumable SAU scoring."""

from __future__ import annotations

import json
from pathlib import Path

from .types import ScoredClaim


class ScoreCache:
    def __init__(self, cache_dir: Path | None) -> None:
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, paper_id: str, claim_id: str) -> ScoredClaim | None:
        path = self._path(paper_id, claim_id)
        if path is None or not path.exists():
            return None
        return ScoredClaim.model_validate_json(path.read_text(encoding="utf-8"))

    def set(self, paper_id: str, claim: ScoredClaim) -> None:
        path = self._path(paper_id, claim.id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(claim.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")

    def _path(self, paper_id: str, claim_id: str) -> Path | None:
        if self.cache_dir is None:
            return None
        safe_claim = claim_id.replace("/", "_")
        return self.cache_dir / paper_id / f"{safe_claim}.json"
