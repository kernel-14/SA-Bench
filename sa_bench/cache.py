"""Small JSON cache for resumable SAU scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .tools.common import iter_text_files, relpath
from .types import ScoredClaim


def repo_fingerprint(repo_path: Path) -> str:
    """Content hash of the files the evidence search can see.

    Scores depend on the repository content, so the cache must be keyed by it: otherwise
    scoring a second repository (another generator, or a later version of the same one)
    for the same paper silently returns the first repository's scores.
    """
    root = Path(repo_path).resolve()
    digest = hashlib.sha256()
    for path in sorted(iter_text_files(root), key=lambda p: relpath(root, p)):
        digest.update(relpath(root, path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


class ScoreCache:
    def __init__(self, cache_dir: Path | None) -> None:
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, paper_id: str, claim_id: str, repo_key: str = "") -> ScoredClaim | None:
        path = self._path(paper_id, claim_id, repo_key)
        if path is None or not path.exists():
            return None
        return ScoredClaim.model_validate_json(path.read_text(encoding="utf-8"))

    def set(self, paper_id: str, claim: ScoredClaim, repo_key: str = "") -> None:
        path = self._path(paper_id, claim.id, repo_key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(claim.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")

    def _path(self, paper_id: str, claim_id: str, repo_key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        safe_claim = claim_id.replace("/", "_")
        base = self.cache_dir / paper_id
        if repo_key:
            base = base / repo_key
        return base / f"{safe_claim}.json"
