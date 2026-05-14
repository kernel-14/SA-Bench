"""SAU scoring pipeline orchestration."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .aggregator import aggregate_result
from .cache import ScoreCache
from .cc_agent import ClaudeCodeAgent
from .claims import filter_claims, load_claims, sau_path_for_paper
from .judge import judge_claim_batch
from .types import Dimension, EvidenceReport, PipelineConfig, SAUClaim, SAUScoreResult, ScoredClaim


class SAUScoringPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.cache = ScoreCache(self.config.cache_dir)

    def score_paper(
        self,
        *,
        paper_id: str,
        repo_path: str | Path,
        dimensions: list[Dimension] | None = None,
    ) -> SAUScoreResult:
        repo = Path(repo_path).resolve()
        if not repo.exists() or not repo.is_dir():
            raise FileNotFoundError(f"repo_path must be an existing directory: {repo}")
        sau_path = sau_path_for_paper(self.config.data_dir, paper_id)
        loaded_paper_id, claims = load_claims(sau_path)
        actual_paper_id = loaded_paper_id or paper_id
        dims = dimensions or list(Dimension)
        claims = filter_claims(claims, dims)
        if not claims:
            raise ValueError(f"No claims found for dimensions {[dim.value for dim in dims]} in {sau_path}")

        scored_by_dim: dict[Dimension, list[ScoredClaim]] = {dim: [] for dim in dims}
        agent = ClaudeCodeAgent(repo, self.config)
        pending: list[tuple[SAUClaim, EvidenceReport]] = []
        for claim in claims:
            cached = self.cache.get(actual_paper_id, claim.id)
            if cached is not None:
                scored_by_dim[claim.dimension].append(cached)
                continue
            report = agent.collect_evidence(claim)
            pending.append((claim, report))

        judged = judge_claim_batch(pending, config=self.config)
        judged_by_id = {item.claim_id: item for item in judged}
        for claim, report in pending:
            judge = judged_by_id[claim.id]
            scored = ScoredClaim(
                id=claim.id,
                claim=claim.claim,
                source=claim.source,
                score=judge.score,
                evidence=report.evidence,
                search_rounds=report.search_rounds,
                search_trace=report.search_trace,
                judge_reasoning=judge.reasoning,
            )
            self.cache.set(actual_paper_id, scored)
            scored_by_dim[claim.dimension].append(scored)
        return aggregate_result(
            paper_id=actual_paper_id,
            repo_path=repo,
            model=self.config.model,
            scored_claims=scored_by_dim,
        )

    def score_batch(
        self,
        *,
        papers: list[str],
        repo_root: str | Path,
        dimensions: list[Dimension] | None = None,
        max_workers: int = 1,
    ) -> list[SAUScoreResult]:
        root = Path(repo_root).resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"repo_root must be an existing directory: {root}")
        paper_ids = _discover_paper_ids(root) if papers == ["all"] else papers
        if max_workers <= 1:
            return [
                self.score_paper(paper_id=paper_id, repo_path=resolve_repo_for_paper(root, paper_id), dimensions=dimensions)
                for paper_id in paper_ids
            ]

        results: list[SAUScoreResult] = []
        config_payload = self.config.model_dump(mode="json")
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _score_paper_worker,
                    config_payload,
                    paper_id,
                    str(resolve_repo_for_paper(root, paper_id)),
                    [dim.value for dim in dimensions] if dimensions else None,
                ): paper_id
                for paper_id in paper_ids
            }
            for future in as_completed(futures):
                results.append(future.result())
        return sorted(results, key=lambda item: item.paper_id)

def resolve_repo_for_paper(repo_root: Path, paper_id: str) -> Path:
    repo = repo_root / paper_id / "repo"
    if repo.exists():
        return repo
    legacy = repo_root / paper_id / "workspace" / "repo"
    if legacy.exists():
        return legacy
    legacy_suffix = repo_root / paper_id / f"{paper_id}_repo"
    if legacy_suffix.exists():
        return legacy_suffix
    raise FileNotFoundError(f"Could not resolve repo for paper {paper_id!r}; tried {repo}, {legacy}, and {legacy_suffix}")


def _discover_paper_ids(repo_root: Path) -> list[str]:
    paper_ids: list[str] = []
    for child in sorted(repo_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        if (child / "repo").exists() or (child / "workspace" / "repo").exists() or (child / f"{child.name}_repo").exists():
            paper_ids.append(child.name)
    return paper_ids


def _score_paper_worker(
    config_payload: dict[str, object],
    paper_id: str,
    repo_path: str,
    dimensions: list[str] | None,
) -> SAUScoreResult:
    config = PipelineConfig.model_validate(config_payload)
    pipeline = SAUScoringPipeline(config)
    dims = [Dimension(dim) for dim in dimensions] if dimensions else None
    return pipeline.score_paper(paper_id=paper_id, repo_path=repo_path, dimensions=dims)
