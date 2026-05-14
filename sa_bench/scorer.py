"""Public Python SDK for SAU scoring."""

from __future__ import annotations

import os
from pathlib import Path

from .pipeline import SAUScoringPipeline
from .types import Dimension, PipelineConfig, SAUScoreResult


class SAUScorer:
    def __init__(
        self,
        *,
        model: str = "deepseek-v4-pro",
        data_dir: str | Path = "data/papers",
        cache_dir: str | Path | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        use_llm_search: bool = False,
        use_llm_judge: bool | None = None,
        judge_batch_size: int = 4,
        judge_workers: int = 2,
        search_model: str | None = None,
    ) -> None:
        if use_llm_judge is None:
            use_llm_judge = bool(api_key or os.getenv("OPENAI_API_KEY"))
        self.config = PipelineConfig(
            data_dir=Path(data_dir),
            cache_dir=Path(cache_dir) if cache_dir else None,
            model=model,
            search_model=search_model,
            api_key=api_key,
            base_url=base_url,
            use_llm_search=use_llm_search,
            use_llm_judge=use_llm_judge,
            judge_batch_size=judge_batch_size,
            judge_workers=judge_workers,
        )
        self.pipeline = SAUScoringPipeline(self.config)

    def score(
        self,
        *,
        paper_id: str,
        repo_path: str | Path,
        dimension: str | Dimension = "all",
    ) -> SAUScoreResult:
        dimensions = _parse_dimensions(dimension)
        return self.pipeline.score_paper(paper_id=paper_id, repo_path=repo_path, dimensions=dimensions)

    def score_batch(
        self,
        *,
        papers: list[str],
        dimensions: list[str | Dimension] | str | Dimension = "all",
        repo_root: str | Path,
        max_workers: int = 1,
    ) -> list[SAUScoreResult]:
        dims = _parse_dimensions(dimensions)
        return self.pipeline.score_batch(
            papers=papers,
            repo_root=repo_root,
            dimensions=dims,
            max_workers=max_workers,
        )


def _parse_dimensions(value: list[str | Dimension] | str | Dimension) -> list[Dimension] | None:
    if value == "all":
        return None
    if isinstance(value, Dimension):
        return [value]
    if isinstance(value, str):
        return [Dimension(value)]
    dims = [Dimension(str(dim)) for dim in value]
    return None if len(dims) == len(Dimension) else dims
