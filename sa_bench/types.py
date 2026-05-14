"""Pydantic models for the SAU scoring pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Dimension(str, Enum):
    D1 = "D1"
    D2 = "D2"
    D3 = "D3"
    D4 = "D4"


VALID_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}


class SAUClaim(BaseModel):
    id: str
    claim: str
    source: str = ""
    dimension: Dimension


class SearchTraceEntry(BaseModel):
    round: int
    tool: str
    query: str = ""
    results: int = 0
    error: str | None = None


class EvidenceItem(BaseModel):
    file: str
    lines: str
    snippet: str
    relevance: str = "candidate"
    kind: str = "other"


class EvidenceReport(BaseModel):
    claim_id: str
    status: str
    search_rounds: int
    evidence: list[EvidenceItem] = Field(default_factory=list)
    search_trace: list[SearchTraceEntry] = Field(default_factory=list)


class JudgeResult(BaseModel):
    claim_id: str
    score: float
    reasoning: str
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if value not in VALID_SCORES:
            raise ValueError("score must be one of 0, 0.25, 0.5, 0.75, or 1")
        return value


class ScoredClaim(BaseModel):
    id: str
    claim: str
    source: str
    score: float
    evidence: list[EvidenceItem] = Field(default_factory=list)
    search_rounds: int
    search_trace: list[SearchTraceEntry] = Field(default_factory=list)
    judge_reasoning: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if value not in VALID_SCORES:
            raise ValueError("score must be one of 0, 0.25, 0.5, 0.75, or 1")
        return value


class DimensionResult(BaseModel):
    score: float
    total_claims: int
    claims: list[ScoredClaim] = Field(default_factory=list)


class SAUScoreResult(BaseModel):
    paper_id: str
    repo_path: str
    scored_at: datetime
    model: str
    dimensions: dict[str, DimensionResult]
    overall_score: float


class PipelineConfig(BaseModel):
    data_dir: Path = Path("data/papers")
    cache_dir: Path | None = None
    model: str = "deepseek-v4-pro"
    search_model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    use_llm_search: bool = False
    use_llm_judge: bool = True
    judge_batch_size: int = 4
    judge_workers: int = 2
    max_evidence_per_claim: int = 8
    max_snippet_lines: int = 8
    max_files_listed: int = 200


def model_dump_jsonable(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")
