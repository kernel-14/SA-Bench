"""DeepSeek/OpenAI-compatible judge for five-level SAU scoring."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .evidence import direct_evidence_items, has_direct_implementation, is_stub_like_item, only_docs_or_tests, rank_evidence_items
from .types import EvidenceReport, JudgeResult, PipelineConfig, SAUClaim, VALID_SCORES


JUDGE_SYSTEM = """You are a strict SAU code-reproduction judge.
Return JSON only. Score must be exactly one of: 0, 0.25, 0.5, 0.75, 1.

Judge using these rules:
- Evidence from code/config matters most; docs/tests may only corroborate.
- Score the implementation, not the paper text or the repo README.
- Prefer lower scores unless the claim's concrete mechanism is visible in code/config.
- If the evidence mostly shows placeholders, stubs, comments, or toy code, keep the score low.
- Return concise reasoning with file:line references only for the strongest matches.
- Every result object must include claim_id, score, reasoning, and evidence_refs.
- Do not omit fields and do not return null for any required field.
- Write reasoning as plain text only. Do not use LaTeX commands, backslashes, or unescaped math markup.
"""


def judge_claim(
    claim: SAUClaim,
    report: EvidenceReport,
    *,
    config: PipelineConfig,
    client: Any | None = None,
) -> JudgeResult:
    return judge_claim_batch([(claim, report)], config=config, client=client)[0]


def judge_claim_batch(
    items: list[tuple[SAUClaim, EvidenceReport]],
    *,
    config: PipelineConfig,
    client: Any | None = None,
) -> list[JudgeResult]:
    results: list[JudgeResult] = []
    pending: list[tuple[SAUClaim, EvidenceReport]] = []
    for claim, report in items:
        evidence = rank_evidence_items(report.evidence)
        kinds = [item.kind for item in evidence]
        if not report.evidence or not has_direct_implementation(kinds) or only_docs_or_tests(kinds):
            results.append(
                JudgeResult(
                    claim_id=claim.id,
                    score=0.0,
                    reasoning=_zero_reason(report) if report.evidence else "No evidence found in repository code/config after search.",
                    evidence_refs=[],
                )
            )
            continue
        if _is_experiment_style_claim(claim) and not _has_numeric_support(claim, evidence):
            results.append(
                JudgeResult(
                    claim_id=claim.id,
                    score=0.0,
                    reasoning="Experiment-parameter claim has no direct numeric or script evidence in code/config.",
                    evidence_refs=[],
                )
            )
            continue
        strong_items = [
            item
            for item in direct_evidence_items(evidence)
            if item.relevance in {"direct", "config", "semantic"} and not is_stub_like_item(item)
        ]
        if not strong_items:
            results.append(
                JudgeResult(
                    claim_id=claim.id,
                    score=0.0,
                    reasoning="Only weak or indirect implementation evidence found; claim not passed to LLM judge.",
                    evidence_refs=[],
                )
            )
            continue
        if _is_theory_heavy(claim.claim) and len(strong_items) < 2:
            results.append(
                JudgeResult(
                    claim_id=claim.id,
                    score=0.0,
                    reasoning="Theorem/proof-style claim only has one strong implementation signal; treated as unsupported.",
                    evidence_refs=[],
                )
            )
            continue
        pending.append((claim, report))

    if not pending:
        return results

    if not config.use_llm_judge:
        return _ordered_results(
            items,
            [
                JudgeResult(
                    claim_id=claim.id,
                    score=0.0,
                    reasoning="LLM judge disabled; use explicit offline inspection only.",
                    evidence_refs=[f"{item.file}:{item.lines}" for item in direct_evidence_items(report.evidence)[:5]],
                )
                for claim, report in pending
            ],
        )

    judged: list[JudgeResult] = []
    batches = _chunked(pending, max(1, config.judge_batch_size))
    if len(batches) == 1 or config.judge_workers <= 1:
        for batch in batches:
            judged.extend(_judge_batch_with_llm(batch, config=config, client=client))
    else:
        with ThreadPoolExecutor(max_workers=min(config.judge_workers, len(batches))) as pool:
            futures = {pool.submit(_judge_batch_with_llm, batch, config=config, client=client): idx for idx, batch in enumerate(batches)}
            ordered_batches: list[list[JudgeResult] | None] = [None] * len(batches)
            for future in as_completed(futures):
                idx = futures[future]
                ordered_batches[idx] = future.result()
        for batch_results in ordered_batches:
            if batch_results is None:
                continue
            judged.extend(batch_results)
    judged_by_id = {result.claim_id: result for result in judged}
    for claim, report in pending:
        result = judged_by_id.get(claim.id)
        if result is None:
            raise ValueError(f"Batch judge did not return a score for claim {claim.id}")
        results.append(result)

    return _ordered_results(items, results)


def _judge_with_llm(
    claim: SAUClaim,
    report: EvidenceReport,
    *,
    config: PipelineConfig,
    client: Any | None,
) -> JudgeResult:
    import time as _time
    from openai import APITimeoutError, APIError

    max_attempts = int(os.getenv("SAU_JUDGE_MAX_ATTEMPTS", "3"))
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            llm = client or _make_openai_client(config)
            response = llm.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": build_judge_prompt(claim, report)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            payload = parse_judge_response(response.choices[0].message.content or "")
            if not payload.get("claim_id"):
                payload["claim_id"] = claim.id
            return JudgeResult.model_validate(payload)
        except (APITimeoutError, APIError, ValueError) as exc:
            last_exc = exc
            if attempt < max_attempts:
                _time.sleep(min(2 ** attempt, 10))
    raise last_exc  # type: ignore[misc]


def _judge_batch_with_llm(
    items: list[tuple[SAUClaim, EvidenceReport]],
    *,
    config: PipelineConfig,
    client: Any | None,
) -> list[JudgeResult]:
    if len(items) == 1:
        claim, report = items[0]
        return [_judge_with_llm(claim, report, config=config, client=client)]

    import time as _time
    from openai import APITimeoutError, APIError

    max_attempts = int(os.getenv("SAU_JUDGE_MAX_ATTEMPTS", "3"))
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            llm = client or _make_openai_client(config)
            response = llm.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": build_batch_judge_prompt(items)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            payload = _extract_json_object(content)
            if payload is None:
                raise ValueError(f"judge did not return a JSON object; response excerpt: {content[:500]!r}")
            raw_results = payload.get("results", [])
            if not isinstance(raw_results, list):
                raise ValueError("judge batch response must contain a results array")
            results: list[JudgeResult] = []
            for raw in raw_results:
                if not isinstance(raw, dict):
                    continue
                if not raw.get("claim_id"):
                    continue
                results.append(JudgeResult.model_validate(parse_judge_response(json.dumps(raw, ensure_ascii=False))))
            if len(results) != len(items):
                raise ValueError(
                    f"judge batch returned {len(results)} results for {len(items)} claims"
                )
            return results
        except (APITimeoutError, APIError, ValueError) as exc:
            last_exc = exc
            if attempt < max_attempts:
                _time.sleep(min(2 ** attempt, 10))
        except Exception:
            raise
    raise last_exc  # type: ignore[misc]


def build_judge_prompt(claim: SAUClaim, report: EvidenceReport) -> str:
    direct_items = direct_evidence_items(report.evidence)
    return json.dumps(
        {
            "claim_id": claim.id,
            "dimension": claim.dimension.value,
            "claim": claim.claim,
            "source": claim.source,
            "direct_evidence": [item.model_dump(mode="json") for item in direct_items[:2]],
            "search_trace": [entry.model_dump(mode="json") for entry in report.search_trace[:1]],
            "expected_output": {
                "claim_id": claim.id,
                "score": "0|0.25|0.5|0.75|1",
                "reasoning": "concise explanation with file:line refs",
                "evidence_refs": ["path:line-or-range"],
            },
            "required_fields": ["claim_id", "score", "reasoning", "evidence_refs"],
        },
        ensure_ascii=False,
        indent=2,
    )


def build_batch_judge_prompt(items: list[tuple[SAUClaim, EvidenceReport]]) -> str:
    payload = []
    for claim, report in items:
        direct_items = direct_evidence_items(report.evidence)
        payload.append(
            {
                "claim_id": claim.id,
                "dimension": claim.dimension.value,
                "claim": claim.claim,
                "source": claim.source,
                "direct_evidence": [item.model_dump(mode="json") for item in direct_items[:2]],
                "search_trace": [entry.model_dump(mode="json") for entry in report.search_trace[:1]],
            }
        )
    return json.dumps(
        {
            "claims": payload,
            "expected_output": {
                "results": [
                    {
                        "claim_id": "string",
                        "score": "0|0.25|0.5|0.75|1",
                        "reasoning": "concise explanation with file:line refs",
                        "evidence_refs": ["path:line-or-range"],
                    }
                ],
                "required_fields": ["claim_id", "score", "reasoning", "evidence_refs"],
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def parse_judge_response(text: str) -> dict[str, Any]:
    payload = _extract_json_object(text)
    if payload is None:
        raise ValueError(f"judge did not return a JSON object; response excerpt: {text[:500]!r}")
    if "score" not in payload:
        raise ValueError(f"judge response missing score; response excerpt: {text[:500]!r}")
    score_raw = payload.get("score")
    try:
        score = float(score_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"judge score must be numeric; got {score_raw!r}; response excerpt: {text[:500]!r}") from exc
    if score not in VALID_SCORES:
        raise ValueError(f"judge score must be one of 0, 0.25, 0.5, 0.75, or 1; got {score!r}; response excerpt: {text[:500]!r}")
    payload["score"] = score
    payload["reasoning"] = str(payload.get("reasoning", "")).strip()
    refs = payload.get("evidence_refs", [])
    if not isinstance(refs, list):
        refs = []
    payload["evidence_refs"] = [str(ref) for ref in refs]
    return payload


def _zero_reason(report: EvidenceReport) -> str:
    if not report.evidence:
        return "No evidence found in repository code/config after search."
    return "No direct implementation evidence found in repository code/config."


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _ordered_results(
    items: list[tuple[SAUClaim, EvidenceReport]],
    results: list[JudgeResult],
) -> list[JudgeResult]:
    by_id = {result.claim_id: result for result in results}
    ordered: list[JudgeResult] = []
    for claim, report in items:
        result = by_id.get(claim.id)
        if result is None:
            result = JudgeResult(
                claim_id=claim.id,
                score=0.0,
                reasoning="No judge result returned for this claim.",
                evidence_refs=[f"{item.file}:{item.lines}" for item in direct_evidence_items(report.evidence)[:5]],
            )
        ordered.append(result)
    return ordered


def _chunked(items: list[tuple[SAUClaim, EvidenceReport]], size: int) -> list[list[tuple[SAUClaim, EvidenceReport]]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _is_theory_heavy(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in ("theorem", "lemma", "proof", "convergence", "bound", "smoothness", "sublinear", "exponential", "optimal"))


def _is_experiment_style_claim(claim: SAUClaim) -> bool:
    text = f"{claim.claim} {claim.source}".lower()
    return any(
        term in text
        for term in (
            "experiment",
            "simulation",
            "setup",
            "figure",
            "appendix c",
            "state/action",
            "transition kernel",
            "reward variance",
            "training iterations",
            "run for",
            "varying state",
            "varying reward",
            "varying transition",
        )
    )


def _has_numeric_support(claim: SAUClaim, evidence: list[Any]) -> bool:
    numbers = {
        match
        for match in re.findall(r"\b\d+(?:\.\d+)?\b", f"{claim.claim} {claim.source}")
        if len(match) > 1 or match not in {"1", "2"}
    }
    if not numbers:
        return True
    haystacks = []
    for item in direct_evidence_items(evidence):
        haystacks.append(" ".join([item.file, item.lines, item.snippet]).lower())
    for number in numbers:
        if any(number in haystack for haystack in haystacks):
            return True
    return False


def _make_openai_client(config: PipelineConfig) -> Any:
    import httpx
    from openai import OpenAI

    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for LLM judge")
    timeout_sec = float(os.getenv("SAU_LLM_TIMEOUT", "30"))
    max_retries = int(os.getenv("SAU_LLM_MAX_RETRIES", "0"))
    return OpenAI(
        api_key=api_key,
        base_url=config.base_url or os.getenv("OPENAI_BASE_URL"),
        timeout=httpx.Timeout(timeout_sec, connect=10.0, read=timeout_sec, write=10.0, pool=10.0),
        max_retries=max_retries,
    )
