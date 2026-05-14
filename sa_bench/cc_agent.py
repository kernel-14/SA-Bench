"""Claim-level evidence collection with the SAU ReAct tool protocol."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .evidence import classify_path, direct_evidence_items, rank_evidence_items
from .index import RepoIndex
from .keywords import claim_queries, claim_regexes
from .tools import grep_exact, grep_regex, list_files, read_file, search_config, trace_imports
from .tools.common import count_results
from .types import EvidenceItem, EvidenceReport, PipelineConfig, SAUClaim, SearchTraceEntry


TOOL_PROTOCOL = """You have these tools:
  grep_exact(pattern)     -> exact string match, returns file:line
  grep_regex(pattern)     -> regex search, returns file:line
  read_file(path, start, end) -> read file lines
  list_files(dir)         -> list directory contents
  trace_imports(file)     -> find imports and dependents of a file
  search_config(key)      -> parse yaml/json/argparse config for key

Tool use format:
<<TOOL:tool_name>>
arg=value
<<END>>
"""


DIMENSION_ROLES = {
    "D1": "You are a numerical hunter. Focus on config files, argparse, hyperparameters, hard-coded values, and numeric equivalence.",
    "D2": "You are a formula detective. Map mathematical terms to code operations, losses, forward passes, norms, schedules, and gradients.",
    "D3": "You are a protocol auditor. Focus on datasets, training loops, evaluation scripts, metrics, augmentation, and experiment configs.",
    "D4": "You are an architecture analyst. Focus on module boundaries, pipeline orchestration, phase order, imports, and call chains.",
}

CLAIM_RANK_STOPWORDS = {
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
}

CODE_MATH_TERMS = {
    "sqrt",
    "sigma",
    "alpha",
    "beta",
    "eta",
    "epsilon",
    "lambda",
    "grad",
    "norm",
    "loss",
    "mse",
    "drift",
    "control",
    "velocity",
    "adjoint",
    "schedule",
    "memoryless",
    "reward",
    "guidance",
    "forward",
}

RELEVANCE_STRENGTH = {"direct": 5, "config": 4, "semantic": 3, "supporting": 2, "weak": 1, "candidate": 0}


class ClaudeCodeAgent:
    """Collect evidence for one claim.

    The default deterministic collector implements the three-round protocol with
    local tools. Optional LLM search uses the same text ReAct protocol through an
    OpenAI-compatible chat endpoint.
    """

    def __init__(self, repo_path: Path, config: PipelineConfig) -> None:
        self.repo_path = repo_path.resolve()
        self.config = config
        self.index = RepoIndex(self.repo_path)

    def collect_evidence(self, claim: SAUClaim) -> EvidenceReport:
        if self.config.use_llm_search:
            return self._collect_with_llm(claim)
        return self._collect_deterministic(claim)

    def _collect_deterministic(self, claim: SAUClaim) -> EvidenceReport:
        trace: list[SearchTraceEntry] = []
        evidence: dict[tuple[str, str], EvidenceItem] = {}

        root_listing = self.index.list_files(".", limit=self.config.max_files_listed)
        trace.append(SearchTraceEntry(round=1, tool="list_files", query=".", results=count_results(root_listing)))

        exact_queries = claim_queries(claim, max_queries=12)
        for query in exact_queries[:5]:
            output = self.index.grep_exact(query, limit=30)
            trace.append(SearchTraceEntry(round=1, tool="grep_exact", query=query, results=count_results(output)))
            self._add_evidence_from_matches(evidence, output, relevance=self._relevance_for_query(query))

        for query in exact_queries[:4]:
            output = self.index.search_config(query, limit=20)
            trace.append(SearchTraceEntry(round=1, tool="search_config", query=query, results=count_results(output)))
            self._add_evidence_from_matches(evidence, output, relevance="config")

        for regex in claim_regexes(claim, max_queries=8):
            output = self.index.grep_regex(regex, limit=30)
            trace.append(SearchTraceEntry(round=2, tool="grep_regex", query=regex, results=count_results(output)))
            self._add_evidence_from_matches(evidence, output, relevance="semantic")

        for query in exact_queries[5:10]:
            output = self.index.grep_exact(query, limit=20)
            trace.append(SearchTraceEntry(round=2, tool="grep_exact", query=query, results=count_results(output)))
            self._add_evidence_from_matches(evidence, output, relevance="supporting")

        for item in list(evidence.values())[:3]:
            output = trace_imports(self.repo_path, item.file)
            trace.append(SearchTraceEntry(round=3, tool="trace_imports", query=item.file, results=count_results(output)))

        if not evidence:
            for fallback in self._fallback_queries(claim):
                output = self.index.grep_regex(fallback, limit=25)
                trace.append(SearchTraceEntry(round=3, tool="grep_regex", query=fallback, results=count_results(output)))
                self._add_evidence_from_matches(evidence, output, relevance="weak")
                if evidence:
                    break

        self._add_called_definitions(evidence, trace)
        evidence_items = _select_evidence_for_claim(
            claim,
            list(evidence.values()),
            limit=self.config.max_evidence_per_claim,
        )
        direct_items = direct_evidence_items(evidence_items)
        evidence_items = evidence_items[: self.config.max_evidence_per_claim]
        status = "found" if direct_items else "partial"
        if not evidence_items:
            status = "not_found"
        return EvidenceReport(
            claim_id=claim.id,
            status=status,
            search_rounds=3,
            evidence=evidence_items,
            search_trace=trace,
        )

    def _collect_with_llm(self, claim: SAUClaim) -> EvidenceReport:
        client = _make_openai_client(self.config)
        model = self.config.search_model or os.getenv("SAU_SEARCH_MODEL") or self.config.model
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": "\n".join(
                    [
                        DIMENSION_ROLES[claim.dimension.value],
                        TOOL_PROTOCOL,
                        "Use at most three search rounds. Finish with an Evidence Report JSON object only.",
                    ]
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Claim ID: {claim.id}\n"
                    f"Dimension: {claim.dimension.value}\n"
                    f"Claim: {claim.claim}\n"
                    f"Source: {claim.source}\n"
                    f"Repository: {self.repo_path}\n"
                ),
            },
        ]
        trace: list[SearchTraceEntry] = []
        for round_no in range(1, 4):
            response = client.chat.completions.create(model=model, messages=messages, temperature=0)
            text = response.choices[0].message.content or ""
            calls = parse_tool_calls(text)
            if not calls:
                parsed = parse_evidence_report(text)
                if parsed is not None:
                    return parsed
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": "Return the Evidence Report JSON now."})
                continue
            messages.append({"role": "assistant", "content": text})
            tool_outputs: list[str] = []
            for tool_name, args in calls:
                output = self._execute_tool(tool_name, args)
                trace.append(
                    SearchTraceEntry(
                        round=round_no,
                        tool=tool_name,
                        query=str(args.get("pattern") or args.get("key") or args.get("path") or args.get("dir") or ""),
                        results=count_results(output),
                    )
                )
                tool_outputs.append(f"<<RESULT:{tool_name}>>\n{output}\n<<END_RESULT>>")
            messages.append({"role": "user", "content": "\n\n".join(tool_outputs)})

        response = client.chat.completions.create(
            model=model,
            messages=messages + [{"role": "user", "content": "Return final Evidence Report JSON only."}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = parse_evidence_report(response.choices[0].message.content or "")
        if parsed is None:
            raise ValueError("LLM search did not return a valid Evidence Report")
        parsed.search_trace.extend(trace)
        return parsed

    def _execute_tool(self, tool_name: str, args: dict[str, str]) -> str:
        if tool_name == "grep_exact":
            return grep_exact(self.repo_path, args.get("pattern", ""), limit=50)
        if tool_name == "grep_regex":
            return grep_regex(self.repo_path, args.get("pattern", ""), limit=50)
        if tool_name == "read_file":
            return read_file(
                self.repo_path,
                args.get("path", ""),
                start=int(args.get("start", args.get("start_line", "1"))),
                end=int(args["end"]) if args.get("end") else None,
            )
        if tool_name == "list_files":
            return list_files(self.repo_path, args.get("dir", args.get("path", ".")))
        if tool_name == "trace_imports":
            return trace_imports(self.repo_path, args.get("file", args.get("path", "")))
        if tool_name == "search_config":
            return search_config(self.repo_path, args.get("key", args.get("pattern", "")))
        return f"Error: unknown tool {tool_name}"

    def _add_evidence_from_matches(
        self,
        evidence: dict[tuple[str, str], EvidenceItem],
        output: str,
        *,
        relevance: str,
    ) -> None:
        for line in output.splitlines():
            parsed = _parse_match_line(line)
            if parsed is None:
                continue
            file, line_no = parsed
            if classify_path(file) in {"docs", "test"}:
                continue
            start = max(1, line_no - 2)
            end = line_no + max(1, self.config.max_snippet_lines - 3)
            snippet = _strip_line_numbers(self.index.read_file(file, start=start, end=end))
            key = (file, f"{start}-{end}")
            existing = evidence.get(key)
            candidate = EvidenceItem(
                    file=file,
                    lines=f"{start}-{end}",
                    snippet=snippet,
                    relevance=relevance,
                    kind=classify_path(file),
            )
            if existing is None or RELEVANCE_STRENGTH.get(candidate.relevance, 0) > RELEVANCE_STRENGTH.get(existing.relevance, 0):
                evidence[key] = candidate

    def _add_called_definitions(
        self,
        evidence: dict[tuple[str, str], EvidenceItem],
        trace: list[SearchTraceEntry],
    ) -> None:
        names: list[str] = []
        for item in list(evidence.values()):
            for name in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", item.snippet):
                if len(name) >= 4 and name not in {"range", "print", "super"}:
                    names.append(name)
        for name in sorted(set(names))[:6]:
            pattern = rf"(def|class)\s+{re.escape(name)}\b"
            output = self.index.grep_regex(pattern, limit=10)
            trace.append(SearchTraceEntry(round=3, tool="grep_regex", query=pattern, results=count_results(output)))
            self._add_evidence_from_matches(evidence, output, relevance="direct")

    @staticmethod
    def _relevance_for_query(query: str) -> str:
        if re.search(r"\d", query) or len(query.split()) >= 2:
            return "direct"
        return "supporting"

    @staticmethod
    def _fallback_queries(claim: SAUClaim) -> list[str]:
        if claim.dimension.value == "D1":
            return [r"\d+(?:\.\d+)?(?:e[+-]?\d+)?", r"(config|args|param|hyper)"]
        if claim.dimension.value == "D2":
            return [r"(loss|forward|sigma|alpha|beta|norm|grad)", r"(torch|numpy|jax)\."]
        if claim.dimension.value == "D3":
            return [r"(train|eval|dataset|metric|epoch)", r"(DataLoader|Trainer|evaluate)"]
        return [r"(pipeline|phase|runner|trainer|workflow)", r"(class|def)\s+\w+"]


def parse_tool_calls(text: str) -> list[tuple[str, dict[str, str]]]:
    calls: list[tuple[str, dict[str, str]]] = []
    pattern = re.compile(r"<<TOOL:([A-Za-z_][A-Za-z0-9_]*)>>\s*(.*?)\s*<<END>>", re.DOTALL)
    for match in pattern.finditer(text):
        args: dict[str, str] = {}
        for line in match.group(2).splitlines():
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            args[key.strip()] = value.strip()
        calls.append((match.group(1), args))
    return calls


def parse_evidence_report(text: str) -> EvidenceReport | None:
    payload = _extract_json_object(text)
    if payload is None:
        return None
    try:
        return EvidenceReport.model_validate(payload)
    except Exception:
        return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_match_line(line: str) -> tuple[str, int] | None:
    match = re.match(r"^(.+?):(\d+):", line)
    if not match:
        return None
    try:
        return match.group(1), int(match.group(2))
    except ValueError:
        return None


def _strip_line_numbers(snippet: str) -> str:
    lines = []
    for line in snippet.splitlines():
        lines.append(re.sub(r"^\d+:\s?", "", line))
    return "\n".join(lines)


def _select_evidence_for_claim(claim: SAUClaim, items: list[EvidenceItem], *, limit: int) -> list[EvidenceItem]:
    ranked = rank_evidence_items(items)
    scored = sorted(
        ranked,
        key=lambda item: (
            -_evidence_claim_score(claim, item),
            item.file,
            item.lines,
        ),
    )
    return _dedupe_overlapping(scored)[:limit]


def _evidence_claim_score(claim: SAUClaim, item: EvidenceItem) -> int:
    haystack = _normalize_for_rank(f"{item.file}\n{item.snippet}")
    terms = _rank_terms(claim.claim)
    numbers = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![A-Za-z])", claim.claim, re.IGNORECASE)

    score = 0
    if item.kind == "code":
        score += 20
    elif item.kind == "config":
        score += 16
    elif item.kind in {"docs", "test"}:
        score -= 20

    score += {"direct": 16, "config": 12, "semantic": 8, "supporting": 4, "weak": 0}.get(item.relevance, 2)
    score += min(40, sum(4 for term in terms if term in haystack))
    score += min(24, sum(6 for number in numbers if number and number.lower() in haystack))
    score += min(30, sum(5 for term in CODE_MATH_TERMS if term in haystack and term in _normalize_for_rank(claim.claim)))

    basename = Path(item.file).stem.lower().replace("_", " ")
    score += min(20, sum(5 for term in terms if term in basename))
    if re.search(r"\b(def|class)\s+[A-Za-z_][A-Za-z0-9_]*", item.snippet):
        score += 8
    if "pass" in haystack or "placeholder" in haystack or "todo" in haystack:
        score -= 12
    return score


def _rank_terms(text: str) -> set[str]:
    normalized = _normalize_for_rank(text)
    terms = {
        token
        for token in re.findall(r"[a-z][a-z0-9_+-]{2,}", normalized)
        if token not in CLAIM_RANK_STOPWORDS and len(token) >= 3
    }
    expanded = set(terms)
    for term in terms:
        expanded.update(part for part in re.split(r"[_+-]+", term) if len(part) >= 3)
    return expanded


def _normalize_for_rank(text: str) -> str:
    replacements = {
        "β": "beta",
        "α": "alpha",
        "σ": "sigma",
        "λ": "lambda",
        "ε": "epsilon",
        "η": "eta",
        "θ": "theta",
        "∇": "grad",
        "×": "x",
        "−": "-",
        "²": "2",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.lower()


def _dedupe_overlapping(items: list[EvidenceItem]) -> list[EvidenceItem]:
    selected: list[EvidenceItem] = []
    for item in items:
        span = _line_span(item.lines)
        overlaps = False
        for existing in selected:
            if item.file != existing.file:
                continue
            existing_span = _line_span(existing.lines)
            if span and existing_span and span[0] <= existing_span[1] and existing_span[0] <= span[1]:
                overlaps = True
                break
        if not overlaps:
            selected.append(item)
    return selected


def _line_span(lines: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)(?:-(\d+))?$", lines)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2) or start)
    return start, end


def _make_openai_client(config: PipelineConfig) -> Any:
    import httpx
    from openai import OpenAI

    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for LLM search")
    timeout_sec = float(os.getenv("SAU_LLM_TIMEOUT", "30"))
    max_retries = int(os.getenv("SAU_LLM_MAX_RETRIES", "0"))
    return OpenAI(
        api_key=api_key,
        base_url=config.base_url or os.getenv("OPENAI_BASE_URL"),
        timeout=httpx.Timeout(timeout_sec, connect=10.0, read=timeout_sec, write=10.0, pool=10.0),
        max_retries=max_retries,
    )
