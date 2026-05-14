"""PaperBench-dev judge — replicates official SimpleJudge logic.

- code_only=True: only grades 'Code Development' leaf nodes
- Uses official prompt structure from paperbench/judge/constants.py
- Scores each leaf 0/1, propagates up via weighted average
- Model: o3-mini (or configurable)
"""
from __future__ import annotations

import json
import os
import re
import time
import concurrent.futures
from pathlib import Path
from typing import Any

from openai import OpenAI

# ── Official paperbench prompts (verbatim) ────────────────────────────────────

JUDGE_TASK_PROMPT = """You are judging a candidate's attempt to reproduce a research paper.

You will first be given the research paper, which you should read carefully and refer to as the ground truth for what constitutes a correct implementation and reproduction.

Following that, you will be presented with the candidate's submission: a set of files that attempts
to reproduce the paper.

To judge the submission, we have prepared a full rubric describing fine-grained criteria for different aspects of the paper.

Your task is to check the submission and its outputs for ONE specific criterion from this rubric."""

GRADING_PROMPT = """You will now evaluate the submission above for the given criterion.

Please present your answer in 3 parts as follows:

# Expectations
Read the paper carefully to identify the parts relevant to your criterion. Describe what you expect correct resolution to look like, and what you will specifically look for in the submission.

# Reality
Explore the files provided for the submission along with the output logs to identify the parts that are relevant to the resolution criteria. When you have identified the relevant parts, comment on how they compare to your expectations above, and whether the attempt meets the resolution criteria. Be sure to describe only what has been given to you in the submission, and be explicit about which files you are referring to when you discuss them.

# Score
Reason about whether the criteria has been met. Give a score of either 0 or 1 depending on whether the criteria has been met, and explain why.

Other notes:
- You must always provide a score. If you have any uncertainties, make them clear in your discussion.
- All the files and logs from the submission attempt have been provided to you. If anything appears to be missing, assume that the attempt failed to produce it (treat it as a failure).
- Be strict and thorough in grading your resolution criteria, but do not check for things that are outside of your scope."""

CRITERION_PROMPT = """The full grading rubric for the paper hierarchically decomposes various aspects of the paper. To provide more context for the criterion you are currently grading, here are some of the preceding criteria (DO NOT grade this, this is just for context):
<preceding_criteria>
{preceding_criteria}
</preceding_criteria>

Now, here is the criterion that you are grading:
<criterion>
{criterion}
</criterion>

The criterion you are grading is of type {task_category} - {task_category_question}"""

TASK_CATEGORY_QUESTION = (
    "Does the code in the submission contain a correct implementation of this?"
    " Note: mere instructions or descriptions of implementations are not sufficient;"
    " you should check whether there is actual code that implements the requirement."
)

# ── Rubric helpers ────────────────────────────────────────────────────────────

def _filter_code_only(node: dict) -> dict | None:
    children = node.get("sub_tasks", [])
    if not children:
        return node if node.get("task_category") == "Code Development" else None
    filtered = [c for c in (_filter_code_only(ch) for ch in children) if c is not None]
    if not filtered:
        return None
    return {**node, "sub_tasks": filtered}


def _get_leaves(node: dict) -> list[dict]:
    children = node.get("sub_tasks", [])
    if not children:
        return [node]
    result = []
    for c in children:
        result.extend(_get_leaves(c))
    return result


def _get_ancestors(node: dict, rubric: dict, path: list[dict] | None = None) -> list[dict]:
    """Return ancestor nodes (path from root to parent of node)."""
    if path is None:
        path = []
    if rubric["id"] == node["id"]:
        return path
    for child in rubric.get("sub_tasks", []):
        result = _get_ancestors(node, child, path + [rubric])
        if result is not None:
            return result
    return None  # type: ignore


def _compute_score(node: dict, leaf_scores: dict[str, float]) -> float:
    nid = node["id"]
    children = node.get("sub_tasks", [])
    if not children:
        return leaf_scores.get(nid, 0.0)
    total_weight = sum(c["weight"] for c in children)
    if total_weight == 0:
        return 0.0
    return sum(_compute_score(c, leaf_scores) * c["weight"] for c in children) / total_weight


# ── File reading ──────────────────────────────────────────────────────────────

def _read_repo_files(repo_dir: Path, max_chars: int = 60000) -> str:
    parts = []
    total = 0
    exts = {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".json", ".cfg", ".toml", ".rst"}
    for f in sorted(repo_dir.rglob("*")):
        if not f.is_file() or f.suffix not in exts:
            continue
        rel = f.relative_to(repo_dir)
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        header = f"### {rel}\n```\n"
        footer = "\n```\n"
        available = max_chars - total - len(header) - len(footer)
        if available <= 0:
            break
        chunk = header + content[:available] + footer
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts) if parts else "(empty submission)"


# ── Grading ───────────────────────────────────────────────────────────────────

def _grade_leaf(
    client: OpenAI,
    model: str,
    paper_md: str,
    addendum: str,
    repo_files: str,
    leaf: dict,
    ancestors: list[dict],
) -> tuple[int, str]:
    preceding = "".join(f" -> {a['requirements']}\n" for a in ancestors[-5:])
    criterion_block = CRITERION_PROMPT.format(
        preceding_criteria=preceding,
        criterion=leaf["requirements"],
        task_category="Code Development",
        task_category_question=TASK_CATEGORY_QUESTION,
    )
    messages = [
        {"role": "system", "content": JUDGE_TASK_PROMPT},
        {"role": "user", "content": f"The paper is below:\n{paper_md}"},
        {"role": "user", "content": f"Addendum:\n<addendum>\n{addendum}\n</addendum>"},
        {"role": "user", "content": f"Submission files:\n<files>\n{repo_files}\n</files>"},
        {"role": "user", "content": criterion_block},
        {"role": "user", "content": GRADING_PROMPT},
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    text = resp.choices[0].message.content or ""
    score = 0
    # Strip markdown bold markers and search for score
    clean = text.replace("**", "").replace("__", "")
    for line in reversed(clean.splitlines()):
        s = line.strip().lower()
        if "score:" in s:
            # extract first digit after "score:"
            after = s.split("score:", 1)[1].strip()
            m = re.search(r"[01]", after)
            if m:
                score = int(m.group())
            break
        if re.fullmatch(r"[01]", s):
            score = int(s)
            break
    return score, text


def grade(
    case_id: str,
    repo_dir: Path,
    rubric_path: Path,
    paper_md_path: Path,
    addendum_path: Path | None,
    out_path: Path,
    model: str = "o3-mini",
    api_key: str | None = None,
    api_base_url: str | None = None,
) -> dict[str, Any]:
    client = OpenAI(
        api_key=api_key or os.environ["OPENAI_API_KEY"],
        base_url=api_base_url or os.environ.get("OPENAI_BASE_URL"),
    )

    rubric_full = json.loads(rubric_path.read_text())
    rubric = _filter_code_only(rubric_full) or rubric_full
    paper_md = paper_md_path.read_text(encoding="utf-8", errors="ignore")
    addendum = addendum_path.read_text(encoding="utf-8", errors="ignore") if addendum_path and addendum_path.exists() else ""
    repo_files = _read_repo_files(repo_dir)

    leaves = _get_leaves(rubric)
    print(f"[{case_id}] Grading {len(leaves)} Code Development leaf nodes with {model} (parallel)...")

    results_map: dict[str, tuple[int, str, float]] = {}
    done_count = 0

    def _grade_one(leaf: dict) -> tuple[str, int, str, float]:
        ancestors = _get_ancestors(leaf, rubric_full) or []
        t0 = time.time()
        score, reasoning = _grade_leaf(client, model, paper_md, addendum, repo_files, leaf, ancestors)
        return leaf["id"], score, reasoning, round(time.time() - t0, 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_grade_one, leaf): leaf for leaf in leaves}
        for future in concurrent.futures.as_completed(futures):
            nid, score, reasoning, elapsed = future.result()
            results_map[nid] = (score, reasoning, elapsed)
            done_count += 1
            leaf = futures[future]
            print(f"  [{done_count}/{len(leaves)}] {score} ({elapsed}s) | {leaf['requirements'][:70]}")

    leaf_scores: dict[str, float] = {}
    node_results: list[dict] = []
    for leaf in leaves:
        nid = leaf["id"]
        score, reasoning, elapsed = results_map[nid]
        leaf_scores[nid] = float(score)
        node_results.append({
            "id": nid,
            "requirements": leaf["requirements"],
            "weight": leaf["weight"],
            "finegrained_task_category": leaf.get("finegrained_task_category"),
            "score": score,
            "reasoning": reasoning,
            "elapsed_s": elapsed,
        })

    final_score = _compute_score(rubric, leaf_scores)
    pass_count = sum(1 for v in leaf_scores.values() if v == 1)

    result = {
        "case_id": case_id,
        "model": model,
        "judge_mode": "paperbench-dev (code_only)",
        "num_leaves": len(leaves),
        "pass_count": pass_count,
        "final_score": round(final_score, 4),
        "node_results": node_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[{case_id}] Score: {final_score:.4f} ({pass_count}/{len(leaves)} leaves) -> {out_path}")
    return result


if __name__ == "__main__":
    import argparse, dotenv
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="o3-mini")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    run_dir = base / "experiments" / "runs" / args.case / args.run_id
    repo_dir = run_dir / "workspace" / "repo"
    rubric_path = base / "data" / "papers" / args.case / "rubric.json"
    paper_md_path = base / "data" / "papers" / args.case / "paper.md"
    addendum_path = base / "data" / "papers" / args.case / "addendum.md"
    out_path = run_dir / "grade.json"

    grade(args.case, repo_dir, rubric_path, paper_md_path, addendum_path, out_path, model=args.model)
