"""Command line interface for SAU scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .scorer import SAUScorer


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    scorer = SAUScorer(
        model=args.model,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        api_key=args.api_key,
        base_url=args.base_url,
        use_llm_search=args.llm_search,
        use_llm_judge=not args.no_llm_judge,
        judge_batch_size=args.judge_batch_size,
        judge_workers=args.judge_workers,
        search_model=args.search_model,
    )

    if args.paper == "all":
        if not args.repo_root:
            parser.error("--repo-root is required when --paper all")
        results = scorer.score_batch(
            papers=["all"],
            dimensions=args.dim,
            repo_root=args.repo_root,
            max_workers=args.workers,
        )
        payload: object = [result.model_dump(mode="json") for result in results]
    else:
        if not args.repo:
            parser.error("--repo is required for a single paper")
        result = scorer.score(paper_id=args.paper, dimension=args.dim, repo_path=args.repo)
        payload = result.model_dump(mode="json")

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sau-score", description="Score generated repos against SAU claims.")
    parser.add_argument("--paper", required=True, help="Paper id, or 'all'.")
    parser.add_argument("--dim", default="all", help="Dimension D1/D2/D3/D4, or 'all'.")
    parser.add_argument("--repo", help="Generated repository path for a single paper.")
    parser.add_argument("--repo-root", help="Root directory containing one generated repo per paper.")
    parser.add_argument("--data-dir", default="data/papers", help="Directory containing paper SAU JSON files.")
    parser.add_argument("--output", "-o", help="Write JSON result to this file.")
    parser.add_argument("--model", default="deepseek-v4-pro", help="Judge model name.")
    parser.add_argument("--search-model", default=None, help="Optional LLM search model.")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--cache-dir", default=None, help="Optional claim-level JSON cache directory.")
    parser.add_argument("--workers", type=int, default=1, help="Paper-level parallel workers for --paper all.")
    parser.add_argument("--judge-batch-size", type=int, default=4, help="Number of claims per LLM judge batch.")
    parser.add_argument("--judge-workers", type=int, default=2, help="Number of concurrent LLM judge requests per paper.")
    parser.add_argument("--llm-search", action="store_true", help="Use LLM ReAct search instead of deterministic search.")
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM judge and use heuristic five-level scoring fallback.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
