from __future__ import annotations

import argparse
from pathlib import Path

from baselines.BasicAgent.config import AppConfig
from baselines.BasicAgent.runner import run_case


DEFAULT_CASES = [
    "fre",
    "sample-specific-masks",
    "mechanistic-understanding",
    "pinn",
    "all-in-one",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PaperBench-lite static BasicAgent runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one case")
    run_parser.add_argument("--case", required=True)
    run_parser.add_argument("--model", default=None)
    run_parser.add_argument("--api-base-url", default=None)
    run_parser.add_argument("--api-key", default=None)
    run_parser.add_argument("--max-steps", type=int, default=80)
    run_parser.add_argument("--time-limit-seconds", type=int, default=900)
    run_parser.add_argument("--dry-run", action="store_true")

    batch_parser = subparsers.add_parser("batch", help="Run the default five cases")
    batch_parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    batch_parser.add_argument("--model", default=None)
    batch_parser.add_argument("--api-base-url", default=None)
    batch_parser.add_argument("--api-key", default=None)
    batch_parser.add_argument("--max-steps", type=int, default=80)
    batch_parser.add_argument("--time-limit-seconds", type=int, default=900)
    batch_parser.add_argument("--dry-run", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    config = AppConfig.from_env(
        repo_root=repo_root,
        model_name=args.model,
        api_key=args.api_key,
        api_base_url=args.api_base_url,
        max_steps=args.max_steps,
        time_limit_seconds=args.time_limit_seconds,
    )

    if args.command == "run":
        run_case(args.case, config=config, dry_run=args.dry_run)
        return

    for case_id in args.cases:
        run_case(case_id, config=config, dry_run=args.dry_run)
