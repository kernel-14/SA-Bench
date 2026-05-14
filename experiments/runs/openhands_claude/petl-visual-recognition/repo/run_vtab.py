"""
Run all VTAB-1K experiments for all PEFT methods.

This script reproduces Table 1 from the paper by running all 14 PEFT methods
plus linear probing and full fine-tuning on all 19 VTAB-1K tasks.

Usage:
  python run_vtab.py --data_dir ./data --output_dir ./outputs --device cuda
  python run_vtab.py --method lora --task cifar100  # single method/task
  python run_vtab.py --methods lora bitfit --tasks cifar100 dtd  # subset
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import torch

from config import (
    VTABConfig,
    VTAB_ALL_TASKS,
    VTAB_NUM_CLASSES,
    ALL_PEFT_METHODS,
)
from train import run_vtab_experiment
from evaluate import (
    aggregate_vtab_results,
    compute_ranking_frequency,
    compute_relative_std_dev,
    load_evaluation_results,
)
from utils import set_seed, format_results_table


def run_all_vtab(
    methods: List[str],
    tasks: List[str],
    config: VTABConfig,
    device: torch.device,
    data_dir: str,
    output_dir: str,
    do_hparam_search: bool = True,
    skip_existing: bool = True,
) -> None:
    """Run VTAB-1K experiments for all method/task combinations."""
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for method in methods:
        all_results[method] = {}
        for task in tasks:
            results_path = Path(output_dir) / f"{method}_{task}_results.json"

            if skip_existing and results_path.exists():
                with open(results_path) as f:
                    data = json.load(f)
                all_results[method][task] = data.get("test_acc", 0.0)
                print(f"Skipping {method}/{task} (already done): {all_results[method][task]:.2f}%")
                continue

            try:
                results = run_vtab_experiment(
                    peft_method=method,
                    task_name=task,
                    config=config,
                    device=device,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    do_hparam_search=do_hparam_search,
                )
                all_results[method][task] = results["test_acc"]
            except Exception as e:
                print(f"ERROR: {method}/{task}: {e}")
                all_results[method][task] = 0.0

    # Save aggregated results
    summary_path = Path(output_dir) / "vtab_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print results table
    print("\n" + "=" * 80)
    print("VTAB-1K Results (Top-1 Accuracy %)")
    print("=" * 80)
    print(format_results_table(all_results, methods, tasks))

    # Print group averages
    aggregated = aggregate_vtab_results(all_results)
    print("\nGroup Averages:")
    for method in methods:
        agg = aggregated.get(method, {})
        print(f"  {method:<20} Natural: {agg.get('natural', 0):.1f}  "
              f"Specialized: {agg.get('specialized', 0):.1f}  "
              f"Structured: {agg.get('structured', 0):.1f}  "
              f"Overall: {agg.get('overall', 0):.1f}")

    # Print relative standard deviations
    rel_std = compute_relative_std_dev(all_results)
    print("\nRelative Std Dev across methods (per task):")
    for task, std in rel_std.items():
        print(f"  {task:<30}: {std:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Run VTAB-1K PEFT experiments")
    parser.add_argument("--methods", nargs="+", default=ALL_PEFT_METHODS,
                        help="PEFT methods to evaluate")
    parser.add_argument("--tasks", nargs="+", default=VTAB_ALL_TASKS,
                        help="VTAB tasks to evaluate")
    parser.add_argument("--data_dir", type=str, default="./data/vtab")
    parser.add_argument("--output_dir", type=str, default="./outputs/vtab")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_hparam_search", action="store_true")
    parser.add_argument("--no_skip_existing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = VTABConfig()

    run_all_vtab(
        methods=args.methods,
        tasks=args.tasks,
        config=config,
        device=device,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        do_hparam_search=not args.no_hparam_search,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
