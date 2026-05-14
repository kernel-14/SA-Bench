"""
Main entry point for running all experiments from
"Conformal Prediction as Bayesian Quadrature" (Snell & Griffiths).

Usage:
    python run_experiments.py [--exp {binomial,heteroskedastic,coco,all}]
                              [--output_dir RESULTS_DIR]
                              [--seed SEED]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from config import get_default_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run experiments from 'Conformal Prediction as Bayesian Quadrature'"
    )
    parser.add_argument(
        "--exp",
        choices=["binomial", "heteroskedastic", "coco", "all"],
        default="all",
        help="Which experiment(s) to run (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results and figures (default: results/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for all experiments",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=None,
        help="Override number of trials M (default: 10000 per paper)",
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.output_dir = args.output_dir

    if args.seed is not None:
        cfg.synthetic_binomial.seed = args.seed
        cfg.synthetic_heteroskedastic.seed = args.seed
        cfg.coco.seed = args.seed

    if args.M is not None:
        cfg.synthetic_binomial.M = args.M
        cfg.synthetic_heteroskedastic.M = args.M
        cfg.coco.M = args.M

    os.makedirs(cfg.output_dir, exist_ok=True)

    run_binomial = args.exp in ("binomial", "all")
    run_heteroskedastic = args.exp in ("heteroskedastic", "all")
    run_coco = args.exp in ("coco", "all")

    if run_binomial:
        from experiments.synthetic_binomial import run_and_report as run_binomial_exp
        t0 = time.time()
        run_binomial_exp(cfg.synthetic_binomial, cfg.output_dir)
        print(f"  [Elapsed: {time.time() - t0:.1f}s]\n")

    if run_heteroskedastic:
        from experiments.synthetic_heteroskedastic import run_and_report as run_hetero_exp
        t0 = time.time()
        run_hetero_exp(cfg.synthetic_heteroskedastic, cfg.output_dir)
        print(f"  [Elapsed: {time.time() - t0:.1f}s]\n")

    if run_coco:
        from experiments.coco import run_and_report as run_coco_exp
        t0 = time.time()
        run_coco_exp(cfg.coco, cfg.output_dir)
        print(f"  [Elapsed: {time.time() - t0:.1f}s]\n")


if __name__ == "__main__":
    main()
