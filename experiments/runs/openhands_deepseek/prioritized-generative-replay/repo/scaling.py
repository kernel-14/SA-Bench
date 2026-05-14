"""Scaling experiments from Section 5.3.

Evaluates PGR scaling behavior:
(1) Larger networks (Fig. 7a)
(2) Higher synthetic data ratios (Fig. 7b)
(3) Combined with higher UTD (Fig. 7c)
"""

import os
import argparse
from typing import List, Tuple
import numpy as np
import torch

from config import RunConfig
from train import run_experiment, run_synther_baseline


def experiment_larger_network(base_config: RunConfig, log_dir: str, seeds: List[int] = [0, 1, 2]):
    """Fig. 7a: Scale network size 6x (2->3 layers, 256->512 width)."""
    results = {"pgr": [], "synther": []}

    for seed in seeds:
        config = base_config
        config.seed = seed
        config.policy.hidden_dims = 512
        config.policy.n_hidden_layers = 3
        config.policy.batch_size = 1024  # maintain per-parameter throughput

        # PGR with larger network
        pgr_result = run_experiment(config, os.path.join(log_dir, f"pgr_larger_seed{seed}"))
        results["pgr"].append(pgr_result[0])

        # SYNTHER with larger network
        syn_result = run_synther_baseline(config, os.path.join(log_dir, f"synther_larger_seed{seed}"))
        results["synther"].append(syn_result[0])

    print(f"[Larger Network] PGR: {np.mean(results['pgr']):.2f} ± {np.std(results['pgr']):.2f}")
    print(f"[Larger Network] SYNTHER: {np.mean(results['synther']):.2f} ± {np.std(results['synther']):.2f}")
    return results


def experiment_higher_ratios(base_config: RunConfig, log_dir: str, seeds: List[int] = [0, 1, 2]):
    """Fig. 7b: Increase synthetic data ratio while keeping real data fixed.

    Configurations:
    - r=0.5, batch=256 (baseline: 128 real + 128 syn)
    - r=0.75, batch=512 (128 real + 384 syn)
    - r=0.875, batch=1024 (128 real + 896 syn)
    """
    ratios = [0.5, 0.75, 0.875]
    batches = [256, 512, 1024]

    results = {"pgr": {r: [] for r in ratios}, "synther": {r: [] for r in ratios}}

    for r, bs in zip(ratios, batches):
        for seed in seeds:
            config = base_config
            config.seed = seed
            config.policy.synthetic_ratio = r
            config.policy.batch_size = bs

            # PGR
            pgr_return, _ = run_experiment(config, os.path.join(log_dir, f"pgr_r{r}_seed{seed}"))
            results["pgr"][r].append(pgr_return)

            # SYNTHER
            syn_return, _ = run_synther_baseline(config, os.path.join(log_dir, f"synther_r{r}_seed{seed}"))
            results["synther"][r].append(syn_return)

    for model in ["pgr", "synther"]:
        for r in ratios:
            vals = results[model][r]
            print(f"[Ratio r={r}] {model.upper()}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")

    return results


def experiment_combined_scaling(base_config: RunConfig, log_dir: str, seeds: List[int] = [0, 1, 2]):
    """Fig. 7c: Combine larger network + higher ratio + higher UTD.

    Configuration:
    - 3 layers, 512 width
    - r=0.75, batch=512
    - UTD=40 (doubled from 20)
    - D_syn capacity = 2M (doubled from 1M)
    """
    results = {"pgr": [], "synther": []}

    for seed in seeds:
        config = base_config
        config.seed = seed
        config.policy.hidden_dims = 512
        config.policy.n_hidden_layers = 3
        config.policy.batch_size = 512
        config.policy.synthetic_ratio = 0.75
        config.scaling_utd = 40
        config.replay.syn_buffer_capacity = 2_000_000

        # PGR
        pgr_return, _ = run_experiment(config, os.path.join(log_dir, f"pgr_combined_seed{seed}"))
        results["pgr"].append(pgr_return)

        # SYNTHER
        syn_return, _ = run_synther_baseline(config, os.path.join(log_dir, f"synther_combined_seed{seed}"))
        results["synther"].append(syn_return)

    print(f"[Combined] PGR: {np.mean(results['pgr']):.2f} ± {np.std(results['pgr']):.2f}")
    print(f"[Combined] SYNTHER: {np.mean(results['synther']):.2f} ± {np.std(results['synther']):.2f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGR Scaling Experiments")
    parser.add_argument("--experiment", type=str, default="larger_network",
                        choices=["larger_network", "higher_ratios", "combined"])
    parser.add_argument("--env_domain", type=str, default="quadruped")
    parser.add_argument("--env_task", type=str, default="walk")
    parser.add_argument("--log_dir", type=str, default="./logs")
    args = parser.parse_args()

    from config import get_dmc_state_config
    base_config = get_dmc_state_config()
    base_config.env.dmc_domain = args.env_domain
    base_config.env.dmc_task = args.env_task

    os.makedirs(args.log_dir, exist_ok=True)

    if args.experiment == "larger_network":
        experiment_larger_network(base_config, args.log_dir)
    elif args.experiment == "higher_ratios":
        experiment_higher_ratios(base_config, args.log_dir)
    elif args.experiment == "combined":
        experiment_combined_scaling(base_config, args.log_dir)
