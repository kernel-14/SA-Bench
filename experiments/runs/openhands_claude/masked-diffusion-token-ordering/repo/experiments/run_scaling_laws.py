"""
π-learner scaling law experiment (Section 3.2, Figure 2 left).

Trains π-learners with different permutation types and measures their
scaling laws (validation loss vs. FLOPs) to demonstrate that:
  - ARM (identity permutation) achieves the best scaling law on text
  - Random permutation (MDM-like) achieves the worst
  - Intermediate permutations interpolate between the two

This reproduces Figure 2 (left) from the paper.

Usage:
  python experiments/run_scaling_laws.py --permutation_type random
  python experiments/run_scaling_laws.py --run_all  # runs all permutation types
"""

import sys
import os
import argparse
import json
import math
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_CONFIGS
from arm import PiLearner
from data import get_text_loaders, sample_permutation
from train_arm import train as train_arm
from evaluate import evaluate_pi_learner_likelihood
from utils import get_logger, load_checkpoint, set_seed, compute_flops_per_token

logger = get_logger("run_scaling_laws")


# ---------------------------------------------------------------------------
# IsoFLOP analysis (Section C.1)
# ---------------------------------------------------------------------------

# Model sizes for IsoFLOP analysis (non-embedding parameters)
# Following TinyLlama / Nie et al. (2024) configurations
ISOFLOP_MODEL_SIZES = ["6M", "19M", "42M", "170M"]

# FLOP budgets for IsoFLOP curves (log-spaced)
FLOP_BUDGETS = [
    1e17, 3e17, 1e18, 3e18, 1e19, 3e19, 1e20
]


def compute_optimal_tokens(
    flop_budget: float,
    n_params: int,
) -> int:
    """
    Compute optimal number of training tokens for a given FLOP budget.
    Following Hoffmann et al. (2022): tokens = C / (6 * N)
    """
    return int(flop_budget / (6.0 * n_params))


def run_isoflop_point(
    permutation_type: str,
    model_size: str,
    flop_budget: float,
    data_path: str,
    output_dir: str,
    device: str,
    seed: int,
    permutation_seed: int,
    use_wandb: bool = False,
) -> Tuple[float, float]:
    """
    Train a π-learner for a given FLOP budget and return (log_flops, val_loss).
    """
    model_config = MODEL_CONFIGS[model_size]
    # Approximate non-embedding parameter count
    d = model_config["d_model"]
    n_layers = model_config["n_layers"]
    d_ff = model_config["d_ff"]
    n_heads = model_config["n_heads"]
    # Rough estimate: 12 * n_layers * d^2 (attention + FFN)
    n_params = 12 * n_layers * d * d

    max_tokens = compute_optimal_tokens(flop_budget, n_params)
    seq_len = 2048
    batch_size = 512
    max_iters = max_tokens // (seq_len * batch_size)
    max_iters = max(max_iters, 100)

    run_name = f"{permutation_type}_{model_size}_flops{flop_budget:.0e}"
    run_output_dir = os.path.join(output_dir, run_name)

    logger.info(
        f"Training {run_name}: n_params={n_params:,}, "
        f"max_iters={max_iters}, flop_budget={flop_budget:.2e}"
    )

    train_arm(
        task="text",
        model_size=model_size,
        max_iters=max_iters,
        lr=4e-4,
        batch_size=batch_size,
        data_path=data_path,
        output_dir=run_output_dir,
        device=device,
        seed=seed,
        pi_learner=True,
        permutation_type=permutation_type,
        permutation_seed=permutation_seed,
        use_wandb=use_wandb,
        log_every=max(max_iters // 20, 10),
        save_every=max_iters + 1,  # only save at end
    )

    # Evaluate
    checkpoint_path = os.path.join(run_output_dir, "final_model.pt")
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return math.log10(flop_budget), float("nan")

    pi = sample_permutation(seq_len, permutation_type, seed=permutation_seed)
    model = PiLearner(
        vocab_size=32_000,
        seq_len=seq_len,
        model_config=model_config,
        pi=pi.to(device),
    ).to(device)
    load_checkpoint(model, None, None, checkpoint_path, device)

    _, val_loader = get_text_loaders(data_path, seq_len=seq_len, batch_size=64)
    val_loss = evaluate_pi_learner_likelihood(model, val_loader, device)

    log_flops = math.log10(flop_budget)
    logger.info(f"  {run_name}: log_flops={log_flops:.2f}, val_loss={val_loss:.4f}")

    return log_flops, val_loss


# ---------------------------------------------------------------------------
# Full scaling law experiment
# ---------------------------------------------------------------------------

def run_scaling_laws(
    permutation_type: str = "random",
    data_path: str = "data/slimpajama",
    output_dir: str = "outputs/scaling_laws",
    device: str = "cuda",
    seed: int = 42,
    n_permutation_samples: int = 3,
    use_wandb: bool = False,
):
    """
    Run IsoFLOP analysis for a given permutation type.

    Trains multiple models at different FLOP budgets and plots the scaling law.
    Repeats with n_permutation_samples different permutations (Section C.1).
    """
    set_seed(seed)
    device = device if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for sample_idx in range(n_permutation_samples):
        perm_seed = seed + sample_idx * 100
        sample_key = f"{permutation_type}_sample{sample_idx}"
        results = []

        for flop_budget in FLOP_BUDGETS:
            # Find optimal model size for this FLOP budget
            model_size = _select_model_size_for_flops(flop_budget)

            log_flops, val_loss = run_isoflop_point(
                permutation_type=permutation_type,
                model_size=model_size,
                flop_budget=flop_budget,
                data_path=data_path,
                output_dir=os.path.join(output_dir, sample_key),
                device=device,
                seed=seed,
                permutation_seed=perm_seed,
                use_wandb=use_wandb,
            )
            results.append({"log_flops": log_flops, "val_loss": val_loss})

        all_results[sample_key] = results

    # Save results
    results_path = os.path.join(output_dir, f"scaling_law_{permutation_type}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Scaling law results saved to {results_path}")
    return all_results


def _select_model_size_for_flops(flop_budget: float) -> str:
    """Select appropriate model size for a given FLOP budget."""
    if flop_budget < 1e18:
        return "6M"
    elif flop_budget < 1e19:
        return "19M"
    elif flop_budget < 1e20:
        return "42M"
    else:
        return "170M"


# ---------------------------------------------------------------------------
# Run all permutation types (Figure 2 left)
# ---------------------------------------------------------------------------

def run_all_scaling_laws(
    data_path: str = "data/slimpajama",
    output_dir: str = "outputs/scaling_laws",
    device: str = "cuda",
    seed: int = 42,
    use_wandb: bool = False,
):
    """
    Run scaling law experiments for all permutation types from Figure 2 (left).

    Permutation types:
      - "identity"    → ARM (left-to-right)
      - "random"      → MDM (Unif(S_L))
      - "closer"      → L/10 swaps from identity
      - "much_closer" → sqrt(L) swaps from identity
    """
    permutation_types = ["identity", "much_closer", "closer", "random"]
    all_results = {}

    for perm_type in permutation_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running scaling laws for permutation type: {perm_type}")
        results = run_scaling_laws(
            permutation_type=perm_type,
            data_path=data_path,
            output_dir=output_dir,
            device=device,
            seed=seed,
            n_permutation_samples=3,  # 3 samples per type (Section C.1)
            use_wandb=use_wandb,
        )
        all_results[perm_type] = results

    # Save combined results
    combined_path = os.path.join(output_dir, "all_scaling_laws.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nAll scaling law results saved to {combined_path}")
    _print_scaling_law_summary(all_results)

    return all_results


def _print_scaling_law_summary(all_results: Dict):
    """Print a summary of scaling law results."""
    logger.info("\nScaling Law Summary (Figure 2 left):")
    logger.info(f"{'Permutation Type':<20} {'Best Val Loss':<15}")
    logger.info("-" * 35)

    for perm_type, samples in all_results.items():
        all_losses = []
        for sample_results in samples.values():
            for point in sample_results:
                if not math.isnan(point["val_loss"]):
                    all_losses.append(point["val_loss"])
        if all_losses:
            best_loss = min(all_losses)
            logger.info(f"{perm_type:<20} {best_loss:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="π-learner scaling law experiment")
    parser.add_argument("--permutation_type", type=str, default="random",
                        choices=["identity", "random", "closer", "much_closer"])
    parser.add_argument("--data_path", type=str, default="data/slimpajama")
    parser.add_argument("--output_dir", type=str, default="outputs/scaling_laws")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_permutation_samples", type=int, default=3)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all permutation types (Figure 2 left)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.run_all:
        run_all_scaling_laws(
            data_path=args.data_path,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            use_wandb=args.use_wandb,
        )
    else:
        run_scaling_laws(
            permutation_type=args.permutation_type,
            data_path=args.data_path,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            n_permutation_samples=args.n_permutation_samples,
            use_wandb=args.use_wandb,
        )
