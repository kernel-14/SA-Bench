"""
L&O-NAE-SAT experiment (Section 3.3, 4.2, Table 1, Figure 2 right).

Trains an MDM on the L&O-NAE-SAT distribution and evaluates:
  1. Error imbalance across masking problems (Figure 2 right)
  2. Vanilla vs. adaptive inference accuracy (Table 1)

Usage:
  python experiments/run_nae_sat.py --N 25 --P 275
  python experiments/run_nae_sat.py --N 100 --P 200 --eval_only --checkpoint outputs/nae_sat/best_model.pt
"""

import sys
import os
import argparse
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_CONFIGS
from mdm import MDM
from data import NAESATDataset, get_nae_sat_loaders
from inference import mdm_solve_puzzle, ORACLE_REGISTRY
from train_mdm import train
from utils import get_logger, load_checkpoint, set_seed

logger = get_logger("run_nae_sat")


# ---------------------------------------------------------------------------
# Error imbalance analysis (Section 3.3, Figure 2 right)
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_error_imbalance(
    model: MDM,
    proxy_model: MDM,
    dataset: NAESATDataset,
    N: int,
    P: int,
    ell: int = 11,
    n_repeats: int = 1000,
    device: str = "cuda",
) -> Dict[str, List[float]]:
    """
    Measure error imbalance across masking problems (Section C.2.1).

    For each ℓ ∈ [1, N-1], randomly mask ℓ latent tokens and ℓ×(P/N) observation tokens.
    Measure the squared error between model predictions and proxy (Bayes-optimal) predictions.

    Returns:
        dict with 'latent_errors' and 'obs_errors' lists
    """
    model.eval()
    proxy_model.eval()

    latent_errors = []
    obs_errors = []

    for _ in tqdm(range(n_repeats), desc="Measuring error imbalance"):
        idx = np.random.randint(len(dataset))
        item = dataset[idx]
        x0 = item["x0"].unsqueeze(0).to(device)

        # Mask ℓ latent tokens and ℓ×(P/N) observation tokens
        n_obs_mask = int(ell * P / N)

        latent_indices = torch.randperm(N)[:ell]
        obs_indices = N + torch.randperm(P)[:n_obs_mask]

        x_masked = x0.clone()
        x_masked[0, latent_indices] = MDM.MASK_TOKEN_ID
        x_masked[0, obs_indices] = MDM.MASK_TOKEN_ID

        # Get predictions from both models
        model_probs = model.get_token_probs(x_masked)    # (1, L, vocab_size)
        proxy_probs = proxy_model.get_token_probs(x_masked)

        # Compute squared error in log-probability space
        model_log_probs = torch.log(model_probs + 1e-10)
        proxy_log_probs = torch.log(proxy_probs + 1e-10)

        # Error at latent positions
        for i in latent_indices:
            err = ((model_log_probs[0, i] - proxy_log_probs[0, i]) ** 2).sum().item()
            latent_errors.append(err)

        # Error at observation positions
        for i in obs_indices:
            err = ((model_log_probs[0, i] - proxy_log_probs[0, i]) ** 2).sum().item()
            obs_errors.append(err)

    return {"latent_errors": latent_errors, "obs_errors": obs_errors}


# ---------------------------------------------------------------------------
# Vanilla vs. adaptive inference accuracy (Table 1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_inference_strategies(
    model: MDM,
    dataset: NAESATDataset,
    N: int,
    P: int,
    num_steps: int = 50,
    n_eval: int = 1000,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compare vanilla and adaptive inference on L&O-NAE-SAT (Table 1).

    Measures accuracy in predicting observation tokens.
    """
    model.eval()
    results = {}

    for strategy in ["vanilla", "top_prob_margin"]:
        correct = 0
        total = 0

        for idx in tqdm(range(min(n_eval, len(dataset))),
                        desc=f"Evaluating {strategy}"):
            item = dataset[idx]
            x0 = item["x0"].unsqueeze(0).to(device)

            # Fully masked input
            puzzle = torch.zeros_like(x0)

            pred = mdm_solve_puzzle(
                model=model,
                puzzle=puzzle,
                num_steps=num_steps,
                strategy=strategy,
                gumbel_noise_coeff=0.0,
                mask_token_id=MDM.MASK_TOKEN_ID,
            )

            # Evaluate on observation positions
            obs_pred = pred[0, N:N + P]
            obs_true = x0[0, N:N + P]
            correct += (obs_pred == obs_true).float().sum().item()
            total += P

        results[strategy] = correct / max(total, 1)

    return results


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_nae_sat_experiment(
    N: int = 25,
    P: int = 275,
    model_size: str = "19M",
    output_dir: str = "outputs/nae_sat",
    device: str = "cuda",
    seed: int = 42,
    eval_only: bool = False,
    checkpoint: str = None,
    proxy_checkpoint: str = None,
    num_steps: int = 50,
    use_wandb: bool = False,
):
    set_seed(seed)
    device = device if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"L&O-NAE-SAT experiment: N={N}, P={P}")

    if not eval_only:
        # Train MDM for 2000 iterations (Section C.2.1)
        logger.info("Training MDM (2000 iterations)...")
        train(
            task="nae_sat",
            model_size=model_size,
            max_iters=2_000,
            lr=4e-4,
            batch_size=128,
            output_dir=os.path.join(output_dir, f"N{N}_P{P}"),
            device=device,
            seed=seed,
            N=N,
            P=P,
            use_wandb=use_wandb,
        )

        # Train proxy MDM for 50000 iterations (Bayes-optimal proxy)
        logger.info("Training proxy MDM (50000 iterations)...")
        train(
            task="nae_sat",
            model_size=model_size,
            max_iters=50_000,
            lr=4e-4,
            batch_size=128,
            output_dir=os.path.join(output_dir, f"N{N}_P{P}_proxy"),
            device=device,
            seed=seed + 1,
            N=N,
            P=P,
            use_wandb=use_wandb,
        )

        checkpoint = os.path.join(output_dir, f"N{N}_P{P}", "final_model.pt")
        proxy_checkpoint = os.path.join(output_dir, f"N{N}_P{P}_proxy", "final_model.pt")

    # Load models
    model_config = MODEL_CONFIGS[model_size]
    vocab_size = 5
    seq_len = 512

    model = MDM(vocab_size=vocab_size, seq_len=seq_len, model_config=model_config).to(device)
    load_checkpoint(model, None, None, checkpoint, device)

    # Evaluate inference strategies (Table 1)
    logger.info("Evaluating inference strategies...")
    test_dataset = NAESATDataset(N, P, num_samples=1000, seed=seed + 2)
    results = evaluate_inference_strategies(
        model, test_dataset, N, P, num_steps=num_steps, device=device
    )

    logger.info(f"\nTable 1 results for (N={N}, P={P}):")
    logger.info(f"  Vanilla inference:  {results['vanilla']*100:.2f}%")
    logger.info(f"  Adaptive inference: {results['top_prob_margin']*100:.2f}%")

    # Error imbalance analysis (Figure 2 right)
    if proxy_checkpoint and os.path.exists(proxy_checkpoint):
        proxy_model = MDM(
            vocab_size=vocab_size, seq_len=seq_len, model_config=model_config
        ).to(device)
        load_checkpoint(proxy_model, None, None, proxy_checkpoint, device)

        logger.info("Measuring error imbalance...")
        train_dataset = NAESATDataset(N, P, num_samples=10_000, seed=seed)
        errors = measure_error_imbalance(
            model, proxy_model, train_dataset, N, P, ell=11, n_repeats=1000, device=device
        )

        mean_latent_err = np.mean(errors["latent_errors"])
        mean_obs_err = np.mean(errors["obs_errors"])
        logger.info(f"\nError imbalance (ℓ=11):")
        logger.info(f"  Mean latent error:      {mean_latent_err:.4f}")
        logger.info(f"  Mean observation error: {mean_obs_err:.4f}")
        logger.info(f"  Ratio (latent/obs):     {mean_latent_err/max(mean_obs_err, 1e-10):.2f}x")

    return results


# ---------------------------------------------------------------------------
# Run all (N, P) configurations from Table 1
# ---------------------------------------------------------------------------

def run_all_nae_sat_configs(
    output_dir: str = "outputs/nae_sat",
    device: str = "cuda",
    seed: int = 42,
    use_wandb: bool = False,
):
    configs = [(25, 275), (30, 270), (40, 260), (50, 250), (100, 200)]

    all_results = {}
    for N, P in configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running N={N}, P={P}")
        results = run_nae_sat_experiment(
            N=N, P=P,
            output_dir=os.path.join(output_dir, f"N{N}_P{P}"),
            device=device, seed=seed, use_wandb=use_wandb,
        )
        all_results[(N, P)] = results

    logger.info("\n" + "="*60)
    logger.info("Table 1: L&O-NAE-SAT Results")
    logger.info(f"{'(N, P)':<15} {'Vanilla':<20} {'Adaptive':<20}")
    logger.info("-" * 55)
    for (N, P), res in all_results.items():
        logger.info(
            f"({N}, {P}){'':<8} "
            f"{res['vanilla']*100:.2f}%{'':<12} "
            f"{res['top_prob_margin']*100:.2f}%"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="L&O-NAE-SAT experiment")
    parser.add_argument("--N", type=int, default=25)
    parser.add_argument("--P", type=int, default=275)
    parser.add_argument("--model_size", type=str, default="19M")
    parser.add_argument("--output_dir", type=str, default="outputs/nae_sat")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--proxy_checkpoint", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all (N, P) configurations from Table 1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.run_all:
        run_all_nae_sat_configs(
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            use_wandb=args.use_wandb,
        )
    else:
        run_nae_sat_experiment(
            N=args.N,
            P=args.P,
            model_size=args.model_size,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            eval_only=args.eval_only,
            checkpoint=args.checkpoint,
            proxy_checkpoint=args.proxy_checkpoint,
            num_steps=args.num_steps,
            use_wandb=args.use_wandb,
        )
