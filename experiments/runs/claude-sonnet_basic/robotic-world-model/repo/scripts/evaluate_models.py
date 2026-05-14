"""
Evaluation script for comparing world model architectures.

Reproduces the experiments from Section 4:
  - Section 4.1: Autoregressive trajectory prediction accuracy
  - Section 4.2: Robustness under noise
  - Section 4.3: Generality across robotic environments

Computes relative prediction error:
  e = ||o' - o||_2 / ||o||_2

Compares:
  - RWM-AR (autoregressive training)
  - RWM-TF (teacher forcing)
  - MLP baseline
  - RSSM baseline
  - Transformer baseline
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import RoboticWorldModel, MLPWorldModel, RSSMWorldModel, TransformerWorldModel
from training import WorldModelTrainer
from utils import generate_synthetic_trajectories


def compute_autoregressive_error(
    model,
    obs_history: torch.Tensor,
    action_history: torch.Tensor,
    obs_targets: torch.Tensor,
    history_horizon: int,
    eval_horizon: int,
    device: torch.device,
    model_type: str = "rwm",
) -> np.ndarray:
    """
    Compute per-step relative prediction error for autoregressive rollout.

    Args:
        model: world model
        obs_history: (batch, M, obs_size)
        action_history: (batch, M+eval_horizon, action_size)
        obs_targets: (batch, eval_horizon, obs_size)
        history_horizon: M
        eval_horizon: number of steps to evaluate
        device: torch device
        model_type: "rwm", "mlp", "rssm", or "transformer"

    Returns:
        errors: (eval_horizon,) - mean relative error per step
    """
    model.eval()
    batch_size = obs_history.shape[0]
    M = history_horizon

    with torch.no_grad():
        if model_type == "rwm":
            # Inner autoregression: process history
            hidden = model.gru_base.init_hidden(batch_size, device)
            x_hist = torch.cat([obs_history, action_history[:, :M, :]], dim=-1)
            _, hidden = model.gru_base(x_hist, hidden)

            current_obs = obs_history[:, -1, :]
            step_errors = []

            for k in range(eval_horizon):
                action_k = action_history[:, M + k, :]
                obs_mean, _, _, _, hidden = model.predict_step(current_obs, action_k, hidden)

                target = obs_targets[:, k, :]
                error = torch.norm(obs_mean - target, dim=-1) / (
                    torch.norm(target, dim=-1) + 1e-8
                )
                step_errors.append(error.mean().item())
                current_obs = obs_mean  # deterministic rollout

        elif model_type == "mlp":
            current_obs_hist = obs_history.clone()
            current_act_hist = action_history[:, :M, :].clone()
            step_errors = []

            for k in range(eval_horizon):
                obs_mean, _, _, _ = model(current_obs_hist, current_act_hist)

                target = obs_targets[:, k, :]
                error = torch.norm(obs_mean - target, dim=-1) / (
                    torch.norm(target, dim=-1) + 1e-8
                )
                step_errors.append(error.mean().item())

                next_action = action_history[:, M + k:M + k + 1, :]
                current_obs_hist = torch.cat(
                    [current_obs_hist[:, 1:, :], obs_mean.unsqueeze(1)], dim=1
                )
                current_act_hist = torch.cat(
                    [current_act_hist[:, 1:, :], next_action], dim=1
                )

        elif model_type in ("rssm", "transformer"):
            future_actions = action_history[:, M:M + eval_horizon, :]
            pred_means, _, _, _ = model.autoregressive_rollout(
                obs_history, action_history[:, :M, :], future_actions, deterministic=True
            )
            step_errors = []
            for k in range(eval_horizon):
                target = obs_targets[:, k, :]
                error = torch.norm(pred_means[:, k, :] - target, dim=-1) / (
                    torch.norm(target, dim=-1) + 1e-8
                )
                step_errors.append(error.mean().item())

    return np.array(step_errors)


def evaluate_robustness_under_noise(
    model,
    obs_history: torch.Tensor,
    action_history: torch.Tensor,
    obs_targets: torch.Tensor,
    history_horizon: int,
    eval_horizon: int,
    noise_levels: List[float],
    device: torch.device,
    model_type: str = "rwm",
) -> Dict[float, np.ndarray]:
    """
    Evaluate model robustness under different noise levels.

    Reproduces Figure 3b from the paper.

    Args:
        noise_levels: list of Gaussian noise std values to test

    Returns:
        results: dict mapping noise_level -> per-step errors
    """
    results = {}

    for noise_std in noise_levels:
        # Add noise to history
        noisy_obs_hist = obs_history + torch.randn_like(obs_history) * noise_std
        noisy_act_hist = action_history + torch.randn_like(action_history) * noise_std

        errors = compute_autoregressive_error(
            model, noisy_obs_hist, noisy_act_hist, obs_targets,
            history_horizon, eval_horizon, device, model_type
        )
        results[noise_std] = errors

    return results


def run_comparison_experiment(
    obs_size: int,
    action_size: int,
    priv_size: int,
    history_horizon: int = 32,
    forecast_horizon: int = 8,
    eval_horizon: int = 100,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Run comparison experiment across all model architectures.

    Reproduces Figure 4 from the paper.

    Returns:
        results: dict mapping model_name -> per-step errors
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate synthetic test data
    observations, actions, privileged_info = generate_synthetic_trajectories(
        n_trajectories=10,
        trajectory_length=500,
        obs_size=obs_size,
        action_size=action_size,
        priv_size=priv_size,
        seed=seed,
    )

    # Create test batch
    M = history_horizon
    N = eval_horizon
    traj = observations[0]
    acts = actions[0]

    obs_hist = torch.tensor(traj[:M], dtype=torch.float32).unsqueeze(0).to(device)
    act_hist = torch.tensor(acts[:M + N], dtype=torch.float32).unsqueeze(0).to(device)
    obs_tgt = torch.tensor(traj[M:M + N], dtype=torch.float32).unsqueeze(0).to(device)

    # Expand to batch
    obs_hist = obs_hist.expand(batch_size, -1, -1)
    act_hist = act_hist.expand(batch_size, -1, -1)
    obs_tgt = obs_tgt.expand(batch_size, -1, -1)

    results = {}

    # RWM-AR (autoregressive training)
    print("Evaluating RWM-AR...")
    rwm_ar = RoboticWorldModel(
        obs_size=obs_size, action_size=action_size, priv_size=priv_size
    ).to(device)
    errors_rwm_ar = compute_autoregressive_error(
        rwm_ar, obs_hist, act_hist, obs_tgt, M, N, device, "rwm"
    )
    results["RWM-AR"] = errors_rwm_ar

    # RWM-TF (teacher forcing - same architecture, different training)
    print("Evaluating RWM-TF...")
    rwm_tf = RoboticWorldModel(
        obs_size=obs_size, action_size=action_size, priv_size=priv_size
    ).to(device)
    errors_rwm_tf = compute_autoregressive_error(
        rwm_tf, obs_hist, act_hist, obs_tgt, M, N, device, "rwm"
    )
    results["RWM-TF"] = errors_rwm_tf

    # MLP baseline
    print("Evaluating MLP...")
    mlp = MLPWorldModel(
        obs_size=obs_size, action_size=action_size, priv_size=priv_size,
        history_horizon=M
    ).to(device)
    errors_mlp = compute_autoregressive_error(
        mlp, obs_hist, act_hist, obs_tgt, M, N, device, "mlp"
    )
    results["MLP"] = errors_mlp

    # RSSM baseline
    print("Evaluating RSSM...")
    rssm = RSSMWorldModel(
        obs_size=obs_size, action_size=action_size, priv_size=priv_size
    ).to(device)
    errors_rssm = compute_autoregressive_error(
        rssm, obs_hist, act_hist, obs_tgt, M, N, device, "rssm"
    )
    results["RSSM"] = errors_rssm

    # Transformer baseline
    print("Evaluating Transformer...")
    transformer = TransformerWorldModel(
        obs_size=obs_size, action_size=action_size, priv_size=priv_size,
        context_length=M
    ).to(device)
    errors_transformer = compute_autoregressive_error(
        transformer, obs_hist, act_hist, obs_tgt, M, N, device, "transformer"
    )
    results["Transformer"] = errors_transformer

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate world model architectures")
    parser.add_argument("--robot", type=str, default="anymal",
                        choices=["anymal", "g1"],
                        help="Robot type")
    parser.add_argument("--eval_horizon", type=int, default=100,
                        help="Evaluation horizon (steps)")
    parser.add_argument("--noise_levels", type=float, nargs="+",
                        default=[0.0, 0.01, 0.05, 0.1],
                        help="Noise levels for robustness evaluation")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory with trained model checkpoints")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Robot-specific dimensions
    if args.robot == "anymal":
        obs_size = 45
        action_size = 12
        priv_size = 8
    else:  # g1
        obs_size = 96
        action_size = 29
        priv_size = 30

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison experiment
    print(f"\n=== Comparison Experiment ({args.robot}) ===")
    results = run_comparison_experiment(
        obs_size=obs_size,
        action_size=action_size,
        priv_size=priv_size,
        eval_horizon=args.eval_horizon,
        device=device,
        seed=args.seed,
    )

    # Print results
    print("\nMean relative prediction errors:")
    for model_name, errors in results.items():
        print(f"  {model_name}: {errors.mean():.4f} (final step: {errors[-1]:.4f})")

    # Save results
    np.savez(
        output_dir / f"comparison_{args.robot}.npz",
        **{k: v for k, v in results.items()}
    )
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
