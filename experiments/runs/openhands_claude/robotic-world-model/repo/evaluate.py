"""
Evaluation utilities for RWM.

Implements:
  - relative_prediction_error: primary metric from Fig. 3, 4 (Sec. 4.1-4.3)
  - autoregressive_rollout_evaluation: evaluate model over long horizons
  - noise_robustness_evaluation: evaluate under Gaussian noise (Sec. 4.2)
  - model_error_tracking: track model error during policy training (Fig. 5)
  - ablation_heatmap: M x N ablation study (Fig. S8)
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ExperimentConfig
from data import Trajectory, TrajectoryDataset, build_dataloader
from model import MLPBaseline, RSSMBaseline, RWM, TransformerBaseline


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def relative_prediction_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Relative prediction error e = ||pred - target||_2 / (||target||_2 + eps).

    Used as the primary evaluation metric in Fig. 3b, 4.

    Args:
        pred:   (B, ...) predicted values
        target: (B, ...) ground truth values

    Returns:
        error: (B,) per-sample relative error
    """
    diff_norm = (pred - target).norm(dim=-1)
    target_norm = target.norm(dim=-1) + eps
    return diff_norm / target_norm


def mean_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean(dim=-1)


def mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean(dim=-1)


# ---------------------------------------------------------------------------
# Autoregressive Rollout Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_autoregressive_rollout(
    model: nn.Module,
    dataset: TrajectoryDataset,
    device: torch.device,
    batch_size: int = 256,
    max_rollout_steps: int = 200,
    use_mean: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Evaluate autoregressive prediction accuracy over extended rollouts.

    Corresponds to the evaluation in Sec. 4.1 and Fig. 3a.

    Returns:
        Dict with keys:
          - "relative_error": (max_rollout_steps,) mean relative error per step
          - "mse": (max_rollout_steps,) mean MSE per step
          - "mae": (max_rollout_steps,) mean MAE per step
    """
    model.eval()
    M = dataset.history_horizon
    N = dataset.forecast_horizon

    all_rel_errors = [[] for _ in range(max_rollout_steps)]
    all_mse = [[] for _ in range(max_rollout_steps)]

    dataloader = build_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for obs_batch, action_batch, _ in dataloader:
        obs_batch = obs_batch.to(device)
        action_batch = action_batch.to(device)

        obs_history = obs_batch[:, :M]
        action_history = action_batch[:, :M]

        # Roll out autoregressively beyond the training forecast horizon
        total_steps = min(max_rollout_steps, obs_batch.size(1) - M)
        action_forecast = action_batch[:, M : M + total_steps]
        obs_targets = obs_batch[:, M : M + total_steps]

        obs_preds, _ = _predict_model(
            model, obs_history, action_history, action_forecast, use_mean
        )

        for k in range(min(total_steps, obs_preds.size(1))):
            pred_k = obs_preds[:, k]
            target_k = obs_targets[:, k]
            rel_err = relative_prediction_error(pred_k, target_k)
            mse = mean_squared_error(pred_k, target_k)
            all_rel_errors[k].extend(rel_err.cpu().numpy().tolist())
            all_mse[k].extend(mse.cpu().numpy().tolist())

    rel_errors = np.array([np.mean(e) if e else np.nan for e in all_rel_errors])
    mse_vals = np.array([np.mean(e) if e else np.nan for e in all_mse])

    return {"relative_error": rel_errors, "mse": mse_vals}


def _predict_model(
    model: nn.Module,
    obs_history: torch.Tensor,
    action_history: torch.Tensor,
    action_forecast: torch.Tensor,
    use_mean: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified prediction interface for all model types."""
    if hasattr(model, "predict"):
        return model.predict(obs_history, action_history, action_forecast, use_mean)
    elif hasattr(model, "core"):
        obs_means, obs_stds, priv_means, priv_stds = model.core.autoregressive_rollout(
            obs_history, action_history, action_forecast, use_mean=use_mean
        )
        return torch.stack(obs_means, dim=1), torch.stack(priv_means, dim=1)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


# ---------------------------------------------------------------------------
# Noise Robustness Evaluation (Sec. 4.2, Fig. 3b)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_noise_robustness(
    model: nn.Module,
    dataset: TrajectoryDataset,
    device: torch.device,
    noise_levels: List[float],
    batch_size: int = 256,
    max_rollout_steps: int = 100,
) -> Dict[float, np.ndarray]:
    """
    Evaluate model robustness under Gaussian noise perturbations.

    Corresponds to Sec. 4.2 and Fig. 3b.

    Args:
        noise_levels: list of noise standard deviations to test

    Returns:
        Dict mapping noise_level → relative_error array (max_rollout_steps,)
    """
    model.eval()
    M = dataset.history_horizon
    results = {}

    dataloader = build_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for noise_std in noise_levels:
        all_rel_errors = [[] for _ in range(max_rollout_steps)]

        for obs_batch, action_batch, _ in dataloader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)

            # Add Gaussian noise to observations and actions
            obs_noisy = obs_batch + torch.randn_like(obs_batch) * noise_std
            action_noisy = action_batch + torch.randn_like(action_batch) * noise_std

            obs_history = obs_noisy[:, :M]
            action_history = action_noisy[:, :M]
            total_steps = min(max_rollout_steps, obs_batch.size(1) - M)
            action_forecast = action_noisy[:, M : M + total_steps]
            obs_targets = obs_batch[:, M : M + total_steps]  # clean targets

            obs_preds, _ = _predict_model(
                model, obs_history, action_history, action_forecast, use_mean=False
            )

            for k in range(min(total_steps, obs_preds.size(1))):
                rel_err = relative_prediction_error(obs_preds[:, k], obs_targets[:, k])
                all_rel_errors[k].extend(rel_err.cpu().numpy().tolist())

        results[noise_std] = np.array([np.mean(e) if e else np.nan for e in all_rel_errors])

    return results


# ---------------------------------------------------------------------------
# Multi-Model Comparison (Sec. 4.3, Fig. 4)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compare_models(
    models: Dict[str, nn.Module],
    dataset: TrajectoryDataset,
    device: torch.device,
    batch_size: int = 256,
    max_rollout_steps: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compare multiple models on autoregressive prediction error.

    Corresponds to Fig. 4 experiments.

    Args:
        models: dict mapping model_name → model

    Returns:
        Dict mapping model_name → relative_error array
    """
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_autoregressive_rollout(
            model, dataset, device, batch_size, max_rollout_steps
        )
        results[name] = metrics["relative_error"]
    return results


# ---------------------------------------------------------------------------
# Model Error During Policy Training (Fig. 5)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_model_error(
    world_model: RWM,
    real_trajectories: List[Trajectory],
    device: torch.device,
    history_horizon: int = 32,
    eval_steps: int = 50,
) -> float:
    """
    Compute world model prediction error on real trajectories.

    Used to track model error during MBPO-PPO training (Fig. 5).

    Returns:
        mean relative prediction error over eval_steps
    """
    world_model.eval()
    errors = []

    for traj in real_trajectories:
        T = len(traj)
        if T < history_horizon + eval_steps:
            continue

        start = np.random.randint(0, T - history_horizon - eval_steps)
        obs_h = torch.tensor(
            traj.obs[start : start + history_horizon], dtype=torch.float32, device=device
        ).unsqueeze(0)
        act_h = torch.tensor(
            traj.actions[start : start + history_horizon], dtype=torch.float32, device=device
        ).unsqueeze(0)
        act_f = torch.tensor(
            traj.actions[start + history_horizon : start + history_horizon + eval_steps],
            dtype=torch.float32, device=device
        ).unsqueeze(0)
        obs_target = torch.tensor(
            traj.obs[start + history_horizon : start + history_horizon + eval_steps],
            dtype=torch.float32, device=device
        ).unsqueeze(0)

        obs_pred, _ = world_model.predict(obs_h, act_h, act_f, use_mean=True)
        rel_err = relative_prediction_error(
            obs_pred.squeeze(0), obs_target.squeeze(0)
        ).mean().item()
        errors.append(rel_err)

    return float(np.mean(errors)) if errors else float("nan")


# ---------------------------------------------------------------------------
# Ablation Study: M x N Heatmap (Fig. S8)
# ---------------------------------------------------------------------------

def ablation_horizon_heatmap(
    build_model_fn,
    trajectories: List[Trajectory],
    device: torch.device,
    history_horizons: List[int],
    forecast_horizons: List[int],
    train_iterations: int = 500,
    batch_size: int = 256,
    eval_rollout_steps: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ablation study over history horizon M and forecast horizon N.

    Corresponds to Fig. S8.

    Returns:
        error_heatmap:    (len(M_values), len(N_values)) relative prediction errors
        time_heatmap:     (len(M_values), len(N_values)) training times in seconds
    """
    import time as time_module
    from train import WorldModelTrainer
    from config import RWMTrainingConfig

    n_M = len(history_horizons)
    n_N = len(forecast_horizons)
    error_heatmap = np.zeros((n_M, n_N))
    time_heatmap = np.zeros((n_M, n_N))

    for i, M in enumerate(history_horizons):
        for j, N in enumerate(forecast_horizons):
            print(f"Ablation M={M}, N={N}...")
            train_cfg = RWMTrainingConfig(
                history_horizon=M,
                forecast_horizon=N,
                max_iterations=train_iterations,
                batch_size=batch_size,
                checkpoint_dir=f"/tmp/ablation_M{M}_N{N}",
            )
            model = build_model_fn(M, N)
            dataset = TrajectoryDataset(trajectories, M, N)
            if len(dataset) == 0:
                error_heatmap[i, j] = float("nan")
                continue

            trainer = WorldModelTrainer(
                model=model,
                cfg=train_cfg,
                device=device,
                autoregressive=True,
                output_dir=f"/tmp/ablation_M{M}_N{N}",
            )

            t_start = time_module.time()
            trainer.train(dataset)
            elapsed = time_module.time() - t_start
            time_heatmap[i, j] = elapsed

            eval_dataset = TrajectoryDataset(trajectories, M, N)
            metrics = evaluate_autoregressive_rollout(
                model, eval_dataset, device, batch_size, eval_rollout_steps
            )
            valid = metrics["relative_error"][~np.isnan(metrics["relative_error"])]
            error_heatmap[i, j] = float(np.mean(valid)) if len(valid) > 0 else float("nan")

    return error_heatmap, time_heatmap


# ---------------------------------------------------------------------------
# Policy Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_policy_in_imagination(
    world_model: RWM,
    actor_critic: nn.Module,
    starting_states: Tuple[torch.Tensor, torch.Tensor],
    reward_computer,
    device: torch.device,
    horizon: int = 100,
    num_episodes: int = 100,
) -> Dict[str, float]:
    """
    Evaluate policy performance in imagination.

    Returns mean episode reward and other statistics.
    """
    from train_policy import ImaginaryEnvironment

    world_model.eval()
    actor_critic.eval()

    obs_history, act_history = starting_states
    B = obs_history.size(0)

    imag_env = ImaginaryEnvironment(
        world_model=world_model,
        num_envs=B,
        history_horizon=obs_history.size(1),
        device=device,
    )
    wm_obs = imag_env.reset(obs_history, act_history)

    cmd = torch.zeros(B, 3, device=device)
    cmd[:, 0] = 1.0  # forward velocity command
    a_prev = torch.zeros(B, actor_critic.policy.action_dim, device=device)

    episode_rewards = torch.zeros(B, device=device)
    episode_lengths = torch.zeros(B, device=device)
    all_rewards = []

    for t in range(horizon):
        policy_obs = reward_computer.wm_obs_to_policy_obs(wm_obs, cmd, a_prev)
        action, _, _ = actor_critic.act(policy_obs, deterministic=True)
        next_wm_obs, done = imag_env.step(action)
        reward = reward_computer.compute_from_wm_obs(
            next_wm_obs, action, cmd, a_prev, torch.zeros(action.size(-1), device=device)
        )
        episode_rewards += reward
        episode_lengths += 1
        all_rewards.append(reward.mean().item())
        wm_obs = next_wm_obs
        a_prev = action

    return {
        "mean_reward": float(episode_rewards.mean().item()),
        "std_reward": float(episode_rewards.std().item()),
        "mean_episode_length": float(episode_lengths.mean().item()),
        "mean_step_reward": float(np.mean(all_rewards)),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from config import ExperimentConfig
    from model import build_rwm, build_mlp_baseline, build_rssm_baseline, build_transformer_baseline

    parser = argparse.ArgumentParser(description="Evaluate RWM models")
    parser.add_argument("--robot", type=str, default="anymal_d")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="rwm",
                        choices=["rwm", "mlp", "rssm", "transformer"])
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.0, 0.01, 0.05, 0.1])
    parser.add_argument("--max-rollout-steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="eval_outputs")
    args = parser.parse_args()

    cfg = ExperimentConfig(robot=args.robot)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    M = cfg.rwm_training.history_horizon
    N = cfg.rwm_training.forecast_horizon

    dataset = TrajectoryDataset.from_directory(args.data_dir, M, N)
    print(f"Evaluation dataset: {len(dataset)} windows")

    model_builders = {
        "rwm": build_rwm,
        "mlp": build_mlp_baseline,
        "rssm": build_rssm_baseline,
        "transformer": build_transformer_baseline,
    }
    model = model_builders[args.model](cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {args.checkpoint}")

    print("\n--- Autoregressive Rollout Evaluation ---")
    metrics = evaluate_autoregressive_rollout(
        model, dataset, device, max_rollout_steps=args.max_rollout_steps
    )
    valid = metrics["relative_error"][~np.isnan(metrics["relative_error"])]
    print(f"Mean relative error: {np.mean(valid):.4f} ± {np.std(valid):.4f}")

    print("\n--- Noise Robustness Evaluation ---")
    noise_results = evaluate_noise_robustness(
        model, dataset, device, args.noise_levels, max_rollout_steps=50
    )
    for noise_std, errors in noise_results.items():
        valid = errors[~np.isnan(errors)]
        print(f"  noise={noise_std:.3f}: mean_error={np.mean(valid):.4f}")

    np.save(os.path.join(args.output_dir, "relative_errors.npy"), metrics["relative_error"])
    np.save(os.path.join(args.output_dir, "mse.npy"), metrics["mse"])
    for noise_std, errors in noise_results.items():
        np.save(os.path.join(args.output_dir, f"noise_{noise_std:.3f}_errors.npy"), errors)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
