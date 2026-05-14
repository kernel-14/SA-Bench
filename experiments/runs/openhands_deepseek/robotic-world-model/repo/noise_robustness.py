"""Evaluate robustness of RWM and baselines under Gaussian noise perturbations.

Implements the noise robustness experiments from Sec 4.2, Fig 3b.
Gaussian noise is applied to both observations and actions during autoregressive
rollout, and the relative prediction error is measured per forecast step.

Usage:
    python noise_robustness.py --robot anymal_d \
        --rwm_checkpoint checkpoints/rwm_anymal_d.pt \
        --mlp_checkpoint checkpoints/mlp_anymal_d.pt \
        --data_path data/test_trajectories.npz
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    RWMConfig,
    RWMArchConfig,
    ANYMAL_D_SPEC,
    UNITREE_G1_SPEC,
)
from model.rwm import RoboticWorldModel
from model.baselines import MLPBaseline
from data.dataset import TrajectoryBuffer, SlidingWindowDataset


def get_spec(robot_name: str):
    if robot_name == "anymal_d":
        return ANYMAL_D_SPEC
    elif robot_name == "unitree_g1":
        return UNITREE_G1_SPEC
    else:
        raise ValueError(f"Unknown robot: {robot_name}")


def load_test_data(data_path: str, history_horizon: int, forecast_horizon: int):
    data = np.load(data_path)
    buffer = TrajectoryBuffer()

    observations = data["test_observations"]
    actions = data["test_actions"]
    privileged = data.get("test_privileged")

    if observations.ndim == 3:
        for i in range(len(observations)):
            obs = observations[i]
            acts = actions[i]
            priv = privileged[i] if privileged is not None else None
            mask = (np.abs(obs).sum(axis=-1) > 1e-8)
            obs = obs[mask]
            acts = acts[mask[:len(acts)]]
            if priv is not None:
                priv = priv[mask]
            buffer.add_trajectory(obs, acts, priv)
    else:
        buffer.add_trajectory(observations, actions, privileged)

    dataset = SlidingWindowDataset(
        buffer,
        history_horizon=history_horizon,
        forecast_horizon=forecast_horizon,
        use_privileged=False,
    )
    return DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)


def evaluate_with_noise(
    model,
    dataloader: DataLoader,
    M: int,
    max_horizon: int,
    noise_std: float,
    device: str,
) -> np.ndarray:
    """Evaluate model under Gaussian noise on observations and actions.

    Noise is added independently to each observation and action during
    autoregressive rollout, simulating the "noise" condition in Sec 4.2.

    Returns:
        mean_relative_errors: (max_horizon,) array
    """
    model.eval()
    horizon_errors = []
    horizon_counts = np.zeros(max_horizon)

    with torch.no_grad():
        for batch in dataloader:
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device)

            B = observations.shape[0]
            effective_horizon = min(max_horizon, observations.shape[1] - M - 1)

            if effective_horizon < 1:
                continue

            # Add noise to historical observations
            obs_history = observations[:, :M]
            noise_obs = noise_std * torch.randn_like(obs_history)
            obs_noisy = obs_history + noise_obs

            # Add noise to actions
            actions_for_model = actions[:, :M - 1 + effective_horizon]
            noise_act = noise_std * torch.randn_like(actions_for_model)
            actions_noisy = actions_for_model + noise_act

            obs_target = observations[:, M:M + effective_horizon]

            result = model.forward(
                observations=obs_noisy,
                actions=actions_noisy,
                forecast_horizon=effective_horizon,
            )

            error = torch.norm(
                result["obs_means"] - obs_target, dim=-1
            ) / (torch.norm(obs_target, dim=-1) + 1e-8)

            horizon_errors.append(error.cpu().numpy())
            horizon_counts[:effective_horizon] += B

    # Average
    max_h = max(e.shape[1] for e in horizon_errors)
    mean_errors = np.zeros(max_h)

    total_samples = sum(e.shape[0] for e in horizon_errors)
    for e in horizon_errors:
        h = e.shape[1]
        mean_errors[:h] += e.mean(axis=0) * (e.shape[0] / total_samples)

    return mean_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="anymal_d", choices=["anymal_d", "unitree_g1"])
    parser.add_argument("--rwm_checkpoint", type=str, required=True)
    parser.add_argument("--mlp_checkpoint", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_horizon", type=int, default=100)
    parser.add_argument("--noise_levels", type=float, nargs="+",
                        default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--output", type=str, default="results/noise_robustness.npz")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    robot_spec = get_spec(args.robot)
    M = 32

    # Load test data
    dataloader = load_test_data(args.data_path, M, args.max_horizon)
    print(f"Test windows: {len(dataloader.dataset)}")

    # Load RWM
    rwm = RoboticWorldModel(
        obs_dim=robot_spec.obs_dim,
        action_dim=robot_spec.action_dim,
        privileged_dim=robot_spec.privileged_dim,
    ).to(args.device)
    rwm_ckpt = torch.load(args.rwm_checkpoint, map_location=args.device)
    rwm.load_state_dict(rwm_ckpt["model_state_dict"])
    print("RWM loaded.")

    # Load MLP baseline
    mlp = None
    if args.mlp_checkpoint:
        mlp = MLPBaseline(
            obs_dim=robot_spec.obs_dim,
            action_dim=robot_spec.action_dim,
            history_horizon=M,
            forecast_horizon=args.max_horizon,
        ).to(args.device)
        mlp_ckpt = torch.load(args.mlp_checkpoint, map_location=args.device)
        mlp.load_state_dict(mlp_ckpt["model_state_dict"])
        print("MLP baseline loaded.")

    results = {}
    print("\n=== Noise Robustness Evaluation ===\n")

    for noise_std in args.noise_levels:
        print(f"Noise std = {noise_std:.3f}")

        rwm_errors = evaluate_with_noise(
            rwm, dataloader, M, args.max_horizon, noise_std, args.device,
        )
        results[f"rwm_noise_{noise_std}"] = rwm_errors
        print(f"  RWM:    t=10: {rwm_errors[min(9, len(rwm_errors)-1)]:.4f}, "
              f"t=50: {rwm_errors[min(49, len(rwm_errors)-1)]:.4f}, "
              f"t=100: {rwm_errors[min(99, len(rwm_errors)-1)]:.4f}")

        if mlp is not None:
            mlp_errors = evaluate_with_noise(
                mlp, dataloader, M, args.max_horizon, noise_std, args.device,
            )
            results[f"mlp_noise_{noise_std}"] = mlp_errors
            print(f"  MLP:    t=10: {mlp_errors[min(9, len(mlp_errors)-1)]:.4f}, "
                  f"t=50: {mlp_errors[min(49, len(mlp_errors)-1)]:.4f}, "
                  f"t=100: {mlp_errors[min(99, len(mlp_errors)-1)]:.4f}")

    # Save results
    np.savez(args.output, **results)
    print(f"\nResults saved to {args.output}")

    # Summary table
    print("\n=== Summary: Relative Error at t=100 ===")
    print(f"{'Std':>8} {'RWM':>10} {'MLP':>10}")
    print("-" * 30)
    for noise_std in args.noise_levels:
        r_key = f"rwm_noise_{noise_std}"
        m_key = f"mlp_noise_{noise_std}"
        r_val = results[r_key][min(99, len(results[r_key]) - 1)] if r_key in results else float("nan")
        m_val = results[m_key][min(99, len(results[m_key]) - 1)] if m_key in results else float("nan")
        print(f"{noise_std:>8.3f} {r_val:>10.4f} {m_val:>10.4f}")


if __name__ == "__main__":
    main()
