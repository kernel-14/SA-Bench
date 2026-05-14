"""Evaluate autoregressive prediction accuracy of world models.

Implements the evaluation in Sec 4.1 (Trajectory Prediction) and Sec 4.3 (Generality).
Compares RWM against MLP, RSSM, and Transformer baselines.

Usage:
    python evaluate.py --robot anymal_d --model rwm --checkpoint checkpoints/rwm.pt
    python evaluate.py --robot anymal_d --model all  # Compare all models
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    RWMConfig,
    RWMArchConfig,
    MLPBaselineConfig,
    RSSMBaselineConfig,
    TransformerBaselineConfig,
    ANYMAL_D_SPEC,
    UNITREE_G1_SPEC,
)
from model.rwm import RoboticWorldModel
from model.baselines import MLPBaseline, RSSMBaseline, TransformerBaseline
from data.dataset import TrajectoryBuffer, SlidingWindowDataset
from utils.metrics import evaluate_rollout


def get_spec(robot_name: str):
    if robot_name == "anymal_d":
        return ANYMAL_D_SPEC
    elif robot_name == "unitree_g1":
        return UNITREE_G1_SPEC
    else:
        raise ValueError(f"Unknown robot: {robot_name}")


def load_test_data(data_path: str, history_horizon: int, forecast_horizon: int, use_privileged: bool = False):
    """Load test trajectory data and create a DataLoader.

    Expected npz keys:
        test_observations: (num_trajs, total_len, obs_dim) or (total_len, obs_dim)
        test_actions: matching actions
        test_privileged: optional
    """
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
        use_privileged=use_privileged,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
    return dataloader


def build_model(model_type: str, robot_spec, config: RWMConfig, device: str):
    """Build the specified world model variant."""
    M = config.history_horizon
    N = config.forecast_horizon

    if model_type == "rwm":
        model = RoboticWorldModel(
            obs_dim=robot_spec.obs_dim,
            action_dim=robot_spec.action_dim,
            privileged_dim=robot_spec.privileged_dim,
            gru_hidden_size=config.arch.gru_hidden_size,
            gru_num_layers=config.arch.gru_num_layers,
            head_hidden_size=config.arch.head_hidden_size,
        )
    elif model_type == "mlp":
        model = MLPBaseline(
            obs_dim=robot_spec.obs_dim,
            action_dim=robot_spec.action_dim,
            history_horizon=M,
            forecast_horizon=N,
        )
    elif model_type == "rssm":
        model = RSSMBaseline(
            obs_dim=robot_spec.obs_dim,
            action_dim=robot_spec.action_dim,
            history_horizon=M,
        )
    elif model_type == "transformer":
        model = TransformerBaseline(
            obs_dim=robot_spec.obs_dim,
            action_dim=robot_spec.action_dim,
            context_length=M,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="anymal_d", choices=["anymal_d", "unitree_g1"])
    parser.add_argument("--model", type=str, nargs="+", default=["rwm"],
                        choices=["rwm", "mlp", "rssm", "transformer", "all"])
    parser.add_argument("--checkpoint", type=str, nargs="+", default=None,
                        help="Path(s) to model checkpoint(s)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test data npz file")
    parser.add_argument("--max_horizon", type=int, default=100,
                        help="Maximum forecast horizon for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results npz")
    args = parser.parse_args()

    robot_spec = get_spec(args.robot)

    config = RWMConfig(
        robot=robot_spec,
        arch=RWMArchConfig(),
    )

    M = config.history_horizon
    N = config.forecast_horizon

    # Resolve model list
    model_types = args.model
    if "all" in model_types:
        model_types = ["rwm", "mlp", "rssm", "transformer"]

    checkpoints = args.checkpoint
    if checkpoints is None:
        checkpoints = [None] * len(model_types)
    elif len(checkpoints) == 1 and len(model_types) > 1:
        checkpoints = checkpoints * len(model_types)

    print(f"Evaluating on {args.robot}")
    print(f"  Models: {model_types}")
    print(f"  Max horizon: {args.max_horizon}")

    # Load test data
    dataloader = load_test_data(args.data_path, M, max(N, args.max_horizon),
                                use_privileged=(robot_spec.privileged_dim > 0))
    print(f"  Test windows: {len(dataloader.dataset)}")

    all_results = {}

    for model_type, ckpt_path in zip(model_types, checkpoints):
        print(f"\n--- {model_type} ---")

        model = build_model(model_type, robot_spec, config, args.device)

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"  Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("  Using untrained model (random weights)")

        model.eval()

        # Collect errors across batches
        horizon_errors = []
        horizon_counts = np.zeros(args.max_horizon)

        with torch.no_grad():
            for batch in dataloader:
                observations = batch["observations"].to(args.device)
                actions = batch["actions"].to(args.device)

                B = observations.shape[0]
                effective_horizon = min(args.max_horizon, observations.shape[1] - M - 1)

                if effective_horizon < 1:
                    continue

                obs_history = observations[:, :M]
                obs_target = observations[:, M:M + effective_horizon]

                result = model.forward(
                    observations=obs_history,
                    actions=actions[:, :M - 1 + effective_horizon],
                    forecast_horizon=effective_horizon,
                )

                # Relative error per step
                error = torch.norm(
                    result["obs_means"] - obs_target, dim=-1
                ) / (torch.norm(obs_target, dim=-1) + 1e-8)

                horizon_errors.append(error.cpu().numpy())
                horizon_counts[:effective_horizon] += B

        # Average over batches
        max_h = max(e.shape[1] for e in horizon_errors)
        mean_errors = np.zeros(max_h)

        for e in horizon_errors:
            h = e.shape[1]
            mean_errors[:h] += e.mean(axis=0) * (e.shape[0] / sum(e.shape[0] for e in horizon_errors))

        all_results[model_type] = mean_errors

        print(f"  Mean error at step {N}: {mean_errors[min(N-1, len(mean_errors)-1)]:.4f}")
        print(f"  Mean error at step {min(50, len(mean_errors)-1)}: {mean_errors[min(49, len(mean_errors)-1)]:.4f}")
        if len(mean_errors) > 80:
            print(f"  Mean error at step 100: {mean_errors[99]:.4f}")

    # Print comparison table (like Fig 4)
    print("\n=== Relative Prediction Error Comparison ===")
    header = f"{'Model':<15}"
    for step in [1, 5, 10, 20, 50, 100]:
        header += f"{'t=' + str(step):>10}"
    print(header)
    print("-" * len(header))

    for mtype, errors in all_results.items():
        row = f"{mtype:<15}"
        for step in [1, 5, 10, 20, 50, 100]:
            idx = min(step - 1, len(errors) - 1)
            row += f"{errors[idx]:>10.4f}"
        print(row)

    # Save
    if args.output:
        np.savez(args.output, **all_results)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
