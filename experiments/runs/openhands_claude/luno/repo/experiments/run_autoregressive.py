"""
Autoregressive rollout evaluation for FNO uncertainty quantification.

Reproduces Figure 4 from the LUNO paper: averaged performance of different UQ
methods on autoregressive rollout of the FNO on 50 trajectories.

In autoregressive rollout, the model's prediction at time t is fed back as
input for time t+1. This causes a distribution shift as prediction errors
accumulate, which tests whether UQ methods can adapt their uncertainty.

Usage:
  python experiments/run_autoregressive.py --pde burgers
  python experiments/run_autoregressive.py --variant pos_neg_flip
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_CONFIG
from evaluate import compute_all_metrics


def autoregressive_rollout_1d(
    model,
    initial_input: np.ndarray,
    n_steps: int,
    predict_fn=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform autoregressive rollout for 1D FNO.

    At each step, the model's prediction replaces the oldest history frame.

    Args:
        model: trained FNO1d
        initial_input: (1, n_x, n_history) initial input
        n_steps: number of rollout steps
        predict_fn: optional callable(a) -> (mean, std); if None, uses model directly
    Returns:
        means: (n_steps, n_x, 1) predicted means
        stds: (n_steps, n_x, 1) predicted stds (zeros if predict_fn is None)
        inputs: (n_steps, n_x, n_history) inputs at each step
    """
    current_input = jnp.array(initial_input)  # (1, n_x, n_history)
    means = []
    stds = []
    inputs_list = []

    for _ in range(n_steps):
        inputs_list.append(np.array(current_input))

        if predict_fn is not None:
            mean, std = predict_fn(current_input)
        else:
            mean = model(current_input)
            std = jnp.zeros_like(mean)

        means.append(np.array(mean))
        stds.append(np.array(std))

        # Update input: shift history window and append new prediction
        # current_input: (1, n_x, n_history)
        # mean: (1, n_x, 1)
        current_input = jnp.concatenate(
            [current_input[:, :, 1:], mean], axis=-1
        )

    return (
        np.concatenate(means, axis=0),   # (n_steps, n_x, 1)
        np.concatenate(stds, axis=0),    # (n_steps, n_x, 1)
        np.concatenate(inputs_list, axis=0),  # (n_steps, n_x, n_history)
    )


def evaluate_autoregressive(
    predict_fns: Dict[str, callable],
    initial_inputs: np.ndarray,
    ground_truth: np.ndarray,
    n_rollout_steps: int,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Evaluate multiple UQ methods on autoregressive rollout.

    Args:
        predict_fns: dict mapping method name to predict_fn(a) -> (mean, std)
        initial_inputs: (n_traj, n_x, n_history) initial inputs
        ground_truth: (n_traj, n_steps, n_x, 1) ground truth trajectories
        n_rollout_steps: number of rollout steps
    Returns:
        results: dict mapping method -> {"rmse": [...], "nll": [...], "chi2": [...]}
                 where each list has length n_rollout_steps
    """
    n_traj = initial_inputs.shape[0]
    results = {name: {"rmse": [], "nll": [], "chi2": []} for name in predict_fns}

    for step in range(n_rollout_steps):
        step_means = {name: [] for name in predict_fns}
        step_stds = {name: [] for name in predict_fns}
        step_targets = []

        for traj_idx in range(n_traj):
            # Get input at this step (requires running rollout up to this point)
            # For efficiency, we run the full rollout per trajectory
            pass

        # Run full rollouts for all trajectories and methods
        for name, predict_fn in predict_fns.items():
            all_means_at_step = []
            all_stds_at_step = []
            all_targets_at_step = []

            for traj_idx in range(n_traj):
                init_inp = initial_inputs[traj_idx:traj_idx + 1]
                means, stds, _ = autoregressive_rollout_1d(
                    None, init_inp, n_rollout_steps, predict_fn=predict_fn
                )
                all_means_at_step.append(means[step])
                all_stds_at_step.append(stds[step])
                all_targets_at_step.append(ground_truth[traj_idx, step])

            mean_pred = np.stack(all_means_at_step, axis=0)
            std_pred = np.stack(all_stds_at_step, axis=0)
            target = np.stack(all_targets_at_step, axis=0)

            metrics = compute_all_metrics(mean_pred, std_pred, target)
            results[name]["rmse"].append(metrics["rmse"])
            results[name]["nll"].append(metrics["nll"])
            results[name]["chi2"].append(metrics["chi2"])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoregressive rollout evaluation")
    parser.add_argument("--pde", type=str, default="burgers",
                        choices=["burgers", "hyper_diffusion", "ks_conservative"])
    parser.add_argument("--n_traj", type=int, default=50, help="Number of test trajectories")
    parser.add_argument("--n_rollout", type=int, default=40, help="Number of rollout steps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Autoregressive rollout evaluation: {args.pde}")
    print("Note: This script requires a trained model. Run run_low_data.py first.")
    print("Full autoregressive evaluation is integrated in run_low_data.py.")
