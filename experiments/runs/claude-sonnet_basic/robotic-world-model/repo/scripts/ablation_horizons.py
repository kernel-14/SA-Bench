"""
Ablation study on history horizon M and forecast horizon N.

Reproduces Figure S8 from the paper (Appendix A.4.1):
  - Heatmap of prediction error vs (M, N) combinations
  - Heatmap of training time vs (M, N) combinations

Key findings from the paper:
  - Larger M reduces error (diminishing returns beyond M=32)
  - Larger N significantly improves long-horizon accuracy
  - N=1 (teacher forcing) leads to poor autoregressive performance
  - Optimal trade-off: M=32, N=8
"""

import sys
import time
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import RoboticWorldModel
from training import WorldModelTrainer, TeacherForcingTrainer
from utils import TrajectoryDataset, generate_synthetic_trajectories
from torch.utils.data import DataLoader


def train_and_evaluate(
    obs_size: int,
    action_size: int,
    priv_size: int,
    history_horizon: int,
    forecast_horizon: int,
    n_iterations: int = 100,
    batch_size: int = 256,
    eval_horizon: int = 100,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Train RWM with given M and N, return (prediction_error, training_time).

    Args:
        history_horizon: M
        forecast_horizon: N
        n_iterations: number of training iterations (reduced for ablation)

    Returns:
        (mean_relative_error, training_time_seconds)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate synthetic data
    observations, actions, privileged_info = generate_synthetic_trajectories(
        n_trajectories=20,
        trajectory_length=300,
        obs_size=obs_size,
        action_size=action_size,
        priv_size=priv_size,
        seed=seed,
    )

    # Create dataset
    dataset = TrajectoryDataset(
        observations, actions, history_horizon, forecast_horizon, privileged_info
    )

    if len(dataset) == 0:
        return float("inf"), 0.0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Build model
    model = RoboticWorldModel(
        obs_size=obs_size,
        action_size=action_size,
        priv_size=priv_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Use teacher forcing for N=1, autoregressive for N>1
    if forecast_horizon == 1:
        trainer = TeacherForcingTrainer(
            model=model,
            optimizer=optimizer,
            history_horizon=history_horizon,
            forecast_horizon=forecast_horizon,
            device=device,
        )
    else:
        trainer = WorldModelTrainer(
            model=model,
            optimizer=optimizer,
            history_horizon=history_horizon,
            forecast_horizon=forecast_horizon,
            device=device,
        )

    # Train and measure time
    start_time = time.time()
    for _ in range(n_iterations):
        trainer.train_epoch(dataloader)
    training_time = time.time() - start_time

    # Evaluate
    eval_metrics = trainer.evaluate(dataloader, eval_horizon=eval_horizon)
    mean_error = eval_metrics["mean_relative_error"]

    return mean_error, training_time


def run_ablation(
    obs_size: int,
    action_size: int,
    priv_size: int,
    M_values: list,
    N_values: list,
    n_iterations: int = 50,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run ablation study over all (M, N) combinations.

    Returns:
        error_heatmap: (len(M_values), len(N_values)) - prediction errors
        time_heatmap: (len(M_values), len(N_values)) - training times
    """
    error_heatmap = np.zeros((len(M_values), len(N_values)))
    time_heatmap = np.zeros((len(M_values), len(N_values)))

    for i, M in enumerate(M_values):
        for j, N in enumerate(N_values):
            print(f"  M={M}, N={N}...", end=" ", flush=True)
            error, t = train_and_evaluate(
                obs_size=obs_size,
                action_size=action_size,
                priv_size=priv_size,
                history_horizon=M,
                forecast_horizon=N,
                n_iterations=n_iterations,
                device=device,
                seed=seed,
            )
            error_heatmap[i, j] = error
            time_heatmap[i, j] = t
            print(f"error={error:.4f}, time={t:.1f}s")

    return error_heatmap, time_heatmap


def main():
    parser = argparse.ArgumentParser(description="Ablation study on M and N")
    parser.add_argument("--robot", type=str, default="anymal",
                        choices=["anymal", "g1"])
    parser.add_argument("--M_values", type=int, nargs="+",
                        default=[4, 8, 16, 32, 64],
                        help="History horizon values to test")
    parser.add_argument("--N_values", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16],
                        help="Forecast horizon values to test")
    parser.add_argument("--n_iterations", type=int, default=50,
                        help="Training iterations per configuration")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.robot == "anymal":
        obs_size, action_size, priv_size = 45, 12, 8
    else:
        obs_size, action_size, priv_size = 96, 29, 30

    print(f"\n=== Ablation Study: M x N ({args.robot}) ===")
    print(f"M values: {args.M_values}")
    print(f"N values: {args.N_values}")

    error_heatmap, time_heatmap = run_ablation(
        obs_size=obs_size,
        action_size=action_size,
        priv_size=priv_size,
        M_values=args.M_values,
        N_values=args.N_values,
        n_iterations=args.n_iterations,
        device=device,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / f"ablation_{args.robot}.npz",
        error_heatmap=error_heatmap,
        time_heatmap=time_heatmap,
        M_values=np.array(args.M_values),
        N_values=np.array(args.N_values),
    )

    print("\nError heatmap (rows=M, cols=N):")
    print(error_heatmap)
    print("\nTime heatmap (rows=M, cols=N):")
    print(time_heatmap)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
