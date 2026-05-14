"""Train the Robotic World Model (RWM) on collected trajectory data.

This script implements the world model training phase described in Sec 3.2.
It performs autoregressive training with dual-autoregressive mechanism.

Usage:
    python train_world_model.py --robot anymal_d --data_path data/trajectories.npz
    python train_world_model.py --robot unitree_g1 --data_path data/trajectories.npz
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import (
    RWMConfig,
    RWMArchConfig,
    ANYMAL_D_SPEC,
    UNITREE_G1_SPEC,
)
from model.rwm import RoboticWorldModel
from data.dataset import TrajectoryBuffer, SlidingWindowDataset
from training.world_model_trainer import WorldModelTrainer, create_dataloader


def get_spec(robot_name: str):
    if robot_name == "anymal_d":
        return ANYMAL_D_SPEC
    elif robot_name == "unitree_g1":
        return UNITREE_G1_SPEC
    else:
        raise ValueError(f"Unknown robot: {robot_name}")


def load_trajectories(data_path: str) -> TrajectoryBuffer:
    """Load trajectory data from npz file.

    Expected keys:
        observations: (num_trajs, max_len, obs_dim) or (total_steps, obs_dim)
        actions: matching actions
        privileged: (optional) privileged info
    """
    data = np.load(data_path)
    buffer = TrajectoryBuffer()

    observations = data["observations"]
    actions = data["actions"]
    privileged = data.get("privileged")

    if observations.ndim == 3:
        # (num_trajs, max_len, obs_dim)
        for i in range(len(observations)):
            obs = observations[i]
            acts = actions[i]
            priv = privileged[i] if privileged is not None else None
            # Remove padding (zero rows at end)
            mask = (np.abs(obs).sum(axis=-1) > 1e-8)
            obs = obs[mask]
            acts = acts[mask[:len(acts)]] if sum(mask) > 0 else acts[:len(obs) - 1]
            if priv is not None:
                priv = priv[mask]
            buffer.add_trajectory(obs, acts, priv)
    else:
        # Single long trajectory; split into segments
        buffer.add_trajectory(observations, actions, privileged)

    return buffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="anymal_d", choices=["anymal_d", "unitree_g1"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs/rwm")
    parser.add_argument("--max_iterations", type=int, default=2500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher_forcing", action="store_true",
                        help="Use teacher-forcing (N=1) instead of autoregressive training")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    robot_spec = get_spec(args.robot)

    config = RWMConfig(
        robot=robot_spec,
        arch=RWMArchConfig(),
        max_iterations=args.max_iterations,
    )

    if args.teacher_forcing:
        config.forecast_horizon = 1
        print("Using teacher-forcing mode (N=1)")

    print(f"Training RWM for {args.robot}")
    print(f"  Obs dim: {robot_spec.obs_dim}, Action dim: {robot_spec.action_dim}")
    print(f"  History horizon M={config.history_horizon}, Forecast horizon N={config.forecast_horizon}")
    print(f"  Batch size: {config.batch_size}, LR: {config.learning_rate}")

    # Load data
    print(f"Loading trajectories from {args.data_path}")
    buffer = load_trajectories(args.data_path)
    print(f"  Loaded {len(buffer)} trajectories")

    # Create dataloader
    dataloader = create_dataloader(buffer, config, shuffle=True)
    print(f"  Dataset size: {len(dataloader.dataset)} windows")

    if len(dataloader.dataset) == 0:
        raise ValueError("No valid windows found! Ensure trajectories are long enough "
                         f"(need at least {config.history_horizon + config.forecast_horizon} steps).")

    # Build model
    model = RoboticWorldModel(
        obs_dim=robot_spec.obs_dim,
        action_dim=robot_spec.action_dim,
        privileged_dim=robot_spec.privileged_dim,
        gru_hidden_size=config.arch.gru_hidden_size,
        gru_num_layers=config.arch.gru_num_layers,
        head_hidden_size=config.arch.head_hidden_size,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # Train
    trainer = WorldModelTrainer(model, config, device=args.device)
    writer = SummaryWriter(log_dir=args.log_dir)
    trainer.set_writer(writer)

    print("Starting training...")
    history = trainer.train(
        dataloader,
        num_iterations=config.max_iterations,
        log_interval=50,
    )

    # Save model
    checkpoint_path = os.path.join(args.output_dir, f"rwm_{args.robot}.pt")
    trainer.save(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Print final metrics
    if history["total_loss"]:
        print(f"Final total loss: {history['total_loss'][-1]:.6f}")
        print(f"Final obs loss: {history['obs_loss'][-1]:.6f}")

    writer.close()


if __name__ == "__main__":
    main()
