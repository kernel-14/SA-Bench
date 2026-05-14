"""
Script to train a policy using MBPO-PPO on a learned world model.

Usage:
    python scripts/train_mbpo_ppo.py \
        --wm_checkpoint outputs/world_model/seed_42/best_model.pt \
        --config configs/mbpo_ppo_anymal.yaml

Algorithm 1 from the paper:
  1. Initialize policy, world model, and replay buffer D
  2. For each iteration:
     a. Collect observation-action pairs in D using current policy
     b. Update world model with autoregressive training
     c. Initialize imagination agents from D
     d. Roll out imagination trajectories for T steps
     e. Update policy using PPO
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import RoboticWorldModel, PolicyNetwork, ValueNetwork
from training import WorldModelTrainer, MBPOPPOTrainer, PPOTrainer, ReplayBuffer
from envs import VelocityTrackingReward


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_reward_fn(robot: str, device: torch.device):
    """Build reward function for the given robot."""
    reward_fn_obj = VelocityTrackingReward(robot=robot)

    def reward_fn(obs, action, next_obs, priv_info):
        """
        Compute reward from world model observations.

        For imagination rollouts, we use the world model obs space.
        The velocity command is embedded in the observation for the policy,
        but the world model obs doesn't include it.

        Here we use a simplified reward based on available information.
        """
        # Use next_obs for reward computation (world model obs space)
        # Extract base velocities from world model obs
        base_lin_vel = next_obs[:, :3]
        base_ang_vel = next_obs[:, 3:6]
        projected_gravity = next_obs[:, 6:9]

        # Use zero velocity command as default (can be overridden)
        # In practice, the velocity command would be part of the policy obs
        velocity_command = torch.zeros(next_obs.shape[0], 3, device=device)

        # Compute reward components
        sigma_vxy = 0.25
        sigma_wz = 0.25

        w = reward_fn_obj.weights

        cmd_xy = velocity_command[:, :2]
        vel_xy = base_lin_vel[:, :2]
        r_vxy = w["w_vxy"] * torch.exp(
            -torch.sum((cmd_xy - vel_xy) ** 2, dim=-1) / (sigma_vxy ** 2)
        )

        cmd_z = velocity_command[:, 2]
        ang_vel_z = base_ang_vel[:, 2]
        r_wz = w["w_wz"] * torch.exp(
            -(cmd_z - ang_vel_z) ** 2 / (sigma_wz ** 2)
        )

        vel_z = base_lin_vel[:, 2]
        r_vz = w["w_vz"] * vel_z ** 2

        ang_vel_xy = base_ang_vel[:, :2]
        r_wxy = w["w_wxy"] * torch.sum(ang_vel_xy ** 2, dim=-1)

        g_xy = projected_gravity[:, :2]
        r_g = w["w_g"] * torch.sum(g_xy ** 2, dim=-1)

        return r_vxy + r_wz + r_vz + r_wxy + r_g

    return reward_fn


def build_termination_fn(robot: str):
    """Build termination function based on privileged info (contact detection)."""

    def termination_fn(next_obs, priv_info):
        """
        Detect termination from privileged info.

        For locomotion tasks, termination occurs when the base contacts the ground.
        This is predicted by the privileged info head of RWM.
        """
        if priv_info is None or priv_info.shape[-1] == 0:
            return torch.zeros(next_obs.shape[0], dtype=torch.bool, device=next_obs.device)

        # For ANYmal: priv_info[:, :4] = knee contacts, priv_info[:, 4:8] = foot contacts
        # Termination if any knee contact detected (base contact)
        if robot == "anymal":
            knee_contacts = priv_info[:, :4]
            terminated = (knee_contacts > 0.5).any(dim=-1)
        elif robot == "g1":
            # For G1: body contacts indicate termination
            body_contacts = priv_info[:, :26]
            terminated = (body_contacts > 0.5).any(dim=-1)
        else:
            terminated = torch.zeros(next_obs.shape[0], dtype=torch.bool, device=next_obs.device)

        return terminated

    return termination_fn


def train(args):
    config = load_config(args.config)
    train_cfg = config["training"]
    robot = config.get("robot", "anymal")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load world model
    print(f"Loading world model from {args.wm_checkpoint}")
    wm_checkpoint = torch.load(args.wm_checkpoint, map_location=device)
    wm_config = wm_checkpoint["config"]
    wm_model_cfg = wm_config["model"]

    world_model = RoboticWorldModel(
        obs_size=wm_model_cfg["obs_size"],
        action_size=wm_model_cfg["action_size"],
        priv_size=wm_model_cfg.get("priv_size", 0),
        hidden_size=wm_model_cfg.get("hidden_size", 256),
        num_gru_layers=wm_model_cfg.get("num_gru_layers", 2),
        head_hidden_size=wm_model_cfg.get("head_hidden_size", 128),
    ).to(device)
    world_model.load_state_dict(wm_checkpoint["model_state_dict"])
    print("World model loaded successfully")

    # Build policy and value function
    policy_cfg = config["policy"]
    policy = PolicyNetwork(
        obs_size=policy_cfg["obs_size"],
        action_size=policy_cfg["action_size"],
        hidden_size=policy_cfg.get("hidden_size", 128),
        num_layers=policy_cfg.get("num_layers", 3),
    ).to(device)

    value_fn = ValueNetwork(
        obs_size=policy_cfg["obs_size"],
        hidden_size=policy_cfg.get("hidden_size", 128),
        num_layers=policy_cfg.get("num_layers", 3),
    ).to(device)

    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Value function parameters: {sum(p.numel() for p in value_fn.parameters()):,}")

    # Optimizers
    ppo_optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    wm_optimizer = torch.optim.Adam(
        world_model.parameters(),
        lr=wm_config["training"]["learning_rate"],
        weight_decay=wm_config["training"].get("weight_decay", 1e-5),
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(max_size=train_cfg["buffer_size"])

    # World model trainer (for online fine-tuning)
    wm_trainer = WorldModelTrainer(
        model=world_model,
        optimizer=wm_optimizer,
        history_horizon=wm_config["training"]["history_horizon"],
        forecast_horizon=wm_config["training"]["forecast_horizon"],
        forecast_decay=wm_config["training"].get("forecast_decay", 1.0),
        device=device,
    )

    # PPO trainer
    ppo_trainer = PPOTrainer(
        policy=policy,
        value_fn=value_fn,
        optimizer=ppo_optimizer,
        clip_range=train_cfg.get("clip_range", 0.2),
        entropy_coef=train_cfg.get("entropy_coef", 0.005),
        learning_epochs=train_cfg.get("learning_epochs", 5),
        num_mini_batches=train_cfg.get("mini_batches", 4),
        gamma=train_cfg.get("discount_factor", 0.99),
        kl_target=train_cfg.get("kl_target", 0.01),
        device=device,
    )

    # Reward and termination functions
    reward_fn = build_reward_fn(robot, device)
    termination_fn = build_termination_fn(robot)

    # MBPO-PPO trainer
    mbpo_trainer = MBPOPPOTrainer(
        world_model=world_model,
        policy=policy,
        value_fn=value_fn,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
        wm_trainer=wm_trainer,
        ppo_trainer=ppo_trainer,
        replay_buffer=replay_buffer,
        history_horizon=wm_config["training"]["history_horizon"],
        imagination_steps=train_cfg.get("imagination_steps", 100),
        n_imagination_envs=train_cfg.get("imagination_envs", 4096),
        wm_train_steps=50,
        wm_batch_size=wm_config["training"]["batch_size"],
        device=device,
    )

    # Output directory
    output_dir = Path(args.output_dir) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate replay buffer with random data if no env available
    if args.env_path is None:
        print("No environment provided. Pre-populating replay buffer with synthetic data.")
        obs_size = wm_model_cfg["obs_size"]
        action_size = wm_model_cfg["action_size"]
        priv_size = wm_model_cfg.get("priv_size", 0)

        # Generate synthetic initial data
        for _ in range(train_cfg["buffer_size"]):
            obs = np.random.randn(obs_size).astype(np.float32) * 0.1
            action = np.random.randn(action_size).astype(np.float32) * 0.1
            priv = np.random.randn(priv_size).astype(np.float32) * 0.1 if priv_size > 0 else None
            replay_buffer.add(obs, action, priv, False)

    # Training loop
    print(f"Starting MBPO-PPO training for {train_cfg['max_iterations']} iterations")
    best_reward = float("-inf")

    for iteration in range(train_cfg["max_iterations"]):
        metrics = mbpo_trainer.train_iteration(env=None)

        if (iteration + 1) % 10 == 0:
            ppo_loss = metrics.get("ppo/policy_loss", 0.0)
            wm_loss = metrics.get("wm/loss", 0.0)
            print(
                f"Iter {iteration + 1}/{train_cfg['max_iterations']} | "
                f"WM Loss: {wm_loss:.4f} | "
                f"PPO Loss: {ppo_loss:.4f}"
            )

        # Save checkpoint
        if (iteration + 1) % args.save_interval == 0:
            torch.save(
                {
                    "iteration": iteration,
                    "policy_state_dict": policy.state_dict(),
                    "value_fn_state_dict": value_fn.state_dict(),
                    "world_model_state_dict": world_model.state_dict(),
                    "ppo_optimizer_state_dict": ppo_optimizer.state_dict(),
                    "config": config,
                    "metrics": metrics,
                },
                output_dir / f"checkpoint_{iteration + 1}.pt",
            )

    print("Training complete!")
    print(f"Checkpoints saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train policy with MBPO-PPO")
    parser.add_argument("--wm_checkpoint", type=str, required=True,
                        help="Path to pretrained world model checkpoint")
    parser.add_argument("--config", type=str, default="configs/mbpo_ppo_anymal.yaml",
                        help="Path to MBPO-PPO config file")
    parser.add_argument("--env_path", type=str, default=None,
                        help="Path to environment (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/mbpo_ppo",
                        help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Checkpoint save interval")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
