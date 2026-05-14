"""Train a policy using MBPO-PPO with the learned Robotic World Model.

Implements Algorithm 1 from the paper: policy optimization on learned world models.

Usage:
    python train_policy.py --robot anymal_d --world_model checkpoints/rwm_anymal_d.pt
    python train_policy.py --robot unitree_g1 --world_model checkpoints/rwm_unitree_g1.pt
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import (
    RWMConfig,
    RWMArchConfig,
    MBPOPOConfig,
    PolicyArchConfig,
    ANYMAL_D_SPEC,
    UNITREE_G1_SPEC,
    ANYMAL_D_REWARD,
    UNITREE_G1_REWARD,
)
from model.rwm import RoboticWorldModel
from model.policy import PPOActor, PPOCritic
from data.replay_buffer import ReplayBuffer
from training.world_model_trainer import WorldModelTrainer
from training.mbpo_ppo_trainer import MBPOPPOTrainer
from env.rewards import RewardComputer
from env.tasks import VelocityCommandGenerator


def get_spec(robot_name: str):
    if robot_name == "anymal_d":
        return ANYMAL_D_SPEC, ANYMAL_D_REWARD
    elif robot_name == "unitree_g1":
        return UNITREE_G1_SPEC, UNITREE_G1_REWARD
    else:
        raise ValueError(f"Unknown robot: {robot_name}")


class DummyEnv:
    """Minimal dummy environment that uses the world model for imagination.

    In a real setup, this would be replaced by the actual Isaac Lab environment.
    For policy evaluation purposes, it provides an interface compatible with MBPO-PPO.
    """

    def __init__(self, obs_dim: int, action_dim: int, privileged_dim: int = 0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.privileged_dim = privileged_dim
        self._obs = np.zeros(obs_dim, dtype=np.float32)

    def reset(self):
        self._obs = np.random.randn(self.obs_dim).astype(np.float32) * 0.1
        self._obs[6:9] = np.array([0.0, 0.0, -1.0])  # Gravity pointing down
        return self._obs, {}

    def step(self, action: np.ndarray):
        # In a real setup, this would query the simulator
        # For MBPO-PPO, the world model does the imagination; this is only for
        # collecting real data into the replay buffer
        next_obs = self._obs.copy()
        reward = 0.0
        done = False
        truncated = False
        info = {}
        self._obs = next_obs + np.random.randn(self.obs_dim).astype(np.float32) * 0.01
        return self._obs, reward, done, truncated, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="anymal_d", choices=["anymal_d", "unitree_g1"])
    parser.add_argument("--world_model", type=str, required=True, help="Path to pretrained RWM checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs/mbpo_ppo")
    parser.add_argument("--max_iterations", type=int, default=2500)
    parser.add_argument("--collect_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    robot_spec, reward_weights = get_spec(args.robot)

    rwm_config = RWMConfig(
        robot=robot_spec,
        arch=RWMArchConfig(),
    )

    mbpo_config = MBPOPOConfig(
        robot=robot_spec,
        policy_arch=PolicyArchConfig(),
        max_iterations=args.max_iterations,
    )

    print(f"Training MBPO-PPO for {args.robot}")
    print(f"  Policy obs dim: {robot_spec.policy_obs_dim}")
    print(f"  Action dim: {robot_spec.action_dim}")
    print(f"  Imagination envs: {mbpo_config.imagination_envs}")
    print(f"  Imagination steps: {mbpo_config.imagination_steps_per_iteration}")

    # Load world model
    print(f"Loading world model from {args.world_model}")
    world_model = RoboticWorldModel(
        obs_dim=robot_spec.obs_dim,
        action_dim=robot_spec.action_dim,
        privileged_dim=robot_spec.privileged_dim,
        gru_hidden_size=rwm_config.arch.gru_hidden_size,
        gru_num_layers=rwm_config.arch.gru_num_layers,
        head_hidden_size=rwm_config.arch.head_hidden_size,
    )
    checkpoint = torch.load(args.world_model, map_location=args.device)
    world_model.load_state_dict(checkpoint["model_state_dict"])
    world_model.to(args.device)
    print("  World model loaded.")

    # Initialize policy networks
    actor = PPOActor(
        obs_dim=robot_spec.policy_obs_dim,
        action_dim=robot_spec.action_dim,
        hidden_shape=mbpo_config.policy_arch.hidden_shape,
        activation=mbpo_config.policy_arch.activation,
    )

    critic = PPOCritic(
        obs_dim=robot_spec.policy_obs_dim,
        hidden_shape=mbpo_config.policy_arch.hidden_shape,
        activation=mbpo_config.policy_arch.activation,
    )

    num_policy_params = sum(p.numel() for p in actor.parameters()) + sum(p.numel() for p in critic.parameters())
    print(f"  Policy parameters: {num_policy_params:,}")

    # Initialize reward computer
    reward_computer = RewardComputer(
        weights=reward_weights,
        robot_spec=robot_spec,
        default_joint_pos=None,  # Set to default joint positions if available
    )

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(
        obs_dim=robot_spec.policy_obs_dim,
        action_dim=robot_spec.action_dim,
        privileged_dim=robot_spec.privileged_dim,
        max_size=mbpo_config.buffer_size,
        device=args.device,
    )

    # Create world model trainer (for fine-tuning during MBPO-PPO)
    wm_trainer = WorldModelTrainer(world_model, rwm_config, device=args.device)

    # Create MBPO-PPO trainer
    mbpo_trainer = MBPOPPOTrainer(
        actor=actor,
        critic=critic,
        world_model=world_model,
        rwm_config=rwm_config,
        mbpo_config=mbpo_config,
        reward_computer=reward_computer,
        device=args.device,
    )

    # Environment (dummy for code structure; real setup would use Isaac Lab)
    env = DummyEnv(
        obs_dim=robot_spec.obs_dim,
        action_dim=robot_spec.action_dim,
        privileged_dim=robot_spec.privileged_dim,
    )

    writer = SummaryWriter(log_dir=args.log_dir)
    mbpo_trainer.set_writer(writer)

    print("Starting MBPO-PPO training...")
    history = mbpo_trainer.train(
        env=env,
        replay_buffer=replay_buffer,
        world_model_trainer=wm_trainer,
        num_iterations=mbpo_config.max_iterations,
        collect_steps_per_iter=args.collect_steps,
        log_interval=10,
    )

    # Save policy
    policy_path = os.path.join(args.output_dir, f"policy_{args.robot}.pt")
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "mbpo_config": mbpo_config,
        },
        policy_path,
    )
    print(f"Policy saved to {policy_path}")

    writer.close()


if __name__ == "__main__":
    main()
