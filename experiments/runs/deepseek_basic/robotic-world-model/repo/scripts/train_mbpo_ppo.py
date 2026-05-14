#!/usr/bin/env python3
"""
Training script for MBPO-PPO (Model-Based Policy Optimization with PPO).

Implements Algorithm 1 from the paper:
1. Collect data in replay buffer D by interacting with the environment
2. Update world model with autoregressive training
3. Initialize imagination agents from D
4. Roll out imagination trajectories using policy and world model
5. Update policy using PPO

Usage:
    python scripts/train_mbpo_ppo.py --robot anymal_d --world_model_checkpoint /path/to/model.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rwm.world_model import RoboticWorldModel
from rwm.mbpo_ppo import MBPOPPO, MBPOPPOConfig, PPOConfig, ReplayBuffer
from rwm.training import WorldModelTrainer
from rwm.rewards import VelocityTrackingReward


def parse_args():
    parser = argparse.ArgumentParser(description='Train policy with MBPO-PPO')
    parser.add_argument('--robot', type=str, default='anymal_d',
                       choices=['anymal_d', 'unitree_g1'],
                       help='Robot type')
    parser.add_argument('--world_model_checkpoint', type=str, required=True,
                       help='Path to pretrained world model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs/policy',
                       help='Output directory')
    parser.add_argument('--imagination_envs', type=int, default=4096,
                       help='Number of parallel imagination environments')
    parser.add_argument('--imagination_steps', type=int, default=100,
                       help='Number of imagination steps per iteration')
    parser.add_argument('--buffer_size', type=int, default=1000,
                       help='Replay buffer size')
    parser.add_argument('--max_iterations', type=int, default=2500,
                       help='Maximum training iterations')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='PPO learning rate')
    parser.add_argument('--discount_factor', type=float, default=0.99,
                       help='Discount factor gamma')
    parser.add_argument('--clip_range', type=float, default=0.2,
                       help='PPO clip range')
    parser.add_argument('--entropy_coef', type=float, default=0.005,
                       help='Entropy coefficient')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--learning_epochs', type=int, default=5,
                       help='PPO learning epochs per iteration')
    parser.add_argument('--mini_batches', type=int, default=4,
                       help='Number of mini-batches')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    return parser.parse_args()


def create_mbpo_ppo(args, world_model):
    """Create MBPO-PPO instance."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Configure PPO
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        learning_epochs=args.learning_epochs,
        mini_batches=args.mini_batches,
        kl_target=0.01,
        discount_factor=args.discount_factor,
        clip_range=args.clip_range,
        entropy_coef=args.entropy_coef,
        gae_lambda=args.gae_lambda,
        max_grad_norm=1.0,
    )
    
    # Configure MBPO-PPO
    mbpo_config = MBPOPPOConfig(
        imagination_envs=args.imagination_envs,
        imagination_steps=args.imagination_steps,
        step_time=0.02,
        buffer_size=args.buffer_size,
        max_iterations=args.max_iterations,
        ppo_config=ppo_config,
    )
    
    # Create reward function
    reward_fn = VelocityTrackingReward(robot_type=args.robot, device=device)
    
    # Get dimensions
    obs_dim = world_model.obs_dim
    act_dim = world_model.act_dim
    
    mbpo = MBPOPPO(
        world_model=world_model,
        obs_dim=obs_dim,
        act_dim=act_dim,
        reward_fn=reward_fn,
        config=mbpo_config,
        device=device,
    )
    
    return mbpo


def main():
    args = parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained world model
    print(f"Loading world model from {args.world_model_checkpoint}")
    checkpoint = torch.load(args.world_model_checkpoint, map_location=device)
    
    # Reconstruct model from checkpoint arguments
    model_args = checkpoint.get('args', {})
    obs_dim = model_args.get('obs_dim', 45)
    act_dim = model_args.get('act_dim', 12)
    priv_dim = model_args.get('priv_dim', 8)
    history_horizon = model_args.get('history_horizon', 32)
    forecast_horizon = model_args.get('forecast_horizon', 8)
    
    # If checkpoint doesn't have args, infer from robot type
    if not model_args:
        if args.robot == 'anymal_d':
            obs_dim, act_dim, priv_dim = 45, 12, 8
        else:
            obs_dim, act_dim, priv_dim = 96, 29, 30
    
    world_model = RoboticWorldModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        priv_dim=priv_dim,
        history_horizon=history_horizon,
        forecast_horizon=forecast_horizon,
    )
    world_model.load_state_dict(checkpoint['model_state_dict'])
    world_model.to(device)
    world_model.eval()
    
    # Create MBPO-PPO
    mbpo = create_mbpo_ppo(args, world_model)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nMBPO-PPO Configuration:")
    print(f"  Imagination envs: {args.imagination_envs}")
    print(f"  Imagination steps: {args.imagination_steps}")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"  Max iterations: {args.max_iterations}")
    
    # Training loop
    # Note: This requires an actual environment to interact with.
    # In practice, we would integrate with Isaac Lab or a similar simulator.
    # Here we provide the skeleton for the Algorithm 1 loop.
    
    print("\nMBPO-PPO training loop structure (Algorithm 1):")
    print("  1. Collect observation-action pairs in D by interacting with environment")
    print("  2. Update world model p_phi with autoregressive training")
    print("  3. Initialize imagination agents with observations sampled from D")
    print("  4. Roll out imagination trajectories using policy and world model")
    print("  5. Update policy using PPO")
    
    # Save initial policy
    torch.save({
        'policy_state_dict': mbpo.ppo.policy.state_dict(),
        'value_state_dict': mbpo.ppo.value.state_dict(),
        'config': vars(args),
    }, output_dir / 'initial_policy.pt')
    
    print(f"\nInitial policy saved to {output_dir}/initial_policy.pt")
    print("Ready for integration with Isaac Lab or custom environment.")


if __name__ == '__main__':
    main()
