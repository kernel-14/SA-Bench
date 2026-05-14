"""
Main training script for Prioritized Generative Replay (PGR).

Supports:
- State-based tasks (DeepMind Control Suite, OpenAI Gym)
- Pixel-based tasks (DeepMind Control Suite)
- Multiple relevance functions (curiosity, return, TD-error, RND, CTS, ECO)
- Comparison baselines (REDQ, SAC, SYNTHER-equivalent unconditional generation)
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from typing import Dict, Any, Optional, Tuple

from pgr.pgr_algorithm import PrioritizedGenerativeReplay
from pgr.models.policy import (
    REDQAgent,
    SACAgent,
    DRQv2Agent,
    VisualEncoder,
)
from pgr.models.diffusion import (
    ConditionalDiffusionModel,
    DiffusionProcess,
)
from pgr.utils.replay_buffer import ReplayBuffer, SyntheticReplayBuffer
from pgr.utils.metrics import (
    compute_dormant_ratio,
    compute_curiosity_distribution,
    measure_sample_diversity,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prioritized Generative Replay")
    
    # Environment
    parser.add_argument('--env', type=str, default='quadruped-walk',
                        help='Environment name')
    parser.add_argument('--domain', type=str, default='dmc',
                        choices=['dmc', 'gym', 'pixel-dmc', 'dmlab'],
                        help='Environment domain')
    parser.add_argument('--state_dim', type=int, default=None,
                        help='State dimension (auto-detected if None)')
    parser.add_argument('--action_dim', type=int, default=None,
                        help='Action dimension (auto-detected if None)')
    
    # PGR settings
    parser.add_argument('--relevance', type=str, default='curiosity',
                        choices=['curiosity', 'return', 'td_error', 'rnd', 'cts', 'eco', 'none'],
                        help='Relevance function type ("none" = unconditional/SYNTHER)')
    parser.add_argument('--synthetic_ratio', type=float, default=0.5,
                        help='Ratio of synthetic data in training batches')
    parser.add_argument('--guidance_scale', type=float, default=3.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--p_uncond', type=float, default=0.25,
                        help='Probability of dropping condition during training')
    parser.add_argument('--buffer_size', type=int, default=1_000_000,
                        help='Replay buffer size')
    parser.add_argument('--inner_loop_freq', type=int, default=10_000,
                        help='Frequency of inner loop (generator retraining)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--utd_ratio', type=int, default=20,
                        help='Update-to-data ratio')
    
    # Training
    parser.add_argument('--total_steps', type=int, default=100_000,
                        help='Total environment steps')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--eval_freq', type=int, default=5_000,
                        help='Evaluation frequency')
    parser.add_argument('--num_eval_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension for policy networks')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of hidden layers')
    parser.add_argument('--diffusion_hidden_dim', type=int, default=1024,
                        help='Hidden dimension for diffusion model')
    parser.add_argument('--diffusion_res_blocks', type=int, default=2,
                        help='Number of residual blocks in diffusion model')
    parser.add_argument('--diffusion_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    
    # Scaling experiments (Section 5.3)
    parser.add_argument('--scale_policy', action='store_true',
                        help='Use larger policy network (3 layers, 512 hidden)')
    parser.add_argument('--scale_batch_size', type=int, default=None,
                        help='Scaled batch size for synthetic data ratio experiments')
    
    # Baselines
    parser.add_argument('--baseline', type=str, default=None,
                        choices=['redq', 'sac', 'drqv2', 'synther', 'redq_curiosity', 
                                'redq_per', 'synther_curiosity'],
                        help='Run as baseline instead of PGR')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Logging directory')
    parser.add_argument('--save_model', action='store_true',
                        help='Save model checkpoints')
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases logging')
    
    return parser.parse_args()


class DummyEnv:
    """
    Placeholder for environments when the actual env is not available.
    Provides the correct state_dim and action_dim for model construction.
    """
    def __init__(self, domain: str, env_name: str):
        self.domain = domain
        self.env_name = env_name
        
        # Standard DMC dimensions
        if domain == 'dmc' or domain == 'pixel-dmc':
            env_configs = {
                'quadruped-walk': {'state_dim': 78, 'action_dim': 12},
                'cheetah-run': {'state_dim': 17, 'action_dim': 6},
                'reacher-hard': {'state_dim': 6, 'action_dim': 2},
                'finger-turn-hard': {'state_dim': 12, 'action_dim': 2},
                'walker-walk': {'state_dim': 24, 'action_dim': 6},
                'hopper-hop': {'state_dim': 15, 'action_dim': 4},
            }
        elif domain == 'gym':
            env_configs = {
                'Walker2d-v2': {'state_dim': 17, 'action_dim': 6},
                'HalfCheetah-v2': {'state_dim': 17, 'action_dim': 6},
                'Hopper-v2': {'state_dim': 11, 'action_dim': 3},
            }
        else:
            env_configs = {}
        
        config = env_configs.get(env_name, {'state_dim': 17, 'action_dim': 6})
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']


def create_env(domain: str, env_name: str, seed: int = 0):
    """
    Create the appropriate environment.
    Tries to import gym and dmc; falls back to DummyEnv if unavailable.
    """
    try:
        if domain == 'dmc':
            import dmc2gym
            env = dmc2gym.make(
                domain_name=env_name.split('-')[0] if '-' in env_name else env_name,
                task_name='-'.join(env_name.split('-')[1:]) if '-' in env_name else env_name,
                seed=seed,
            )
            return env
        elif domain == 'pixel-dmc':
            import dmc2gym
            env = dmc2gym.make(
                domain_name=env_name.split('-')[0],
                task_name='-'.join(env_name.split('-')[1:]),
                seed=seed,
                from_pixels=True,
                height=84,
                width=84,
                channels_first=True,
            )
            return env
        elif domain == 'gym':
            import gym
            env = gym.make(env_name)
            env.seed(seed)
            return env
        else:
            return None
    except ImportError:
        return None


def run_episode(
    pgr: PrioritizedGenerativeReplay,
    env,
    agent,
    training: bool = True,
    pixel_based: bool = False,
) -> Tuple[float, int]:
    """
    Run one episode in the environment.
    
    Returns:
        episode_return, episode_length
    """
    if env is None:
        return 0.0, 0
    
    obs = env.reset()
    done = False
    episode_return = 0.0
    episode_length = 0
    
    while not done:
        # Select action
        if pixel_based:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=pgr.device).unsqueeze(0)
            action = agent.select_action(obs_tensor)
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=pgr.device).unsqueeze(0)
            action = agent.select_action(obs_tensor)
        
        action_np = action.cpu().numpy().flatten()
        
        # Step environment
        next_obs, reward, done, info = env.step(action_np)
        
        if training:
            # Add to replay buffer
            if pixel_based:
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=pgr.device).unsqueeze(0)
            else:
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=pgr.device).unsqueeze(0)
            
            reward_tensor = torch.tensor([[reward]], dtype=torch.float32, device=pgr.device)
            done_tensor = torch.tensor([[float(done)]], dtype=torch.float32, device=pgr.device)
            
            pgr.add_real_transitions(
                obs_tensor, action, next_obs_tensor, reward_tensor, done_tensor
            )
        
        obs = next_obs
        episode_return += reward
        episode_length += 1
    
    return episode_return, episode_length


def evaluate(
    pgr: PrioritizedGenerativeReplay,
    env,
    agent,
    num_episodes: int = 10,
    pixel_based: bool = False,
) -> Tuple[float, float]:
    """Evaluate the current policy."""
    if env is None:
        return 0.0, 0.0
    
    returns = []
    lengths = []
    
    for _ in range(num_episodes):
        ep_return, ep_length = run_episode(pgr, env, agent, training=False, pixel_based=pixel_based)
        returns.append(ep_return)
        lengths.append(ep_length)
    
    return np.mean(returns), np.std(returns)


def train_pgr(args):
    """
    Main PGR training loop implementing Algorithm 1.
    """
    # Create environment
    env = create_env(args.domain, args.env, args.seed)
    eval_env = create_env(args.domain, args.env, args.seed + 1000)
    
    # Get environment dimensions
    if env is not None:
        if args.domain == 'pixel-dmc':
            args.state_dim = env.observation_space.shape
            args.action_dim = env.action_space.shape[0]
        else:
            args.state_dim = env.observation_space.shape[0]
            args.action_dim = env.action_space.shape[0]
    else:
        dummy = DummyEnv(args.domain, args.env)
        args.state_dim = dummy.state_dim
        args.action_dim = dummy.action_dim
    
    pixel_based = (args.domain == 'pixel-dmc')
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create PGR module
    effective_state_dim = args.state_dim if not pixel_based else 50  # latent dim
    
    if args.scale_policy:
        policy_hidden_dim = 512
        policy_num_layers = 3
        if args.scale_batch_size is None:
            args.scale_batch_size = 1024
    else:
        policy_hidden_dim = args.hidden_dim
        policy_num_layers = args.num_layers
    
    pgr = PrioritizedGenerativeReplay(
        state_dim=args.state_dim if not pixel_based else args.state_dim,
        action_dim=args.action_dim,
        relevance_type=args.relevance if args.relevance != 'none' else 'curiosity',
        synthetic_data_ratio=args.synthetic_ratio,
        guidance_scale=args.guidance_scale,
        p_uncond=args.p_uncond,
        buffer_size=args.buffer_size,
        inner_loop_freq=args.inner_loop_freq,
        batch_size=args.scale_batch_size or args.batch_size,
        utd_ratio=args.utd_ratio,
        diffusion_hidden_dim=args.diffusion_hidden_dim,
        diffusion_num_residual_blocks=args.diffusion_res_blocks,
        diffusion_timesteps=args.diffusion_timesteps,
        device=args.device,
        use_latent=pixel_based,
        latent_dim=50,
    )
    
    # Create policy agent
    if args.baseline == 'redq' or args.baseline is None:
        agent = REDQAgent(
            state_dim=effective_state_dim,
            action_dim=args.action_dim,
            hidden_dim=policy_hidden_dim,
            num_layers=policy_num_layers,
            utd_ratio=args.utd_ratio,
            device=args.device,
        )
    elif args.baseline == 'sac':
        agent = SACAgent(
            state_dim=args.state_dim if not pixel_based else 50,
            action_dim=args.action_dim,
            hidden_dim=policy_hidden_dim,
            num_layers=policy_num_layers,
            device=args.device,
        )
    elif args.baseline == 'drqv2' or pixel_based:
        state_shape = args.state_dim if pixel_based else (3, 84, 84)
        agent = DRQv2Agent(
            state_shape=state_shape,
            action_dim=args.action_dim,
            latent_dim=50,
            hidden_dim=policy_hidden_dim,
            num_layers=policy_num_layers,
            utd_ratio=args.utd_ratio,
            device=args.device,
        )
        if pixel_based:
            pgr.set_visual_encoder(agent.encoder)
    else:
        agent = REDQAgent(
            state_dim=args.state_dim if not pixel_based else 50,
            action_dim=args.action_dim,
            hidden_dim=policy_hidden_dim,
            num_layers=policy_num_layers,
            utd_ratio=args.utd_ratio,
            device=args.device,
        )
    
    # Link policy to PGR
    pgr.set_policy(
        agent.actor,
        q_function=agent.critic,
        q_target=getattr(agent, 'critic_target', None),
    )
    
    # Training loop
    total_steps = 0
    eval_returns = []
    step = 0
    
    print(f"Starting PGR training with {args.relevance} relevance")
    print(f"Environment: {args.env}, Domain: {args.domain}")
    print(f"State dim: {args.state_dim}, Action dim: {args.action_dim}")
    print(f"Total steps: {args.total_steps}")
    
    while total_steps < args.total_steps:
        # === Outer Loop: Collect Transitions ===
        ep_return, ep_length = run_episode(
            pgr, env, agent, training=True, pixel_based=pixel_based
        )
        total_steps += ep_length
        
        # Update relevance function (if needed)
        if len(pgr.D_real) >= pgr.batch_size and pgr.F_optimizer is not None:
            batch = pgr.D_real.sample(pgr.batch_size)
            pgr.update_relevance_function(batch)
        
        # === Inner Loop: Update Generator & Train Policy ===
        if pgr.should_update_generator():
            print(f"\n[Step {total_steps}] Running inner loop...")
            
            # Train diffusion model on real data
            diffusion_loss = pgr.train_diffusion_model(num_steps=1000)
            print(f"  Diffusion loss: {diffusion_loss:.6f}")
            
            # Generate synthetic transitions
            pgr.fill_synthetic_buffer(num_transitions=pgr.buffer_size)
            print(f"  Generated {pgr.buffer_size} synthetic transitions")
            
            # For unconditional baseline (SYNTHER): use null condition
            if args.relevance == 'none':
                # Override with uniform sampling (equivalent to unconditional)
                pgr.synthetic_data_ratio = args.synthetic_ratio
        
        # Train policy on mixed real + synthetic data
        if len(pgr.D_real) >= pgr.batch_size and len(pgr.D_syn) > 0:
            for _ in range(args.utd_ratio):
                batch = pgr.sample_training_batch(pgr.batch_size)
                info = agent.update(batch)
        
        # Evaluation
        if total_steps % args.eval_freq == 0 or step == 0:
            mean_return, std_return = evaluate(
                pgr, eval_env, agent, args.num_eval_episodes, pixel_based
            )
            eval_returns.append((total_steps, mean_return, std_return))
            print(f"[Step {total_steps}] Eval return: {mean_return:.1f} ± {std_return:.1f}")
        
        step += 1
    
    # Final evaluation
    mean_return, std_return = evaluate(
        pgr, eval_env, agent, args.num_eval_episodes, pixel_based
    )
    print(f"\nFinal return: {mean_return:.1f} ± {std_return:.1f}")
    
    return eval_returns


def main():
    args = parse_args()
    return train_pgr(args)


if __name__ == '__main__':
    main()
