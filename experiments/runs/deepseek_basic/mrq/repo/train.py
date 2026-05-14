"""
Training script for MR.Q algorithm.

Usage:
    python train.py --env HalfCheetah-v4 --total_steps 1000000
    python train.py --domain dmc --task cheetah_run --total_steps 500000
    python train.py --domain atari --task Alien --total_steps 2500000
"""

import argparse
import os
import numpy as np
import torch
import gymnasium as gym

from mrq.agent import MRQ


def make_env(domain, task, image_obs=False):
    """Create environment based on domain and task."""
    if domain == 'gym':
        env = gym.make(task)
    elif domain == 'dmc':
        try:
            import dmc2gym
            env = dmc2gym.make(
                domain_name=task.split('_')[0],
                task_name='_'.join(task.split('_')[1:]),
                from_pixels=image_obs,
                frame_skip=2  # action repeat of 2
            )
        except ImportError:
            raise ImportError("dmc2gym is required for DM Control tasks. "
                            "Install with: pip install dmc2gym")
    elif domain == 'atari':
        env = gym.make(task, render_mode=None)
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    return env


def get_env_info(env):
    """Get state dim, action dim, and discrete flag from environment."""
    if isinstance(env.observation_space, gym.spaces.Box):
        state_dim = env.observation_space.shape[0]
        image_obs = len(env.observation_space.shape) == 3
    else:
        # Atari or other image-based
        state_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 84 * 84
        image_obs = True
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        discrete = True
    else:
        action_dim = env.action_space.shape[0]
        discrete = False
    
    return state_dim, action_dim, discrete, image_obs


def evaluate(agent, env, num_episodes=10):
    """Evaluate agent performance."""
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action_eval(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), np.std(total_rewards)


def main():
    parser = argparse.ArgumentParser(description='Train MR.Q agent')
    parser.add_argument('--domain', type=str, default='gym',
                        choices=['gym', 'dmc', 'atari'],
                        help='Benchmark domain')
    parser.add_argument('--task', type=str, default='HalfCheetah-v4',
                        help='Environment task name')
    parser.add_argument('--total_steps', type=int, default=1000000,
                        help='Total training steps')
    parser.add_argument('--eval_freq', type=int, default=5000,
                        help='Evaluation frequency (steps)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create environment
    env = make_env(args.domain, args.task)
    state_dim, action_dim, discrete, image_obs = get_env_info(env)
    
    print(f"Environment: {args.task}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Discrete actions: {discrete}, Image observations: {image_obs}")
    
    # Create MR.Q agent
    agent = MRQ(
        state_dim=state_dim,
        action_dim=action_dim,
        discrete_action_space=discrete,
        image_observations=image_obs,
        state_channels=1 if image_obs and args.domain == 'atari' else 3,
        device=args.device
    )
    
    # Training loop
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    eval_history = []
    
    for step in range(args.total_steps):
        # Select action (random for initial steps)
        if step < agent.initial_random_steps:
            if discrete:
                action = np.zeros(action_dim)
                action[np.random.randint(action_dim)] = 1.0
            else:
                action = env.action_space.sample()
        else:
            action = agent.select_action(state, explore=True)
        
        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        if discrete:
            # Convert integer action to one-hot
            action_one_hot = np.zeros(action_dim)
            action_one_hot[int(np.argmax(action))] = 1.0
            agent.replay_buffer.add(state, action_one_hot, reward, next_state, done)
        else:
            agent.replay_buffer.add(state, action, reward, next_state, done)
        
        episode_reward += reward
        episode_steps += 1
        agent.total_steps += 1
        
        state = next_state
        
        if done:
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
        
        # Training update
        if step >= agent.initial_random_steps:
            info = agent.update()
        
        # Evaluation
        if (step + 1) % args.eval_freq == 0:
            mean_reward, std_reward = evaluate(agent, env)
            eval_history.append((step + 1, mean_reward, std_reward))
            print(f"Step {step + 1}: Eval reward = {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Save checkpoint
            os.makedirs(args.save_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                args.save_dir, 
                f"mrq_{args.domain}_{args.task}_seed{args.seed}_step{step+1}.pt"
            )
            agent.save(checkpoint_path)
    
    # Final evaluation
    mean_reward, std_reward = evaluate(agent, env, num_episodes=10)
    print(f"\nFinal evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Save results
    np.savez(
        os.path.join(args.save_dir, f"eval_history_{args.domain}_{args.task}_seed{args.seed}.npz"),
        eval_history=eval_history
    )
    
    env.close()


if __name__ == '__main__':
    main()
