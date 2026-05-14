"""
Main training script for MR.Q.

Usage:
    # Gym locomotion
    python train.py --env_type gym --env_name HalfCheetah-v4 --seed 0

    # DMC proprioceptive
    python train.py --env_type dmc_proprio --env_name cheetah-run --seed 0

    # DMC visual
    python train.py --env_type dmc_visual --env_name cheetah-run --seed 0

    # Atari
    python train.py --env_type atari --env_name Pong --seed 0
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrq.agent import MRQ
from utils.replay_buffer import ReplayBuffer
from envs.wrappers import make_env


# ============================================================================
# Benchmark-specific settings
# ============================================================================

BENCHMARK_CONFIGS = {
    "gym": {
        "total_steps": 1_000_000,
        "eval_freq": 5_000,
        "eval_episodes": 10,
        "random_steps": 10_000,
        "replay_ratio": 1,  # 1 update per env step
    },
    "dmc_proprio": {
        "total_steps": 500_000,
        "eval_freq": 5_000,
        "eval_episodes": 10,
        "random_steps": 10_000,
        "replay_ratio": 1,
    },
    "dmc_visual": {
        "total_steps": 500_000,
        "eval_freq": 5_000,
        "eval_episodes": 10,
        "random_steps": 10_000,
        "replay_ratio": 1,
    },
    "atari": {
        "total_steps": 2_500_000,
        "eval_freq": 100_000,
        "eval_episodes": 10,
        "random_steps": 10_000,
        "replay_ratio": 1,
    },
}

# Gym locomotion environments
GYM_ENVS = [
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "Humanoid-v4",
    "Walker2d-v4",
]

# DMC environments (28 tasks)
DMC_ENVS = [
    "acrobot-swingup",
    "ball_in_cup-catch",
    "cartpole-balance",
    "cartpole-balance_sparse",
    "cartpole-swingup",
    "cartpole-swingup_sparse",
    "cheetah-run",
    "dog-run",
    "dog-stand",
    "dog-trot",
    "dog-walk",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "fish-swim",
    "hopper-hop",
    "hopper-stand",
    "humanoid-run",
    "humanoid-stand",
    "humanoid-walk",
    "pendulum-swingup",
    "quadruped-run",
    "quadruped-walk",
    "reacher-easy",
    "reacher-hard",
    "walker-run",
    "walker-stand",
    "walker-walk",
]

# Atari 57 games
ATARI_ENVS = [
    "Alien", "Amidar", "Assault", "Asterix", "Asteroids", "Atlantis",
    "BankHeist", "BattleZone", "BeamRider", "Berzerk", "Bowling", "Boxing",
    "Breakout", "Centipede", "ChopperCommand", "CrazyClimber", "Defender",
    "DemonAttack", "DoubleDunk", "Enduro", "FishingDerby", "Freeway",
    "Frostbite", "Gopher", "Gravitar", "Hero", "IceHockey", "Jamesbond",
    "Kangaroo", "Krull", "KungFuMaster", "MontezumaRevenge", "MsPacman",
    "NameThisGame", "Phoenix", "Pitfall", "Pong", "PrivateEye", "Qbert",
    "Riverraid", "RoadRunner", "Robotank", "Seaquest", "Skiing", "Solaris",
    "SpaceInvaders", "StarGunner", "Surround", "Tennis", "TimePilot",
    "Tutankham", "UpNDown", "Venture", "VideoPinball", "WizardOfWor",
    "YarsRevenge", "Zaxxon",
]


def evaluate(agent, env, n_episodes=10):
    """Evaluate agent for n_episodes, return mean episode reward."""
    total_reward = 0.0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(state, explore=False)
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward
        total_reward += ep_reward
    return total_reward / n_episodes


def train(args):
    """Main training loop."""
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    # Create environment
    env = make_env(args.env_type, args.env_name, seed=args.seed)
    eval_env = make_env(args.env_type, args.env_name, seed=args.seed + 100)

    print(f"Environment: {args.env_name} ({args.env_type})")
    print(f"  State dim: {env.state_dim}, Action dim: {env.action_dim}")
    print(f"  Discrete: {env.discrete}, Image obs: {env.image_obs}")

    # Benchmark config
    config = BENCHMARK_CONFIGS[args.env_type]
    total_steps = args.total_steps or config["total_steps"]
    eval_freq = args.eval_freq or config["eval_freq"]
    random_steps = args.random_steps or config["random_steps"]

    # Sequence length for buffer: max(H_enc, H_Q) + 1
    enc_horizon = 5
    q_horizon = 3
    seq_len = max(enc_horizon, q_horizon) + 1

    # Create replay buffer
    buffer = ReplayBuffer(
        state_dim=env.state_dim or 1,
        action_dim=env.action_dim,
        max_size=1_000_000,
        batch_size=256,
        image_obs=env.image_obs,
        state_channels=env.state_channels,
        image_size=84,
        seq_len=seq_len,
        lap_alpha=0.4,
        min_priority=1.0,
        device=device,
    )

    # Create agent
    agent = MRQ(
        state_dim=env.state_dim or 1,
        action_dim=env.action_dim,
        discrete=env.discrete,
        image_obs=env.image_obs,
        state_channels=env.state_channels,
        action_scale=env.action_scale,
        # Encoder
        enc_horizon=enc_horizon,
        lambda_reward=0.1,
        lambda_dynamics=1.0,
        lambda_terminal=0.1,
        reward_bins=65,
        reward_range=(-10.0, 10.0),
        # Value
        q_horizon=q_horizon,
        gamma=0.99,
        target_noise_std=0.2,
        target_noise_clip=0.3,
        # Policy
        lambda_preactiv=1e-5,
        gumbel_tau=10.0,
        # LAP
        lap_alpha=0.4,
        min_priority=1.0,
        # Training
        batch_size=256,
        enc_lr=1e-4,
        value_lr=3e-4,
        policy_lr=3e-4,
        weight_decay=1e-4,
        grad_clip_value=20.0,
        target_update_freq=250,
        # Exploration
        expl_noise_std=0.2,
        device=device,
    )

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(
        args.output_dir,
        f"{args.env_type}_{args.env_name}_seed{args.seed}.csv"
    )

    # Training loop
    state = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    episode_num = 0
    seen_terminal = False

    eval_rewards = []
    eval_steps = []

    print(f"\nStarting training for {total_steps} steps...")
    print(f"Random exploration for first {random_steps} steps")

    start_time = time.time()

    for t in range(1, total_steps + 1):
        # Select action
        if t <= random_steps:
            if env.discrete:
                action = np.random.randint(env.action_dim)
            else:
                action = np.random.uniform(-1.0, 1.0, env.action_dim).astype(np.float32)
        else:
            action = agent.select_action(state, explore=True)

        # Step environment
        next_state, reward, done, terminated, info = env.step(action)
        episode_reward += reward
        episode_steps += 1

        # Track terminal
        if terminated:
            seen_terminal = True
            agent.update_terminal_weight(True)

        # Store action as one-hot for discrete, raw for continuous
        if env.discrete:
            action_stored = np.zeros(env.action_dim, dtype=np.float32)
            action_stored[action] = 1.0
        else:
            action_stored = action

        # Add to buffer
        buffer.add(state, action_stored, reward, done)

        state = next_state

        # Episode end
        if done:
            episode_num += 1
            if episode_num % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Step {t:7d} | Episode {episode_num:4d} | "
                      f"Reward {episode_reward:8.1f} | "
                      f"Steps {episode_steps:4d} | "
                      f"Time {elapsed:.0f}s")
            state = env.reset()
            episode_reward = 0.0
            episode_steps = 0

        # Training
        if t > random_steps and len(buffer) >= 256:
            losses = agent.train(buffer)

        # Evaluation
        if t % eval_freq == 0:
            eval_reward = evaluate(agent, eval_env, n_episodes=10)
            eval_rewards.append(eval_reward)
            eval_steps.append(t)
            print(f"  >> Eval at step {t:7d}: mean reward = {eval_reward:.2f}")

            # Save results
            with open(results_file, "w") as f:
                f.write("step,reward\n")
                for s, r in zip(eval_steps, eval_rewards):
                    f.write(f"{s},{r:.4f}\n")

    # Final evaluation
    final_reward = evaluate(agent, eval_env, n_episodes=10)
    print(f"\nFinal evaluation: {final_reward:.2f}")

    # Save model
    model_path = os.path.join(
        args.output_dir,
        f"{args.env_type}_{args.env_name}_seed{args.seed}_final.pt"
    )
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    env.close()
    eval_env.close()

    return eval_rewards, eval_steps


def main():
    parser = argparse.ArgumentParser(description="Train MR.Q")

    # Environment
    parser.add_argument("--env_type", type=str, required=True,
                        choices=["gym", "dmc_proprio", "dmc_visual", "atari"],
                        help="Environment type")
    parser.add_argument("--env_name", type=str, required=True,
                        help="Environment name")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")

    # Training
    parser.add_argument("--total_steps", type=int, default=None,
                        help="Total training steps (default: benchmark-specific)")
    parser.add_argument("--eval_freq", type=int, default=None,
                        help="Evaluation frequency (default: benchmark-specific)")
    parser.add_argument("--random_steps", type=int, default=None,
                        help="Initial random exploration steps")

    # Output
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")

    # Device
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
