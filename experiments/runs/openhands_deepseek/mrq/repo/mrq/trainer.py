"""Training loop for MR.Q across Gym, DMC, and Atari benchmarks."""

import os
import time
import random
import numpy as np
from collections import deque
from typing import Optional, Dict, Any

import gymnasium as gym
import torch

from mrq.config import MRQConfig, gym_locomotion_config, dmc_proprio_config, dmc_visual_config, atari_config
from mrq.agent import MRQAgent
from mrq.replay_buffer import ReplayBuffer, ImageReplayBuffer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(config: MRQConfig) -> gym.Env:
    """Create environment based on config settings."""
    env_name = config.env_name

    if config.observation_type == "image":
        if "Atari" in env_name or any(a in env_name.lower() for a in ["alien", "amidar"]):
            # Atari environment
            try:
                env = gym.make(f"ALE/{env_name}-v5", render_mode="rgb_array")
            except Exception:
                env = gym.make(env_name, render_mode="rgb_array")
        else:
            # DMC visual
            try:
                env = gym.make(f"dm_control/{env_name}", render_mode="rgb_array")
            except Exception:
                env = gym.make(env_name, render_mode="rgb_array")
    else:
        if "dm_control" in env_name.lower() or env_name in [
            "acrobot-swingup", "ball_in_cup-catch", "cartpole-balance", "cartpole-balance_sparse",
            "cartpole-swingup", "cartpole-swingup_sparse", "cheetah-run", "dog-run", "dog-stand",
            "dog-trot", "dog-walk", "finger-spin", "finger-turn_easy", "finger-turn_hard",
            "fish-swim", "hopper-hop", "hopper-stand", "humanoid-run", "humanoid-stand",
            "humanoid-walk", "pendulum-swingup", "quadruped-run", "quadruped-walk",
            "reacher-easy", "reacher-hard", "walker-run", "walker-stand", "walker-walk",
        ]:
            try:
                env = gym.make(f"dm_control/{env_name}")
            except Exception:
                # Try gymnasium MuJoCo environments
                try:
                    env = gym.make(env_name)
                except Exception:
                    env = gym.make(f"{env_name}", render_mode="rgb_array")
        else:
            try:
                env = gym.make(env_name)
            except Exception:
                env = gym.make(env_name, render_mode="rgb_array")

    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def evaluate(agent: MRQAgent, eval_env: gym.Env, num_episodes: int = 10) -> float:
    returns = []
    for _ in range(num_episodes):
        state, _ = eval_env.reset()
        done = False
        truncated = False
        total = 0.0
        while not done and not truncated:
            action = agent.select_action(state, deterministic=True)
            state, reward, done, truncated, _ = eval_env.step(action)
            total += reward
        returns.append(total)
    return np.mean(returns)


def run_training(
    config: MRQConfig,
    env_name: str,
    log_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Main training loop."""
    set_seed(config.seed)
    config.env_name = env_name

    env = make_env(config)
    eval_env = make_env(config)

    obs_space = env.observation_space
    act_space = env.action_space

    if config.observation_type == "image":
        if hasattr(obs_space, 'shape') and len(obs_space.shape) == 3:
            state_dim = int(np.prod(obs_space.shape))
            img_shape = obs_space.shape
        else:
            img_shape = (1, 84, 84)
            state_dim = 1 * 84 * 84
    else:
        state_dim = obs_space.shape[0]
        img_shape = None

    if isinstance(act_space, gym.spaces.Discrete):
        action_dim = act_space.n
        config.discrete_actions = True
    else:
        action_dim = act_space.shape[0]
        config.discrete_actions = False

    if config.discrete_actions:
        action_low = None
        action_high = None
    else:
        action_low = act_space.low.astype(np.float32)
        action_high = act_space.high.astype(np.float32)

    agent = MRQAgent(config, state_dim, action_dim,
                     action_space_low=action_low, action_space_high=action_high,
                     device=device)

    if img_shape is not None:
        replay_buffer = ImageReplayBuffer(config.replay_buffer_capacity,
                                          config.lap_alpha, config.lap_min_priority,
                                          device, img_shape)
    else:
        replay_buffer = ReplayBuffer(state_dim, action_dim, config.replay_buffer_capacity,
                                     config.lap_alpha, config.lap_min_priority, device)

    state, _ = env.reset()
    total_steps = config.total_timesteps
    eval_results = []
    episode_return = 0.0
    episode_len = 0

    print(f"Training {env_name} for {total_steps} steps, action_dim={action_dim}")

    for step in range(total_steps):
        if step < config.start_timesteps:
            if config.discrete_actions:
                a_idx = env.action_space.sample()
                action = np.zeros(action_dim, dtype=np.float32)
                action[a_idx] = 1.0
            else:
                action = env.action_space.sample()
        else:
            action = agent.select_action(state, explore=True)

        next_state, reward, done, truncated, info = env.step(action)
        episode_return += reward
        episode_len += 1

        if config.discrete_actions and not isinstance(action, np.ndarray):
            action = np.array([action], dtype=np.float32)

        replay_buffer.push(state, action, float(reward), float(done), next_state)

        state = next_state

        if done or truncated:
            state, _ = env.reset()
            episode_return = 0.0
            episode_len = 0

        if step >= config.start_timesteps and len(replay_buffer) >= config.batch_size:
            for _ in range(config.replay_ratio):
                metrics = agent.update(replay_buffer)

        if (step + 1) % config.eval_freq == 0:
            eval_return = evaluate(agent, eval_env, config.eval_episodes)
            eval_results.append((step + 1, eval_return))
            print(f"Step {step+1}/{total_steps} | Eval: {eval_return:.2f}")

    env.close()
    eval_env.close()
    return {"eval_results": eval_results, "agent": agent}


# Environment lists for each benchmark
GYM_LOCOMOTION_ENVS = [
    "Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Humanoid-v4", "Walker2d-v4",
]

DMC_PROPRIO_ENVS = [
    "acrobot-swingup", "ball_in_cup-catch", "cartpole-balance", "cartpole-balance_sparse",
    "cartpole-swingup", "cartpole-swingup_sparse", "cheetah-run", "dog-run", "dog-stand",
    "dog-trot", "dog-walk", "finger-spin", "finger-turn_easy", "finger-turn_hard",
    "fish-swim", "hopper-hop", "hopper-stand", "humanoid-run", "humanoid-stand",
    "humanoid-walk", "pendulum-swingup", "quadruped-run", "quadruped-walk",
    "reacher-easy", "reacher-hard", "walker-run", "walker-stand", "walker-walk",
]

DMC_VISUAL_ENVS = DMC_PROPRIO_ENVS

ATARI_ENVS = [
    "Alien", "Amidar", "Assault", "Asterix", "Asteroids", "Atlantis", "BankHeist",
    "BattleZone", "BeamRider", "Berzerk", "Bowling", "Boxing", "Breakout", "Centipede",
    "ChopperCommand", "CrazyClimber", "Defender", "DemonAttack", "DoubleDunk", "Enduro",
    "FishingDerby", "Freeway", "Frostbite", "Gopher", "Gravitar", "Hero", "IceHockey",
    "Jamesbond", "Kangaroo", "Krull", "KungFuMaster", "MontezumaRevenge", "MsPacman",
    "NameThisGame", "Phoenix", "Pitfall", "Pong", "PrivateEye", "Qbert", "Riverraid",
    "RoadRunner", "Robotank", "Seaquest", "Skiing", "Solaris", "SpaceInvaders",
    "StarGunner", "Surround", "Tennis", "TimePilot", "Tutankham", "UpNDown",
    "Venture", "VideoPinball", "WizardOfWor", "YarsRevenge", "Zaxxon",
]
