"""Baseline training scripts for comparison with PGR.

Implements:
- SAC: Soft Actor-Critic (Haarnoja et al., 2018)
- REDQ: Randomized Ensembled Double Q-Learning (Chen et al., 2021)
- SYNTHER: Unconditional generative replay (Lu et al., 2024)
- PER (TD-error): Prioritized Experience Replay with TD-error priority
- PER (Curiosity): Prioritized Experience Replay with curiosity priority
- REDQ + Curiosity: REDQ with intrinsic curiosity exploration bonus
- SYNTHER + Curiosity: SYNTHER with intrinsic curiosity exploration bonus
- NoisyNets: REDQ with noisy network layers (Fortunato et al., 2018)
- Boot-DQN: REDQ with bootstrapped Q-value exploration (Osband et al., 2016)
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, MixedReplayBuffer
from models.diffusion import (
    ConditionalDiffusion,
    TransitionNormalizer,
    build_transition_tensor,
    unpack_transition_tensor,
)
from models.rl_agents import REDQ, SAC
from models.relevance import ICMRelevance, build_relevance_fn
from utils import make_env, set_seed, Logger, evaluate_policy
from config import SPARSE_REWARD_ENVS, FINGER_TURN_HARD_STEPS


def train_sac(
    env_name: str,
    seed: int = 0,
    total_env_steps: int = 100_000,
    hidden_dim: int = 256,
    n_hidden: int = 2,
    batch_size: int = 256,
    seed_steps: int = 5_000,
    eval_freq: int = 5_000,
    eval_episodes: int = 10,
    device: str = "cuda",
    log_dir: str = "logs",
    use_wandb: bool = False,
):
    """Train SAC baseline."""
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if env_name in SPARSE_REWARD_ENVS:
        total_env_steps = FINGER_TURN_HARD_STEPS

    env = make_env(env_name, seed=seed)
    eval_env = make_env(env_name, seed=seed + 100)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    logger = Logger(
        log_dir=os.path.join(log_dir, f"sac_{env_name}_seed{seed}"),
        use_wandb=use_wandb,
        config={"method": "sac", "env": env_name, "seed": seed},
    )

    buffer = ReplayBuffer(state_dim, action_dim, 1_000_000, str(device))
    agent = SAC(state_dim, action_dim, hidden_dim, n_hidden, device=str(device))

    state, _ = env.reset()
    episode_return = 0.0

    for step in range(total_env_steps):
        if step < seed_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_return += reward
        buffer.add(state, action, next_state, reward, float(terminated))
        state = next_state

        if done:
            logger.log({"train/episode_return": episode_return}, step)
            state, _ = env.reset()
            episode_return = 0.0

        if step >= seed_steps and buffer.size >= batch_size:
            batch = buffer.sample(batch_size)
            logs = agent.update(batch)
            logger.log(logs, step)

        if (step + 1) % eval_freq == 0:
            eval_return = evaluate_policy(agent, eval_env, eval_episodes)
            logger.log({"eval/episode_return": eval_return}, step)
            print(f"[SAC] Step {step + 1} | Eval: {eval_return:.2f}")

    env.close()
    eval_env.close()
    logger.close()


def train_redq(
    env_name: str,
    seed: int = 0,
    total_env_steps: int = 100_000,
    hidden_dim: int = 256,
    n_hidden: int = 2,
    utd_ratio: int = 20,
    batch_size: int = 256,
    seed_steps: int = 5_000,
    eval_freq: int = 5_000,
    eval_episodes: int = 10,
    device: str = "cuda",
    log_dir: str = "logs",
    use_wandb: bool = False,
    noisy: bool = False,
    use_bootstrap_mask: bool = False,
    curiosity_bonus: bool = False,
    curiosity_weight: float = 0.1,
):
    """Train REDQ baseline (optionally with NoisyNets, Boot-DQN, or curiosity bonus)."""
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if env_name in SPARSE_REWARD_ENVS:
        total_env_steps = FINGER_TURN_HARD_STEPS

    env = make_env(env_name, seed=seed)
    eval_env = make_env(env_name, seed=seed + 100)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    method_name = "redq"
    if noisy:
        method_name = "noisynets"
    elif use_bootstrap_mask:
        method_name = "boot_dqn"
    elif curiosity_bonus:
        method_name = "redq_curiosity"

    logger = Logger(
        log_dir=os.path.join(log_dir, f"{method_name}_{env_name}_seed{seed}"),
        use_wandb=use_wandb,
        config={"method": method_name, "env": env_name, "seed": seed},
    )

    buffer = ReplayBuffer(state_dim, action_dim, 1_000_000, str(device))
    agent = REDQ(
        state_dim, action_dim, hidden_dim, n_hidden,
        utd_ratio=utd_ratio, noisy=noisy,
        use_bootstrap_mask=use_bootstrap_mask,
        device=str(device),
    )

    icm = None
    if curiosity_bonus:
        icm = ICMRelevance(state_dim, action_dim, device=str(device))

    state, _ = env.reset()
    episode_return = 0.0

    for step in range(total_env_steps):
        if step < seed_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if curiosity_bonus and icm is not None:
            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                a_t = torch.FloatTensor(action).unsqueeze(0).to(device)
                sp_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                intr = icm.compute({"states": s_t, "actions": a_t, "next_states": sp_t})
                reward = reward + curiosity_weight * intr.item()

        episode_return += reward
        buffer.add(state, action, next_state, reward, float(terminated))
        state = next_state

        if done:
            logger.log({"train/episode_return": episode_return}, step)
            state, _ = env.reset()
            episode_return = 0.0

        if step >= seed_steps and buffer.size >= batch_size:
            for _ in range(utd_ratio):
                batch = buffer.sample(batch_size)
                logs = agent.update(batch)

            if curiosity_bonus and icm is not None:
                icm_batch = buffer.sample(batch_size)
                icm_logs = icm.update(icm_batch)
                logger.log(icm_logs, step)

            logger.log(logs, step)

        if (step + 1) % eval_freq == 0:
            eval_return = evaluate_policy(agent, eval_env, eval_episodes)
            logger.log({"eval/episode_return": eval_return}, step)
            print(f"[{method_name.upper()}] Step {step + 1} | Eval: {eval_return:.2f}")

    env.close()
    eval_env.close()
    logger.close()


def train_synther(
    env_name: str,
    seed: int = 0,
    total_env_steps: int = 100_000,
    hidden_dim: int = 256,
    n_hidden: int = 2,
    utd_ratio: int = 20,
    batch_size: int = 256,
    synthetic_ratio: float = 0.5,
    real_buffer_size: int = 1_000_000,
    syn_buffer_size: int = 1_000_000,
    inner_loop_freq: int = 10_000,
    diffusion_train_steps: int = 50_000,
    diffusion_batch_size: int = 256,
    n_diffusion_steps: int = 1000,
    seed_steps: int = 5_000,
    eval_freq: int = 5_000,
    eval_episodes: int = 10,
    device: str = "cuda",
    log_dir: str = "logs",
    use_wandb: bool = False,
    curiosity_bonus: bool = False,
    curiosity_weight: float = 0.1,
):
    """Train SYNTHER baseline: unconditional generative replay (Lu et al., 2024).

    SYNTHER is PGR without guidance — uses an unconditional diffusion model
    (guidance_scale=1.0, no conditioning on relevance).
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if env_name in SPARSE_REWARD_ENVS:
        total_env_steps = FINGER_TURN_HARD_STEPS

    env = make_env(env_name, seed=seed)
    eval_env = make_env(env_name, seed=seed + 100)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    transition_dim = 2 * state_dim + action_dim + 1

    method_name = "synther_curiosity" if curiosity_bonus else "synther"
    logger = Logger(
        log_dir=os.path.join(log_dir, f"{method_name}_{env_name}_seed{seed}"),
        use_wandb=use_wandb,
        config={"method": method_name, "env": env_name, "seed": seed},
    )

    real_buffer = ReplayBuffer(state_dim, action_dim, real_buffer_size, str(device))
    syn_buffer = ReplayBuffer(state_dim, action_dim, syn_buffer_size, str(device))
    mixed_buffer = MixedReplayBuffer(real_buffer, syn_buffer, synthetic_ratio)

    agent = REDQ(state_dim, action_dim, hidden_dim, n_hidden, utd_ratio=utd_ratio, device=str(device))

    diffusion = ConditionalDiffusion(
        transition_dim=transition_dim,
        hidden_dim=256,
        n_hidden_layers=4,
        time_embed_dim=128,
        cond_embed_dim=128,
        n_diffusion_steps=n_diffusion_steps,
        p_uncond=1.0,
        guidance_scale=1.0,
    ).to(device)
    diffusion_optimizer = torch.optim.Adam(diffusion.parameters(), lr=3e-4)
    normalizer = TransitionNormalizer()

    icm = None
    if curiosity_bonus:
        icm = ICMRelevance(state_dim, action_dim, device=str(device))

    state, _ = env.reset()
    episode_return = 0.0
    diffusion_trained = False

    for step in range(total_env_steps):
        if step < seed_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if curiosity_bonus and icm is not None:
            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                a_t = torch.FloatTensor(action).unsqueeze(0).to(device)
                sp_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                intr = icm.compute({"states": s_t, "actions": a_t, "next_states": sp_t})
                reward = reward + curiosity_weight * intr.item()

        episode_return += reward
        real_buffer.add(state, action, next_state, reward, float(terminated))
        state = next_state

        if done:
            logger.log({"train/episode_return": episode_return}, step)
            state, _ = env.reset()
            episode_return = 0.0

        if step < seed_steps:
            continue

        if (step + 1) % inner_loop_freq == 0 and real_buffer.size >= batch_size:
            transitions_arr = real_buffer.as_transitions_array()
            normalizer.fit(transitions_arr)

            diffusion.train()
            null_cond = torch.zeros(diffusion_batch_size, 1, device=device)
            total_loss = 0.0
            for _ in range(diffusion_train_steps):
                idx = np.random.randint(0, real_buffer.size, size=diffusion_batch_size)
                s = torch.FloatTensor(real_buffer.states[idx]).to(device)
                a = torch.FloatTensor(real_buffer.actions[idx]).to(device)
                sp = torch.FloatTensor(real_buffer.next_states[idx]).to(device)
                r = torch.FloatTensor(real_buffer.rewards[idx]).to(device)
                trans = build_transition_tensor(s, a, sp, r)
                trans_norm = normalizer.normalize_tensor(trans)
                loss = diffusion.loss(trans_norm, null_cond)
                diffusion_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
                diffusion_optimizer.step()
                total_loss += loss.item()
            diffusion.eval()
            logger.log({"diffusion/train_loss": total_loss / diffusion_train_steps}, step)

            syn_buffer.ptr = 0
            syn_buffer.size = 0
            n_generate = syn_buffer_size
            batch_gen = 1024
            for start in range(0, n_generate, batch_gen):
                n_batch = min(batch_gen, n_generate - start)
                cond = torch.zeros(n_batch, 1, device=device)
                with torch.no_grad():
                    gen_norm = diffusion.sample(n_batch, cond, use_guidance=False)
                gen = normalizer.denormalize_tensor(gen_norm).cpu().numpy()
                s_g = gen[:, :state_dim]
                a_g = gen[:, state_dim: state_dim + action_dim]
                sp_g = gen[:, state_dim + action_dim: 2 * state_dim + action_dim]
                r_g = gen[:, 2 * state_dim + action_dim:]
                syn_buffer.add_batch(s_g, a_g, sp_g, r_g, np.zeros((len(s_g), 1)))
            diffusion_trained = True

        if curiosity_bonus and icm is not None and real_buffer.size >= batch_size:
            icm_batch = real_buffer.sample(batch_size)
            icm_logs = icm.update(icm_batch)
            logger.log(icm_logs, step)

        if diffusion_trained and real_buffer.size >= batch_size:
            for _ in range(utd_ratio):
                batch = mixed_buffer.sample(batch_size)
                logs = agent.update(batch)
            logger.log(logs, step)
        elif real_buffer.size >= batch_size:
            batch = real_buffer.sample(batch_size)
            logs = agent.update(batch)

        if (step + 1) % eval_freq == 0:
            eval_return = evaluate_policy(agent, eval_env, eval_episodes)
            logger.log({"eval/episode_return": eval_return}, step)
            print(f"[{method_name.upper()}] Step {step + 1} | Eval: {eval_return:.2f}")

    env.close()
    eval_env.close()
    logger.close()


def train_per(
    env_name: str,
    priority_type: str = "td_error",
    seed: int = 0,
    total_env_steps: int = 100_000,
    hidden_dim: int = 256,
    n_hidden: int = 2,
    utd_ratio: int = 20,
    batch_size: int = 256,
    seed_steps: int = 5_000,
    eval_freq: int = 5_000,
    eval_episodes: int = 10,
    device: str = "cuda",
    log_dir: str = "logs",
    use_wandb: bool = False,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
):
    """Train REDQ with Prioritized Experience Replay (PER).

    Supports TD-error priority (Schaul et al., 2015) and curiosity priority.
    Used as a baseline to show PGR goes beyond simple prioritized replay.
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if env_name in SPARSE_REWARD_ENVS:
        total_env_steps = FINGER_TURN_HARD_STEPS

    env = make_env(env_name, seed=seed)
    eval_env = make_env(env_name, seed=seed + 100)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    method_name = f"per_{priority_type}"
    logger = Logger(
        log_dir=os.path.join(log_dir, f"{method_name}_{env_name}_seed{seed}"),
        use_wandb=use_wandb,
        config={"method": method_name, "env": env_name, "seed": seed},
    )

    buffer = PrioritizedReplayBuffer(
        state_dim, action_dim, 1_000_000, str(device),
        alpha=per_alpha, beta=per_beta,
    )
    agent = REDQ(state_dim, action_dim, hidden_dim, n_hidden, utd_ratio=utd_ratio, device=str(device))

    icm = None
    if priority_type == "curiosity":
        icm = ICMRelevance(state_dim, action_dim, device=str(device))

    state, _ = env.reset()
    episode_return = 0.0

    for step in range(total_env_steps):
        if step < seed_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_return += reward
        buffer.add(state, action, next_state, reward, float(terminated))
        state = next_state

        if done:
            logger.log({"train/episode_return": episode_return}, step)
            state, _ = env.reset()
            episode_return = 0.0

        if step < seed_steps or buffer.size < batch_size:
            continue

        for _ in range(utd_ratio):
            batch = buffer.sample(batch_size)
            indices = batch.pop("indices")
            weights = batch.pop("weights")

            if priority_type == "td_error":
                with torch.no_grad():
                    td_errors = agent.get_td_error(
                        batch["states"], batch["actions"],
                        batch["next_states"], batch["rewards"], batch["dones"]
                    ).cpu().numpy().flatten()
                buffer.update_priorities(indices, td_errors)
            elif priority_type == "curiosity" and icm is not None:
                with torch.no_grad():
                    curiosity = icm.compute(batch).cpu().numpy().flatten()
                buffer.update_priorities(indices, curiosity)

            logs = agent.update(batch)

        if icm is not None:
            icm_batch = buffer.sample(batch_size)
            icm_batch.pop("indices", None)
            icm_batch.pop("weights", None)
            icm_logs = icm.update(icm_batch)
            logger.log(icm_logs, step)

        logger.log(logs, step)

        if (step + 1) % eval_freq == 0:
            eval_return = evaluate_policy(agent, eval_env, eval_episodes)
            logger.log({"eval/episode_return": eval_return}, step)
            print(f"[{method_name.upper()}] Step {step + 1} | Eval: {eval_return:.2f}")

    env.close()
    eval_env.close()
    logger.close()


def main():
    parser = argparse.ArgumentParser(description="Baseline training for PGR comparison")
    parser.add_argument("--method", type=str, default="redq",
                        choices=["sac", "redq", "synther", "per_td", "per_curiosity",
                                 "redq_curiosity", "synther_curiosity", "noisynets", "boot_dqn"])
    parser.add_argument("--env", type=str, default="quadruped-walk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_env_steps", type=int, default=100_000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--utd_ratio", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--synthetic_ratio", type=float, default=0.5)
    parser.add_argument("--seed_steps", type=int, default=5_000)
    parser.add_argument("--eval_freq", type=int, default=5_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    common_kwargs = dict(
        env_name=args.env,
        seed=args.seed,
        total_env_steps=args.total_env_steps,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        batch_size=args.batch_size,
        seed_steps=args.seed_steps,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        device=args.device,
        log_dir=args.log_dir,
        use_wandb=args.use_wandb,
    )

    if args.method == "sac":
        train_sac(**common_kwargs)
    elif args.method == "redq":
        train_redq(utd_ratio=args.utd_ratio, **common_kwargs)
    elif args.method == "synther":
        train_synther(utd_ratio=args.utd_ratio, synthetic_ratio=args.synthetic_ratio, **common_kwargs)
    elif args.method == "per_td":
        train_per(priority_type="td_error", utd_ratio=args.utd_ratio, **common_kwargs)
    elif args.method == "per_curiosity":
        train_per(priority_type="curiosity", utd_ratio=args.utd_ratio, **common_kwargs)
    elif args.method == "redq_curiosity":
        train_redq(utd_ratio=args.utd_ratio, curiosity_bonus=True, **common_kwargs)
    elif args.method == "synther_curiosity":
        train_synther(utd_ratio=args.utd_ratio, synthetic_ratio=args.synthetic_ratio,
                      curiosity_bonus=True, **common_kwargs)
    elif args.method == "noisynets":
        train_redq(utd_ratio=args.utd_ratio, noisy=True, **common_kwargs)
    elif args.method == "boot_dqn":
        train_redq(utd_ratio=args.utd_ratio, use_bootstrap_mask=True, **common_kwargs)


if __name__ == "__main__":
    main()
