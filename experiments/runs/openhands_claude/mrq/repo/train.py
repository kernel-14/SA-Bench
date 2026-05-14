"""
MR.Q Training Loop
==================
Implements the full training procedure described in Section 4.2 of the paper.

Update schedule (Algorithm box in the paper)
--------------------------------------------
Every T_target steps:
  1. Hard-copy target networks: θ' ← θ, φ' ← φ, ω' ← ω
  2. Update reward scaling:     r̄' ← r̄, r̄ ← mean_D(|r|)

For each environment step:
  1. Collect transition (random action for first init_random_steps steps,
     then policy + exploration noise)
  2. Store in replay buffer
  3. If buffer ready: sample batch and perform one gradient update for
     encoder, value, and policy networks
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim

from config import MRQConfig, BENCHMARK_CONFIGS
from envs import make_env
from model import MRQAgent
from replay_buffer import LAPReplayBuffer
from utils import set_seed


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    agent: MRQAgent,
    cfg: MRQConfig,
    device: torch.device,
    n_episodes: int = 10,
) -> float:
    """Run n_episodes with the greedy policy and return mean episode return."""
    eval_env = make_env(cfg)
    returns = []

    for _ in range(n_episodes):
        obs = eval_env.reset()
        ep_return = 0.0
        done = False

        while not done:
            obs_t = _obs_to_tensor(obs, device)
            action = agent.select_action(obs_t, explore=False)
            action_np = action.cpu().numpy().squeeze(0)
            obs, reward, done, _ = eval_env.step(action_np)
            ep_return += reward

        returns.append(ep_return)

    eval_env.close()
    return float(np.mean(returns))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a single observation to a batched tensor (batch size 1)."""
    t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    return t


def _batch_to_device(
    states: np.ndarray,
    actions: np.ndarray,
    seq_actions: np.ndarray,
    seq_rewards: np.ndarray,
    seq_dones: np.ndarray,
    seq_next_states: np.ndarray,
    device: torch.device,
    enc_horizon: int,
    q_horizon: int,
):
    """Move sampled numpy arrays to torch tensors on the target device."""
    def t(x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=device)

    states_t = t(states)
    actions_t = t(actions)

    # Encoder uses first enc_horizon steps
    enc_actions = t(seq_actions[:, :enc_horizon])
    enc_rewards = t(seq_rewards[:, :enc_horizon])
    enc_dones = t(seq_dones[:, :enc_horizon])
    enc_next_states = t(seq_next_states[:, :enc_horizon])

    # Value uses first q_horizon steps
    q_rewards = t(seq_rewards[:, :q_horizon])
    q_dones = t(seq_dones[:, :q_horizon])
    # State at t=H_Q (for bootstrap)
    next_states_hq = t(seq_next_states[:, q_horizon - 1])

    return (
        states_t,
        actions_t,
        enc_actions,
        enc_rewards,
        enc_dones,
        enc_next_states,
        q_rewards,
        q_dones,
        next_states_hq,
    )


def _make_optimisers(agent: MRQAgent, cfg: MRQConfig):
    """Create AdamW optimisers for encoder, value, and policy networks."""
    enc_params = (
        list(agent.state_enc.parameters())
        + list(agent.sa_enc.parameters())
    )
    enc_opt = optim.AdamW(enc_params, lr=cfg.enc_lr, weight_decay=cfg.enc_weight_decay)

    value_params = list(agent.Q1.parameters()) + list(agent.Q2.parameters())
    value_opt = optim.AdamW(
        value_params, lr=cfg.value_lr, weight_decay=cfg.value_weight_decay
    )

    policy_opt = optim.AdamW(
        agent.pi.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_weight_decay
    )

    return enc_opt, value_opt, policy_opt


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: MRQConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Environment ───────────────────────────────────────────────────────────
    env = make_env(cfg)
    obs_shape = env.obs_shape
    action_dim = env.action_dim
    print(f"Env: {cfg.env_name}  obs={obs_shape}  action_dim={action_dim}  "
          f"action_type={env.action_type}")

    # ── Agent ─────────────────────────────────────────────────────────────────
    # For image obs the encoder takes the number of input channels
    if cfg.obs_type == "image":
        state_dim_or_channels = obs_shape[0]  # channels
    else:
        state_dim_or_channels = obs_shape[0]  # feature dim

    agent = MRQAgent(cfg, state_dim_or_channels, action_dim).to(device)

    enc_opt, value_opt, policy_opt = _make_optimisers(agent, cfg)

    # ── Replay buffer ─────────────────────────────────────────────────────────
    seq_len = max(cfg.enc_horizon, cfg.q_horizon)
    obs_dtype = np.uint8 if cfg.obs_type == "image" else np.float32
    replay = LAPReplayBuffer(
        capacity=cfg.buffer_size,
        obs_shape=obs_shape,
        action_dim=action_dim,
        seq_len=seq_len,
        lap_alpha=cfg.lap_alpha,
        min_priority=cfg.lap_min_priority,
        obs_dtype=obs_dtype,
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    os.makedirs(cfg.save_dir, exist_ok=True)
    log_path = os.path.join(cfg.save_dir, f"{cfg.env_name}_seed{cfg.seed}.csv")
    log_file = open(log_path, "w")
    log_file.write("step,eval_return,time\n")

    # ── Training state ────────────────────────────────────────────────────────
    obs = env.reset()
    ep_return = 0.0
    ep_steps = 0
    start_time = time.time()

    best_eval = -float("inf")

    for step in range(1, cfg.total_steps + 1):

        # ── Collect transition ────────────────────────────────────────────────
        if step <= cfg.init_random_steps:
            if cfg.action_type == "discrete":
                action_np = np.zeros(action_dim, dtype=np.float32)
                action_np[np.random.randint(action_dim)] = 1.0
            else:
                action_np = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
        else:
            obs_t = _obs_to_tensor(obs, device)
            with torch.no_grad():
                action_t = agent.select_action(
                    obs_t, explore=True, expl_noise_std=cfg.expl_noise_std
                )
            action_np = action_t.cpu().numpy().squeeze(0)

        next_obs, reward, done, _ = env.step(action_np)
        ep_return += reward
        ep_steps += 1

        # Track whether any terminal transition has been seen
        if done and not agent.terminal_loss_active:
            agent.terminal_loss_active = True

        # Store transition (use float done for terminal loss)
        replay.add(obs, action_np, reward, float(done), next_obs)

        obs = next_obs if not done else env.reset()
        if done:
            ep_return = 0.0
            ep_steps = 0

        # ── Gradient updates ──────────────────────────────────────────────────
        if step > cfg.init_random_steps and replay.ready:

            # Periodic target network update
            if step % cfg.target_update_freq == 0:
                mean_r = replay.mean_abs_reward()
                agent.update_targets(mean_r)

            for _ in range(cfg.replay_ratio):
                (
                    states_np,
                    actions_np,
                    seq_actions_np,
                    seq_rewards_np,
                    seq_dones_np,
                    seq_next_states_np,
                    indices,
                ) = replay.sample(cfg.batch_size)

                (
                    states_t,
                    actions_t,
                    enc_actions_t,
                    enc_rewards_t,
                    enc_dones_t,
                    enc_next_states_t,
                    q_rewards_t,
                    q_dones_t,
                    next_states_hq_t,
                ) = _batch_to_device(
                    states_np,
                    actions_np,
                    seq_actions_np,
                    seq_rewards_np,
                    seq_dones_np,
                    seq_next_states_np,
                    device,
                    cfg.enc_horizon,
                    cfg.q_horizon,
                )

                # ── Encoder update ────────────────────────────────────────────
                enc_opt.zero_grad()
                enc_loss = agent.encoder_loss(
                    states_t,
                    enc_actions_t,
                    enc_rewards_t,
                    enc_dones_t,
                    enc_next_states_t,
                )
                enc_loss.backward()
                enc_opt.step()

                # ── Value update ──────────────────────────────────────────────
                value_opt.zero_grad()
                val_loss, td_errors = agent.value_loss(
                    states_t,
                    actions_t,
                    q_rewards_t,
                    q_dones_t,
                    next_states_hq_t,
                )
                val_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(agent.Q1.parameters()) + list(agent.Q2.parameters()),
                    cfg.value_grad_clip,
                )
                value_opt.step()

                # ── Policy update ─────────────────────────────────────────────
                policy_opt.zero_grad()
                pol_loss = agent.policy_loss(states_t)
                pol_loss.backward()
                policy_opt.step()

                # ── Update LAP priorities ─────────────────────────────────────
                replay.update_priorities(indices, td_errors.cpu().numpy())

        # ── Logging ───────────────────────────────────────────────────────────
        if step % cfg.log_freq == 0:
            elapsed = time.time() - start_time
            print(
                f"Step {step:>8d} | "
                f"Buffer {len(replay):>7d} | "
                f"Elapsed {elapsed:.0f}s"
            )

        # ── Evaluation ────────────────────────────────────────────────────────
        if step % cfg.eval_freq == 0:
            eval_return = evaluate(agent, cfg, device, cfg.eval_episodes)
            elapsed = time.time() - start_time
            print(
                f"  ↳ Eval @ step {step:>8d}: "
                f"return = {eval_return:.2f}  "
                f"({elapsed:.0f}s)"
            )
            log_file.write(f"{step},{eval_return:.4f},{elapsed:.1f}\n")
            log_file.flush()

            if cfg.save_model and eval_return > best_eval:
                best_eval = eval_return
                model_path = os.path.join(
                    cfg.save_dir, f"{cfg.env_name}_seed{cfg.seed}_best.pt"
                )
                torch.save(agent.state_dict(), model_path)

    log_file.close()
    env.close()
    print("Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> MRQConfig:
    parser = argparse.ArgumentParser(description="MR.Q training")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="gym",
        choices=["gym", "dmc_proprio", "dmc_visual", "atari"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = BENCHMARK_CONFIGS[args.benchmark]
    # Dataclasses are frozen-like but we can copy and modify
    import dataclasses
    cfg = dataclasses.replace(
        cfg,
        env_name=args.env,
        seed=args.seed,
        save_dir=args.save_dir,
        save_model=args.save_model,
    )
    if args.total_steps is not None:
        cfg = dataclasses.replace(cfg, total_steps=args.total_steps)

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    print(cfg)
    train(cfg)
