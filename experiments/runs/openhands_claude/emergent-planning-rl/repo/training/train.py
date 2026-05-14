import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import deque

from config import Config, CELL_TYPES
from model.drc import DRCAgent
from training.impala import impala_loss
from environment.sokoban import SokobanEnv
from data.boxoban import LevelSampler, create_level_sampler


class RolloutBuffer:
    """Stores a single unroll of T steps for B actors."""

    def __init__(self, unroll_length: int, num_actors: int, num_actions: int, device: str):
        self.T = unroll_length
        self.B = num_actors
        self.A = num_actions
        self.device = device
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.behavior_log_probs = []
        self.hidden_states = None
        self.cell_states = None
        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        behavior_log_prob: torch.Tensor,
    ):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.behavior_log_probs.append(behavior_log_prob)
        self.step += 1

    def get(self) -> Dict[str, torch.Tensor]:
        return {
            "obs": torch.stack(self.obs, dim=0),
            "actions": torch.stack(self.actions, dim=0),
            "rewards": torch.stack(self.rewards, dim=0),
            "dones": torch.stack(self.dones, dim=0),
            "behavior_log_probs": torch.stack(self.behavior_log_probs, dim=0),
        }


class ActorEnvironment:
    """Single actor environment wrapper."""

    def __init__(self, level_sampler: LevelSampler, env_config, seed: int = 0):
        self.env = SokobanEnv(
            grid_size=env_config.grid_size,
            min_steps=env_config.min_episode_steps,
            max_steps=env_config.max_episode_steps,
            reward_step=env_config.reward_step,
            reward_box_on_target=env_config.reward_box_on_target,
            reward_box_off_target=env_config.reward_box_off_target,
            reward_solved=env_config.reward_solved,
        )
        self.level_sampler = level_sampler
        self.obs = None
        self.done = True
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.rng = random.Random(seed)

    def reset(self) -> np.ndarray:
        level = self.level_sampler.next_level()
        self.obs = self.env.reset(level)
        self.done = False
        self.episode_reward = 0.0
        self.episode_steps = 0
        return self.obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self.done:
            self.reset()
        obs, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_steps += 1
        self.done = done
        self.obs = obs
        return obs, reward, done

    def get_obs(self) -> np.ndarray:
        if self.done or self.obs is None:
            return self.reset()
        return self.obs


def collect_rollout(
    agent: DRCAgent,
    actors: List[ActorEnvironment],
    hidden_states: List[torch.Tensor],
    cell_states: List[torch.Tensor],
    unroll_length: int,
    device: torch.device,
    num_actions: int,
) -> Tuple[Dict, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Collect T steps from B actors using the current policy.
    hidden_states / cell_states: list of D tensors each (B, C, H, W).
    Returns rollout data, updated hidden/cell states, and bootstrap obs.
    """
    B = len(actors)
    num_layers = len(hidden_states)
    obs_list, action_list, reward_list, done_list, log_prob_list = [], [], [], [], []

    for t in range(unroll_length):
        obs_batch = np.stack([a.get_obs() for a in actors], axis=0)
        obs_tensor = torch.from_numpy(obs_batch).float().to(device)

        with torch.no_grad():
            out = agent.forward(obs_tensor, hidden_states, cell_states)

        logits = out["policy_logits"]
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        hidden_states = out["hidden_states"]
        cell_states = out["cell_states"]

        rewards = []
        dones = []
        for i, actor in enumerate(actors):
            _, r, d = actor.step(actions[i].item())
            rewards.append(r)
            dones.append(d)

            if d:
                # Reset hidden/cell state for this actor to zeros
                for d_idx in range(num_layers):
                    hidden_states[d_idx][i] = torch.zeros_like(hidden_states[d_idx][i])
                    cell_states[d_idx][i] = torch.zeros_like(cell_states[d_idx][i])

        obs_list.append(obs_tensor)
        action_list.append(actions)
        reward_list.append(torch.tensor(rewards, dtype=torch.float32, device=device))
        done_list.append(torch.tensor(dones, dtype=torch.float32, device=device))
        log_prob_list.append(log_probs)

    bootstrap_obs = np.stack([a.get_obs() for a in actors], axis=0)
    bootstrap_obs_tensor = torch.from_numpy(bootstrap_obs).float().to(device)

    rollout = {
        "obs": torch.stack(obs_list, dim=0),
        "actions": torch.stack(action_list, dim=0),
        "rewards": torch.stack(reward_list, dim=0),
        "dones": torch.stack(done_list, dim=0),
        "behavior_log_probs": torch.stack(log_prob_list, dim=0),
        "bootstrap_obs": bootstrap_obs_tensor,
    }

    return rollout, hidden_states, cell_states, bootstrap_obs_tensor


def train(config: Config, checkpoint_dir: str = "checkpoints"):
    """
    Main training loop for the DRC agent using IMPALA.
    
    Training setup (from paper Appendix E.4):
      - 250M transitions on Boxoban unfiltered training set
      - IMPALA with V-trace
      - Adam optimizer, lr linear decay 4e-4 -> 0
      - Batch size 16, unroll length 20
      - Discount 0.97, V-trace lambda 0.97
      - Entropy penalty 1e-2
      - L2 penalty 1e-3 on logits
      - L2 reg 1e-5 on policy/value heads
    """
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    agent = DRCAgent(
        obs_channels=config.env.obs_channels,
        num_actions=config.env.num_actions,
        num_layers=config.drc.num_layers,
        num_ticks=config.drc.num_ticks,
        hidden_channels=config.drc.hidden_channels,
        encoder_channels=config.drc.encoder_channels,
        kernel_size=config.drc.kernel_size,
        padding=config.drc.padding,
        grid_size=config.env.grid_size,
    ).to(device)

    optimizer = optim.Adam(
        agent.parameters(),
        lr=config.train.learning_rate_start,
    )

    total_steps = config.train.total_transitions
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=total_steps // (config.train.batch_size * config.train.unroll_length),
    )

    level_sampler = create_level_sampler(
        config.data.boxoban_path,
        config.data.train_split,
        seed=config.train.seed,
    )

    B = config.train.batch_size
    actors = [
        ActorEnvironment(level_sampler, config.env, seed=config.train.seed + i)
        for i in range(B)
    ]

    for actor in actors:
        actor.reset()

    hidden_states, cell_states = agent.init_hidden(B, device)

    total_transitions = 0
    update_count = 0
    episode_rewards = deque(maxlen=100)
    episode_solved = deque(maxlen=100)
    start_time = time.time()

    print(f"Training DRC({config.drc.num_layers},{config.drc.num_ticks}) agent")
    print(f"Device: {device}")
    print(f"Total transitions: {total_steps:,}")

    while total_transitions < total_steps:
        rollout, hidden_states, cell_states, bootstrap_obs = collect_rollout(
            agent=agent,
            actors=actors,
            hidden_states=hidden_states,
            cell_states=cell_states,
            unroll_length=config.train.unroll_length,
            device=device,
            num_actions=config.env.num_actions,
        )

        T = config.train.unroll_length
        obs_flat = rollout["obs"].view(T * B, *rollout["obs"].shape[2:])

        h_init, c_init = agent.init_hidden(B, device)
        learner_out = agent.forward(obs_flat, h_init, c_init)
        policy_logits = learner_out["policy_logits"].view(T, B, config.env.num_actions)
        values = learner_out["value"].view(T, B)

        with torch.no_grad():
            bootstrap_out = agent.forward(bootstrap_obs, hidden_states, cell_states)
        bootstrap_value = bootstrap_out["value"]

        losses = impala_loss(
            policy_logits=policy_logits,
            values=values,
            actions=rollout["actions"],
            rewards=rollout["rewards"],
            bootstrap_value=bootstrap_value,
            behavior_log_probs=rollout["behavior_log_probs"],
            discount=config.train.discount,
            vtrace_lambda=config.train.vtrace_lambda,
            entropy_coef=config.train.entropy_coef,
            logit_l2_penalty=config.train.logit_l2_penalty,
        )

        # L2 regularization on policy and value heads
        head_l2 = config.train.head_l2_reg * (
            sum(p.pow(2).sum() for p in agent.policy_head.parameters()) +
            sum(p.pow(2).sum() for p in agent.value_head.parameters())
        )
        total_loss = losses["total_loss"] + head_l2

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_norm=40.0)
        optimizer.step()
        scheduler.step()

        total_transitions += T * B
        update_count += 1

        if update_count % (config.train.log_interval // (T * B) + 1) == 0:
            elapsed = time.time() - start_time
            fps = total_transitions / elapsed
            print(
                f"Steps: {total_transitions:,} | "
                f"Updates: {update_count} | "
                f"Loss: {losses['total_loss'].item():.4f} | "
                f"Entropy: {losses['entropy'].item():.4f} | "
                f"FPS: {fps:.0f}"
            )

        if total_transitions % config.train.checkpoint_interval < T * B:
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_{total_transitions // 1_000_000}M.pt"
            )
            torch.save({
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "total_transitions": total_transitions,
                "update_count": update_count,
                "config": config,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        "model_state_dict": agent.state_dict(),
        "total_transitions": total_transitions,
        "config": config,
    }, final_path)
    print(f"Training complete. Final model saved to {final_path}")

    return agent


def load_checkpoint(checkpoint_path: str, config: Config, device: str = "cuda") -> DRCAgent:
    """Load a DRC agent from a checkpoint."""
    agent = DRCAgent(
        obs_channels=config.env.obs_channels,
        num_actions=config.env.num_actions,
        num_layers=config.drc.num_layers,
        num_ticks=config.drc.num_ticks,
        hidden_channels=config.drc.hidden_channels,
        encoder_channels=config.drc.encoder_channels,
        kernel_size=config.drc.kernel_size,
        padding=config.drc.padding,
        grid_size=config.env.grid_size,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent = agent.to(device)
    agent.eval()
    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--boxoban_path", type=str, default="data/boxoban-levels")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config()
    cfg.data.boxoban_path = args.boxoban_path
    cfg.device = args.device
    cfg.train.seed = args.seed

    train(cfg, checkpoint_dir=args.checkpoint_dir)
