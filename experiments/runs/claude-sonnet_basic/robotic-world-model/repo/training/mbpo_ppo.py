"""
MBPO-PPO: Model-Based Policy Optimization with PPO.

Algorithm 1 from the paper:
  1. Initialize policy, world model, and replay buffer D
  2. For each iteration:
     a. Collect observation-action pairs in D using current policy
     b. Update world model with autoregressive training
     c. Initialize imagination agents from D
     d. Roll out imagination trajectories using policy and world model for T steps
     e. Update policy using PPO

Training parameters (Table S11):
  - imagination environments: 4096
  - imagination steps per iteration: 100
  - buffer size: 1000
  - max iterations: 2500
  - learning rate: 0.001
  - weight decay: 0.0
  - learning epochs: 5
  - mini-batches: 4
  - KL divergence target: 0.01
  - discount factor: 0.99
  - clip range: 0.2
  - entropy coefficient: 0.005
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Callable
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Replay buffer storing real environment interaction data.

    Stores observation-action pairs for world model training and
    imagination initialization.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.observations = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.privileged_info = deque(maxlen=max_size)
        self.dones = deque(maxlen=max_size)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        priv: Optional[np.ndarray] = None,
        done: bool = False,
    ):
        """Add a single transition."""
        self.observations.append(obs)
        self.actions.append(action)
        self.privileged_info.append(priv if priv is not None else np.zeros(1))
        self.dones.append(done)

    def add_trajectory(
        self,
        obs_traj: np.ndarray,
        action_traj: np.ndarray,
        priv_traj: Optional[np.ndarray] = None,
        done_traj: Optional[np.ndarray] = None,
    ):
        """Add a full trajectory."""
        T = len(obs_traj)
        for t in range(T):
            priv_t = priv_traj[t] if priv_traj is not None else None
            done_t = done_traj[t] if done_traj is not None else False
            self.add(obs_traj[t], action_traj[t], priv_t, done_t)

    def sample_windows(
        self,
        batch_size: int,
        history_horizon: int,
        forecast_horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample sliding windows of size M + N from the buffer.

        Returns:
            obs_history: (batch, M, obs_size)
            action_history: (batch, M+N, action_size)
            obs_targets: (batch, N, obs_size)
            priv_targets: (batch, N, priv_size) or None
        """
        window_size = history_horizon + forecast_horizon
        n = len(self.observations)

        if n < window_size + 1:
            raise ValueError(f"Buffer too small: {n} < {window_size + 1}")

        obs_arr = np.array(list(self.observations))
        act_arr = np.array(list(self.actions))
        priv_arr = np.array(list(self.privileged_info))

        # Sample random starting indices
        max_start = n - window_size
        indices = np.random.randint(0, max_start, size=batch_size)

        obs_history = []
        action_history = []
        obs_targets = []
        priv_targets = []

        for idx in indices:
            obs_history.append(obs_arr[idx: idx + history_horizon])
            action_history.append(act_arr[idx: idx + history_horizon + forecast_horizon])
            obs_targets.append(obs_arr[idx + history_horizon: idx + window_size])
            priv_targets.append(priv_arr[idx + history_horizon: idx + window_size])

        obs_history = torch.tensor(np.array(obs_history), dtype=torch.float32)
        action_history = torch.tensor(np.array(action_history), dtype=torch.float32)
        obs_targets = torch.tensor(np.array(obs_targets), dtype=torch.float32)
        priv_targets_tensor = torch.tensor(np.array(priv_targets), dtype=torch.float32)

        return obs_history, action_history, obs_targets, priv_targets_tensor

    def sample_initial_states(
        self,
        n_envs: int,
        history_horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample initial states for imagination rollouts.

        Returns:
            obs_history: (n_envs, M, obs_size)
            action_history: (n_envs, M, action_size)
        """
        n = len(self.observations)
        obs_arr = np.array(list(self.observations))
        act_arr = np.array(list(self.actions))

        indices = np.random.randint(0, max(1, n - history_horizon), size=n_envs)

        obs_history = []
        action_history = []

        for idx in indices:
            end = min(idx + history_horizon, n)
            obs_seq = obs_arr[idx:end]
            act_seq = act_arr[idx:end]

            # Pad if necessary
            if len(obs_seq) < history_horizon:
                pad_len = history_horizon - len(obs_seq)
                obs_seq = np.concatenate([
                    np.zeros((pad_len, obs_seq.shape[-1])), obs_seq
                ], axis=0)
                act_seq = np.concatenate([
                    np.zeros((pad_len, act_seq.shape[-1])), act_seq
                ], axis=0)

            obs_history.append(obs_seq)
            action_history.append(act_seq)

        obs_history = torch.tensor(np.array(obs_history), dtype=torch.float32)
        action_history = torch.tensor(np.array(action_history), dtype=torch.float32)

        return obs_history, action_history

    def __len__(self):
        return len(self.observations)


class PPOBuffer:
    """Buffer for storing PPO rollout data from imagination."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get(self) -> Dict[str, torch.Tensor]:
        return {
            "observations": torch.stack(self.observations, dim=0),  # (T, N, obs_size)
            "actions": torch.stack(self.actions, dim=0),
            "log_probs": torch.stack(self.log_probs, dim=0),
            "rewards": torch.stack(self.rewards, dim=0),
            "values": torch.stack(self.values, dim=0),
            "dones": torch.stack(self.dones, dim=0),
        }

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: (T, N)
        values: (T, N)
        dones: (T, N)
        last_value: (N,)
        gamma: discount factor
        lam: GAE lambda

    Returns:
        advantages: (T, N)
        returns: (T, N)
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
            next_done = torch.zeros(N, device=rewards.device)
        else:
            next_value = values[t + 1]
            next_done = dones[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
        last_gae = delta + gamma * lam * (1 - next_done) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class PPOTrainer:
    """
    PPO trainer for policy optimization on imagination rollouts.

    Parameters from Table S11:
      - learning rate: 0.001
      - learning epochs: 5
      - mini-batches: 4
      - KL divergence target: 0.01
      - discount factor: 0.99
      - clip range: 0.2
      - entropy coefficient: 0.005
    """

    def __init__(
        self,
        policy: nn.Module,
        value_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_range: float = 0.2,
        entropy_coef: float = 0.005,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        learning_epochs: int = 5,
        num_mini_batches: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95,
        kl_target: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ):
        self.policy = policy
        self.value_fn = value_fn
        self.optimizer = optimizer
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.learning_epochs = learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.lam = lam
        self.kl_target = kl_target
        self.device = device

    def update(
        self,
        rollout_data: Dict[str, torch.Tensor],
        last_value: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update policy and value function using PPO.

        Args:
            rollout_data: dict with keys observations, actions, log_probs, rewards, values, dones
            last_value: (N,) - value estimate for last state

        Returns:
            metrics: dict of training metrics
        """
        obs = rollout_data["observations"]       # (T, N, obs_size)
        actions = rollout_data["actions"]         # (T, N, action_size)
        old_log_probs = rollout_data["log_probs"] # (T, N)
        rewards = rollout_data["rewards"]         # (T, N)
        values = rollout_data["values"]           # (T, N)
        dones = rollout_data["dones"]             # (T, N)

        T, N = rewards.shape

        # Compute advantages and returns
        with torch.no_grad():
            advantages, returns = compute_gae(
                rewards, values, dones, last_value, self.gamma, self.lam
            )
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten for mini-batch training
        obs_flat = obs.reshape(T * N, -1)
        actions_flat = actions.reshape(T * N, -1)
        old_log_probs_flat = old_log_probs.reshape(T * N)
        advantages_flat = advantages.reshape(T * N)
        returns_flat = returns.reshape(T * N)

        total_samples = T * N
        mini_batch_size = total_samples // self.num_mini_batches

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl_divergence": 0.0,
            "clip_fraction": 0.0,
        }
        num_updates = 0

        for epoch in range(self.learning_epochs):
            # Shuffle indices
            indices = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]

                mb_obs = obs_flat[mb_indices]
                mb_actions = actions_flat[mb_indices]
                mb_old_log_probs = old_log_probs_flat[mb_indices]
                mb_advantages = advantages_flat[mb_indices]
                mb_returns = returns_flat[mb_indices]

                # Evaluate actions under current policy
                log_probs, entropy, _ = self.policy.evaluate_actions(mb_obs, mb_actions)
                values_pred = self.value_fn(mb_obs).squeeze(-1)

                # PPO clipped objective
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                value_loss = F.mse_loss(values_pred, mb_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_fn.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    kl = (mb_old_log_probs - log_probs).mean()
                    clip_frac = ((ratio - 1).abs() > self.clip_range).float().mean()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.mean().item()
                metrics["kl_divergence"] += kl.item()
                metrics["clip_fraction"] += clip_frac.item()
                num_updates += 1

            # Early stopping based on KL divergence
            if metrics["kl_divergence"] / max(num_updates, 1) > self.kl_target * 1.5:
                break

        for k in metrics:
            metrics[k] /= max(num_updates, 1)

        return metrics


class MBPOPPOTrainer:
    """
    Full MBPO-PPO training loop (Algorithm 1).

    Combines:
      1. Real environment data collection
      2. World model training (autoregressive)
      3. Imagination rollouts
      4. PPO policy updates

    Args:
        world_model: RoboticWorldModel
        policy: PolicyNetwork
        value_fn: ValueNetwork
        reward_fn: callable(obs, priv_info) -> reward
        termination_fn: callable(obs, priv_info) -> done
        wm_trainer: WorldModelTrainer
        ppo_trainer: PPOTrainer
        replay_buffer: ReplayBuffer
        history_horizon: M
        imagination_steps: T - steps per imagination rollout
        n_imagination_envs: number of parallel imagination environments
        device: torch device
    """

    def __init__(
        self,
        world_model: nn.Module,
        policy: nn.Module,
        value_fn: nn.Module,
        reward_fn: Callable,
        termination_fn: Callable,
        wm_trainer,
        ppo_trainer: PPOTrainer,
        replay_buffer: ReplayBuffer,
        history_horizon: int = 32,
        imagination_steps: int = 100,
        n_imagination_envs: int = 4096,
        wm_train_steps: int = 100,
        wm_batch_size: int = 1024,
        device: torch.device = torch.device("cpu"),
    ):
        self.world_model = world_model
        self.policy = policy
        self.value_fn = value_fn
        self.reward_fn = reward_fn
        self.termination_fn = termination_fn
        self.wm_trainer = wm_trainer
        self.ppo_trainer = ppo_trainer
        self.replay_buffer = replay_buffer
        self.history_horizon = history_horizon
        self.imagination_steps = imagination_steps
        self.n_imagination_envs = n_imagination_envs
        self.wm_train_steps = wm_train_steps
        self.wm_batch_size = wm_batch_size
        self.device = device

    def collect_real_data(self, env, n_steps: int = 1000):
        """
        Collect data from real environment using current policy.

        Args:
            env: environment with step() and reset() methods
            n_steps: number of steps to collect
        """
        obs = env.reset()
        obs_history = [obs]
        act_history = []

        for _ in range(n_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, _ = self.policy.get_action(obs_tensor, deterministic=False)
            action_np = action.cpu().numpy()[0]

            next_obs, reward, done, info = env.step(action_np)
            priv = info.get("privileged_info", None)

            self.replay_buffer.add(obs, action_np, priv, done)

            obs = next_obs
            if done:
                obs = env.reset()

    def train_world_model(self):
        """Update world model using data from replay buffer."""
        if len(self.replay_buffer) < self.history_horizon + self.wm_trainer.forecast_horizon + 1:
            return {}

        metrics_list = []
        for _ in range(self.wm_train_steps):
            obs_history, action_history, obs_targets, priv_targets = (
                self.replay_buffer.sample_windows(
                    self.wm_batch_size,
                    self.history_horizon,
                    self.wm_trainer.forecast_horizon,
                )
            )
            obs_history = obs_history.to(self.device)
            action_history = action_history.to(self.device)
            obs_targets = obs_targets.to(self.device)
            priv_targets = priv_targets.to(self.device)

            metrics = self.wm_trainer.train_step(
                obs_history, action_history, obs_targets, priv_targets
            )
            metrics_list.append(metrics)

        avg_metrics = {}
        for k in metrics_list[0]:
            avg_metrics[k] = np.mean([m[k] for m in metrics_list])

        return avg_metrics

    @torch.no_grad()
    def rollout_imagination(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Roll out imagination trajectories using world model and policy.

        Returns:
            rollout_data: dict for PPO update
            last_value: (N,) value estimate for last state
        """
        self.world_model.eval()
        self.policy.eval()
        self.value_fn.eval()

        N = self.n_imagination_envs
        T = self.imagination_steps
        M = self.history_horizon

        # Initialize from replay buffer
        obs_history, action_history = self.replay_buffer.sample_initial_states(N, M)
        obs_history = obs_history.to(self.device)
        action_history = action_history.to(self.device)

        # Get initial hidden state from history (inner autoregression)
        hidden = self.world_model.get_hidden_from_history(obs_history, action_history)

        # Start from last observed observation
        current_obs = obs_history[:, -1, :]  # (N, obs_size)

        ppo_buffer = PPOBuffer()
        dones = torch.zeros(N, device=self.device)

        for t in range(T):
            # Policy action
            action, log_prob = self.policy.get_action(current_obs, deterministic=False)
            value = self.value_fn(current_obs).squeeze(-1)

            # World model step (outer autoregression)
            obs_mean, obs_std, priv_mean, priv_std, hidden = self.world_model.predict_step(
                current_obs, action, hidden
            )

            # Sample next observation
            next_obs = self.world_model.sample_obs(obs_mean, obs_std)

            # Compute reward from imagined observations
            priv_info = priv_mean if priv_mean is not None else torch.zeros(N, 1, device=self.device)
            reward = self.reward_fn(current_obs, action, next_obs, priv_info)

            # Check termination
            done = self.termination_fn(next_obs, priv_info)

            ppo_buffer.add(current_obs, action, log_prob, reward, value, dones)

            # Reset terminated environments (keep hidden state but reset obs)
            dones = done.float()
            current_obs = next_obs

        # Last value for GAE
        last_value = self.value_fn(current_obs).squeeze(-1)

        rollout_data = ppo_buffer.get()
        return rollout_data, last_value

    def train_iteration(self, env=None, collect_steps: int = 1000) -> Dict[str, float]:
        """
        Single training iteration of MBPO-PPO.

        Steps:
          1. Collect real data (if env provided)
          2. Train world model
          3. Roll out imagination
          4. Update policy with PPO

        Returns:
            metrics: combined training metrics
        """
        all_metrics = {}

        # Step 1: Collect real data
        if env is not None:
            self.collect_real_data(env, collect_steps)

        # Step 2: Train world model
        wm_metrics = self.train_world_model()
        all_metrics.update({f"wm/{k}": v for k, v in wm_metrics.items()})

        # Step 3: Roll out imagination
        if len(self.replay_buffer) >= self.history_horizon + 1:
            rollout_data, last_value = self.rollout_imagination()

            # Step 4: Update policy with PPO
            self.policy.train()
            self.value_fn.train()
            ppo_metrics = self.ppo_trainer.update(rollout_data, last_value)
            all_metrics.update({f"ppo/{k}": v for k, v in ppo_metrics.items()})

        return all_metrics
