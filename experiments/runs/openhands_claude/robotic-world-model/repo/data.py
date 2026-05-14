"""
Data utilities for RWM:
  - Trajectory: container for a single episode
  - TrajectoryDataset: sliding-window dataset for world model training (Sec. 3.2)
  - ReplayBuffer: fixed-size FIFO buffer for MBPO-PPO (Algorithm 1)
  - ImaginaryRolloutBuffer: stores imagined trajectories for PPO updates
  - collate_fn / data loading helpers
"""

import os
import random
from collections import deque
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Trajectory Container
# ---------------------------------------------------------------------------

class Trajectory:
    """
    Stores a single episode of (obs, action, privileged, done) tuples.

    obs:        (T, obs_dim)
    actions:    (T, action_dim)
    privileged: (T, priv_dim)
    dones:      (T,)  — True when episode terminates (e.g., base contact)
    """

    def __init__(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        privileged: np.ndarray,
        dones: Optional[np.ndarray] = None,
    ):
        self.obs = obs.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.privileged = privileged.astype(np.float32)
        self.dones = dones.astype(np.float32) if dones is not None else np.zeros(len(obs), dtype=np.float32)
        assert len(obs) == len(actions) == len(privileged)

    def __len__(self) -> int:
        return len(self.obs)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            obs=self.obs,
            actions=self.actions,
            privileged=self.privileged,
            dones=self.dones,
        )

    @classmethod
    def load(cls, path: str) -> "Trajectory":
        data = np.load(path)
        return cls(
            obs=data["obs"],
            actions=data["actions"],
            privileged=data["privileged"],
            dones=data.get("dones", np.zeros(len(data["obs"]), dtype=np.float32)),
        )


# ---------------------------------------------------------------------------
# Sliding-Window Dataset for World Model Training
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """
    Sliding-window dataset over collected trajectories.

    Each sample is a window of size M+N:
      - obs:        (M+N, obs_dim)
      - actions:    (M+N, action_dim)
      - privileged: (M+N, priv_dim)

    The first M steps are the history context; the next N steps are targets.
    Windows that span episode boundaries (done=True) are excluded.
    """

    def __init__(
        self,
        trajectories: List[Trajectory],
        history_horizon: int,
        forecast_horizon: int,
    ):
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        window_size = history_horizon + forecast_horizon

        self.obs_windows: List[np.ndarray] = []
        self.action_windows: List[np.ndarray] = []
        self.priv_windows: List[np.ndarray] = []

        for traj in trajectories:
            T = len(traj)
            if T < window_size:
                continue
            for start in range(T - window_size + 1):
                end = start + window_size
                # Skip windows that cross a terminal state
                if traj.dones[start:end - 1].any():
                    continue
                self.obs_windows.append(traj.obs[start:end])
                self.action_windows.append(traj.actions[start:end])
                self.priv_windows.append(traj.privileged[start:end])

    def __len__(self) -> int:
        return len(self.obs_windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = torch.from_numpy(self.obs_windows[idx])
        actions = torch.from_numpy(self.action_windows[idx])
        privileged = torch.from_numpy(self.priv_windows[idx])
        return obs, actions, privileged

    @classmethod
    def from_directory(
        cls,
        data_dir: str,
        history_horizon: int,
        forecast_horizon: int,
    ) -> "TrajectoryDataset":
        trajectories = []
        data_path = Path(data_dir)
        for f in sorted(data_path.glob("*.npz")):
            trajectories.append(Trajectory.load(str(f)))
        return cls(trajectories, history_horizon, forecast_horizon)


def build_dataloader(
    dataset: TrajectoryDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Replay Buffer for MBPO-PPO
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-size FIFO replay buffer storing real environment transitions.

    Stores full trajectories (or partial episodes) as Trajectory objects.
    Supports sampling starting states for imagination rollouts.

    Buffer size |D| = 1000 episodes (Table S11).
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)

    def add(self, trajectory: Trajectory) -> None:
        self._buffer.append(trajectory)

    def __len__(self) -> int:
        return len(self._buffer)

    def sample_starting_states(
        self,
        n: int,
        history_horizon: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n starting (obs_history, action_history) pairs for imagination.

        Returns:
            obs_history:    (n, M, obs_dim)
            action_history: (n, M, action_dim)
        """
        obs_list, act_list = [], []
        trajectories = list(self._buffer)
        valid = [t for t in trajectories if len(t) >= history_horizon]
        if not valid:
            raise RuntimeError("No trajectories with sufficient length in replay buffer.")

        for _ in range(n):
            traj = random.choice(valid)
            T = len(traj)
            start = random.randint(0, T - history_horizon)
            end = start + history_horizon
            obs_list.append(traj.obs[start:end])
            act_list.append(traj.actions[start:end])

        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=device)
        act_tensor = torch.tensor(np.stack(act_list), dtype=torch.float32, device=device)
        return obs_tensor, act_tensor

    def build_dataset(
        self, history_horizon: int, forecast_horizon: int
    ) -> TrajectoryDataset:
        return TrajectoryDataset(
            list(self._buffer), history_horizon, forecast_horizon
        )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for i, traj in enumerate(self._buffer):
            traj.save(os.path.join(path, f"traj_{i:06d}"))

    @classmethod
    def load(cls, path: str, max_size: int = 1000) -> "ReplayBuffer":
        buf = cls(max_size)
        data_path = Path(path)
        for f in sorted(data_path.glob("*.npz")):
            buf.add(Trajectory.load(str(f)))
        return buf


# ---------------------------------------------------------------------------
# Imaginary Rollout Buffer for PPO
# ---------------------------------------------------------------------------

class ImaginaryRolloutBuffer:
    """
    Stores imagined trajectories for PPO policy updates.

    Collects (obs, action, log_prob, reward, value, done) tuples
    from imagination rollouts and computes GAE advantages.
    """

    def __init__(
        self,
        num_envs: int,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.num_envs = num_envs
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self) -> None:
        self.obs = torch.zeros(self.horizon, self.num_envs, self.obs_dim, device=self.device)
        self.actions = torch.zeros(self.horizon, self.num_envs, self.action_dim, device=self.device)
        self.log_probs = torch.zeros(self.horizon, self.num_envs, device=self.device)
        self.rewards = torch.zeros(self.horizon, self.num_envs, device=self.device)
        self.values = torch.zeros(self.horizon, self.num_envs, device=self.device)
        self.dones = torch.zeros(self.horizon, self.num_envs, device=self.device)
        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.dones[self.step] = done
        self.step += 1

    def compute_returns_and_advantages(
        self, last_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns."""
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return returns, advantages

    def get_mini_batches(
        self,
        last_value: torch.Tensor,
        num_mini_batches: int,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield mini-batches for PPO updates."""
        returns, advantages = self.compute_returns_and_advantages(last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten (horizon, num_envs) → (horizon * num_envs,)
        obs_flat = self.obs.view(-1, self.obs_dim)
        actions_flat = self.actions.view(-1, self.action_dim)
        log_probs_flat = self.log_probs.view(-1)
        returns_flat = returns.view(-1)
        advantages_flat = advantages.view(-1)
        values_flat = self.values.view(-1)

        total = obs_flat.size(0)
        indices = torch.randperm(total, device=self.device)
        batch_size = total // num_mini_batches

        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": obs_flat[idx],
                "actions": actions_flat[idx],
                "log_probs_old": log_probs_flat[idx],
                "returns": returns_flat[idx],
                "advantages": advantages_flat[idx],
                "values_old": values_flat[idx],
            }


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class RunningNormalizer:
    """
    Online running mean/std normalizer using Welford's algorithm.
    Used to normalize observations for stable training.
    """

    def __init__(self, dim: int, epsilon: float = 1e-8):
        self.dim = dim
        self.epsilon = epsilon
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        if x.ndim == 1:
            x = x[np.newaxis]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)

    def normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.tensor(np.sqrt(self.var) + self.epsilon, dtype=x.dtype, device=x.device)
        return (x - mean) / std

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, var=self.var, count=np.array([self.count]))

    @classmethod
    def load(cls, path: str, dim: int) -> "RunningNormalizer":
        data = np.load(path)
        norm = cls(dim)
        norm.mean = data["mean"]
        norm.var = data["var"]
        norm.count = int(data["count"][0])
        return norm


# ---------------------------------------------------------------------------
# Synthetic Data Generation (for testing without a simulator)
# ---------------------------------------------------------------------------

def generate_synthetic_trajectory(
    obs_dim: int,
    action_dim: int,
    privileged_dim: int,
    length: int = 1000,
    seed: Optional[int] = None,
) -> Trajectory:
    """
    Generate a synthetic trajectory with simple linear dynamics + noise.
    Useful for unit testing without a real simulator.
    """
    rng = np.random.default_rng(seed)
    obs = np.zeros((length, obs_dim), dtype=np.float32)
    actions = rng.uniform(-1, 1, (length, action_dim)).astype(np.float32)
    privileged = rng.uniform(0, 1, (length, privileged_dim)).astype(np.float32)
    dones = np.zeros(length, dtype=np.float32)

    obs[0] = rng.normal(0, 0.1, obs_dim).astype(np.float32)
    for t in range(1, length):
        action_contrib = np.zeros(obs_dim, dtype=np.float32)
        action_contrib[:action_dim] = actions[t - 1]
        obs[t] = (0.95 * obs[t - 1] + 0.05 * action_contrib
                  + rng.normal(0, 0.01, obs_dim).astype(np.float32))

    return Trajectory(obs, actions, privileged, dones)


def generate_synthetic_dataset(
    obs_dim: int,
    action_dim: int,
    privileged_dim: int,
    num_trajectories: int = 10,
    trajectory_length: int = 1000,
    save_dir: Optional[str] = None,
) -> List[Trajectory]:
    trajectories = [
        generate_synthetic_trajectory(obs_dim, action_dim, privileged_dim, trajectory_length, seed=i)
        for i in range(num_trajectories)
    ]
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for i, traj in enumerate(trajectories):
            traj.save(os.path.join(save_dir, f"traj_{i:06d}"))
    return trajectories
