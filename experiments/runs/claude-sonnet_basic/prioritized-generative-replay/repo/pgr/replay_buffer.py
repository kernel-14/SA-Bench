"""
Replay buffers for Prioritized Generative Replay.

Two buffers:
- RealReplayBuffer: stores actual environment transitions
- SyntheticReplayBuffer: stores generated transitions from the diffusion model

Both support efficient sampling and normalization for diffusion model training.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple


class ReplayBuffer:
    """
    Standard circular replay buffer for storing environment transitions.
    Supports both real and synthetic transitions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_size: int = 1_000_000,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        # Relevance values (computed by F)
        self.relevance = np.zeros((max_size, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        done: float,
        relevance: float = 0.0,
    ):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.relevance[self.ptr] = relevance

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        relevance: Optional[np.ndarray] = None,
    ):
        """Add a batch of transitions."""
        B = obs.shape[0]
        if relevance is None:
            relevance = np.zeros((B, 1), dtype=np.float32)

        indices = np.arange(self.ptr, self.ptr + B) % self.max_size
        self.obs[indices] = obs
        self.action[indices] = action
        self.next_obs[indices] = next_obs
        self.reward[indices] = reward.reshape(-1, 1) if reward.ndim == 1 else reward
        self.done[indices] = done.reshape(-1, 1) if done.ndim == 1 else done
        self.relevance[indices] = relevance.reshape(-1, 1) if relevance.ndim == 1 else relevance

        self.ptr = (self.ptr + B) % self.max_size
        self.size = min(self.size + B, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Uniformly sample a batch of transitions."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return self._get_batch(idx)

    def sample_top_k_relevance(self, k: int) -> Dict[str, torch.Tensor]:
        """
        Sample from the top-k highest relevance transitions.
        Used for the 'prompting' strategy in PGR (Section 4.3).
        """
        top_k_idx = np.argsort(self.relevance[:self.size, 0])[-k:]
        # Randomly sample from top-k
        idx = np.random.choice(top_k_idx, size=k, replace=False)
        return self._get_batch(idx)

    def get_top_k_relevance_values(self, k: int) -> np.ndarray:
        """Get the relevance values of the top-k transitions."""
        top_k_idx = np.argsort(self.relevance[:self.size, 0])[-k:]
        return self.relevance[top_k_idx]

    def _get_batch(self, idx: np.ndarray) -> Dict[str, torch.Tensor]:
        return {
            "obs": torch.FloatTensor(self.obs[idx]).to(self.device),
            "action": torch.FloatTensor(self.action[idx]).to(self.device),
            "next_obs": torch.FloatTensor(self.next_obs[idx]).to(self.device),
            "reward": torch.FloatTensor(self.reward[idx]).to(self.device),
            "done": torch.FloatTensor(self.done[idx]).to(self.device),
            "relevance": torch.FloatTensor(self.relevance[idx]).to(self.device),
        }

    def get_all_as_tensor(self) -> Dict[str, torch.Tensor]:
        """Get all stored transitions as tensors (for diffusion training)."""
        return self._get_batch(np.arange(self.size))

    def update_relevance(self, idx: np.ndarray, relevance: np.ndarray):
        """Update relevance values for specific indices."""
        self.relevance[idx] = relevance.reshape(-1, 1)

    def get_transition_dim(self) -> int:
        """Total dimension of a flattened transition (s, a, s', r)."""
        return self.obs_dim + self.action_dim + self.obs_dim + 1

    def to_flat_transitions(self, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return transitions as flat arrays [s, a, s', r] for diffusion training.
        Excludes 'done' flag as in SynthER.
        """
        if idx is None:
            idx = np.arange(self.size)
        return np.concatenate([
            self.obs[idx],
            self.action[idx],
            self.next_obs[idx],
            self.reward[idx],
        ], axis=-1)

    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Compute mean and std for normalizing transitions."""
        flat = self.to_flat_transitions()
        return {
            "mean": flat.mean(axis=0),
            "std": flat.std(axis=0) + 1e-8,
        }

    def __len__(self):
        return self.size


class NormalizedReplayBuffer(ReplayBuffer):
    """
    Replay buffer with online normalization for diffusion model training.
    Transitions are normalized to zero mean and unit variance.
    """

    def __init__(self, obs_dim, action_dim, max_size=1_000_000, device="cpu"):
        super().__init__(obs_dim, action_dim, max_size, device)
        self._mean = None
        self._std = None

    def update_normalization(self):
        """Recompute normalization statistics from current buffer."""
        stats = self.get_normalization_stats()
        self._mean = stats["mean"]
        self._std = stats["std"]

    def normalize(self, flat_transitions: np.ndarray) -> np.ndarray:
        if self._mean is None:
            self.update_normalization()
        return (flat_transitions - self._mean) / self._std

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("Normalization stats not computed yet.")
        return normalized * self._std + self._mean

    def denormalize_tensor(self, normalized: torch.Tensor) -> torch.Tensor:
        if self._mean is None:
            raise RuntimeError("Normalization stats not computed yet.")
        mean = torch.FloatTensor(self._mean).to(normalized.device)
        std = torch.FloatTensor(self._std).to(normalized.device)
        return normalized * std + mean

    def sample_normalized(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample normalized transitions and their relevance values.
        Returns: (normalized_transitions [B, transition_dim], relevance [B, 1])
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        flat = self.to_flat_transitions(idx)
        normalized = self.normalize(flat)
        relevance = self.relevance[idx]
        return (
            torch.FloatTensor(normalized).to(self.device),
            torch.FloatTensor(relevance).to(self.device),
        )

    def sample_top_k_normalized(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample top-k relevance transitions (normalized).
        Used for the prompting strategy in PGR.
        """
        top_k_idx = np.argsort(self.relevance[:self.size, 0])[-k:]
        flat = self.to_flat_transitions(top_k_idx)
        normalized = self.normalize(flat)
        relevance = self.relevance[top_k_idx]
        return (
            torch.FloatTensor(normalized).to(self.device),
            torch.FloatTensor(relevance).to(self.device),
        )
