import numpy as np
import torch
from typing import Dict, Optional, Tuple


class ReplayBuffer:
    """Fixed-size replay buffer storing transitions (s, a, s', r, done)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = 1_000_000,
        device: str = "cuda",
    ):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        self.relevance = np.zeros((max_size, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: float,
        relevance: float = 0.0,
    ):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.relevance[self.ptr] = relevance
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        relevances: Optional[np.ndarray] = None,
    ):
        n = len(states)
        if relevances is None:
            relevances = np.zeros((n, 1), dtype=np.float32)
        if relevances.ndim == 1:
            relevances = relevances[:, None]
        if rewards.ndim == 1:
            rewards = rewards[:, None]
        if dones.ndim == 1:
            dones = dones[:, None]

        indices = np.arange(self.ptr, self.ptr + n) % self.max_size
        self.states[indices] = states
        self.actions[indices] = actions
        self.next_states[indices] = next_states
        self.rewards[indices] = rewards
        self.dones[indices] = dones
        self.relevance[indices] = relevances
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return self._get_batch(idx)

    def _get_batch(self, idx: np.ndarray) -> Dict[str, torch.Tensor]:
        return {
            "states": torch.FloatTensor(self.states[idx]).to(self.device),
            "actions": torch.FloatTensor(self.actions[idx]).to(self.device),
            "next_states": torch.FloatTensor(self.next_states[idx]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[idx]).to(self.device),
            "dones": torch.FloatTensor(self.dones[idx]).to(self.device),
            "relevance": torch.FloatTensor(self.relevance[idx]).to(self.device),
        }

    def get_top_k_relevance(self, k: int) -> Dict[str, np.ndarray]:
        """Return the top-k transitions by relevance value (for prompting)."""
        valid_relevance = self.relevance[: self.size, 0]
        top_k_idx = np.argpartition(valid_relevance, -k)[-k:]
        return {
            "states": self.states[top_k_idx],
            "actions": self.actions[top_k_idx],
            "next_states": self.next_states[top_k_idx],
            "rewards": self.rewards[top_k_idx],
            "dones": self.dones[top_k_idx],
            "relevance": self.relevance[top_k_idx],
        }

    def get_all_as_tensor(self) -> Dict[str, torch.Tensor]:
        idx = np.arange(self.size)
        return self._get_batch(idx)

    def update_relevance(self, idx: np.ndarray, relevance: np.ndarray):
        if relevance.ndim == 1:
            relevance = relevance[:, None]
        self.relevance[idx] = relevance

    def update_all_relevance(self, relevance_fn, batch_size: int = 1024):
        """Recompute relevance for all stored transitions in batches."""
        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)
            idx = np.arange(start, end)
            batch = {
                "states": torch.FloatTensor(self.states[idx]).to(self.device),
                "actions": torch.FloatTensor(self.actions[idx]).to(self.device),
                "next_states": torch.FloatTensor(self.next_states[idx]).to(self.device),
                "rewards": torch.FloatTensor(self.rewards[idx]).to(self.device),
            }
            with torch.no_grad():
                rel = relevance_fn(batch).cpu().numpy()
            self.update_relevance(idx, rel)

    def as_transitions_array(self) -> np.ndarray:
        """Return all transitions as a flat array (s, a, s', r) for diffusion training."""
        s = self.states[: self.size]
        a = self.actions[: self.size]
        sp = self.next_states[: self.size]
        r = self.rewards[: self.size]
        return np.concatenate([s, a, sp, r], axis=-1)

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer using sum-tree for O(log n) sampling."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = 1_000_000,
        device: str = "cuda",
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-6,
        epsilon: float = 1e-6,
    ):
        super().__init__(state_dim, action_dim, max_size, device)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, state, action, next_state, reward, done, relevance=0.0):
        self.priorities[self.ptr] = self.max_priority
        super().add(state, action, next_state, reward, done, relevance)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        valid_priorities = self.priorities[: self.size]
        probs = valid_priorities ** self.alpha
        probs /= probs.sum()
        idx = np.random.choice(self.size, size=batch_size, replace=False, p=probs)
        weights = (self.size * probs[idx]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        batch = self._get_batch(idx)
        batch["weights"] = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        batch["indices"] = idx
        return batch

    def update_priorities(self, idx: np.ndarray, priorities: np.ndarray):
        priorities = np.abs(priorities) + self.epsilon
        self.priorities[idx] = priorities
        self.max_priority = max(self.max_priority, priorities.max())


class MixedReplayBuffer:
    """Samples from real and synthetic buffers at a fixed ratio r.

    Per Algorithm 1: train π on samples from D_real ∪ D_syn mixed with ratio r.
    """

    def __init__(
        self,
        real_buffer: ReplayBuffer,
        syn_buffer: ReplayBuffer,
        synthetic_ratio: float = 0.5,
    ):
        self.real_buffer = real_buffer
        self.syn_buffer = syn_buffer
        self.synthetic_ratio = synthetic_ratio

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        n_syn = int(batch_size * self.synthetic_ratio)
        n_real = batch_size - n_syn

        real_batch = self.real_buffer.sample(n_real)

        if self.syn_buffer.size > 0 and n_syn > 0:
            syn_batch = self.syn_buffer.sample(min(n_syn, self.syn_buffer.size))
            return {
                k: torch.cat([real_batch[k], syn_batch[k]], dim=0)
                for k in real_batch
                if k in syn_batch
            }
        return real_batch

    def __len__(self) -> int:
        return len(self.real_buffer) + len(self.syn_buffer)


class PixelReplayBuffer:
    """Replay buffer for pixel-based observations.

    Stores raw pixel observations and generates in the latent space of the CNN encoder,
    following Lu et al. (2024) and Esser et al. (2021).
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        max_size: int = 1_000_000,
        device: str = "cuda",
    ):
        self.max_size = max_size
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.next_observations = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
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
        self.observations[self.ptr] = obs
        self.next_observations[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.relevance[self.ptr] = relevance
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.FloatTensor(self.observations[idx]).to(self.device) / 255.0
        next_obs = torch.FloatTensor(self.next_observations[idx]).to(self.device) / 255.0
        return {
            "observations": obs,
            "next_observations": next_obs,
            "actions": torch.FloatTensor(self.actions[idx]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[idx]).to(self.device),
            "dones": torch.FloatTensor(self.dones[idx]).to(self.device),
            "relevance": torch.FloatTensor(self.relevance[idx]).to(self.device),
        }

    def get_top_k_relevance(self, k: int) -> Dict[str, np.ndarray]:
        valid_relevance = self.relevance[: self.size, 0]
        top_k_idx = np.argpartition(valid_relevance, -k)[-k:]
        return {
            "observations": self.observations[top_k_idx],
            "next_observations": self.next_observations[top_k_idx],
            "actions": self.actions[top_k_idx],
            "rewards": self.rewards[top_k_idx],
            "dones": self.dones[top_k_idx],
            "relevance": self.relevance[top_k_idx],
        }

    def __len__(self) -> int:
        return self.size


class LatentReplayBuffer(ReplayBuffer):
    """Replay buffer storing latent-space transitions for pixel-based tasks.

    Stores (f(s), a, f(s'), r) where f is the CNN visual encoder.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        max_size: int = 1_000_000,
        device: str = "cuda",
    ):
        super().__init__(latent_dim, action_dim, max_size, device)
