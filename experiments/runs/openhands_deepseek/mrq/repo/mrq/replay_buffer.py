"""Episode-based replay buffer with LAP prioritized sampling for MR.Q."""

import numpy as np
import torch
from collections import deque
from typing import Tuple, Optional, List


class EpisodeBuffer:
    """Stores a single episode's transitions."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.next_states: List[np.ndarray] = []

    def add(self, state, action, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def __len__(self) -> int:
        return len(self.states)


class ReplayBuffer:
    """Replay buffer with LAP prioritized experience replay and episode tracking."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
        alpha: float = 0.4,
        min_priority: float = 1.0,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.min_priority = min_priority
        self.device = device
        self.action_dim = action_dim

        self._ptr = 0
        self._size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)

        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self._max_priority = min_priority

        self._current_episode = EpisodeBuffer()
        self._episode_lengths: List[int] = []
        self._episode_start_indices: List[int] = []
        self._seen_terminal = False

    def push(self, state, action, reward, done, next_state):
        idx = self._ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.next_states[idx] = next_state
        self.priorities[idx] = self._max_priority

        self._current_episode.add(state, action, reward, done, next_state)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

        if done:
            self._seen_terminal = True
            ep_len = len(self._current_episode)
            start_idx = (idx - ep_len + 1) % self.capacity
            self._episode_lengths.append(ep_len)
            self._episode_start_indices.append(start_idx)
            self._current_episode = EpisodeBuffer()
            while len(self._episode_lengths) > 200:
                self._episode_lengths.pop(0)
                self._episode_start_indices.pop(0)

    @property
    def has_seen_terminal(self) -> bool:
        return self._seen_terminal

    def sample(self, batch_size: int) -> Tuple:
        if self._size < batch_size:
            raise ValueError("Not enough samples in replay buffer")
        probs = self.priorities[: self._size] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self._size, batch_size, p=probs, replace=False)
        states = torch.as_tensor(self.states[indices], device=self.device)
        actions = torch.as_tensor(self.actions[indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device)
        dones = torch.as_tensor(self.dones[indices], device=self.device)
        next_states = torch.as_tensor(self.next_states[indices], device=self.device)
        return states, actions, rewards, dones, next_states, indices

    def sample_subsequences(self, batch_size: int, seq_len: int) -> Optional[Tuple]:
        if not self._episode_lengths:
            return None
        valid_eps = [(i, l) for i, l in enumerate(self._episode_lengths) if l >= seq_len]
        if len(valid_eps) < 1:
            return None
        chosen = np.random.choice(len(valid_eps), batch_size, replace=True)
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_next_states = []
        for ep_idx in [valid_eps[c][0] for c in chosen]:
            start = self._episode_start_indices[ep_idx]
            ep_len = self._episode_lengths[ep_idx]
            t0 = np.random.randint(0, ep_len - seq_len + 1)
            seq_s = np.zeros((seq_len, *self.states.shape[1:]), dtype=np.float32)
            seq_a = np.zeros((seq_len, self.action_dim), dtype=np.float32)
            seq_r = np.zeros((seq_len, 1), dtype=np.float32)
            seq_d = np.zeros((seq_len, 1), dtype=np.float32)
            seq_ns = np.zeros((seq_len, *self.next_states.shape[1:]), dtype=np.float32)
            for k in range(seq_len):
                idx = (start + t0 + k) % self.capacity
                seq_s[k] = self.states[idx]
                seq_a[k] = self.actions[idx]
                seq_r[k] = self.rewards[idx]
                seq_d[k] = self.dones[idx]
                seq_ns[k] = self.next_states[idx]
            batch_states.append(seq_s)
            batch_actions.append(seq_a)
            batch_rewards.append(seq_r)
            batch_dones.append(seq_d)
            batch_next_states.append(seq_ns)
        return (
            torch.as_tensor(np.stack(batch_states), device=self.device),
            torch.as_tensor(np.stack(batch_actions), device=self.device),
            torch.as_tensor(np.stack(batch_rewards), device=self.device),
            torch.as_tensor(np.stack(batch_dones), device=self.device),
            torch.as_tensor(np.stack(batch_next_states), device=self.device),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        new_priorities = np.abs(td_errors) + self.min_priority
        self.priorities[indices] = new_priorities
        self._max_priority = max(self._max_priority, new_priorities.max())

    def __len__(self) -> int:
        return self._size


class ImageReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha, min_priority, device, image_shape):
        c, h, w = image_shape
        flat_dim = c * h * w
        super().__init__(flat_dim, 0, capacity, alpha, min_priority, device)
        self.image_shape = image_shape

    def push(self, state, action, reward, done, next_state):
        idx = self._ptr
        self.states[idx] = state.ravel()
        if np.isscalar(action):
            self.actions[idx] = np.array([action], dtype=np.float32)
        else:
            self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.next_states[idx] = next_state.ravel()
        self.priorities[idx] = self._max_priority
        self._current_episode.add(state, action, reward, done, next_state)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        if done:
            self._seen_terminal = True
            ep_len = len(self._current_episode)
            start_idx = (idx - ep_len + 1) % self.capacity
            self._episode_lengths.append(ep_len)
            self._episode_start_indices.append(start_idx)
            self._current_episode = EpisodeBuffer()
            while len(self._episode_lengths) > 200:
                self._episode_lengths.pop(0)
                self._episode_start_indices.pop(0)

    def sample(self, batch_size: int):
        states, actions, rewards, dones, next_states, indices = super().sample(batch_size)
        states = states.reshape(batch_size, *self.image_shape).float()
        next_states = next_states.reshape(batch_size, *self.image_shape).float()
        return states, actions, rewards, dones, next_states, indices

    def sample_subsequences(self, batch_size: int, seq_len: int):
        result = super().sample_subsequences(batch_size, seq_len)
        if result is None:
            return None
        states, actions, rewards, dones, next_states = result
        B, T = states.shape[:2]
        states = states.reshape(B, T, *self.image_shape).float()
        next_states = next_states.reshape(B, T, *self.image_shape).float()
        return states, actions, rewards, dones, next_states
