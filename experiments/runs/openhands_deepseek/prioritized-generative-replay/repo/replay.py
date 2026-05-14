"""Replay buffers for real and synthetic transitions."""

import random
from collections import deque
from typing import Optional, Tuple
import torch
import numpy as np


class Transition:
    """A single environment transition."""

    __slots__ = ("state", "action", "reward", "next_state", "done")

    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class ReplayBuffer:
    """Standard FIFO replay buffer for real transitions (D_real)."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, pixel_based: bool = False):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pixel_based = pixel_based

        # Use deque for efficient FIFO
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: str = "cpu") -> Tuple[torch.Tensor, ...]:
        """Randomly sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)

        states = torch.FloatTensor(np.stack([t.state for t in batch])).to(device)
        actions = torch.FloatTensor(np.stack([t.action for t in batch])).to(device)
        rewards = torch.FloatTensor(np.array([t.reward for t in batch])).to(device).unsqueeze(-1)
        next_states = torch.FloatTensor(np.stack([t.next_state for t in batch])).to(device)
        dones = torch.FloatTensor(np.array([t.done for t in batch])).to(device).unsqueeze(-1)

        return states, actions, rewards, next_states, dones

    def sample_all(self, device: str = "cpu") -> Tuple[torch.Tensor, ...]:
        """Return all transitions in the buffer (for training generative model)."""
        return self.sample(min(len(self.buffer), self.capacity), device)

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_empty(self) -> bool:
        return len(self.buffer) == 0


class SyntheticBuffer:
    """Synthetic replay buffer for generated transitions (D_syn).

    Stores generated transitions as raw tensors for efficient sampling.
    Matches SYNTHER's buffer design.
    """

    def __init__(self, capacity: int, transition_dim: int):
        self.capacity = capacity
        self.transition_dim = transition_dim

        self.buffer = torch.zeros(capacity, transition_dim)
        self.size = 0
        self.ptr = 0

    def add(self, transitions: torch.Tensor):
        """Add generated transitions (FIFO)."""
        n = transitions.shape[0]
        if n > self.capacity:
            transitions = transitions[-self.capacity:]
            n = self.capacity

        end_idx = min(self.ptr + n, self.capacity)
        self.buffer[self.ptr:end_idx] = transitions[:end_idx - self.ptr].detach().cpu()

        remainder = n - (end_idx - self.ptr)
        if remainder > 0:
            self.buffer[:remainder] = transitions[end_idx - self.ptr:].detach().cpu()

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> Tuple[torch.Tensor, ...]:
        """Sample random transitions and split into components."""
        indices = torch.randint(0, self.size, (batch_size,))
        transitions = self.buffer[indices].to(device)

        # Split into state, action, reward, next_state
        # Transition format: (s, a, s', r)
        # We need to know the split points. Store split info.
        state_dim = self.state_dim
        action_dim = self.action_dim

        states = transitions[:, :state_dim]
        actions = transitions[:, state_dim:state_dim + action_dim]
        next_states = transitions[:, state_dim + action_dim:state_dim + action_dim + state_dim]
        rewards = transitions[:, state_dim + action_dim + state_dim:]

        return states, actions, rewards, next_states

    def sample_raw(self, batch_size: int) -> torch.Tensor:
        """Sample raw transitions without splitting."""
        indices = torch.randint(0, self.size, (batch_size,))
        return self.buffer[indices]

    def set_split_dims(self, state_dim: int, action_dim: int):
        """Set dimensions for splitting transitions."""
        self.state_dim = state_dim
        self.action_dim = action_dim

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer (Schaul et al. 2015).

    Used for PER baselines.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 1e-4

        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
    ):
        self.buffer.append(Transition(state, action, reward, next_state, done))
        self.priorities.append(priority if priority is not None else self.max_priority)

    def sample(self, batch_size: int, device: str = "cpu") -> Tuple[torch.Tensor, ...]:
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]

        # Compute importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        states = torch.FloatTensor(np.stack([t.state for t in batch])).to(device)
        actions = torch.FloatTensor(np.stack([t.action for t in batch])).to(device)
        rewards = torch.FloatTensor(np.array([t.reward for t in batch])).to(device).unsqueeze(-1)
        next_states = torch.FloatTensor(np.stack([t.next_state for t in batch])).to(device)
        dones = torch.FloatTensor(np.array([t.done for t in batch])).to(device).unsqueeze(-1)
        weights = torch.FloatTensor(weights).to(device).unsqueeze(-1)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer)
