## replay_buffer.py

"""
Replay buffer implementation for Prioritized Generative Replay (PGR).

Provides a ring‑buffer (`ReplayBuffer`) that stores transitions along with
a scalar relevance condition.  Efficient random sampling for policy training
and bulk retrieval for diffusion model training are supported.

All data is stored as NumPy arrays on CPU; conversion to PyTorch tensors
occurs only when requested (via `sample` and `all_data`).  The recommended
usage is to move the returned tensors to the appropriate device via
`utils.to_tensor`.
"""

from collections import deque
from typing import Tuple

import numpy as np
import torch

# Avoid circular imports by using local type hints only
# from utils import get_device, to_tensor  # not needed here; caller handles conversion


class ReplayBuffer:
    """Circular buffer for transitions with relevance condition.

    Parameters
    ----------
    capacity : int
        Total number of transitions the buffer can hold.
    state_shape : tuple
        Shape of a single state observation (e.g. (state_dim,) or (3, 84, 84)).
    action_shape : tuple
        Shape of a single action (e.g. (action_dim,)).
    condition_dim : int, optional
        Dimensionality of the relevance condition (default 1 for scalar).
    """

    def __init__(
        self,
        capacity: int,
        state_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        condition_dim: int = 1,
    ) -> None:
        self.capacity = capacity
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.condition_dim = condition_dim

        # Pre‑allocate NumPy arrays for all components
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.conditions = np.zeros((capacity, condition_dim), dtype=np.float32)

        # Internal pointers for the ring buffer
        self._pos = 0          # next write location (physical index)
        self._size = 0         # number of valid transitions currently stored
        self._full = False     # whether the buffer has wrapped around

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        condition: float,
    ) -> None:
        """Store a single transition and its relevance condition.

        Parameters
        ----------
        state : np.ndarray
            Current state (shape matches `state_shape`).
        action : np.ndarray
            Action taken.
        reward : float
            Immediate reward.
        next_state : np.ndarray
            Resulting next state.
        done : bool
            Whether the episode terminated.
        condition : float
            Scalar relevance value computed by the current `RelevanceFunction`.
        """
        # Write data into the current physical slot (overwriting oldest if full)
        self.states[self._pos] = state.astype(np.float32, copy=False)
        self.actions[self._pos] = action.astype(np.float32, copy=False)
        self.rewards[self._pos] = float(reward)
        self.next_states[self._pos] = next_state.astype(np.float32, copy=False)
        self.dones[self._pos] = float(done)
        self.conditions[self._pos] = float(condition)

        # Update pointer and size tracking
        self._pos = (self._pos + 1) % self.capacity
        if self._full:
            # Buffer already full; size stays at capacity, no need to change _size
            pass
        else:
            self._size += 1
            if self._size == self.capacity:
                self._full = True

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Randomly sample a mini‑batch of transitions (including conditions).

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        tuple of torch.Tensor
            (states, actions, rewards, next_states, dones, conditions) each with
            leading dimension `batch_size`.

        Raises
        ------
        ValueError
            If the buffer is empty.
        """
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        # Logical indices (0 .. _size-1) in order of insertion
        logical_indices = np.random.choice(self._size, size=batch_size, replace=False)

        # Map to physical indices taking the circular wrap into account.
        # The oldest valid transition is at physical position:
        #   start = (self._pos - self._size) % self.capacity   (if buffer not full)
        start = (self._pos - self._size) % self.capacity
        physical_indices = (start + logical_indices) % self.capacity

        # Extract the sliced data and convert to tensors.
        # Using np.take along axis 0 is concise and efficient.
        states = np.take(self.states, physical_indices, axis=0)
        actions = np.take(self.actions, physical_indices, axis=0)
        rewards = self.rewards[physical_indices]
        next_states = np.take(self.next_states, physical_indices, axis=0)
        dones = self.dones[physical_indices]
        conditions = self.conditions[physical_indices]

        # Convert to torch tensors (on CPU). The caller can move them later.
        return (
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(rewards).unsqueeze(-1),   # keep shape (batch, 1)
            torch.from_numpy(next_states),
            torch.from_numpy(dones).unsqueeze(-1),
            torch.from_numpy(conditions),              # (batch, condition_dim)
        )

    def all_data(self) -> Tuple[torch.Tensor, ...]:
        """Return all stored transitions as tensors (in chronological order).

        This is used for training the diffusion model and for sampling prompt
        conditions during generation.

        Returns
        -------
        tuple of torch.Tensor
            (states, actions, rewards, next_states, dones, conditions) with shapes
            (buffer_size, ...). If the buffer is empty, returns tensors with
            shape (0, ...).

        Note
        ----
        All tensors are on CPU.
        """
        if self._size == 0:
            # Return empty tensors with the proper shapes
            return (
                torch.empty(0, *self.state_shape),
                torch.empty(0, *self.action_shape),
                torch.empty(0, 1),
                torch.empty(0, *self.state_shape),
                torch.empty(0, 1),
                torch.empty(0, self.condition_dim),
            )

        # Logical ordering: oldest first.
        start = (self._pos - self._size) % self.capacity
        # When the buffer is full, we need to wrap around; we can concatenate
        # two slices: the tail part and the head part.
        if self._full:
            # Two contiguous blocks: [start, end) where end = capacity, then [0, self._pos)
            tail_size = self.capacity - start
            head_size = self._pos
            # Build indices for the whole logical range
            all_indices = np.arange(self._size)
            # Map logical to physical: (start + i) % capacity
            physical_indices = (start + all_indices) % self.capacity
        else:
            # Not full: data is contiguous from `start` to `start + self._size - 1`
            physical_indices = np.arange(start, start + self._size)

        # Extract data
        states = self.states[physical_indices]
        actions = self.actions[physical_indices]
        rewards = self.rewards[physical_indices]
        next_states = self.next_states[physical_indices]
        dones = self.dones[physical_indices]
        conditions = self.conditions[physical_indices]

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(rewards).unsqueeze(-1),
            torch.from_numpy(next_states),
            torch.from_numpy(dones).unsqueeze(-1),
            torch.from_numpy(conditions),
        )

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self._size

