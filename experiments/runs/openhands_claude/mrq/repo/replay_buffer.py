"""
LAP Replay Buffer – Loss-Adjusted Prioritized Experience Replay.

Implements the prioritized replay buffer from Fujimoto et al. (2020) used by MR.Q.
Priorities are based on TD errors; sampling is proportional to priority^α.
Sequences of consecutive transitions are returned to support:
  - Encoder unrolling over H_Enc steps
  - Multi-step return computation over H_Q steps
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple


class LAPReplayBuffer:
    """
    Circular replay buffer with LAP-style prioritized sampling.

    Stores individual transitions and supports sampling contiguous sequences
    of length `seq_len = max(enc_horizon, q_horizon)`.

    Priority update rule:
        priority_i = |td_error_i|^α + min_priority
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_dim: int,
        seq_len: int,
        lap_alpha: float = 0.4,
        min_priority: float = 1.0,
        obs_dtype: np.dtype = np.float32,
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.lap_alpha = lap_alpha
        self.min_priority = min_priority
        self.obs_dtype = obs_dtype

        # Storage arrays
        self.states = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        # next_state is stored at index (i+1) % capacity, so we only need states

        # Priority array (initialised to max priority for new transitions)
        self.priorities = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0          # next write position
        self.size = 0         # current number of stored transitions
        self._max_priority = 1.0

    # ── Adding transitions ────────────────────────────────────────────────────

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: float,
        next_state: np.ndarray,
    ) -> None:
        """Store a single transition with maximum priority."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        # next_state is stored at the next slot (will be overwritten eventually)
        next_ptr = (self.ptr + 1) % self.capacity
        self.states[next_ptr] = next_state

        self.priorities[self.ptr] = self._max_priority

        self.ptr = next_ptr
        self.size = min(self.size + 1, self.capacity)

    # ── Sampling ──────────────────────────────────────────────────────────────

    def _valid_start_indices(self) -> np.ndarray:
        """
        Return indices that can serve as the start of a valid sequence of
        length seq_len.

        A sequence starting at i uses transition indices i … i+seq_len-1 and
        next-state index i+seq_len.  All of these must have been written.

        When the buffer is not yet full (size < capacity):
          Written state indices are 0 … ptr  (ptr = next write ptr, but
          states[ptr] holds the next_state of the last added transition).
          Valid starts: i such that i + seq_len <= ptr, i.e. i <= ptr - seq_len.

        When the buffer is full (size == capacity):
          All indices are written.  We exclude the seq_len indices immediately
          before ptr to avoid sequences that straddle the overwrite boundary.
        """
        n = self.size
        if n < self.seq_len:
            return np.array([], dtype=np.int64)

        if n < self.capacity:
            # Buffer not yet full: ptr == n, valid starts are [0, ptr - seq_len]
            max_start = self.ptr - self.seq_len  # inclusive upper bound
            if max_start < 0:
                return np.array([], dtype=np.int64)
            return np.arange(max_start + 1, dtype=np.int64)

        # Buffer full: exclude the seq_len indices just before ptr
        # (those sequences would include ptr which is about to be overwritten)
        all_idx = np.arange(self.capacity, dtype=np.int64)
        # Indices to exclude: [ptr - seq_len, ptr - 1]  (mod capacity)
        lo = (self.ptr - self.seq_len) % self.capacity
        hi = (self.ptr - 1) % self.capacity
        if lo <= hi:
            mask = ~((all_idx >= lo) & (all_idx <= hi))
        else:
            mask = ~((all_idx >= lo) | (all_idx <= hi))
        return all_idx[mask]

    def sample(
        self, batch_size: int
    ) -> Tuple[
        np.ndarray,  # states        (B, *obs_shape)
        np.ndarray,  # actions       (B, action_dim)
        np.ndarray,  # seq_actions   (B, seq_len, action_dim)
        np.ndarray,  # seq_rewards   (B, seq_len)
        np.ndarray,  # seq_dones     (B, seq_len)
        np.ndarray,  # seq_next_states (B, seq_len, *obs_shape)
        np.ndarray,  # indices       (B,)  for priority update
    ]:
        """
        Sample a batch of sequences.

        Returns the initial state (for value/policy updates) plus full
        sequences (for encoder unrolling and multi-step returns).
        """
        valid_idx = self._valid_start_indices()
        n_valid = len(valid_idx)

        # Sampling probabilities proportional to priority^α
        probs = self.priorities[valid_idx] ** self.lap_alpha
        probs = probs / probs.sum()

        chosen = np.random.choice(n_valid, size=batch_size, replace=True, p=probs)
        start_indices = valid_idx[chosen]

        # Gather sequences
        seq_indices = (
            start_indices[:, None] + np.arange(self.seq_len)[None, :]
        ) % self.capacity  # (B, seq_len)

        states = self.states[start_indices]                          # (B, *obs_shape)
        seq_actions = self.actions[seq_indices]                      # (B, seq_len, action_dim)
        seq_rewards = self.rewards[seq_indices]                      # (B, seq_len)
        seq_dones = self.dones[seq_indices]                          # (B, seq_len)

        # next_state for step t is stored at index (seq_indices[:, t] + 1) % capacity
        next_indices = (seq_indices + 1) % self.capacity             # (B, seq_len)
        seq_next_states = self.states[next_indices]                  # (B, seq_len, *obs_shape)

        # Action at t=0 (for value/policy update)
        actions = seq_actions[:, 0]                                  # (B, action_dim)

        return (
            states,
            actions,
            seq_actions,
            seq_rewards,
            seq_dones,
            seq_next_states,
            start_indices,
        )

    # ── Priority update ───────────────────────────────────────────────────────

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.

        priority_i = |td_error_i|^α + min_priority
        (The α exponent is applied during sampling, not here.)
        """
        new_priorities = np.abs(td_errors) + self.min_priority
        self.priorities[indices] = new_priorities
        self._max_priority = max(self._max_priority, new_priorities.max())

    # ── Reward statistics ─────────────────────────────────────────────────────

    def mean_abs_reward(self, sample_size: int = 10_000) -> float:
        """Estimate mean absolute reward from a random sample of the buffer."""
        n = min(self.size, sample_size)
        idx = np.random.randint(0, self.size, size=n)
        return float(np.mean(np.abs(self.rewards[idx])))

    # ── Misc ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.size

    @property
    def ready(self) -> bool:
        """True once the buffer has enough transitions to sample a full sequence."""
        return self.size >= self.seq_len
