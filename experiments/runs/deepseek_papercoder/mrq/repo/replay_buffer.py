# replay_buffer.py

"""
Prioritised experience replay buffer with LAP (Loss Adjusted Prioritisation)
for the MR.Q algorithm.

Implements a circular buffer storing transitions (s, a, r, d, s') and a
SumTree for proportional sampling based on priorities. Supports retrieval
of contiguous temporal sequences with episode‑boundary masking.

Classes:
    SequenceBatch : namedtuple returned by sample().
    ReplayBuffer : the main buffer class.

Dependencies:
    - numpy
    - collections.namedtuple
    - utils.SumTree (binary tree for priority sampling)
"""

from collections import namedtuple
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from utils import SumTree


# ---------------------------------------------------------------------------
# Data structure for sampled batches
# ---------------------------------------------------------------------------

SequenceBatch = namedtuple(
    "SequenceBatch",
    [
        "obs",          # [batch, seq_len+1, *state_shape]  first state + next states
        "actions",      # [batch, seq_len, *action_shape]
        "rewards",      # [batch, seq_len] or [batch, seq_len, 1]
        "dones",        # [batch, seq_len] or [batch, seq_len, 1]
        "next_obs",     # [batch, seq_len, *state_shape]   (identical to obs[:,1:])
        "mask",         # [batch, seq_len]   1.0 for valid, 0.0 for invalid
        "indices",      # list of data indices (len batch)
    ],
)


# ---------------------------------------------------------------------------
# ReplayBuffer class
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Prioritised experience replay buffer with LAP.

    Stores transitions as separate numpy arrays of fixed capacity.
    Uses a SumTree for O(log N) priority sampling.
    New transitions receive the current maximum priority to favour
    recent data.

    Parameters
    ----------
    capacity : int
        Maximum number of stored transitions.
    state_shape : Tuple[int, ...]
        Shape of the state (observation) array (without batch dim).
    action_shape : Tuple[int, ...]
        Shape of the action array (without batch dim).
    alpha : float, default=0.4
        Prioritisation exponent (LAP α).  p_i^α is used for sampling.
    min_priority : float, default=1.0
        Minimum priority (avoids zero‑probability items).
    state_dtype : np.dtype, optional
        Data type for states (default float32).
    action_dtype : np.dtype, optional
        Data type for actions (default float32).
    """

    def __init__(
        self,
        capacity: int,
        state_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        alpha: float = 0.4,
        min_priority: float = 1.0,
        state_dtype: np.dtype = np.float32,
        action_dtype: np.dtype = np.float32,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if min_priority <= 0:
            raise ValueError("min_priority must be positive")

        self.capacity = capacity
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.alpha = alpha
        self.min_priority = min_priority
        self.state_dtype = state_dtype
        self.action_dtype = action_dtype

        # Circular buffer data
        self._states = np.zeros((capacity,) + state_shape, dtype=state_dtype)
        self._actions = np.zeros((capacity,) + action_shape, dtype=action_dtype)
        self._rewards = np.zeros((capacity, 1), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)
        self._next_states = np.zeros((capacity,) + state_shape, dtype=state_dtype)

        # SumTree for priorities
        # Round capacity up to a power of two for the tree
        self._tree_capacity = 1 << (capacity - 1).bit_length()
        self._tree = SumTree(self._tree_capacity)
        # Initialise all priorities to 0 (unused leaves)
        # The tree is already zero from SumTree.__init__

        # Write pointer and size
        self._pos = 0          # next index to write to
        self._size = 0         # number of stored transitions (≤ capacity)
        self._full = False     # whether all slots have been written at least once

        # Current maximum priority (used for new transitions)
        self._max_priority = min_priority

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, state: np.ndarray, action: np.ndarray,
            reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Store a new transition.

        Parameters
        ----------
        state : np.ndarray
        action : np.ndarray
        reward : float
        next_state : np.ndarray
        done : bool
        """
        idx = self._pos
        self._states[idx] = state.astype(self.state_dtype, copy=False)
        self._actions[idx] = action.astype(self.action_dtype, copy=False)
        self._rewards[idx] = reward
        self._dones[idx] = float(done)
        self._next_states[idx] = next_state.astype(self.state_dtype, copy=False)

        # Assign the current maximum priority to the new transition
        priority = self._max_priority ** self.alpha  # p_i^α
        self._tree.set_priority(idx, priority)

        # Advance write pointer
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        if self._size == self.capacity:
            self._full = True

    def sample(self, batch_size: int, seq_len: int) -> SequenceBatch:
        """
        Sample a batch of contiguous sequences of length seq_len transitions.

        Parameters
        ----------
        batch_size : int
            Number of sequences to sample.
        seq_len : int
            Number of transitions (actions) in each sequence. The returned
            observation array will have shape [batch, seq_len+1, ...].

        Returns
        -------
        SequenceBatch
            Named tuple containing the sampled data and the list of
            starting data indices (for priority updates).
        """
        # Ensure we have at least batch_size stored transitions
        assert self._size >= batch_size, (
            f"Not enough transitions in buffer ({self._size} < {batch_size})"
        )

        # Sample starting indices using the priority tree
        indices = self._tree.sample(batch_size)

        # Retrieve each sequence
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_next_obs = []
        batch_mask = []

        for data_idx in indices:
            seq = self._get_sequence(data_idx, seq_len)
            batch_obs.append(seq["states"])
            batch_actions.append(seq["actions"])
            batch_rewards.append(seq["rewards"])
            batch_dones.append(seq["dones"])
            batch_next_obs.append(seq["next_states"])
            batch_mask.append(seq["mask"])

        # Stack into batch tensors
        obs = np.stack(batch_obs, axis=0)          # (B, seq_len+1, *state_shape)
        actions = np.stack(batch_actions, axis=0)  # (B, seq_len, *action_shape)
        rewards = np.stack(batch_rewards, axis=0)  # (B, seq_len, 1)
        dones = np.stack(batch_dones, axis=0)      # (B, seq_len, 1)
        next_obs = np.stack(batch_next_obs, axis=0)  # (B, seq_len, *state_shape)
        mask = np.stack(batch_mask, axis=0)        # (B, seq_len)

        return SequenceBatch(
            obs=obs,
            actions=actions,
            rewards=rewards.squeeze(axis=-1),  # remove last dim to match shape (B, seq_len)
            dones=dones.squeeze(axis=-1),
            next_obs=next_obs,
            mask=mask,
            indices=indices,
        )

    def update_priorities(self, indices: List[int],
                          priorities: List[float]) -> None:
        """
        Update the sampling priorities for the given transitions.

        Parameters
        ----------
        indices : list of int
            Data indices (as returned in SequenceBatch.indices).
        priorities : list of float
            New priorities (raw, usually |TD_error| + 1).
        """
        for idx, p in zip(indices, priorities):
            # Clamp to minimum and update max if needed
            p = max(p, self.min_priority)
            if p > self._max_priority:
                self._max_priority = p
            # Store the priority^α in the tree
            self._tree.set_priority(idx, p ** self.alpha)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sequence(self, data_idx: int, seq_len: int) -> Dict[str, np.ndarray]:
        """
        Retrieve a contiguous sequence of up to seq_len transitions
        starting from data_idx, respecting episode boundaries.

        Returns a dictionary with keys:
            states      : (seq_len+1, *state_shape)
            actions     : (seq_len,   *action_shape)
            rewards     : (seq_len, 1)
            dones       : (seq_len, 1)
            next_states : (seq_len,   *state_shape)
            mask        : (seq_len,)   float 1.0 for valid, 0.0 otherwise

        The mask marks transitions that are either out‑of‑bounds
        (unwritten data) or belong to a new episode (after done==1).
        """
        states = np.zeros((seq_len + 1,) + self.state_shape, dtype=self.state_dtype)
        actions = np.zeros((seq_len,) + self.action_shape, dtype=self.action_dtype)
        rewards = np.zeros((seq_len, 1), dtype=np.float32)
        dones = np.zeros((seq_len, 1), dtype=np.float32)
        next_states = np.zeros((seq_len,) + self.state_shape, dtype=self.state_dtype)
        mask = np.zeros((seq_len,), dtype=np.float32)

        # First state (s0) is always valid because data_idx is known to be a
        # stored transition (the tree only indexes written slots).
        states[0] = self._states[data_idx]

        episode_ended = False
        for t in range(seq_len):
            # Physical index of the t‑th transition
            idx = (data_idx + t) % self.capacity

            # Check validity: the slot must contain data and the episode must
            # not have ended.
            valid = self._is_valid_idx(idx) and not episode_ended

            if valid:
                a = self._actions[idx]
                r = float(self._rewards[idx])
                d = float(self._dones[idx])
                s_next = self._next_states[idx]
                mask[t] = 1.0
                if d == 1.0:
                    episode_ended = True
            else:
                # Use zero-filled tensors (already initialised)
                a = actions[t]    # zero
                r = rewards[t][0] # zero
                d = dones[t][0]   # zero
                s_next = next_states[t]  # zero
                # mask[t] stays 0

            actions[t] = a
            rewards[t] = r
            dones[t] = d
            next_states[t] = s_next
            # The next state becomes the state for the next time step
            states[t + 1] = s_next

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "next_states": next_states,
            "mask": mask,
        }

    def _is_valid_idx(self, idx: int) -> bool:
        """
        Return True if the physical index `idx` currently holds valid data.

        When not yet full, only indices < self._pos are valid.
        Once full, every index is valid.
        """
        if self._full:
            return True
        return idx < self._pos

    # ------------------------------------------------------------------
    # Properties for external use
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of transitions currently stored."""
        return self._size

    @property
    def max_priority(self) -> float:
        """Current maximum priority seen (raw, not raised to α)."""
        return self._max_priority
