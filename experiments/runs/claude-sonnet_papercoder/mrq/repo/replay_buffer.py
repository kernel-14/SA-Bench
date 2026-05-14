## replay_buffer.py
"""Episode-aware replay buffer with LAP prioritized sampling for MR.Q.

This module implements EpisodeBuffer, which supports two distinct sampling
modes required by the MR.Q algorithm:

  1. Single-transition sampling (sample) — for value and policy updates,
     using LAP prioritized replay (Fujimoto et al., 2020). Sampling
     probability is proportional to priority^alpha. No importance sampling
     weights are returned; the Huber loss in the value update implicitly
     corrects for the bias.

  2. Sequential subsequence sampling (sample_sequences) — for encoder
     unrolling over H_Enc = 5 steps. Returns contiguous windows of
     seq_len = H_Enc + 1 = 6 transitions that all belong to the same
     episode. Sampled uniformly over valid episode windows.

All storage is pre-allocated as numpy arrays on CPU for efficient random-
access writes and index-based operations. Conversion to torch tensors
happens only at sample time.

Episode boundaries are tracked via a monotonically increasing episode_ids
array. The circular buffer wrap is handled conservatively: only non-wrapping
windows are considered for sequence sampling, avoiding cross-episode
contamination near the wrap point.

Done signal convention (Gymnasium): done=True means the episode ended
(terminated or truncated). Stored as float32 where 1.0 = terminal.
The value bootstrap masking in agent.py uses (1 - done) to zero out the
bootstrap term at terminal states.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch


class EpisodeBuffer:
    """Circular replay buffer with episode tracking and LAP prioritized sampling.

    Supports both single-transition LAP sampling (for value/policy updates)
    and contiguous episode-window sampling (for encoder unrolling).

    All internal storage uses pre-allocated numpy arrays on CPU. Tensors
    are created on the target device only at sample time.

    Attributes:
        capacity: Maximum number of transitions stored.
        state_shape: Shape tuple of a single observation.
        action_dim: Dimensionality of the action vector.
        lap_alpha: LAP probability smoothing exponent (0.4 per config.yaml).
        lap_min_priority: Minimum priority floor (1.0 per config.yaml).
        device: Torch device for returned tensors.
        ptr: Next write position in the circular buffer.
        size: Number of valid transitions currently stored.
        current_episode_id: Monotonically increasing episode counter.
        current_step: Step index within the current episode.
        terminal_seen: True once any done=True transition has been stored.
            Sticky — never reverts to False.
    """

    def __init__(
        self,
        capacity: int,
        state_shape: Tuple[int, ...],
        action_dim: int,
        lap_alpha: float = 0.4,
        lap_min_priority: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialise the episode buffer with pre-allocated numpy arrays.

        All priority values are initialised to lap_min_priority so that
        newly added transitions (before their first TD error is computed)
        receive the maximum existing priority and are guaranteed to be
        sampled at least once.

        Args:
            capacity: Maximum number of transitions. 1_000_000 per config.yaml.
            state_shape: Shape of a single observation (e.g., (17,) for
                HalfCheetah, (9, 84, 84) for DMC visual, (4, 84, 84) for
                Atari).
            action_dim: Number of action dimensions (or number of discrete
                actions for Atari one-hot encoding).
            lap_alpha: LAP probability smoothing exponent. Sampling
                probability ∝ priority^lap_alpha. Default 0.4 per config.yaml.
            lap_min_priority: Minimum priority floor. Prevents zero-probability
                transitions. Default 1.0 per config.yaml.
            device: Torch device on which sampled tensors will be placed.

        Raises:
            ValueError: If capacity < 1, action_dim < 1, lap_alpha <= 0,
                or lap_min_priority <= 0.
        """
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}.")
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}.")
        if lap_alpha <= 0.0:
            raise ValueError(f"lap_alpha must be > 0, got {lap_alpha}.")
        if lap_min_priority <= 0.0:
            raise ValueError(
                f"lap_min_priority must be > 0, got {lap_min_priority}."
            )

        self.capacity: int = capacity
        self.state_shape: Tuple[int, ...] = state_shape
        self.action_dim: int = action_dim
        self.lap_alpha: float = lap_alpha
        self.lap_min_priority: float = lap_min_priority
        self.device: torch.device = device

        # ---------------------------------------------------------------
        # Circular buffer state
        # ---------------------------------------------------------------
        self.ptr: int = 0
        self.size: int = 0
        self.current_episode_id: int = 0
        self.current_step: int = 0
        self.terminal_seen: bool = False

        # ---------------------------------------------------------------
        # Pre-allocated storage arrays
        # ---------------------------------------------------------------
        # Observations: float32 for both vector and image inputs.
        # Image observations are stored in [0, 255] range; normalization
        # to [-0.5, 0.5] is performed inside the CNN forward pass.
        self.states: np.ndarray = np.zeros(
            (capacity, *state_shape), dtype=np.float32
        )
        self.next_states: np.ndarray = np.zeros(
            (capacity, *state_shape), dtype=np.float32
        )

        # Actions: float32 one-hot for discrete, continuous vector otherwise.
        self.actions: np.ndarray = np.zeros(
            (capacity, action_dim), dtype=np.float32
        )

        # Scalar signals
        self.rewards: np.ndarray = np.zeros(capacity, dtype=np.float32)
        # done: 1.0 = terminal, 0.0 = non-terminal (Gymnasium convention)
        self.dones: np.ndarray = np.zeros(capacity, dtype=np.float32)

        # LAP priorities: initialised to lap_min_priority so that the first
        # batch of transitions is sampled with equal probability.
        self.priorities: np.ndarray = np.full(
            capacity, fill_value=lap_min_priority, dtype=np.float32
        )

        # Episode tracking: monotonically increasing episode IDs.
        # Used by sample_sequences to verify window contiguity.
        self.episode_ids: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self.step_ids: np.ndarray = np.zeros(capacity, dtype=np.int64)

    # ------------------------------------------------------------------
    # Core write operation
    # ------------------------------------------------------------------

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_state: np.ndarray,
    ) -> None:
        """Store a single transition in the circular buffer.

        New transitions receive the maximum existing priority (or
        lap_min_priority if the buffer is empty) to ensure they are
        sampled at least once before their TD error is computed.

        Episode tracking is updated: if done=True, the episode counter
        increments and the step counter resets. terminal_seen is set
        permanently once any done=True is observed.

        Args:
            state: Current observation, shape self.state_shape, float32.
            action: Action taken, shape (action_dim,), float32.
            reward: Scalar reward received.
            done: True if this transition ends the episode.
            next_state: Next observation, shape self.state_shape, float32.
        """
        # Assign priority: max of existing valid priorities or the floor.
        # This guarantees new transitions are sampled at least once.
        if self.size > 0:
            new_priority: float = float(
                max(self.priorities[: self.size].max(), self.lap_min_priority)
            )
        else:
            new_priority = self.lap_min_priority

        # Write transition data at the current pointer position.
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = 1.0 if done else 0.0
        self.next_states[self.ptr] = next_state
        self.priorities[self.ptr] = new_priority
        self.episode_ids[self.ptr] = self.current_episode_id
        self.step_ids[self.ptr] = self.current_step

        # Advance the circular pointer and update valid size.
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # Update episode tracking.
        self.current_step += 1
        if done:
            self.terminal_seen = True
            self.current_episode_id += 1
            self.current_step = 0

    # ------------------------------------------------------------------
    # Single-transition LAP sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> Dict[str, object]:
        """Sample a batch of transitions using LAP prioritized sampling.

        Sampling probability is proportional to priority^lap_alpha.
        No importance sampling weights are returned — the Huber loss in
        the value update implicitly corrects for the prioritization bias
        (Fujimoto et al., 2020).

        Sampling is performed without replacement to avoid duplicate
        transitions within a minibatch.

        Args:
            batch_size: Number of transitions to sample. Must be <= size.

        Returns:
            Dictionary with the following keys:
                'states'      (torch.Tensor): shape (batch, *state_shape), float32.
                'actions'     (torch.Tensor): shape (batch, action_dim), float32.
                'rewards'     (torch.Tensor): shape (batch,), float32.
                'dones'       (torch.Tensor): shape (batch,), float32.
                'next_states' (torch.Tensor): shape (batch, *state_shape), float32.
                'indices'     (np.ndarray):   shape (batch,), int64. Returned as
                    numpy for direct use in update_priorities without device
                    transfer overhead.

        Raises:
            ValueError: If batch_size > self.size.
        """
        if batch_size > self.size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({self.size}). "
                f"Ensure the training loop waits until the buffer has at least "
                f"batch_size transitions before calling update()."
            )

        # Compute LAP sampling probabilities over valid transitions.
        valid_priorities: np.ndarray = self.priorities[: self.size]
        # Raise to power lap_alpha: P(i) ∝ priority_i^alpha
        probs: np.ndarray = valid_priorities ** self.lap_alpha
        # Normalize to a valid probability distribution.
        probs_sum: float = float(probs.sum())
        if probs_sum <= 0.0:
            # Fallback to uniform if all priorities are zero (should not happen
            # given lap_min_priority > 0, but guard defensively).
            probs = np.ones(self.size, dtype=np.float64) / self.size
        else:
            probs = probs.astype(np.float64) / probs_sum

        # Sample without replacement.
        indices: np.ndarray = np.random.choice(
            self.size, size=batch_size, replace=False, p=probs
        )

        # Gather transitions and convert to torch tensors.
        return {
            "states": torch.as_tensor(
                self.states[indices], dtype=torch.float32, device=self.device
            ),
            "actions": torch.as_tensor(
                self.actions[indices], dtype=torch.float32, device=self.device
            ),
            "rewards": torch.as_tensor(
                self.rewards[indices], dtype=torch.float32, device=self.device
            ),
            "dones": torch.as_tensor(
                self.dones[indices], dtype=torch.float32, device=self.device
            ),
            "next_states": torch.as_tensor(
                self.next_states[indices], dtype=torch.float32, device=self.device
            ),
            "indices": indices,  # numpy array for update_priorities
        }

    # ------------------------------------------------------------------
    # Sequential episode-window sampling
    # ------------------------------------------------------------------

    def sample_sequences(
        self, batch_size: int, seq_len: int
    ) -> Dict[str, torch.Tensor]:
        """Sample contiguous episode windows for encoder unrolling.

        Finds all valid start positions where seq_len consecutive transitions
        (i) are within the valid buffer range, (ii) do not wrap around the
        circular buffer boundary, and (iii) all belong to the same episode.

        Sampling over valid windows is uniform (no priority weighting).

        The returned tensors have shape (batch, seq_len, ...) so that
        agent._update_encoder can access:
            batch_seq['states'][:, 0]       — initial state for encoder
            batch_seq['actions'][:, t-1]    — action at step t
            batch_seq['rewards'][:, t-1]    — reward at step t
            batch_seq['dones'][:, t-1]      — done signal at step t
            batch_seq['next_states'][:, t-1] — next state at step t

        for t = 1 .. H_Enc (seq_len = H_Enc + 1 = 6).

        Args:
            batch_size: Number of windows to sample.
            seq_len: Length of each contiguous window (H_Enc + 1 = 6).

        Returns:
            Dictionary with the following keys:
                'states'      (torch.Tensor): shape (batch, seq_len, *state_shape).
                'actions'     (torch.Tensor): shape (batch, seq_len, action_dim).
                'rewards'     (torch.Tensor): shape (batch, seq_len).
                'dones'       (torch.Tensor): shape (batch, seq_len).
                'next_states' (torch.Tensor): shape (batch, seq_len, *state_shape).
            All tensors are float32 on self.device.

        Raises:
            ValueError: If seq_len < 1, or if no valid episode windows exist,
                or if batch_size > number of valid windows.
        """
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}.")

        # ---------------------------------------------------------------
        # Find valid start indices (non-wrapping, same-episode windows).
        # ---------------------------------------------------------------
        # Conservative approach: only consider windows that do not wrap
        # around the circular buffer boundary. This avoids cross-episode
        # contamination near the wrap point at negligible cost (at most
        # seq_len - 1 transitions are excluded near the wrap).
        #
        # A start index i is valid if:
        #   1. i + seq_len <= self.size  (all indices within valid range)
        #   2. i + seq_len <= self.capacity  (no circular wrap)
        #   3. episode_ids[i] == episode_ids[i + seq_len - 1]
        #      (first and last share the same episode; since IDs are
        #       monotonically non-decreasing within a linear segment,
        #       equality of endpoints implies all intermediate are equal)
        # ---------------------------------------------------------------
        max_start: int = min(self.size, self.capacity) - seq_len
        if max_start < 0:
            raise ValueError(
                f"Buffer has {self.size} transitions but seq_len={seq_len} "
                f"requires at least {seq_len} transitions. "
                f"Ensure the buffer is warmed up before calling sample_sequences()."
            )

        # Vectorized validity check over all candidate start indices.
        candidate_starts: np.ndarray = np.arange(max_start + 1, dtype=np.int64)

        # Episode ID at the first and last position of each candidate window.
        first_episode_ids: np.ndarray = self.episode_ids[candidate_starts]
        last_episode_ids: np.ndarray = self.episode_ids[
            candidate_starts + seq_len - 1
        ]

        valid_mask: np.ndarray = first_episode_ids == last_episode_ids
        valid_starts: np.ndarray = candidate_starts[valid_mask]

        if len(valid_starts) == 0:
            raise ValueError(
                f"No valid episode windows of length {seq_len} found in the "
                f"buffer (size={self.size}). The buffer may not contain any "
                f"complete episodes of sufficient length yet."
            )

        if batch_size > len(valid_starts):
            raise ValueError(
                f"batch_size ({batch_size}) exceeds the number of valid "
                f"episode windows ({len(valid_starts)}) of length {seq_len}."
            )

        # Sample batch_size start indices uniformly without replacement.
        sampled_starts: np.ndarray = np.random.choice(
            valid_starts, size=batch_size, replace=False
        )

        # ---------------------------------------------------------------
        # Gather sequences.
        # ---------------------------------------------------------------
        # Build index array of shape (batch_size, seq_len).
        # offsets[j] = sampled_starts[j] + [0, 1, ..., seq_len-1]
        offsets: np.ndarray = np.arange(seq_len, dtype=np.int64)  # (seq_len,)
        # (batch_size, 1) + (1, seq_len) → (batch_size, seq_len)
        all_indices: np.ndarray = (
            sampled_starts[:, np.newaxis] + offsets[np.newaxis, :]
        )

        # Gather all fields using advanced indexing.
        # states: (batch_size, seq_len, *state_shape)
        batch_states: np.ndarray = self.states[all_indices]
        batch_actions: np.ndarray = self.actions[all_indices]
        batch_rewards: np.ndarray = self.rewards[all_indices]
        batch_dones: np.ndarray = self.dones[all_indices]
        batch_next_states: np.ndarray = self.next_states[all_indices]

        return {
            "states": torch.as_tensor(
                batch_states, dtype=torch.float32, device=self.device
            ),
            "actions": torch.as_tensor(
                batch_actions, dtype=torch.float32, device=self.device
            ),
            "rewards": torch.as_tensor(
                batch_rewards, dtype=torch.float32, device=self.device
            ),
            "dones": torch.as_tensor(
                batch_dones, dtype=torch.float32, device=self.device
            ),
            "next_states": torch.as_tensor(
                batch_next_states, dtype=torch.float32, device=self.device
            ),
        }

    # ------------------------------------------------------------------
    # Priority update
    # ------------------------------------------------------------------

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        """Update LAP priorities for previously sampled transitions.

        Sets priority[i] = max(|td_error_i|, lap_min_priority) for each
        index in indices. The floor ensures no transition has zero priority.

        Called by MRQAgent._update_value after computing the value loss,
        using the indices returned by sample().

        Args:
            indices: 1-D numpy array of buffer indices to update, shape
                (batch,). These are the 'indices' returned by sample().
            td_errors: 1-D numpy array of TD errors (absolute differences
                between predicted and target Q-values), shape (batch,).
                May contain negative values; absolute value is taken.

        Raises:
            ValueError: If indices and td_errors have different lengths.
        """
        if len(indices) != len(td_errors):
            raise ValueError(
                f"indices and td_errors must have the same length, "
                f"got {len(indices)} and {len(td_errors)}."
            )

        # Clamp priorities to [lap_min_priority, inf) to prevent zero-
        # probability transitions.
        new_priorities: np.ndarray = np.maximum(
            np.abs(td_errors).astype(np.float32),
            self.lap_min_priority,
        )
        self.priorities[indices] = new_priorities

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def mean_abs_reward(self) -> float:
        """Compute the mean absolute reward over all valid transitions.

        Used by MRQAgent._sync_targets() to update r_bar for reward scaling
        (Section 4.2.2 of the paper). Returns a safe default of 1.0 if the
        buffer is empty to prevent division by zero in the agent.

        Returns:
            Mean of |reward| over all stored transitions, or 1.0 if empty.
        """
        if self.size == 0:
            return 1.0
        return float(np.mean(np.abs(self.rewards[: self.size])))

    def has_terminal(self) -> bool:
        """Return whether any terminal transition has been stored.

        Once True, this never reverts to False (sticky flag). Used by
        MRQAgent to decide whether to apply lambda_terminal in the encoder
        loss (Section 4.2.1: "lambda_Terminal is set to 0 until the first
        terminal transition is viewed").

        Returns:
            True if at least one done=True transition has been stored.
        """
        return self.terminal_seen

    def __len__(self) -> int:
        """Return the number of valid transitions currently stored.

        Used by the training loop in main.py to guard against calling
        agent.update() before the buffer has enough transitions.

        Returns:
            Number of valid transitions in [0, capacity].
        """
        return self.size

    def __repr__(self) -> str:
        """Return a human-readable description of the buffer state.

        Returns:
            String with key buffer properties and current fill level.
        """
        fill_pct: float = 100.0 * self.size / self.capacity
        return (
            f"EpisodeBuffer("
            f"size={self.size}/{self.capacity} ({fill_pct:.1f}%), "
            f"state_shape={self.state_shape}, "
            f"action_dim={self.action_dim}, "
            f"lap_alpha={self.lap_alpha}, "
            f"lap_min_priority={self.lap_min_priority}, "
            f"current_episode_id={self.current_episode_id}, "
            f"terminal_seen={self.terminal_seen}, "
            f"device={self.device}"
            f")"
        )
