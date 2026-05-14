"""
LAP (Loss-Adjusted Prioritized) Replay Buffer for MR.Q.

Based on: "An Equivalence between Loss Functions and Non-Uniform Sampling
in Experience Replay" - Fujimoto et al., 2020

LAP uses TD errors as priorities and the Huber loss to eliminate bias
from prioritized sampling.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Replay buffer with LAP (Loss-Adjusted Prioritized) sampling.
    
    Stores transitions and samples with priority proportional to TD error.
    Uses Huber loss in value updates to eliminate importance sampling bias.
    
    Supports sequence sampling for encoder unrolling and multi-step returns.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_size=1_000_000,
        batch_size=256,
        image_obs=False,
        state_channels=None,
        image_size=84,
        seq_len=1,
        lap_alpha=0.4,
        min_priority=1.0,
        device="cpu",
    ):
        self.max_size = max_size
        self.batch_size = batch_size
        self.image_obs = image_obs
        self.seq_len = seq_len
        self.lap_alpha = lap_alpha
        self.min_priority = min_priority
        self.device = device

        self.ptr = 0
        self.size = 0

        # Storage
        if image_obs:
            # Store as uint8 to save memory (normalized in network forward pass)
            self.states = np.zeros(
                (max_size, state_channels, image_size, image_size), dtype=np.uint8
            )
        else:
            self.states = np.zeros((max_size, state_dim), dtype=np.float32)

        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        # Track episode boundaries for sequence sampling
        self.episode_ends = np.zeros(max_size, dtype=bool)

        # LAP priorities
        self.priorities = np.ones(max_size, dtype=np.float32) * min_priority

    def add(self, state, action, reward, done):
        """Add a transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.episode_ends[self.ptr] = bool(done)

        # New transitions get max priority to ensure they are sampled
        max_prio = self.priorities[:self.size].max() if self.size > 0 else self.min_priority
        self.priorities[self.ptr] = max(max_prio, self.min_priority)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=None):
        """
        Sample a batch of transitions using LAP priorities.
        
        Returns (batch_dict, indices).
        """
        if batch_size is None:
            batch_size = self.batch_size

        # LAP sampling: probability proportional to priority^alpha
        probs = self.priorities[:self.size] ** self.lap_alpha
        probs = probs / probs.sum()

        replace = self.size < batch_size
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=replace)

        return self._get_batch(indices), indices

    def sample_sequences(self, batch_size=None, seq_len=None):
        """
        Sample sequences of transitions for encoder unrolling and multi-step returns.
        
        Samples starting indices using LAP priorities, then returns
        contiguous sequences of length seq_len.
        
        Returns dict with tensors of shape (seq_len, batch_size, ...).
        """
        if batch_size is None:
            batch_size = self.batch_size
        if seq_len is None:
            seq_len = self.seq_len

        # Maximum valid starting index (need seq_len consecutive transitions)
        max_start = max(1, self.size - seq_len)

        # LAP sampling for starting indices
        probs = self.priorities[:max_start] ** self.lap_alpha
        probs = probs / probs.sum()

        start_indices = np.random.choice(
            max_start,
            size=batch_size,
            p=probs,
            replace=True
        )

        # Build sequences
        seq_states = []
        seq_actions = []
        seq_rewards = []
        seq_dones = []

        for t in range(seq_len):
            indices = (start_indices + t) % self.max_size
            seq_states.append(self.states[indices])
            seq_actions.append(self.actions[indices])
            seq_rewards.append(self.rewards[indices])
            seq_dones.append(self.dones[indices])

        def to_tensor(arr_list, dtype=torch.float32):
            arr = np.stack(arr_list, axis=0)  # (seq_len, batch, ...)
            return torch.tensor(arr, dtype=dtype, device=self.device)

        return {
            "states": to_tensor(seq_states),
            "actions": to_tensor(seq_actions),
            "rewards": to_tensor(seq_rewards),
            "dones": to_tensor(seq_dones),
            "start_indices": start_indices,
        }

    def _get_batch(self, indices):
        """Get a batch of transitions at given indices."""
        next_indices = (indices + 1) % self.max_size

        states = torch.tensor(self.states[indices], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions[indices], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(self.states[next_indices], dtype=torch.float32, device=self.device)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "next_states": next_states,
        }

    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors.
        
        Priority = max(|td_error|, min_priority)
        """
        if np.isscalar(td_errors):
            td_errors = np.array([td_errors])
        td_errors = np.abs(td_errors).flatten()
        priorities = td_errors + self.min_priority
        # Clip to valid range
        priorities = np.clip(priorities, self.min_priority, 1e6)
        self.priorities[indices] = priorities

    def get_mean_abs_reward(self):
        """Get mean absolute reward in buffer (used for reward scaling)."""
        if self.size == 0:
            return 1.0
        mean_abs = np.abs(self.rewards[:self.size]).mean()
        return max(float(mean_abs), 1e-8)

    def __len__(self):
        return self.size
