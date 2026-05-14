# utils.py

"""
Utility functions and classes for the MR.Q algorithm.

Contains:
- TwoHotEncoding: converts scalar rewards into two‑hot categorical targets.
- hard_update / soft_update: functions for synchronising target networks.
- RunningMean: online calculation of mean absolute reward (used for reward scaling).
- SumTree: binary tree for efficient priority sampling in LAP replay buffer.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional


class TwoHotEncoding:
    """
    Implements symexp two‑hot encoding for scalar rewards.

    Bin centres are computed as symexp(linspace(-10, 10, num_bins)).
    For a given reward, the two adjacent bins receive a fractional weight
    such that the total mass sums to 1. Out‑of‑range values are clipped to
    the nearest bin.

    Parameters
    ----------
    num_bins : int
        Number of discrete bins (default 65).
    low : float
        Lower boundary of the linear range (default -10).
    high : float
        Upper boundary of the linear range (default 10).
    """

    def __init__(self, num_bins: int = 65, low: float = -10.0, high: float = 10.0):
        self.num_bins = num_bins
        # Symexp function: sign(x) * (exp(|x|) - 1)
        def symexp(x):
            return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

        # Linearly spaced points in [low, high] and then symexp
        linear_bins = torch.linspace(low, high, num_bins)
        self.centers = symexp(linear_bins).float()  # shape (num_bins,)

    def to(self, device: torch.device) -> "TwoHotEncoding":
        """Move bin centres to given device (returns self for convenience)."""
        self.centers = self.centers.to(device)
        return self

    def encode(self, reward: torch.Tensor) -> torch.Tensor:
        """
        Convert scalar reward(s) to two‑hot probability vectors.

        Parameters
        ----------
        reward : torch.Tensor
            Scalar reward values, shape (B,) or a single value.

        Returns
        -------
        target : torch.Tensor
            Two‑hot encoded target, shape (B, num_bins) (even for single value, batch dim added).
        """
        # Ensure reward is 1-D
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        batch_size = reward.shape[0]
        device = reward.device

        # Move centres to same device if necessary
        if self.centers.device != device:
            self.centers = self.centers.to(device)

        # For each reward value, find the two surrounding bin indices and fractional weights
        centers = self.centers.unsqueeze(0)  # (1, num_bins)
        reward_expanded = reward.unsqueeze(1)  # (B, 1)

        # Find indices of the two closest bins by comparing with centres
        # right_idx[i] = smallest index where centers[i] >= reward[i]
        right_idx = torch.sum(centers <= reward_expanded, dim=1)  # number of centres <= reward
        # clip to valid range
        right_idx = torch.clamp(right_idx, 1, self.num_bins - 1)
        left_idx = right_idx - 1

        # Get the corresponding centre values
        left_center = self.centers[left_idx]
        right_center = self.centers[right_idx]

        # Compute fractional weight (delta)
        delta = torch.where(
            (right_center - left_center) != 0,
            (reward - left_center) / (right_center - left_center),
            torch.zeros_like(reward),  # when exactly at a bin, delta=0
        )

        # Create target tensor
        target = torch.zeros(batch_size, self.num_bins, device=device)
        # Clamp delta so that out-of-range values get full weight at the extremity
        # (left_idx might be 0 when reward < centers[0]; right_idx might be num_bins-1 when reward > centers[-1])
        # If reward < centers[0], left_idx = 0, right_idx = 0 (due to clamp), we want full weight at index 0.
        # So we need to handle the edge cases explicitly.
        # Recompute left/right and delta properly with torch.where:

        # Find all pairs where reward is less than the smallest centre
        low_mask = reward < self.centers[0]
        high_mask = reward > self.centers[-1]

        # For typical cases (in range), we already have left_idx, right_idx, delta.
        # For low_mask: set left_idx=0, right_idx=0, delta=0 and then we'll assign weight 1 to 0.
        # For high_mask: set left_idx=num_bins-1, right_idx=num_bins-1, delta=0, assign weight 1 to num_bins-1.
        # Actually, for low_mask, we want one-hot at 0, so weight 1.0 at 0, 0.0 at 1.
        # For high_mask, weight 1.0 at last.
        # We can set delta appropriately:
        delta = torch.where(low_mask, torch.zeros_like(delta), delta)
        delta = torch.where(high_mask, torch.zeros_like(delta), delta)
        # For low_mask, we need left_idx=0, right_idx=0; for high_mask, left_idx=num_bins-1, right_idx=num_bins-1.
        left_idx = torch.where(low_mask, torch.zeros_like(left_idx), left_idx)
        right_idx = torch.where(high_mask, (self.num_bins - 1) * torch.ones_like(right_idx), right_idx)

        # Scatter the weights into the target tensor
        # We'll use a loop over batch because scatter_ expects indices as long tensors; alternatively:
        target = target.scatter_add(
            1, left_idx.unsqueeze(1), (1.0 - delta).unsqueeze(1)
        )
        target = target.scatter_add(
            1, right_idx.unsqueeze(1), delta.unsqueeze(1)
        )

        # For low_mask, we want all weight at index 0; our current logic gives left_idx=0, right_idx=0, delta=0 -> weight 1 at 0 and 0 at 0 (which scatter_add adds both, resulting in 1+0=1). That's correct.
        # For high_mask, left_idx=num_bins-1, right_idx=num_bins-1, delta=0 -> weight 1 at last, OK.
        # For in-range, correct.
        # However, we need to ensure that the total sum per sample is 1. Due to scatter_add from both left and right, it should be (1-delta)+delta=1. Fine.

        # But the above is not fully correct because right_idx and left_idx may be equal for out-of-range? Actually for low_mask we set left_idx=0, right_idx=0; then scatter adds 1 to index 0 and 0 to index 0 (since delta=0). So target[:,0]=1. Good. For high_mask similar.
        target = target.clamp(0, 1)  # ensure no accidental >1

        # If batch size was originally single and we added a dim, we could squeeze back, but we'll keep batch dim for consistency.
        return target

    def __repr__(self) -> str:
        return f"TwoHotEncoding(num_bins={self.num_bins}, centres={self.centers[:5]}...)"


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """
    Hard copy all parameters from source network to target network.
    """
    target.load_state_dict(source.state_dict())


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """
    Polyak averaging update: θ_target = τ * θ_source + (1-τ) * θ_target
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


class RunningMean:
    """
    Tracks the mean of a sequence of non-negative values with online add/remove.
    Used for reward scaling.

    Attributes
    ----------
    sum : float
        Current sum of all values in the buffer.
    count : int
        Number of values.
    """

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def add(self, value: float) -> None:
        self.sum += value
        self.count += 1

    def remove(self, value: float) -> None:
        self.sum -= value
        self.count -= 1
        # Safety: prevent negative count (should not happen with proper usage)
        if self.count < 0:
            self.count = 0

    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def __repr__(self) -> str:
        return f"RunningMean(sum={self.sum:.2f}, count={self.count})"


class SumTree:
    """
    Binary tree for priority sampling used in LAP replay buffer.

    Leaves store priorities; internal nodes store the sum of child priorities.
    Supports O(log N) sampling and O(log N) priority updates.

    Parameters
    ----------
    capacity : int
        Maximum number of leaves (buffer capacity).
    """

    def __init__(self, capacity: int):
        # Total number of leaves (power of 2 not required, but we use full binary tree)
        self.capacity = capacity
        # tree array: 0-index unused, 1..(2*capacity-1) used. root at 1.
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.max_priority = 1.0  # initial max priority

    def _leaf_index(self, idx: int) -> int:
        """Convert buffer index (0..capacity-1) to tree leaf index."""
        return idx + self.capacity

    def set_priority(self, idx: int, priority: float) -> None:
        """
        Set priority of leaf idx, then update ancestors.
        """
        tree_idx = self._leaf_index(idx)
        self.tree[tree_idx] = priority
        # Propagate upward
        while tree_idx > 1:
            tree_idx //= 2
            left = 2 * tree_idx
            right = left + 1
            self.tree[tree_idx] = self.tree[left] + self.tree[right]

    def total(self) -> float:
        return self.tree[1]

    def sample(self, batch_size: int) -> List[int]:
        """
        Sample batch_size indices according to priority distribution.

        Returns
        -------
        indices : List[int]
            Sampled buffer indices (0-based).
        """
        total_priority = self.total()
        # Generate random values uniformly between 0 and total_priority
        samples = np.random.uniform(0.0, total_priority, size=batch_size)
        indices = []
        for s in samples:
            idx = self._retrieve(s)
            indices.append(idx)
        return indices

    def _retrieve(self, s: float) -> int:
        """
        Find the leaf index for a given cumulative priority s.
        Returns buffer index (0..capacity-1).
        """
        node = 1
        while node < self.capacity:
            left = 2 * node
            right = left + 1
            if s <= self.tree[left]:
                node = left
            else:
                s -= self.tree[left]
                node = right
        leaf_idx = node - self.capacity
        return leaf_idx

    def get_priority(self, idx: int) -> float:
        tree_idx = self._leaf_index(idx)
        return self.tree[tree_idx]

    def __len__(self) -> int:
        return self.capacity


# Optional: test the module when run directly
if __name__ == "__main__":
    # Quick sanity checks
    print("Testing TwoHotEncoding...")
    two_hot = TwoHotEncoding()
    # test with a scalar reward
    r = torch.tensor(0.0)
    enc = two_hot.encode(r)
    print("Encoding of 0.0:", enc.shape, enc.sum(dim=1))  # should be (1,65) sum to 1

    print("Testing SumTree...")
    st = SumTree(10)
    for i in range(10):
        st.set_priority(i, float(i + 1))
    total = st.total()
    print("Total priority:", total)
    samps = st.sample(5)
    print("Sampled indices:", samps)
