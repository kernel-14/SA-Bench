"""Utility functions for MR.Q: symexp, two-hot encoding, reward scaling."""

import torch
import torch.nn.functional as F
import numpy as np


def symexp(x: torch.Tensor) -> torch.Tensor:
    """symexp(x) = sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


def symlog(x: torch.Tensor) -> torch.Tensor:
    """symlog(x) = sign(x) * log(|x| + 1)."""
    return torch.sign(x) * torch.log(x.abs() + 1)


def two_hot_encode(rewards: torch.Tensor, bins: int, low: float, high: float) -> torch.Tensor:
    """
    Compute two-hot encoding of rewards using symlog / symexp bins.

    The bin centers in symlog space are uniformly spaced between `low` and `high`.
    Each reward is transformed via symlog, and its probability mass is split
    between the two nearest bins.
    """
    batch_size = rewards.shape[0]
    device = rewards.device

    symlog_rewards = symlog(rewards)
    symlog_rewards = torch.clamp(symlog_rewards, low, high)

    bin_width = (high - low) / (bins - 1)
    bin_centers = torch.linspace(low, high, bins, device=device)
    indices_float = (symlog_rewards - low) / bin_width  # shape: (B,)
    indices_float = torch.clamp(indices_float, 0, bins - 1)

    low_idx = indices_float.floor().long()
    high_idx = low_idx + 1
    high_weight = indices_float - low_idx.float()
    low_weight = 1.0 - high_weight

    high_idx = torch.clamp(high_idx, 0, bins - 1)

    target = torch.zeros(batch_size, bins, device=device)
    target.scatter_add_(1, low_idx.unsqueeze(1), low_weight.unsqueeze(1))
    target.scatter_add_(1, high_idx.unsqueeze(1), high_weight.unsqueeze(1))
    return target


def get_symexp_bin_centers(bins: int, low: float, high: float) -> torch.Tensor:
    """Return bin centers in original reward space (after symexp)."""
    linear_bins = torch.linspace(low, high, bins)
    return symexp(linear_bins)


class RewardScaler:
    """Tracks the mean absolute reward in the replay buffer for reward scaling."""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.mean = 1.0

    def update(self, rewards: torch.Tensor):
        batch_mean = rewards.abs().mean().item()
        self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean

    def get_scale(self) -> float:
        return max(self.mean, 1e-6)


def clip_action(action: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return torch.clamp(action, low, high)


def one_hot_encode(actions: torch.Tensor, num_actions: int) -> torch.Tensor:
    """Convert integer actions to one-hot vectors."""
    return F.one_hot(actions.long(), num_classes=num_actions).float()
