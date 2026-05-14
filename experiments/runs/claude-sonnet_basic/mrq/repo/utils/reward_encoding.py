"""
Reward encoding utilities for MR.Q.

Uses symexp-spaced bins and two-hot encoding as described in:
"Towards General-Purpose Model-Free RL (MR.Q)" - Fujimoto et al., 2025

The symexp transformation: symexp(x) = sign(x) * (exp(|x|) - 1)
maps uniform bin indices to non-uniform reward ranges, allowing
the categorical representation to handle a wide range of reward magnitudes.
"""

import torch
import numpy as np


def symexp(x):
    """Symmetric exponential: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def symlog(x):
    """Inverse of symexp: sign(x) * log(|x| + 1)."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp_np(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


def symlog_np(x):
    return np.sign(x) * np.log(np.abs(x) + 1)


def get_reward_bins(num_bins=65, reward_range=(-10.0, 10.0)):
    """
    Compute reward bin locations using symexp spacing.
    
    Bins are uniformly spaced in symlog space over [reward_range[0], reward_range[1]],
    then mapped back via symexp. This gives non-uniform spacing in reward space,
    with finer resolution near zero and coarser at extremes.
    
    With range=[-10, 10] and symexp, effective range is approximately [-22k, 22k].
    
    Args:
        num_bins: Number of bins (default 65 from paper)
        reward_range: Range in symlog space (default [-10, 10])
    
    Returns:
        bins: Tensor of shape (num_bins,) with bin locations in reward space
    """
    # Uniform spacing in symlog space
    symlog_bins = np.linspace(reward_range[0], reward_range[1], num_bins)
    # Map to reward space via symexp
    bins = symexp_np(symlog_bins)
    return torch.FloatTensor(bins)


def two_hot_encode(rewards, bins):
    """
    Two-hot encoding of rewards.
    
    For each reward value, finds the two adjacent bins and assigns
    weights proportional to proximity (linear interpolation).
    
    Args:
        rewards: Tensor of shape (batch,) or (batch, 1)
        bins: Tensor of shape (num_bins,) - bin locations
    
    Returns:
        two_hot: Tensor of shape (batch, num_bins)
    """
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)
    
    bins = bins.to(rewards.device)
    num_bins = len(bins)
    batch_size = rewards.shape[0]
    
    # Clamp rewards to bin range
    rewards_clamped = torch.clamp(rewards, bins[0], bins[-1])
    
    # Find lower bin index for each reward
    # bins[lower] <= reward < bins[lower+1]
    lower = torch.searchsorted(bins, rewards_clamped, right=True) - 1
    lower = torch.clamp(lower, 0, num_bins - 2)
    upper = lower + 1
    
    # Compute interpolation weights
    bin_lower = bins[lower]
    bin_upper = bins[upper]
    
    # Weight for upper bin (how close to upper)
    upper_weight = (rewards_clamped - bin_lower) / (bin_upper - bin_lower + 1e-8)
    upper_weight = torch.clamp(upper_weight, 0.0, 1.0)
    lower_weight = 1.0 - upper_weight
    
    # Create two-hot encoding
    two_hot = torch.zeros(batch_size, num_bins, device=rewards.device)
    two_hot.scatter_(1, lower.unsqueeze(1), lower_weight.unsqueeze(1))
    two_hot.scatter_(1, upper.unsqueeze(1), upper_weight.unsqueeze(1))
    
    return two_hot


def decode_reward(logits, bins):
    """
    Decode reward from categorical logits using expected value.
    
    Args:
        logits: Tensor of shape (batch, num_bins)
        bins: Tensor of shape (num_bins,)
    
    Returns:
        rewards: Tensor of shape (batch, 1)
    """
    bins = bins.to(logits.device)
    probs = torch.softmax(logits, dim=-1)
    return (probs * bins.unsqueeze(0)).sum(dim=-1, keepdim=True)
