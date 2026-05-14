import torch
import torch.nn as nn
import math


def symexp(x):
    """Symmetric exponential function: sign(x) * (exp(|x|) - 1)"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def symlog(x):
    """Symmetric logarithm: sign(x) * log(|x| + 1)"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def two_hot_encode(value, num_bins=65, symexp_bound=10.0, device=None):
    """
    Two-hot encoding as used in DreamerV3 and MR.Q.
    
    Bins are spaced at increasing non-uniform intervals according to symexp.
    The effective range covers [-22026, 22026] when symexp_bound=10.
    
    Args:
        value: scalar or batch of values
        num_bins: number of bins (must be odd for symmetric range)
        symexp_bound: bound for symexp bin centers
        device: torch device
    
    Returns:
        two-hot encoded tensor of shape (..., num_bins)
    """
    if device is None:
        device = value.device if isinstance(value, torch.Tensor) else torch.device('cpu')
    
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, device=device, dtype=torch.float32)
    
    # Create bin centers uniformly in symlog space, then map to real space via symexp
    half_bins = num_bins // 2
    bin_centers_symlog = torch.linspace(-symexp_bound, symexp_bound, num_bins, device=device)
    bin_centers = symexp(bin_centers_symlog)
    
    # Compute distances to bin centers
    value_expanded = value.unsqueeze(-1)  # shape: (..., 1)
    bin_centers_expanded = bin_centers.view(*([1] * (value.dim())), -1)  # shape: (1, ..., num_bins)
    
    diff = value_expanded - bin_centers_expanded
    abs_diff = torch.abs(diff)
    
    # Find the two closest bins
    topk_vals, topk_indices = torch.topk(abs_diff, k=2, dim=-1, largest=False)
    
    # Compute weights (linear interpolation)
    total_dist = topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    weights = 1.0 - topk_vals / total_dist
    
    # Create output
    output = torch.zeros(*value.shape, num_bins, device=device)
    output.scatter_(-1, topk_indices, weights)
    
    return output


def get_output_dim(action_space_type, action_dim, zsa_dim, num_reward_bins):
    """
    Compute the output dimension of the MDP predictor.
    Predicts next state embedding (zsa_dim), reward (num_reward_bins), and terminal (1).
    """
    return zsa_dim + num_reward_bins + 1


def reward_scaling(replay_buffer_avg_reward, target_avg_reward=None):
    """Compute the reward scaling factor."""
    if target_avg_reward is None:
        target_avg_reward = replay_buffer_avg_reward
    return target_avg_reward


class LayerNormActiv(nn.Module):
    """Layer normalization followed by activation, used throughout MR.Q networks."""
    
    def __init__(self, normalized_shape, activation):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)
        self.activation = activation
    
    def forward(self, x):
        return self.activation(self.layer_norm(x))


def ln_activ(x, activ):
    """Functional version: apply LayerNorm then activation."""
    norm = nn.LayerNorm(x.shape[-1:], device=x.device, dtype=x.dtype)
    return activ(norm(x))
