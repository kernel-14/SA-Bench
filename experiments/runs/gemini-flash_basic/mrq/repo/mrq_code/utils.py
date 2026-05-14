import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def init_weights(m, weight_init, bias_init):
    if isinstance(m, nn.Linear):
        if weight_init == "Xavier uniform":
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif weight_init == "He normal":
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu') # Assuming ReLU for He normal
        else:
            raise ValueError(f"Unknown weight initialization: {weight_init}")
        if m.bias is not None:
            nn.init.constant_(m.bias, bias_init)
    elif isinstance(m, nn.Conv2d):
        if weight_init == "Xavier uniform":
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif weight_init == "He normal":
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
            raise ValueError(f"Unknown weight initialization: {weight_init}")
        if m.bias is not None:
            nn.init.constant_(m.bias, bias_init)


class LinearNormalizedActivation(nn.Module):
    def __init__(self, num_features, activation_fn):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_features)
        self.activation = activation_fn

    def forward(self, x):
        return self.activation(self.layer_norm(x))

def symexp(x, alpha=1.0):
    # Hafner et al., 2023 for symexp
    return torch.sign(x) * (torch.exp(torch.abs(x) * alpha) - 1.0) / alpha

def symlog(x, alpha=1.0):
    return torch.sign(x) * (torch.log(torch.abs(x) * alpha + 1.0) / alpha)

def two_hot_encode(rewards, reward_bins, reward_range):
    # Based on DreamerV3's categorical reward encoding
    # rewards: tensor of shape (batch_size,)
    # reward_bins: integer, number of bins for discretization
    # reward_range: [min_val, max_val] for rewards

    min_val, max_val = reward_range

    # Apply symlog to rewards to handle large magnitudes
    rewards = symlog(rewards)
    min_val = symlog(torch.tensor(min_val, device=rewards.device))
    max_val = symlog(torch.tensor(max_val, device=rewards.device))

    # Normalize rewards to [0, 1] range based on the symmetric log scale
    normalized_rewards = (rewards - min_val) / (max_val - min_val)

    # Scale to [0, reward_bins - 1]
    scaled_rewards = normalized_rewards * (reward_bins - 1)

    # Clip values to ensure they are within the valid bin range
    scaled_rewards = torch.clamp(scaled_rewards, 0, reward_bins - 1)

    # Calculate lower and upper bin indices
    lower_bin = torch.floor(scaled_rewards).long()
    upper_bin = torch.ceil(scaled_rewards).long()

    # Create one-hot encoding for lower and upper bins
    one_hot_lower = F.one_hot(lower_bin, num_classes=reward_bins).float()
    one_hot_upper = F.one_hot(upper_bin, num_classes=reward_bins).float()

    # Calculate the interpolation weights
    # If lower_bin == upper_bin, alpha_upper will be 0 and alpha_lower will be 1
    alpha_upper = scaled_rewards - lower_bin.float()
    alpha_lower = 1.0 - alpha_upper

    # Combine to form two-hot encoding
    two_hot = alpha_lower.unsqueeze(-1) * one_hot_lower + alpha_upper.unsqueeze(-1) * one_hot_upper

    return two_hot

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # From Pytorch documentation / common implementations
    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = F.softmax(gumbels, dim=dim)

    if hard:
        # Straight through estimation
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

