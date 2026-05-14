## utils.py
import random
import numpy as np
import torch
import math
from typing import Union, Tuple

def gaussian_nll_loss(
    mean: torch.Tensor, log_std: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the Negative Log Likelihood (NLL) loss for a Gaussian distribution.
    This is used for the RWM's observation and privileged information prediction losses.

    Args:
        mean: Predicted mean of the Gaussian distribution (batch_size, dim).
        log_std: Predicted log standard deviation of the Gaussian distribution (batch_size, dim).
        target: Ground truth values (batch_size, dim).

    Returns:
        A tensor representing the NLL loss for each sample in the batch, summed over dimensions.
    """
    # Clamp log_std for numerical stability to prevent exp(log_std) from becoming too large/small
    # Common range might be -20 to 2, corresponding to std from e^-20 to e^2
    log_std = torch.clamp(log_std, min=-20.0, max=2.0)
    
    # Calculate the variance
    variance = torch.exp(2 * log_std)
    
    # Calculate the NLL loss
    # Formula: 0.5 * (log(2 * pi * variance) + (target - mean)^2 / variance)
    # log(2 * pi * variance) = log(2 * pi) + log(variance) = log(2 * pi) + 2 * log_std
    loss = 0.5 * (math.log(2 * math.pi) + 2 * log_std + ((target - mean) ** 2 / variance))
    
    # Sum over the feature dimensions, return per-sample loss
    return torch.sum(loss, dim=-1)


def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility across numpy, torch, and Python's random module.

    Args:
        seed: The integer seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic operations for CUDA if possible, might slightly impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # For PyTorch 2.0+ and newer versions, might also set:
    # torch.use_deterministic_algorithms(True) # Requires specific ops to be deterministic

    # Set hash environment variable for reproducibility in some cases
    # import os
    # os.environ['PYTHONHASHSEED'] = str(seed)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Generalized Advantage Estimation (GAE) and returns for a trajectory segment.

    Args:
        rewards: Tensor of rewards for each step (T,).
        values: Tensor of value predictions for each state (T,).
        next_values: Tensor of value predictions for the next state of each step (T,).
                     Note: For the last step, next_values should be the value of the
                     terminal state if not done, or 0 if done.
        dones: Tensor indicating if an episode terminated at each step (T,).
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.

    Returns:
        A tuple containing:
        - advantages: Tensor of advantages for each step (T,).
        - returns: Tensor of discounted returns for each step (T,).
    """
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    last_gae_lambda = 0.0

    # Ensure dones is float for calculations
    dones_float = dones.float()

    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            # If at the end of the rollout, use actual next_value (can be 0 for terminal state)
            next_non_terminal = 1.0 - dones_float[t]
            next_value = next_values[t]
        else:
            # Otherwise, use the value of the next state in the rollout
            next_non_terminal = 1.0 - dones_float[t]
            next_value = values[t+1] # This is typically V(s_{t+1})

        # TD error (delta)
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        
        # GAE calculation
        advantages[t] = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
        last_gae_lambda = advantages[t]
    
    # Calculate returns (V(s) + A(s,a))
    returns = advantages + values
    
    return advantages, returns


def normalize_obs(
    obs: np.ndarray, mean: np.ndarray, std: np.ndarray, epsilon: float = 1e-8
) -> np.ndarray:
    """
    Normalizes observations using pre-computed mean and standard deviation.

    Args:
        obs: Raw observation data (e.g., (batch_size, obs_dim) or (obs_dim,)).
        mean: Mean of each observation dimension (obs_dim,).
        std: Standard deviation of each observation dimension (obs_dim,).
        epsilon: Small value to prevent division by zero.

    Returns:
        Normalized observations.
    """
    return (obs - mean) / (std + epsilon)


def unnormalize_action(
    action: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """
    Unnormalizes actions using pre-computed mean and standard deviation.

    Args:
        action: Normalized action data from the policy.
        mean: Mean of the action space dimensions.
        std: Standard deviation of the action space dimensions.

    Returns:
        Unnormalized actions.
    """
    return action * std + mean

