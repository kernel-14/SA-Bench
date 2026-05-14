import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

# Assuming config.py is in the same directory or accessible via Python path
from config import Config


def symlog(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Applies the symmetric logarithm function to the input tensor.

    The symmetric logarithm is defined as sign(x) * log(|x| + 1).
    It compresses large values logarithmically while preserving the sign
    and handling values around zero linearly.

    Args:
        x: Input tensor.
        eps: A small epsilon value for numerical stability within log.

    Returns:
        Tensor after applying the symmetric logarithm.
    """
    sign_x = torch.sign(x)
    abs_x = torch.abs(x)
    # Using log1p (log(1+x)) for numerical stability for small x
    log_val = torch.log1p(abs_x + eps) - torch.log1p(torch.tensor(eps, device=x.device)) # adjust for eps
    return sign_x * log_val


def symexp(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Applies the symmetric exponential function to the input tensor.

    The symmetric exponential is the inverse of symlog, defined as
    sign(x) * (exp(|x|) - 1). It expands log-compressed values back to
    their original scale while preserving the sign.

    Args:
        x: Input tensor.
        eps: A small epsilon value for numerical stability within exp.

    Returns:
        Tensor after applying the symmetric exponential.
    """
    sign_x = torch.sign(x)
    abs_x = torch.abs(x)
    # Using expm1 (exp(x)-1) for numerical stability for small x
    exp_val = torch.expm1(abs_x + torch.log1p(torch.tensor(eps, device=x.device))) # adjusted inverse
    return sign_x * exp_val


def reward_to_categorical(rewards: torch.Tensor, config: Config) -> torch.Tensor:
    """Converts a batch of continuous scalar rewards into a two-hot
    categorical distribution over predefined reward bins.

    Args:
        rewards: A tensor of scalar reward values, shape (batch_size,).
        config: Configuration object containing reward_bins and reward_range.

    Returns:
        A torch.Tensor of shape (batch_size, reward_bins) representing
        the two-hot categorical distribution for each reward.
    """
    reward_bins = config.reward_bins
    reward_range_min, reward_range_max = config.reward_range

    # Define linear bins in the symlog space
    linear_bins = torch.linspace(
        reward_range_min, reward_range_max, reward_bins, device=rewards.device
    )

    # Transform rewards to the symlog space and clamp
    logged_rewards = symlog(rewards)
    logged_rewards = torch.clamp(logged_rewards, reward_range_min, reward_range_max)

    # Initialize categorical distribution
    batch_size = rewards.shape[0]
    categorical_dist = torch.zeros(
        (batch_size, reward_bins), device=rewards.device, dtype=rewards.dtype
    )

    # Find indices for bucketization. bucketize requires sorted boundaries.
    # We use linear_bins as boundaries.
    # The 'right=True' argument means [a, b) behavior, which is common.
    # It returns index i such that linear_bins[i-1] < value <= linear_bins[i].
    # So, for the lower bin, we use i-1.
    indices = torch.bucketize(logged_rewards, linear_bins, right=True) - 1

    # Clamp indices to ensure they are within valid bounds [0, reward_bins - 2]
    # (since we need i and i+1, max i can be reward_bins-2)
    indices = torch.clamp(indices, 0, reward_bins - 2)

    # Get values of the lower and upper bins for interpolation
    lower_bin_values = linear_bins[indices]
    upper_bin_values = linear_bins[indices + 1]

    # Calculate interpolation weights
    bin_widths = upper_bin_values - lower_bin_values
    # Add a small epsilon to bin_widths to prevent division by zero for identical bin values,
    # though linspace should prevent this usually.
    weights_upper = (logged_rewards - lower_bin_values) / (bin_widths + 1e-6)
    weights_lower = 1.0 - weights_upper

    # Use scatter_add to create the two-hot encoding
    # Add weights_lower to index `indices`
    categorical_dist.scatter_add_(
        1, indices.unsqueeze(-1), weights_lower.unsqueeze(-1)
    )
    # Add weights_upper to index `indices + 1`
    categorical_dist.scatter_add_(
        1, (indices + 1).unsqueeze(-1), weights_upper.unsqueeze(-1)
    )

    return categorical_dist


def categorical_to_reward(categorical_probs: torch.Tensor, config: Config) -> torch.Tensor:
    """Recovers a scalar reward value from a predicted categorical probability
    distribution over reward bins.

    Args:
        categorical_probs: A tensor of probabilities, shape (batch_size, reward_bins).
                           Typically obtained by applying softmax to the logits.
        config: Configuration object containing reward_bins and reward_range.

    Returns:
        A torch.Tensor of scalar reward values, shape (batch_size,).
    """
    reward_bins = config.reward_bins
    reward_range_min, reward_range_max = config.reward_range

    # Define linear bins in the symlog space
    linear_bins = torch.linspace(
        reward_range_min, reward_range_max, reward_bins, device=categorical_probs.device
    )

    # Transform linear bins back to the original reward space
    bin_centers_in_reward_space = symexp(linear_bins)

    # Calculate the expected reward by summing (probability * bin_center) for each bin
    expected_rewards = torch.sum(
        categorical_probs * bin_centers_in_reward_space, dim=-1
    )

    return expected_rewards


def calculate_huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Computes the Huber loss between a prediction and a target.

    Args:
        pred: Predicted values tensor.
        target: Target values tensor.
        delta: The threshold where the loss transitions from quadratic (L2) to
               linear (L1). Defaults to 1.0.

    Returns:
        A scalar torch.Tensor representing the mean Huber loss across the batch.
    """
    # F.huber_loss provides a clean and numerically stable implementation
    # The 'reduction' argument ensures it returns the mean loss over the batch.
    return F.huber_loss(pred, target, reduction='mean', delta=delta)


class LayerNormActivation(nn.Module):
    """A reusable neural network block that applies Layer Normalization
    followed by an activation function."""

    def __init__(self, features: int, activation_fn: Callable) -> None:
        """Initializes the LayerNormActivation block.

        Args:
            features: The number of features (dimension) for Layer Normalization.
            activation_fn: The activation function to apply (e.g., nn.ReLU, nn.ELU).
                           It should be an instantiated module or a function.
        """
        super().__init__()
        self.norm = nn.LayerNorm(features)
        # Check if activation_fn is a callable function or an nn.Module instance
        if isinstance(activation_fn, nn.Module):
            self.activation = activation_fn
        elif callable(activation_fn):
            # If it's a function (like F.relu), wrap it in an nn.Module for consistency
            self.activation = activation_fn
        else:
            raise ValueError("activation_fn must be a callable or an nn.Module.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through Layer Normalization and activation.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying Layer Normalization and activation.
        """
        x = self.norm(x)
        if isinstance(self.activation, nn.Module):
            return self.activation(x)
        else: # assume it's a function from F
            return self.activation(x)

