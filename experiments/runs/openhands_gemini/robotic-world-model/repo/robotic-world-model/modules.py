
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple

from layers import MLP
from config import GlobalConfig

class RWMBase(nn.Module):
    """
    The base GRU-based architecture for the Robotic World Model (RWM).
    It processes concatenated observation-action pairs sequentially.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        # Input to GRU is concatenation of observation and action
        self.input_dim = obs_dim + action_dim
        self.gru = nn.GRU(self.input_dim, hidden_size, batch_first=True)

    def forward(self, obs_actions: torch.Tensor, hidden_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GRU.
        Args:
            obs_actions (torch.Tensor): A sequence of concatenated observations and actions.
                                        Shape: (batch_size, sequence_length, obs_dim + action_dim)
            hidden_state (torch.Tensor, optional): Initial hidden state for the GRU.
                                                   Shape: (1, batch_size, hidden_size)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output features from GRU and final hidden state.
                                               Output features shape: (batch_size, sequence_length, hidden_size)
                                               Hidden state shape: (1, batch_size, hidden_size)
        """
        if obs_actions.dim() == 2: # Handle single step input by adding sequence_length dimension
            obs_actions = obs_actions.unsqueeze(1)

        output, hidden = self.gru(obs_actions, hidden_state)
        return output, hidden

class RWMHeads(nn.Module):
    """
    MLP heads for the Robotic World Model, predicting the mean and standard deviation
    of the next observation and privileged information.
    """
    def __init__(self, input_size: int, obs_dim: int, priv_info_dim: int, hidden_size: int, activation: str):
        super().__init__()
        self.obs_dim = obs_dim
        self.priv_info_dim = priv_info_dim

        # MLP for observation mean
        self.obs_mean_mlp = MLP(input_size, obs_dim, [hidden_size], activation)
        # MLP for observation log standard deviation (predict log_std for stability)
        self.obs_log_std_mlp = MLP(input_size, obs_dim, [hidden_size], activation)

        # MLP for privileged information mean
        self.priv_info_mean_mlp = MLP(input_size, priv_info_dim, [hidden_size], activation)
        # MLP for privileged information log standard deviation
        self.priv_info_log_std_mlp = MLP(input_size, priv_info_dim, [hidden_size], activation)

    def forward(self, features: torch.Tensor) -> Tuple[Normal, Normal]:
        """
        Forward pass through the MLP heads to predict distributions.
        Args:
            features (torch.Tensor): Features from the GRU base.
                                     Shape: (batch_size, hidden_size) or (batch_size, sequence_length, hidden_size)
        Returns:
            Tuple[Normal, Normal]: Normal distributions for next observation and privileged information.
        """
        # Ensure features are 2D (batch_size, hidden_size) if they come from a sequence
        if features.dim() == 3:
            features = features[:, -1, :] # Take the last step's features

        obs_mean = self.obs_mean_mlp(features)
        obs_log_std = self.obs_log_std_mlp(features)
        obs_std = torch.exp(obs_log_std.clamp(min=-20, max=2)) # Clamp for numerical stability

        priv_info_mean = self.priv_info_mean_mlp(features)
        priv_info_log_std = self.priv_info_log_std_mlp(features)
        priv_info_std = torch.exp(priv_info_log_std.clamp(min=-20, max=2))

        return Normal(obs_mean, obs_std), Normal(priv_info_mean, priv_info_std)

class PolicyNetwork(nn.Module):
    """
    MLP-based policy network for continuous control.
    Outputs mean and log standard deviation for a Gaussian policy.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list, activation: str):
        super().__init__()
        self.action_dim = action_dim
        self.mlp = MLP(obs_dim, action_dim * 2, hidden_sizes, activation) # Output action_dim for mean and action_dim for log_std

    def forward(self, obs: torch.Tensor) -> Normal:
        """
        Forward pass through the policy network.
        Args:
            obs (torch.Tensor): Observation from the environment.
                                Shape: (batch_size, obs_dim)
        Returns:
            Normal: Gaussian distribution for the actions.
        """
        output = self.mlp(obs)
        mean, log_std = torch.split(output, self.action_dim, dim=-1)
        std = torch.exp(log_std.clamp(min=-20, max=2)) # Clamp for numerical stability
        return Normal(mean, std)

class ValueNetwork(nn.Module):
    """
    MLP-based value network.
    Outputs a single scalar value estimate for the given observation.
    """
    def __init__(self, obs_dim: int, hidden_sizes: list, activation: str):
        super().__init__()
        self.mlp = MLP(obs_dim, 1, hidden_sizes, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value network.
        Args:
            obs (torch.Tensor): Observation from the environment.
                                Shape: (batch_size, obs_dim)
        Returns:
            torch.Tensor: Scalar value estimate. Shape: (batch_size, 1)
        """
        return self.mlp(obs)

