"""
Policy and value function networks for MBPO-PPO.

Architecture (Table S9):
  - Policy: MLP with hidden shape [128, 128, 128], ELU activation
  - Value function: MLP with hidden shape [128, 128, 128], ELU activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    Stochastic policy network for continuous control.

    Outputs mean and log_std of a Gaussian distribution over actions.
    Architecture: 3 hidden layers of size 128, ELU activation.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        in_size = obs_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ELU())
            in_size = hidden_size

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_size, action_size)
        self.log_std_head = nn.Linear(hidden_size, action_size)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, obs_size)

        Returns:
            mean: (batch, action_size)
            std: (batch, action_size)
        """
        h = self.backbone(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: (batch, obs_size)
            deterministic: if True, return mean action

        Returns:
            action: (batch, action_size)
            log_prob: (batch,)
        """
        mean, std = self.forward(obs)
        if deterministic:
            action = mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions.

        Args:
            obs: (batch, obs_size)
            actions: (batch, action_size)

        Returns:
            log_prob: (batch,)
            entropy: (batch,)
            mean: (batch, action_size)
        """
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, mean


class ValueNetwork(nn.Module):
    """
    Value function network for PPO.

    Architecture: 3 hidden layers of size 128, ELU activation.
    """

    def __init__(
        self,
        obs_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.obs_size = obs_size

        layers = []
        in_size = obs_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ELU())
            in_size = hidden_size

        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_size)

        Returns:
            value: (batch, 1)
        """
        h = self.backbone(obs)
        return self.value_head(h)
