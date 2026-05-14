"""PPO Actor and Critic networks for MBPO-PPO policy optimization.

Architectures follow Table S9:
  - Policy (actor): MLP with hidden shape [128, 128, 128], ELU activation
  - Value function (critic): MLP with hidden shape [128, 128, 128], ELU activation

Both use the policy observation space defined in Table S5.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import build_mlp


class PPOActor(nn.Module):
    """Gaussian policy network for continuous action space.

    Outputs mean and log-standard-deviation of action distribution.
    Uses ELU activation throughout.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_shape: Tuple[int, ...] = (128, 128, 128),
        activation: str = "elu",
        log_std_init: float = 0.0,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.backbone = build_mlp(
            input_dim=obs_dim,
            output_dim=hidden_shape[-1],
            hidden_dims=list(hidden_shape[:-1]),
            activation=activation,
        )

        self.mean_head = nn.Linear(hidden_shape[-1], action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        log_std = log_std.expand_as(mean)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from the Gaussian policy.

        Returns:
            action: sampled action
            log_prob: log probability of sampled action
            mean: action mean
        """
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, mean

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy of given actions under current policy.

        Returns:
            log_prob: (B,)
            entropy: (B,)
        """
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def get_entropy(self, obs: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        return dist.entropy().sum(dim=-1)


class PPOCritic(nn.Module):
    """Value function network for PPO.

    Maps observations to scalar state values.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_shape: Tuple[int, ...] = (128, 128, 128),
        activation: str = "elu",
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim=obs_dim,
            output_dim=1,
            hidden_dims=list(hidden_shape),
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
