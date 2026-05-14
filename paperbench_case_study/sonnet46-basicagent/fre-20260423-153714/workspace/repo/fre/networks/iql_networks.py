"""
IQL network components for FRE.

All networks are conditioned on the latent z by concatenating z to the
observation state (per addendum: "the latent embedding is simply concatenated
to the observation state").

Network architecture: MLP with layers [512, 512, 512] (from Table 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(input_dim: int, hidden_dims: list, output_dim: int, activate_final: bool = False):
    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    if activate_final:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ValueNetwork(nn.Module):
    """V(s, z) - state value function conditioned on latent z."""

    def __init__(self, state_dim: int, latent_dim: int = 128, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]
        self.net = mlp(state_dim + latent_dim, hidden_dims, 1)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, z], dim=-1)
        return self.net(x).squeeze(-1)


class QNetwork(nn.Module):
    """Q(s, a, z) - action-value function conditioned on latent z."""

    def __init__(
        self, state_dim: int, action_dim: int, latent_dim: int = 128, hidden_dims: list = None
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]
        self.net = mlp(state_dim + action_dim + latent_dim, hidden_dims, 1)

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([states, actions, z], dim=-1)
        return self.net(x).squeeze(-1)


class TwinQNetwork(nn.Module):
    """Two Q-networks for double Q-learning."""

    def __init__(
        self, state_dim: int, action_dim: int, latent_dim: int = 128, hidden_dims: list = None
    ):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, latent_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, latent_dim, hidden_dims)

    def forward(self, states, actions, z):
        return self.q1(states, actions, z), self.q2(states, actions, z)

    def min(self, states, actions, z):
        q1, q2 = self.forward(states, actions, z)
        return torch.min(q1, q2)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy pi(a | s, z).

    Outputs mean and log_std of a Gaussian distribution over actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = None,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        in_dim = state_dim + latent_dim
        layers = []
        cur_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur_dim, h))
            layers.append(nn.ReLU())
            cur_dim = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(cur_dim, action_dim)
        self.log_std_head = nn.Linear(cur_dim, action_dim)

    def forward(self, states: torch.Tensor, z: torch.Tensor):
        x = torch.cat([states, z], dim=-1)
        h = self.trunk(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def act(self, states: torch.Tensor, z: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.forward(states, z)
        if deterministic:
            return mean
        std = log_std.exp()
        return mean + std * torch.randn_like(std)

    def log_prob(self, states: torch.Tensor, z: torch.Tensor, actions: torch.Tensor):
        mean, log_std = self.forward(states, z)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(-1)
