"""
REDQ (Randomized Ensembled Double Q-Learning) agent.

Reference: Chen et al. (2021) "Randomized Ensembled Double Q-Learning: Learning Fast without a Model"

REDQ uses an ensemble of Q-networks with a high UTD (update-to-data) ratio.
At each update step, a random subset of Q-networks is used to compute the target.
This is the primary RL backbone used in PGR experiments.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Network architectures
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Standard MLP with configurable depth and width."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}[activation]
        layers = [nn.Linear(input_dim, hidden_dim), act_fn()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_fn()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class QNetwork(nn.Module):
    """Single Q-network: (obs, action) -> Q-value."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        self.net = MLP(obs_dim + action_dim, 1, hidden_dim, n_layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class GaussianActor(nn.Module):
    """
    Gaussian policy for SAC/REDQ.
    Outputs mean and log_std, with tanh squashing.
    """

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
    ):
        super().__init__()
        self.action_scale = action_scale
        self.action_bias = action_bias

        self.net = MLP(obs_dim, hidden_dim, hidden_dim, n_layers - 1)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob)."""
        h = self.net(obs)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (mean, no sampling)."""
        h = self.net(obs)
        mean = self.mean_layer(h)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(obs)


# ---------------------------------------------------------------------------
# REDQ Agent
# ---------------------------------------------------------------------------

class REDQAgent:
    """
    REDQ agent with configurable ensemble size and UTD ratio.

    Key hyperparameters (from paper):
    - N = 10 Q-networks in ensemble
    - M = 2 randomly selected for target computation
    - UTD = 20 (update-to-data ratio)
    - Batch size = 256
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_q_networks: int = 10,
        n_target_q: int = 2,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        utd_ratio: int = 20,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.n_q_networks = n_q_networks
        self.n_target_q = n_target_q
        self.utd_ratio = utd_ratio
        self.device = torch.device(device)

        # Actor
        self.actor = GaussianActor(obs_dim, action_dim, hidden_dim, n_layers).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Q-network ensemble
        self.q_networks = nn.ModuleList([
            QNetwork(obs_dim, action_dim, hidden_dim, n_layers)
            for _ in range(n_q_networks)
        ]).to(self.device)
        self.q_target_networks = copy.deepcopy(self.q_networks)
        for p in self.q_target_networks.parameters():
            p.requires_grad = False

        self.q_optimizer = torch.optim.Adam(self.q_networks.parameters(), lr=lr_critic)

        # Entropy temperature
        self.auto_alpha = auto_alpha
        if target_entropy is None:
            target_entropy = -action_dim
        self.target_entropy = target_entropy

        if auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.alpha = alpha

        self.total_updates = 0

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.get_action(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
        return action.cpu().numpy().flatten()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single update step for REDQ."""
        obs = batch["obs"]
        action = batch["action"]
        next_obs = batch["next_obs"]
        reward = batch["reward"]
        done = batch["done"]

        # ---- Critic update ----
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)

            # Randomly select M Q-networks for target
            target_q_indices = np.random.choice(self.n_q_networks, self.n_target_q, replace=False)
            target_q_values = torch.stack([
                self.q_target_networks[i](next_obs, next_action)
                for i in target_q_indices
            ], dim=0)
            min_target_q = target_q_values.min(dim=0).values

            target = reward + self.gamma * (1 - done) * (
                min_target_q - self.alpha * next_log_prob
            )

        # Update all Q-networks
        q_losses = []
        for q_net in self.q_networks:
            q_val = q_net(obs, action)
            q_loss = F.mse_loss(q_val, target)
            q_losses.append(q_loss)

        total_q_loss = sum(q_losses)
        self.q_optimizer.zero_grad()
        total_q_loss.backward()
        self.q_optimizer.step()

        # ---- Actor update ----
        new_action, log_prob = self.actor.sample(obs)

        # Use all Q-networks for actor update (mean)
        q_values = torch.stack([
            q_net(obs, new_action) for q_net in self.q_networks
        ], dim=0).mean(dim=0)

        actor_loss = (self.alpha * log_prob - q_values).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Alpha update ----
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            with torch.no_grad():
                _, log_prob = self.actor.sample(obs)
            alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ---- Soft target update ----
        for q_net, q_target in zip(self.q_networks, self.q_target_networks):
            for param, target_param in zip(q_net.parameters(), q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_updates += 1

        return {
            "q_loss": total_q_loss.item() / self.n_q_networks,
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if self.auto_alpha else 0.0,
            "alpha": self.alpha,
        }

    def get_q_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get mean Q-value across ensemble."""
        with torch.no_grad():
            q_values = torch.stack([
                q_net(obs, action) for q_net in self.q_networks
            ], dim=0).mean(dim=0)
        return q_values

    def get_td_error(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TD error for relevance function."""
        with torch.no_grad():
            next_action, _ = self.actor.sample(next_obs)
            next_q = torch.stack([
                self.q_target_networks[i](next_obs, next_action)
                for i in range(self.n_q_networks)
            ], dim=0).min(dim=0).values

            target = reward + self.gamma * (1 - done) * next_q
            current_q = self.get_q_value(obs, action)
            td_error = (target - current_q).abs()
        return td_error

    def compute_dormant_ratio(self, obs: torch.Tensor, threshold: float = 0.025) -> float:
        """
        Compute dormant neuron ratio (Sokar et al., 2023).
        Fraction of neurons with activation below threshold.
        Used in Section 5.2 of the paper.
        """
        total_neurons = 0
        dormant_neurons = 0

        hooks = []
        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach())

        # Register hooks on ReLU/SiLU layers
        for module in self.actor.modules():
            if isinstance(module, (nn.ReLU, nn.SiLU)):
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            self.actor(obs)

        for h in hooks:
            h.remove()

        for act in activations:
            total_neurons += act.numel()
            dormant_neurons += (act.abs() < threshold).sum().item()

        return dormant_neurons / max(total_neurons, 1)

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "q_networks": self.q_networks.state_dict(),
            "q_target_networks": self.q_target_networks.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "log_alpha": self.log_alpha if self.auto_alpha else None,
            "total_updates": self.total_updates,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.q_networks.load_state_dict(checkpoint["q_networks"])
        self.q_target_networks.load_state_dict(checkpoint["q_target_networks"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        if self.auto_alpha and checkpoint["log_alpha"] is not None:
            self.log_alpha = checkpoint["log_alpha"]
        self.total_updates = checkpoint["total_updates"]
