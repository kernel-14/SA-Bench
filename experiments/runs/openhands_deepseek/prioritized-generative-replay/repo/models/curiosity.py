"""Relevance functions used in PGR: ICM, RND, CTS, ECO.

Each module implements a function F(s, a, s', r) -> scalar relevance value.
Also supports computing intrinsic reward for exploration bonus baselines.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ICM(nn.Module):
    """Intrinsic Curiosity Module (Pathak et al. 2017).

    Relevance function: F(tau) = 1/2 * ||g(h(s), a) - h(s')||^2   (Eq. 5)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        forward_loss_weight: float = 1.0,
        inverse_loss_weight: float = 0.2,
        intrinsic_reward_weight: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.intrinsic_reward_weight = intrinsic_reward_weight

        # Feature encoder h: s -> phi(s)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # Inverse dynamics model: (h(s), h(s')) -> action prediction
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Forward dynamics model g: (h(s), a) -> h(s')
        self.forward_dynamics = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Compute curiosity signal (relevance value) using Eq. (5)."""
        h_state = self.encode(state)
        h_next_state = self.encode(next_state)
        pred_next = self.forward_dynamics(torch.cat([h_state, action], dim=-1))
        return 0.5 * ((pred_next - h_next_state) ** 2).sum(dim=-1, keepdim=True)

    def compute_relevance(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        """Relevance value for PGR conditioning."""
        return self.forward(state, action, next_state)

    def compute_intrinsic_reward(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        """Intrinsic reward for exploration bonus baselines."""
        return self.intrinsic_reward_weight * self.compute_relevance(state, action, next_state)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update ICM: forward + inverse dynamics loss.

        Returns: (forward_loss, inverse_loss)
        """
        h_states = self.encode(states)
        h_next_states = self.encode(next_states)

        # Forward dynamics loss
        pred_next = self.forward_dynamics(torch.cat([h_states, actions], dim=-1))
        forward_loss = 0.5 * ((pred_next - h_next_states) ** 2).mean(dim=-1).mean()

        # Inverse dynamics loss
        pred_actions = self.inverse_dynamics(torch.cat([h_states, h_next_states], dim=-1))
        inverse_loss = F.mse_loss(pred_actions, actions)

        total_loss = self.forward_loss_weight * forward_loss + self.inverse_loss_weight * inverse_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return forward_loss.detach(), inverse_loss.detach()


class PixelICM(nn.Module):
    """ICM for pixel-based observations.

    Uses CNN encoder instead of MLP.
    """

    def __init__(
        self,
        action_dim: int,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        image_channels: int = 3,
        image_size: int = 84,
        lr: float = 1e-3,
        forward_loss_weight: float = 1.0,
        inverse_loss_weight: float = 0.2,
        intrinsic_reward_weight: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.forward_loss_weight = forward_loss_weight
        self.inverse_loss_weight = inverse_loss_weight

        # CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # Forward and inverse dynamics
        self.forward_dynamics = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def compute_relevance(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor:
        h = self.encode(obs)
        h_next = self.encode(next_obs)
        pred_next = self.forward_dynamics(torch.cat([h, action], dim=-1))
        return 0.5 * ((pred_next - h_next) ** 2).sum(dim=-1, keepdim=True)

    def compute_intrinsic_reward(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor:
        return self.intrinsic_reward_weight * self.compute_relevance(obs, action, next_obs)

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(obs)
        h_next = self.encode(next_obs)

        pred_next = self.forward_dynamics(torch.cat([h, actions], dim=-1))
        forward_loss = 0.5 * ((pred_next - h_next) ** 2).mean(dim=-1).mean()

        pred_actions = self.inverse_dynamics(torch.cat([h, h_next], dim=-1))
        inverse_loss = F.mse_loss(pred_actions, actions)

        total_loss = self.forward_loss_weight * forward_loss + self.inverse_loss_weight * inverse_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return forward_loss.detach(), inverse_loss.detach()


class RND(nn.Module):
    """Random Network Distillation (Burda et al. 2018).

    Relevance: F(s, a, s', r) = 1/2 * ||f_hat(s') - f(s')||^2   (Eq. 6)
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 512,
        bottleneck: int = 64,
        lr: float = 1e-3,
        pixel_based: bool = False,
    ):
        super().__init__()
        self.pixel_based = pixel_based

        # Target network f(s) - fixed, randomly initialized
        if pixel_based:
            self.target = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(64 * 11 * 11, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, feature_dim),
            )
            self.predictor = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(64 * 11 * 11, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, feature_dim),
            )
        else:
            self.target = nn.Sequential(
                nn.Linear(input_dim, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, feature_dim),
            )
            self.predictor = nn.Sequential(
                nn.Linear(input_dim, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, feature_dim),
            )

        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    def compute_relevance(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        # RND operates on next_state only (Burda et al. 2018)
        with torch.no_grad():
            target_out = self.target(next_state)
        pred_out = self.predictor(next_state)
        return 0.5 * ((pred_out - target_out) ** 2).sum(dim=-1, keepdim=True)

    def update(self, next_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_out = self.target(next_states)
        pred_out = self.predictor(next_states)
        loss = F.mse_loss(pred_out, target_out)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()


class CTSRelevance(nn.Module):
    """Context-Tree Switching density model for pseudo-counts (Bellemare et al. 2016).

    Relevance: F(s, a, s', r) = (N_hat(s, a) + 0.01)^{-1/2}   (Eq. 7)
    """

    def __init__(self, state_dim: int, action_dim: int, context_bins: int = 8):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_bins = context_bins

        # Simplified CTS: maintain count table
        self.counts = {}
        self.total_count = 0

    def discretize(self, values: torch.Tensor, bins: int) -> torch.Tensor:
        """Discretize continuous values into bins."""
        vmin, vmax = values.min(), values.max()
        if vmin == vmax:
            return torch.zeros_like(values, dtype=torch.long)
        normalized = (values - vmin) / (vmax - vmin + 1e-8)
        return (normalized * bins).long().clamp(0, bins - 1)

    def compute_relevance(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pseudo-count-based relevance following Eq. (7)."""
        batch_size = state.shape[0]
        device = state.device
        relevance = torch.zeros(batch_size, 1, device=device)

        for i in range(batch_size):
            sa_key = (
                tuple(self.discretize(state[i].unsqueeze(0), self.context_bins)[0].tolist()),
                tuple(self.discretize(action[i].unsqueeze(0), self.context_bins)[0].tolist()),
            )
            if sa_key not in self.counts:
                self.counts[sa_key] = 0
            self.counts[sa_key] += 1
            self.total_count += 1

            # Pseudo-count formula: (N_hat(s, a) + 0.01)^{-1/2}
            n_hat = float(self.counts[sa_key]) / max(self.total_count, 1)
            relevance[i] = (n_hat + 0.01) ** (-0.5)

        return relevance

    def update(self, states: torch.Tensor, actions: torch.Tensor):
        """Update counts with new data."""
        pass  # counts updated in compute_relevance


class ECORelevance(nn.Module):
    """Episodic Curiosity (Savinov et al. 2018).

    Relevance: F(tau) = alpha * (beta - F(C(E(s), E(s_i))) for s_i in M   (Eq. 8)
    """

    def __init__(
        self,
        state_dim: int,
        feature_dim: int = 512,
        memory_size: int = 200,
        alpha: float = 0.03,
        beta: float = 0.5,
        percentile: int = 90,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.percentile = percentile
        self.memory_size = memory_size

        # Embedding network E
        self.embedder = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # Comparator network C: predicts if two embeddings are within k steps
        self.comparator = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )

        # Memory buffer M
        self.memory: list = []
        self.reachability_labels: list = []

        self.optimizer = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.comparator.parameters()),
            lr=lr,
        )

    def embed(self, state: torch.Tensor) -> torch.Tensor:
        return self.embedder(state)

    def compute_relevance(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ECO relevance following Eq. (8)."""
        batch_size = state.shape[0]
        device = state.device

        e = self.embed(next_state)  # (B, feature_dim)

        if len(self.memory) == 0:
            return self.alpha * self.beta * torch.ones(batch_size, 1, device=device)

        # Compute similarity to memory
        memory = torch.stack(self.memory, dim=0)  # (M, feature_dim)
        relevance = torch.zeros(batch_size, 1, device=device)

        for i in range(batch_size):
            e_i = e[i].unsqueeze(0).expand(len(self.memory), -1)
            similarities = self.comparator(torch.cat([e_i, memory], dim=-1)).squeeze(-1)
            F_val = torch.quantile(similarities, self.percentile / 100.0)
            relevance[i] = self.alpha * (self.beta - F_val)

        return relevance

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        reachability_label: torch.Tensor,
    ):
        """Update ECO using logistic regression loss."""
        e_s = self.embed(states)
        e_ns = self.embed(next_states)
        cat = torch.cat([e_s, e_ns], dim=-1)
        pred = self.comparator(cat)
        loss = F.binary_cross_entropy(pred, reachability_label.float().unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update memory
        for i in range(next_states.shape[0]):
            self.memory.append(e_ns[i].detach())
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)

        return loss.detach()


class PPOActorCritic(nn.Module):
    """PPO policy for DMLab experiments.

    Used as the backbone when PGR is applied to DMLab (Appendix A.2).
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim

        if len(obs_shape) == 3:
            # Pixel-based
            self.encoder = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, *obs_shape)
                enc_out = self.encoder(dummy).shape[1]
            self.shared = nn.Sequential(
                nn.Linear(enc_out, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            enc_out = obs_shape[0]
            self.encoder = nn.Identity()
            self.shared = nn.Sequential(
                nn.Linear(enc_out, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(obs)
        h = self.shared(x)
        action_mean = self.actor_mean(h)
        value = self.critic(h)
        return action_mean, self.actor_log_std.expand_as(action_mean), value

    def sample(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self.forward(obs)
        dist = Normal(mean, torch.exp(log_std))
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self.forward(obs)
        dist = Normal(mean, torch.exp(log_std))
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, value
