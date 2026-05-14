"""
Relevance functions F(tau) for Prioritized Generative Replay.

Three main variants from the paper:
1. Return-based: F(s,a,s',r) = Q(s, pi(s))
2. TD-error-based: F(s,a,s',r) = r + gamma*Q_target(s', argmax Q(s',a')) - Q(s,a)
3. Curiosity-based (ICM): F(s,a,s',r) = 0.5 * ||g(h(s), a) - h(s')||^2
   where h is a feature encoder and g is a forward dynamics model.

The curiosity-based relevance function is the recommended default (Section 4.2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Curiosity-based relevance (Intrinsic Curiosity Module, Pathak et al. 2017)
# ---------------------------------------------------------------------------

class CuriosityRelevance(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) as a relevance function.

    Learns:
    - Feature encoder h: s -> z
    - Forward dynamics model g: (h(s), a) -> h(s')_pred
    - Inverse dynamics model (optional, for better features): (h(s), h(s')) -> a_pred

    Relevance: F(s,a,s',r) = 0.5 * ||g(h(s), a) - h(s')||^2  (Eq. 5 in paper)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_dim: int = 64,
        hidden_dim: int = 256,
        use_inverse_model: bool = True,
        inverse_loss_weight: float = 0.2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.use_inverse_model = use_inverse_model
        self.inverse_loss_weight = inverse_loss_weight

        # Feature encoder h: obs -> feature
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Forward dynamics model g: (feature, action) -> next_feature
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Inverse dynamics model (optional): (feature, next_feature) -> action
        if use_inverse_model:
            self.inverse_model = nn.Sequential(
                nn.Linear(feature_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def forward_dynamics(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward_model(torch.cat([feat, action], dim=-1))

    def compute_relevance(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute curiosity relevance (Eq. 5):
            F(s,a,s',r) = 0.5 * ||g(h(s), a) - h(s')||^2

        Returns: [B] scalar relevance values
        """
        with torch.no_grad():
            feat = self.encoder(obs)
            next_feat = self.encoder(next_obs)
            pred_next_feat = self.forward_model(torch.cat([feat, action], dim=-1))
            relevance = 0.5 * ((pred_next_feat - next_feat) ** 2).sum(dim=-1)
        return relevance

    def compute_loss(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ICM training loss:
        - Forward model loss: MSE between predicted and actual next features
        - Inverse model loss (optional): MSE between predicted and actual actions
        """
        feat = self.encoder(obs)
        next_feat = self.encoder(next_obs)
        pred_next_feat = self.forward_model(torch.cat([feat, action], dim=-1))

        forward_loss = F.mse_loss(pred_next_feat, next_feat.detach())

        if self.use_inverse_model:
            pred_action = self.inverse_model(torch.cat([feat, next_feat], dim=-1))
            inverse_loss = F.mse_loss(pred_action, action)
            total_loss = (1 - self.inverse_loss_weight) * forward_loss + \
                         self.inverse_loss_weight * inverse_loss
        else:
            total_loss = forward_loss

        return total_loss


# ---------------------------------------------------------------------------
# Return-based relevance
# ---------------------------------------------------------------------------

class ReturnRelevance(nn.Module):
    """
    Return-based relevance function (Eq. 3 in paper):
        F(s,a,s',r) = Q(s, pi(s))

    Uses the current Q-function and policy to estimate value.
    """

    def __init__(self):
        super().__init__()

    def compute_relevance(
        self,
        obs: torch.Tensor,
        q_network,
        actor,
    ) -> torch.Tensor:
        """
        Args:
            obs: [B, obs_dim]
            q_network: callable (obs, action) -> Q-value
            actor: callable obs -> action
        Returns: [B] relevance values
        """
        with torch.no_grad():
            action = actor(obs)
            q_val = q_network(obs, action)
            if isinstance(q_val, (list, tuple)):
                q_val = torch.min(*q_val) if len(q_val) == 2 else q_val[0]
        return q_val.squeeze(-1)

    def compute_loss(self, *args, **kwargs):
        return torch.tensor(0.0)


# ---------------------------------------------------------------------------
# TD-error-based relevance
# ---------------------------------------------------------------------------

class TDErrorRelevance(nn.Module):
    """
    TD-error-based relevance function (Eq. 4 in paper):
        F(s,a,s',r) = r + gamma * Q_target(s', argmax Q(s',a')) - Q(s,a)
    """

    def __init__(self, gamma: float = 0.99):
        super().__init__()
        self.gamma = gamma

    def compute_relevance(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        q_network,
        q_target,
    ) -> torch.Tensor:
        """
        Args:
            obs, action, next_obs, reward, done: transition components
            q_network: current Q-network (obs, action) -> Q
            q_target: target Q-network (obs, action) -> Q
        Returns: [B] TD-error values (absolute)
        """
        with torch.no_grad():
            q_current = q_network(obs, action)
            if isinstance(q_current, (list, tuple)):
                q_current = torch.min(*q_current) if len(q_current) == 2 else q_current[0]

            # Get best next action
            # For continuous action spaces, use actor; here we approximate
            # by using the target network's greedy action via the current policy
            next_q = q_target(next_obs, action)  # simplified; ideally use actor
            if isinstance(next_q, (list, tuple)):
                next_q = torch.min(*next_q) if len(next_q) == 2 else next_q[0]

            target = reward + self.gamma * (1 - done) * next_q
            td_error = (target - q_current).abs().squeeze(-1)
        return td_error

    def compute_loss(self, *args, **kwargs):
        return torch.tensor(0.0)


# ---------------------------------------------------------------------------
# Reward-based relevance (naive baseline from paper)
# ---------------------------------------------------------------------------

class RewardRelevance(nn.Module):
    """
    Simple reward-based relevance (naive baseline, shown to underperform in paper):
        F(s,a,s',r) = r
    """

    def __init__(self):
        super().__init__()

    def compute_relevance(self, reward: torch.Tensor) -> torch.Tensor:
        return reward.squeeze(-1)

    def compute_loss(self, *args, **kwargs):
        return torch.tensor(0.0)
