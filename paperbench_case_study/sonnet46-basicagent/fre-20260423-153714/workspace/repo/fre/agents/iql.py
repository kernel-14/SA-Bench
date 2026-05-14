"""
Implicit Q-Learning (IQL) agent for FRE.

Implements IQL (Kostrikov et al., 2021) with FRE latent conditioning.
All networks (Q, V, policy) are conditioned on the latent z.

Hyperparameters (from Table 3):
- discount: 0.88
- expectile: 0.8
- AWR temperature: 3.0
- target update rate: 0.001
- learning rate: 0.0001
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Optional

from fre.networks.iql_networks import TwinQNetwork, ValueNetwork, GaussianPolicy


def expectile_loss(diff: torch.Tensor, expectile: float = 0.8) -> torch.Tensor:
    """Asymmetric L2 loss for IQL value function training."""
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


class IQLAgent:
    """
    IQL agent conditioned on FRE latent z.

    Trains Q(s, a, z), V(s, z), and pi(a | s, z) using offline IQL.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = None,
        discount: float = 0.88,
        expectile: float = 0.8,
        temperature: float = 3.0,
        tau: float = 0.001,
        lr: float = 1e-4,
        device: str = 'cpu',
    ):
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        self.discount = discount
        self.expectile = expectile
        self.temperature = temperature
        self.tau = tau
        self.device = device

        # Networks
        self.q_net = TwinQNetwork(state_dim, action_dim, latent_dim, hidden_dims).to(device)
        self.q_target = copy.deepcopy(self.q_net).to(device)
        self.v_net = ValueNetwork(state_dim, latent_dim, hidden_dims).to(device)
        self.policy = GaussianPolicy(state_dim, action_dim, latent_dim, hidden_dims).to(device)

        # Freeze target network
        for p in self.q_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        z: torch.Tensor,
    ) -> dict:
        """
        One IQL update step.

        Args:
            states:      (B, state_dim)
            actions:     (B, action_dim)
            rewards:     (B,)
            next_states: (B, state_dim)
            dones:       (B,) float, 1 if terminal
            z:           (B, latent_dim) - frozen FRE encoding
        Returns:
            dict of loss values for logging
        """
        # ---- Value function update ----
        with torch.no_grad():
            q1_target, q2_target = self.q_target(states, actions, z)
            q_target = torch.min(q1_target, q2_target)

        v_pred = self.v_net(states, z)
        v_loss = expectile_loss(q_target - v_pred, self.expectile)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # ---- Q-function update ----
        with torch.no_grad():
            v_next = self.v_net(next_states, z)
            q_backup = rewards + self.discount * (1.0 - dones) * v_next

        q1_pred, q2_pred = self.q_net(states, actions, z)
        q_loss = F.mse_loss(q1_pred, q_backup) + F.mse_loss(q2_pred, q_backup)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ---- Policy update (AWR) ----
        with torch.no_grad():
            q1_t, q2_t = self.q_target(states, actions, z)
            q_t = torch.min(q1_t, q2_t)
            v_t = self.v_net(states, z)
            adv = q_t - v_t
            weights = torch.exp(adv / self.temperature).clamp(max=100.0)

        log_prob = self.policy.log_prob(states, z, actions)
        policy_loss = -(weights * log_prob).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ---- Soft target update ----
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        return {
            'v_loss': v_loss.item(),
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
        }

    @torch.no_grad()
    def act(self, state: np.ndarray, z: torch.Tensor, deterministic: bool = True) -> np.ndarray:
        """Select action given state and latent z."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.policy.act(state_t, z.unsqueeze(0), deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def save(self, path: str):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'v_net': self.v_net.state_dict(),
            'policy': self.policy.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.q_target = copy.deepcopy(self.q_net)
        self.v_net.load_state_dict(ckpt['v_net'])
        self.policy.load_state_dict(ckpt['policy'])
