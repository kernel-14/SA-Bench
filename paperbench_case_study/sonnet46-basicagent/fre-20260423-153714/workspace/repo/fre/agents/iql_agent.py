"""
Implicit Q-Learning (IQL) agent conditioned on FRE latent z.

Implements the IQL algorithm (Kostrikov et al., 2021) with:
- Expectile regression for value function (expectile = 0.8)
- Advantage-weighted regression (AWR) for policy (temperature = 3.0)
- Double Q-networks with target network (tau = 0.001)
- Discount factor gamma = 0.88

All networks are conditioned on the FRE latent z.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Optional

from fre.networks.iql_networks import ValueNetwork, TwinQNetwork, GaussianPolicy


def expectile_loss(diff: torch.Tensor, expectile: float = 0.8) -> torch.Tensor:
    """
    Asymmetric L2 loss for expectile regression.

    L_tau(u) = |tau - 1(u < 0)| * u^2
    """
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


class IQLAgent:
    """
    FRE-conditioned IQL agent.

    Trains Q(s, a, z), V(s, z), and pi(a | s, z) using IQL.
    The latent z is provided externally (from the frozen FRE encoder).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = None,
        lr: float = 1e-4,
        gamma: float = 0.88,
        tau: float = 0.001,
        expectile: float = 0.8,
        awr_temperature: float = 3.0,
        device: str = 'cpu',
    ):
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.awr_temperature = awr_temperature
        self.device = device

        # Networks
        self.value_net = ValueNetwork(state_dim, latent_dim, hidden_dims).to(device)
        self.qnet = TwinQNetwork(state_dim, action_dim, latent_dim, hidden_dims).to(device)
        self.qnet_target = copy.deepcopy(self.qnet).to(device)
        self.policy = GaussianPolicy(state_dim, action_dim, latent_dim, hidden_dims).to(device)

        # Freeze target network
        for p in self.qnet_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.qnet.parameters(), lr=lr)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

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
            z:           (B, latent_dim) frozen FRE latent

        Returns:
            dict of scalar losses for logging
        """
        # ---- Value function update (expectile regression) ----
        with torch.no_grad():
            q1_target, q2_target = self.qnet_target(states, actions, z)
            q_target = torch.min(q1_target, q2_target)

        v = self.value_net(states, z)
        value_loss = expectile_loss(q_target - v, self.expectile)

        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()

        # ---- Q-function update (Bellman backup) ----
        with torch.no_grad():
            v_next = self.value_net(next_states, z)
            q_backup = rewards + self.gamma * (1.0 - dones) * v_next

        q1, q2 = self.qnet(states, actions, z)
        q_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # ---- Policy update (AWR) ----
        with torch.no_grad():
            q1_pi, q2_pi = self.qnet_target(states, actions, z)
            q_pi = torch.min(q1_pi, q2_pi)
            v_pi = self.value_net(states, z)
            adv = q_pi - v_pi
            # Advantage-weighted regression weights
            weights = torch.exp(adv / self.awr_temperature).clamp(max=100.0)

        log_prob = self.policy.log_prob(states, z, actions)
        policy_loss = -(weights * log_prob).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # ---- Soft target update ----
        self._soft_update_target()

        return {
            'value_loss': value_loss.item(),
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
        }

    def _soft_update_target(self):
        for p, p_target in zip(self.qnet.parameters(), self.qnet_target.parameters()):
            p_target.data.lerp_(p.data, self.tau)

    @torch.no_grad()
    def act(self, state: np.ndarray, z: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Select action given state and latent z.

        Args:
            state: (state_dim,) numpy array
            z:     (latent_dim,) numpy array
            deterministic: if True, return mean action
        Returns:
            action: (action_dim,) numpy array
        """
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        z_t = torch.FloatTensor(z).unsqueeze(0).to(self.device)
        action = self.policy.act(s, z_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def save(self, path: str):
        torch.save({
            'value_net': self.value_net.state_dict(),
            'qnet': self.qnet.state_dict(),
            'qnet_target': self.qnet_target.state_dict(),
            'policy': self.policy.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.value_net.load_state_dict(ckpt['value_net'])
        self.qnet.load_state_dict(ckpt['qnet'])
        self.qnet_target.load_state_dict(ckpt['qnet_target'])
        self.policy.load_state_dict(ckpt['policy'])
