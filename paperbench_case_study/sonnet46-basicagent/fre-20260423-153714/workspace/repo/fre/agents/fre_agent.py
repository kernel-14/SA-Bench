"""
FRE Agent: Functional Reward Encoding + IQL.

Implements the two-phase training procedure from Algorithm 1:
1. Train FRE encoder/decoder (frozen RL networks)
2. Freeze encoder, train IQL policy/value/Q networks

The strided training scheme ensures a stationary z mapping during RL training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from fre.networks.encoder import FREEncoder
from fre.networks.decoder import FREDecoder
from fre.networks.iql_networks import ValueNetwork, TwinQNetwork, GaussianPolicy


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Polyak averaging for target network update."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


class FREAgent:
    """
    Full FRE agent combining encoder-decoder VAE with IQL.

    Training is split into two phases:
    - Phase 1: Train encoder + decoder only (encoder_steps)
    - Phase 2: Freeze encoder, train IQL (policy_steps)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        # Encoder hyperparams
        state_embed_dim: int = 64,
        reward_embed_dim: int = 64,
        num_reward_bins: int = 32,
        encoder_layers: int = 4,
        encoder_heads: int = 4,
        encoder_mlp_dim: int = 256,
        # RL hyperparams
        hidden_dims: list = None,
        discount: float = 0.88,
        tau: float = 0.001,
        expectile: float = 0.8,
        awr_temperature: float = 3.0,
        beta_kl: float = 0.01,
        # Encoding context sizes
        K_encode: int = 32,
        K_decode: int = 8,
        # Optimizer
        lr: float = 1e-4,
        device: str = 'cpu',
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.K_encode = K_encode
        self.K_decode = K_decode
        self.discount = discount
        self.tau = tau
        self.expectile = expectile
        self.awr_temperature = awr_temperature
        self.beta_kl = beta_kl
        self.device = torch.device(device)

        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        # Encoder-decoder
        self.encoder = FREEncoder(
            state_dim, latent_dim, state_embed_dim, reward_embed_dim,
            num_reward_bins, encoder_layers, encoder_heads, encoder_mlp_dim
        ).to(self.device)
        self.decoder = FREDecoder(state_dim, latent_dim, hidden_dims=[512, 512, 512]).to(self.device)

        # IQL networks
        self.value_net = ValueNetwork(state_dim, latent_dim, hidden_dims).to(self.device)
        self.q_net = TwinQNetwork(state_dim, action_dim, latent_dim, hidden_dims).to(self.device)
        self.q_target = TwinQNetwork(state_dim, action_dim, latent_dim, hidden_dims).to(self.device)
        self.policy = GaussianPolicy(state_dim, action_dim, latent_dim, hidden_dims).to(self.device)

        # Initialize target networks
        self.q_target.load_state_dict(self.q_net.state_dict())

        # Optimizers
        self.encoder_opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )
        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def encode_reward_function(
        self,
        reward_fn,
        dataset_states: np.ndarray,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Encode a reward function into latent z using K random states from dataset.

        For goal-reaching, ensures at least one sample contains the goal state.

        Args:
            reward_fn: callable (states: np.ndarray) -> rewards: np.ndarray
            dataset_states: (N, state_dim) offline dataset states
            deterministic: if True, return mean of distribution
        Returns:
            z: (1, latent_dim) tensor
        """
        indices = np.random.choice(len(dataset_states), self.K_encode, replace=False)
        enc_states = dataset_states[indices]
        enc_rewards = reward_fn(enc_states)

        # Normalize rewards to [0, 1]
        r_min, r_max = enc_rewards.min(), enc_rewards.max()
        if r_max - r_min > 1e-8:
            enc_rewards_norm = (enc_rewards - r_min) / (r_max - r_min)
        else:
            enc_rewards_norm = np.zeros_like(enc_rewards)

        states_t = torch.FloatTensor(enc_states).unsqueeze(0).to(self.device)
        rewards_t = torch.FloatTensor(enc_rewards_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                z = self.encoder.encode_deterministic(states_t, rewards_t)
            else:
                z = self.encoder.encode(states_t, rewards_t)
        return z

    def train_encoder_step(
        self,
        reward_fn,
        dataset_states: np.ndarray,
    ) -> Dict[str, float]:
        """
        Single training step for the FRE encoder-decoder (Phase 1).

        Maximizes the variational lower bound:
            E[sum_k log q(eta(s_k^d) | s_k^d, z)] - beta * KL(p(z|L^e) || N(0,I))
        """
        # Sample encoder states
        enc_idx = np.random.choice(len(dataset_states), self.K_encode, replace=False)
        enc_states = dataset_states[enc_idx]
        enc_rewards = reward_fn(enc_states)

        # Sample decoder states (different from encoder states)
        dec_idx = np.random.choice(len(dataset_states), self.K_decode, replace=False)
        dec_states = dataset_states[dec_idx]
        dec_rewards = reward_fn(dec_states)

        # Normalize encoder rewards to [0, 1] for discretization
        r_min = enc_rewards.min()
        r_max = enc_rewards.max()
        if r_max - r_min > 1e-8:
            enc_rewards_norm = (enc_rewards - r_min) / (r_max - r_min)
        else:
            enc_rewards_norm = np.zeros_like(enc_rewards)

        # Convert to tensors (add batch dim)
        enc_states_t = torch.FloatTensor(enc_states).unsqueeze(0).to(self.device)
        enc_rewards_t = torch.FloatTensor(enc_rewards_norm).unsqueeze(0).to(self.device)
        dec_states_t = torch.FloatTensor(dec_states).unsqueeze(0).to(self.device)
        dec_rewards_t = torch.FloatTensor(dec_rewards).unsqueeze(0).to(self.device)

        # Encode
        mu, log_std = self.encoder(enc_states_t, enc_rewards_t)
        std = log_std.exp()
        z = mu + std * torch.randn_like(std)

        # Decode: predict rewards for decoder states
        pred_rewards = self.decoder(dec_states_t, z)  # (1, K')

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(pred_rewards, dec_rewards_t)

        # KL divergence: KL(N(mu, std) || N(0, I))
        kl_loss = -0.5 * (1 + 2 * log_std - mu.pow(2) - std.pow(2)).sum(-1).mean()

        loss = recon_loss + self.beta_kl * kl_loss

        self.encoder_opt.zero_grad()
        loss.backward()
        self.encoder_opt.step()

        return {
            'encoder/recon_loss': recon_loss.item(),
            'encoder/kl_loss': kl_loss.item(),
            'encoder/total_loss': loss.item(),
        }

    def train_iql_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single IQL training step (Phase 2).

        IQL objective:
        - Value: expectile regression on Q values
        - Q: Bellman backup using V(s')
        - Policy: AWR (advantage-weighted regression)
        """
        with torch.no_grad():
            # Target Q values using V(s')
            v_next = self.value_net(next_states, z)
            q_target = rewards + self.discount * (1 - dones) * v_next

        # Q-function update
        q1, q2 = self.q_net(states, actions, z)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # Value function update (expectile regression)
        with torch.no_grad():
            q_min = self.q_target.min(states, actions, z)

        v = self.value_net(states, z)
        diff = q_min - v
        # Asymmetric L2 loss (expectile)
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        v_loss = (weight * diff.pow(2)).mean()

        self.value_opt.zero_grad()
        v_loss.backward()
        self.value_opt.step()

        # Policy update (AWR)
        with torch.no_grad():
            q_min_pol = self.q_target.min(states, actions, z)
            v_pol = self.value_net(states, z)
            advantage = q_min_pol - v_pol
            exp_adv = torch.exp(advantage / self.awr_temperature).clamp(max=100.0)

        log_prob = self.policy.log_prob(states, z, actions)
        policy_loss = -(exp_adv * log_prob).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # Soft update target Q
        soft_update(self.q_target, self.q_net, self.tau)

        return {
            'iql/q_loss': q_loss.item(),
            'iql/v_loss': v_loss.item(),
            'iql/policy_loss': policy_loss.item(),
            'iql/mean_q': q_min.mean().item(),
            'iql/mean_v': v.mean().item(),
        }

    def act(
        self,
        state: np.ndarray,
        z: torch.Tensor,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Select action given state and latent z."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy.act(state_t, z, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def save(self, path: str):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'value_net': self.value_net.state_dict(),
            'q_net': self.q_net.state_dict(),
            'q_target': self.q_target.state_dict(),
            'policy': self.policy.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.decoder.load_state_dict(ckpt['decoder'])
        self.value_net.load_state_dict(ckpt['value_net'])
        self.q_net.load_state_dict(ckpt['q_net'])
        self.q_target.load_state_dict(ckpt['q_target'])
        self.policy.load_state_dict(ckpt['policy'])
