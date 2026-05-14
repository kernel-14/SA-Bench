"""
Higher-level modules for RWM and baseline world models, plus the policy/value networks.

Implements:
  - RWMCore: dual-autoregressive GRU world model (Sec. 3.2, Fig. S6)
  - MLPWorldModel: MLP baseline trained autoregressively
  - RSSMWorldModel: Recurrent State Space Model baseline (Dreamer-style)
  - TransformerWorldModel: decoder-only transformer baseline
  - PolicyNetwork / ValueNetwork: actor-critic for MBPO-PPO (Table S9)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, OneHotCategorical

from layers import (
    MLP,
    CausalTransformerDecoder,
    GRUEncoder,
    GaussianHead,
    SinusoidalPositionalEncoding,
    get_activation,
)


# ---------------------------------------------------------------------------
# RWM: Dual-Autoregressive GRU World Model
# ---------------------------------------------------------------------------

class RWMCore(nn.Module):
    """
    Robotic World Model core with dual-autoregressive mechanism.

    Inner autoregression: GRU hidden state updated step-by-step over the
    history horizon M.
    Outer autoregression: predicted observations fed back as input over the
    forecast horizon N.

    Architecture (Table S7):
      base  : GRU  hidden=(256, 256)
      heads : MLP  hidden=128, ReLU  →  (mean, std) for obs and privileged info
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        privileged_dim: int,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 2,
        head_hidden_size: int = 128,
        head_activation: str = "relu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.privileged_dim = privileged_dim
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        input_dim = obs_dim + action_dim
        self.gru = GRUEncoder(input_dim, gru_hidden_size, gru_num_layers)

        self.obs_head = GaussianHead(
            gru_hidden_size, obs_dim, head_hidden_size, head_activation
        )
        self.priv_head = GaussianHead(
            gru_hidden_size, privileged_dim, head_hidden_size, head_activation
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.gru.init_hidden(batch_size, device)

    def step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single GRU step. Returns (obs_mean, obs_std, priv_mean, priv_std, new_hidden)."""
        x = torch.cat([obs, action], dim=-1).unsqueeze(1)  # (B, 1, input_dim)
        gru_out, new_hidden = self.gru(x, hidden)
        feat = gru_out.squeeze(1)  # (B, hidden_size)
        obs_mean, obs_std = self.obs_head(feat)
        priv_mean, priv_std = self.priv_head(feat)
        return obs_mean, obs_std, priv_mean, priv_std, new_hidden

    def forward_history(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process history horizon M with inner autoregression.

        Args:
            obs_history:    (B, M, obs_dim)
            action_history: (B, M, action_dim)
            hidden:         (num_layers, B, hidden_size) or None

        Returns:
            hidden: (num_layers, B, hidden_size) after processing all M steps
        """
        B = obs_history.size(0)
        if hidden is None:
            hidden = self.init_hidden(B, obs_history.device)
        x = torch.cat([obs_history, action_history], dim=-1)  # (B, M, input_dim)
        _, hidden = self.gru(x, hidden)
        return hidden

    def forward_forecast(
        self,
        obs_start: torch.Tensor,
        action_forecast: torch.Tensor,
        hidden: torch.Tensor,
        use_mean: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Outer autoregression over forecast horizon N.

        Args:
            obs_start:       (B, obs_dim) — last observed/predicted obs
            action_forecast: (B, N, action_dim)
            hidden:          (num_layers, B, hidden_size)
            use_mean:        if True, use mean prediction (no sampling)

        Returns:
            obs_means, obs_stds, priv_means, priv_stds — each a list of N tensors (B, dim)
        """
        N = action_forecast.size(1)
        obs_means, obs_stds = [], []
        priv_means, priv_stds = [], []

        current_obs = obs_start
        for k in range(N):
            a_k = action_forecast[:, k]
            obs_mean, obs_std, priv_mean, priv_std, hidden = self.step(
                current_obs, a_k, hidden
            )
            obs_means.append(obs_mean)
            obs_stds.append(obs_std)
            priv_means.append(priv_mean)
            priv_stds.append(priv_std)

            if use_mean:
                current_obs = obs_mean
            else:
                # Reparameterization trick for gradient propagation
                eps = torch.randn_like(obs_mean)
                current_obs = obs_mean + obs_std * eps

        return obs_means, obs_stds, priv_means, priv_stds

    def autoregressive_rollout(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        action_forecast: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        use_mean: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Full dual-autoregressive forward pass: inner (history) + outer (forecast).

        Args:
            obs_history:     (B, M, obs_dim)
            action_history:  (B, M, action_dim)
            action_forecast: (B, N, action_dim)
            hidden:          optional initial hidden state
            use_mean:        deterministic rollout

        Returns:
            obs_means, obs_stds, priv_means, priv_stds — each list of N tensors
        """
        hidden = self.forward_history(obs_history, action_history, hidden)
        obs_start = obs_history[:, -1]
        return self.forward_forecast(obs_start, action_forecast, hidden, use_mean)


# ---------------------------------------------------------------------------
# MLP Baseline World Model
# ---------------------------------------------------------------------------

class MLPWorldModel(nn.Module):
    """
    MLP baseline trained autoregressively (Table S8: hidden=(256,256), ReLU).

    Concatenates flattened history of (obs, action) pairs as input.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        privileged_dim: int,
        history_horizon: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.privileged_dim = privileged_dim
        self.history_horizon = history_horizon

        input_dim = history_horizon * (obs_dim + action_dim)

        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(get_activation(activation))
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        self.obs_mean_head = nn.Linear(in_dim, obs_dim)
        self.obs_log_std_head = nn.Linear(in_dim, obs_dim)
        self.priv_mean_head = nn.Linear(in_dim, privileged_dim)
        self.priv_log_std_head = nn.Linear(in_dim, privileged_dim)

    def _encode(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        B = obs_history.size(0)
        x = torch.cat([obs_history, action_history], dim=-1).view(B, -1)
        return self.backbone(x)

    def forward(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self._encode(obs_history, action_history)
        obs_mean = self.obs_mean_head(feat)
        obs_std = self.obs_log_std_head(feat).clamp(-5, 2).exp()
        priv_mean = self.priv_mean_head(feat)
        priv_std = self.priv_log_std_head(feat).clamp(-5, 2).exp()
        return obs_mean, obs_std, priv_mean, priv_std

    def autoregressive_rollout(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        action_forecast: torch.Tensor,
        use_mean: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        N = action_forecast.size(1)
        M = self.history_horizon
        obs_means, obs_stds, priv_means, priv_stds = [], [], [], []

        obs_buf = obs_history.clone()
        act_buf = action_history.clone()

        for k in range(N):
            obs_mean, obs_std, priv_mean, priv_std = self.forward(obs_buf, act_buf)
            obs_means.append(obs_mean)
            obs_stds.append(obs_std)
            priv_means.append(priv_mean)
            priv_stds.append(priv_std)

            if use_mean:
                next_obs = obs_mean
            else:
                eps = torch.randn_like(obs_mean)
                next_obs = obs_mean + obs_std * eps

            a_k = action_forecast[:, k].unsqueeze(1)
            obs_buf = torch.cat([obs_buf[:, 1:], next_obs.unsqueeze(1)], dim=1)
            act_buf = torch.cat([act_buf[:, 1:], a_k], dim=1)

        return obs_means, obs_stds, priv_means, priv_stds


# ---------------------------------------------------------------------------
# RSSM Baseline World Model
# ---------------------------------------------------------------------------

class RSSMWorldModel(nn.Module):
    """
    Recurrent State Space Model (RSSM) baseline (Table S8).

    Architecture:
      - Deterministic state: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
      - Prior:     p(z_t | h_t)       — categorical (32 categories × 32 dims)
      - Posterior: q(z_t | h_t, o_t)  — categorical
      - Decoder:   p(o_t | h_t, z_t)  — Gaussian MLP

    Parameters (Table S8):
      type=GRU, hidden=256, layers=2, latent_dim=64,
      prior=categorical, categories=32
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        privileged_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        latent_dim: int = 64,
        num_categories: int = 32,
        category_size: int = 32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.privileged_dim = privileged_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_categories = num_categories
        self.category_size = category_size
        self.stoch_dim = num_categories * category_size  # flattened stochastic state

        # Recurrent model: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        self.gru = GRUEncoder(
            self.stoch_dim + action_dim, hidden_size, num_layers
        )

        # Prior: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_categories * category_size),
        )

        # Posterior: q(z_t | h_t, o_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_size + obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_categories * category_size),
        )

        # Observation decoder: p(o_t | h_t, z_t)
        feat_dim = hidden_size + self.stoch_dim
        self.obs_decoder = GaussianHead(feat_dim, obs_dim, hidden_size)
        self.priv_decoder = GaussianHead(feat_dim, privileged_dim, hidden_size)

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, z

    def _categorical_straight_through(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical with straight-through gradient."""
        B = logits.size(0)
        logits = logits.view(B, self.num_categories, self.category_size)
        probs = F.softmax(logits, dim=-1)
        # Straight-through: hard one-hot in forward, soft in backward
        indices = probs.argmax(dim=-1)
        one_hot = F.one_hot(indices, self.category_size).float()
        z = one_hot + probs - probs.detach()
        return z.view(B, self.stoch_dim)

    def prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute prior distribution logits and sample z."""
        feat = h[-1]  # last layer hidden state
        logits = self.prior_net(feat)
        z = self._categorical_straight_through(logits)
        return logits, z

    def posterior(
        self, h: torch.Tensor, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior distribution logits and sample z."""
        feat = torch.cat([h[-1], obs], dim=-1)
        logits = self.posterior_net(feat)
        z = self._categorical_straight_through(logits)
        return logits, z

    def recurrent_step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1).unsqueeze(1)
        _, h_new = self.gru(x, h)
        return h_new

    def decode(
        self, h: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = torch.cat([h[-1], z], dim=-1)
        obs_mean, obs_std = self.obs_decoder(feat)
        priv_mean, priv_std = self.priv_decoder(feat)
        return obs_mean, obs_std, priv_mean, priv_std

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        use_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a sequence using posterior (training) or prior (imagination).

        Args:
            obs_seq:    (B, T, obs_dim)
            action_seq: (B, T, action_dim)
            use_posterior: use posterior for z (True during training)

        Returns dict with keys: obs_means, obs_stds, prior_logits, post_logits
        """
        B, T, _ = obs_seq.shape
        h, z = self.init_state(B, obs_seq.device)

        obs_means, obs_stds = [], []
        prior_logits_list, post_logits_list = [], []

        for t in range(T):
            h = self.recurrent_step(z, action_seq[:, t], h)
            prior_logits, z_prior = self.prior(h)
            if use_posterior:
                post_logits, z = self.posterior(h, obs_seq[:, t])
            else:
                post_logits = prior_logits
                z = z_prior

            obs_mean, obs_std, _, _ = self.decode(h, z)
            obs_means.append(obs_mean)
            obs_stds.append(obs_std)
            prior_logits_list.append(prior_logits)
            post_logits_list.append(post_logits)

        return {
            "obs_means": torch.stack(obs_means, dim=1),
            "obs_stds": torch.stack(obs_stds, dim=1),
            "prior_logits": torch.stack(prior_logits_list, dim=1),
            "post_logits": torch.stack(post_logits_list, dim=1),
        }

    def autoregressive_rollout(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        action_forecast: torch.Tensor,
        use_mean: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Encode history with posterior, then roll out with prior."""
        B, M, _ = obs_history.shape
        h, z = self.init_state(B, obs_history.device)

        # Encode history using posterior
        for t in range(M):
            h = self.recurrent_step(z, action_history[:, t], h)
            _, z = self.posterior(h, obs_history[:, t])

        # Forecast using prior
        N = action_forecast.size(1)
        obs_means, obs_stds, priv_means, priv_stds = [], [], [], []

        for k in range(N):
            h = self.recurrent_step(z, action_forecast[:, k], h)
            _, z = self.prior(h)
            obs_mean, obs_std, priv_mean, priv_std = self.decode(h, z)
            obs_means.append(obs_mean)
            obs_stds.append(obs_std)
            priv_means.append(priv_mean)
            priv_stds.append(priv_std)

        return obs_means, obs_stds, priv_means, priv_stds


# ---------------------------------------------------------------------------
# Transformer Baseline World Model
# ---------------------------------------------------------------------------

class TransformerWorldModel(nn.Module):
    """
    Decoder-only transformer baseline (Table S8).

    Parameters: d_model=64, heads=8, layers=2, context=32, sinusoidal PE.
    Input projection: (obs_dim + action_dim) → d_model
    Output: Gaussian (mean, std) over next observation and privileged info.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        privileged_dim: int,
        d_model: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        context_length: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.privileged_dim = privileged_dim
        self.context_length = context_length

        input_dim = obs_dim + action_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=context_length + 1)
        self.transformer = CausalTransformerDecoder(
            d_model, num_heads, num_layers, context_length, dropout
        )

        self.obs_mean_head = nn.Linear(d_model, obs_dim)
        self.obs_log_std_head = nn.Linear(d_model, obs_dim)
        self.priv_mean_head = nn.Linear(d_model, privileged_dim)
        self.priv_log_std_head = nn.Linear(d_model, privileged_dim)

    def _encode(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([obs_seq, action_seq], dim=-1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        return self.transformer(x)

    def forward(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next obs from history. Returns (obs_mean, obs_std, priv_mean, priv_std)."""
        feat = self._encode(obs_history, action_history)
        last = feat[:, -1]
        obs_mean = self.obs_mean_head(last)
        obs_std = self.obs_log_std_head(last).clamp(-5, 2).exp()
        priv_mean = self.priv_mean_head(last)
        priv_std = self.priv_log_std_head(last).clamp(-5, 2).exp()
        return obs_mean, obs_std, priv_mean, priv_std

    def autoregressive_rollout(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        action_forecast: torch.Tensor,
        use_mean: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        N = action_forecast.size(1)
        obs_means, obs_stds, priv_means, priv_stds = [], [], [], []

        obs_buf = obs_history.clone()
        act_buf = action_history.clone()

        for k in range(N):
            obs_mean, obs_std, priv_mean, priv_std = self.forward(obs_buf, act_buf)
            obs_means.append(obs_mean)
            obs_stds.append(obs_std)
            priv_means.append(priv_mean)
            priv_stds.append(priv_std)

            if use_mean:
                next_obs = obs_mean
            else:
                eps = torch.randn_like(obs_mean)
                next_obs = obs_mean + obs_std * eps

            a_k = action_forecast[:, k].unsqueeze(1)
            obs_buf = torch.cat([obs_buf[:, 1:], next_obs.unsqueeze(1)], dim=1)
            act_buf = torch.cat([act_buf[:, 1:], a_k], dim=1)

        return obs_means, obs_stds, priv_means, priv_stds


# ---------------------------------------------------------------------------
# Policy and Value Networks for MBPO-PPO
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    Gaussian policy network (Table S9): MLP(128, 128, 128, ELU).
    Outputs mean and log-std of a diagonal Gaussian action distribution.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (128, 128, 128),
        activation: str = "elu",
        log_std_init: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(get_activation(activation))
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(obs)
        mean = self.mean_head(feat)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)
        return mean, std

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_distribution(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """
    Value function network (Table S9): MLP(128, 128, 128, ELU).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Tuple[int, ...] = (128, 128, 128),
        activation: str = "elu",
    ):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(get_activation(activation))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
