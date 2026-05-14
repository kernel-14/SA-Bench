"""
Baseline world model architectures for comparison with RWM.

Includes:
  - MLPWorldModel: MLP with history concatenation (trained autoregressively)
  - RSSMWorldModel: Recurrent State Space Model (as in PlaNet/Dreamer)
  - TransformerWorldModel: Decoder-only transformer for sequence modeling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# MLP Baseline
# ---------------------------------------------------------------------------

class MLPWorldModel(nn.Module):
    """
    MLP-based world model.

    Concatenates flattened history of observations and actions as input.
    Predicts mean and std of the next observation.

    Architecture: 2 hidden layers of size 256, ReLU activation.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        priv_size: int,
        history_horizon: int,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.priv_size = priv_size
        self.history_horizon = history_horizon

        input_size = (obs_size + action_size) * history_horizon

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.obs_head = nn.Linear(hidden_size, obs_size * 2)

        if priv_size > 0:
            self.priv_head = nn.Linear(hidden_size, priv_size * 2)
        else:
            self.priv_head = None

    def forward(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            obs_history: (batch, M, obs_size)
            action_history: (batch, M, action_size)

        Returns:
            obs_mean: (batch, obs_size)
            obs_std: (batch, obs_size)
            priv_mean: (batch, priv_size) or None
            priv_std: (batch, priv_size) or None
        """
        batch = obs_history.shape[0]
        x = torch.cat([obs_history, action_history], dim=-1).reshape(batch, -1)
        h = self.net(x)

        obs_out = self.obs_head(h)
        obs_mean, obs_log_std = obs_out.chunk(2, dim=-1)
        obs_std = F.softplus(obs_log_std) + 1e-5

        priv_mean, priv_std = None, None
        if self.priv_head is not None:
            priv_out = self.priv_head(h)
            priv_mean, priv_log_std = priv_out.chunk(2, dim=-1)
            priv_std = F.softplus(priv_log_std) + 1e-5

        return obs_mean, obs_std, priv_mean, priv_std

    def autoregressive_rollout(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        future_actions: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Autoregressive rollout for MLP: slide window forward.

        Args:
            obs_history: (batch, M, obs_size)
            action_history: (batch, M, action_size)
            future_actions: (batch, N, action_size)
            deterministic: use mean if True

        Returns:
            pred_obs_means: (batch, N, obs_size)
            pred_obs_stds: (batch, N, obs_size)
            pred_priv_means: (batch, N, priv_size) or None
            pred_priv_stds: (batch, N, priv_size) or None
        """
        N = future_actions.shape[1]
        M = self.history_horizon

        current_obs_hist = obs_history.clone()
        current_act_hist = action_history.clone()

        pred_obs_means = []
        pred_obs_stds = []
        pred_priv_means = []
        pred_priv_stds = []

        for k in range(N):
            obs_mean, obs_std, priv_mean, priv_std = self.forward(
                current_obs_hist, current_act_hist
            )
            pred_obs_means.append(obs_mean)
            pred_obs_stds.append(obs_std)
            if priv_mean is not None:
                pred_priv_means.append(priv_mean)
                pred_priv_stds.append(priv_std)

            # Slide window
            if deterministic:
                next_obs = obs_mean
            else:
                next_obs = obs_mean + obs_std * torch.randn_like(obs_mean)

            next_action = future_actions[:, k:k+1, :]
            current_obs_hist = torch.cat([current_obs_hist[:, 1:, :], next_obs.unsqueeze(1)], dim=1)
            current_act_hist = torch.cat([current_act_hist[:, 1:, :], next_action], dim=1)

        pred_obs_means = torch.stack(pred_obs_means, dim=1)
        pred_obs_stds = torch.stack(pred_obs_stds, dim=1)

        if pred_priv_means:
            pred_priv_means = torch.stack(pred_priv_means, dim=1)
            pred_priv_stds = torch.stack(pred_priv_stds, dim=1)
        else:
            pred_priv_means = None
            pred_priv_stds = None

        return pred_obs_means, pred_obs_stds, pred_priv_means, pred_priv_stds


# ---------------------------------------------------------------------------
# RSSM Baseline (Recurrent State Space Model, as in PlaNet/Dreamer)
# ---------------------------------------------------------------------------

class RSSMCell(nn.Module):
    """
    Single RSSM cell combining deterministic (GRU) and stochastic (categorical) states.

    Architecture matches Table S8:
      - GRU hidden size: 256
      - Latent dimension: 64
      - Prior type: categorical with 32 categories
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 256,
        latent_dim: int = 64,
        num_categories: int = 32,
        num_gru_layers: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.stoch_size = latent_dim * num_categories

        # Input embedding
        self.input_embed = nn.Linear(self.stoch_size + action_size, hidden_size)

        # GRU for deterministic state
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        # Prior: predict stochastic state from deterministic state
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim * num_categories),
        )

        # Posterior: predict stochastic state from deterministic state + observation
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_size + obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim * num_categories),
        )

    def init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "deter": torch.zeros(2, batch_size, self.hidden_size, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_size, device=device),
        }

    def prior(
        self,
        prev_stoch: torch.Tensor,
        prev_action: torch.Tensor,
        prev_deter: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute prior distribution and next deterministic state.

        Returns:
            logits: (batch, latent_dim * num_categories)
            stoch: (batch, stoch_size) - straight-through sample
            deter: (num_layers, batch, hidden_size)
        """
        x = torch.cat([prev_stoch, prev_action], dim=-1)
        x = F.elu(self.input_embed(x)).unsqueeze(1)
        deter_out, deter = self.gru(x, prev_deter)
        deter_out = deter_out.squeeze(1)

        logits = self.prior_net(deter_out)
        logits = logits.reshape(-1, self.latent_dim, self.num_categories)
        stoch = self._straight_through_sample(logits)
        stoch = stoch.reshape(-1, self.stoch_size)

        return logits, stoch, deter

    def posterior(
        self,
        deter: torch.Tensor,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior distribution given deterministic state and observation.

        Args:
            deter: (num_layers, batch, hidden_size) - use last layer output
            obs: (batch, obs_size)

        Returns:
            logits: (batch, latent_dim * num_categories)
            stoch: (batch, stoch_size)
        """
        deter_last = deter[-1]  # (batch, hidden_size)
        x = torch.cat([deter_last, obs], dim=-1)
        logits = self.posterior_net(x)
        logits = logits.reshape(-1, self.latent_dim, self.num_categories)
        stoch = self._straight_through_sample(logits)
        stoch = stoch.reshape(-1, self.stoch_size)
        return logits, stoch

    def _straight_through_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Straight-through categorical sample."""
        probs = F.softmax(logits, dim=-1)
        # Hard sample
        indices = torch.argmax(probs, dim=-1, keepdim=True)
        hard = torch.zeros_like(probs).scatter_(-1, indices, 1.0)
        # Straight-through: use hard in forward, probs gradient in backward
        return hard + (probs - probs.detach())


class RSSMWorldModel(nn.Module):
    """
    Recurrent State Space Model (RSSM) world model.

    Based on PlaNet/Dreamer architecture. Uses categorical stochastic states
    with straight-through gradients.

    Architecture (Table S8):
      - GRU hidden size: 256, 2 layers
      - Latent dimension: 64
      - Prior type: categorical with 32 categories
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        priv_size: int,
        hidden_size: int = 256,
        latent_dim: int = 64,
        num_categories: int = 32,
        num_gru_layers: int = 2,
        head_hidden_size: int = 128,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.priv_size = priv_size
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.stoch_size = latent_dim * num_categories

        self.rssm_cell = RSSMCell(
            obs_size=obs_size,
            action_size=action_size,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            num_categories=num_categories,
            num_gru_layers=num_gru_layers,
        )

        # Decoder: predict observation from (deter, stoch)
        decoder_input = hidden_size + self.stoch_size
        self.obs_decoder = nn.Sequential(
            nn.Linear(decoder_input, head_hidden_size),
            nn.ReLU(),
            nn.Linear(head_hidden_size, obs_size * 2),
        )

        if priv_size > 0:
            self.priv_decoder = nn.Sequential(
                nn.Linear(decoder_input, head_hidden_size),
                nn.ReLU(),
                nn.Linear(head_hidden_size, priv_size * 2),
            )
        else:
            self.priv_decoder = None

    def _decode(
        self,
        deter: torch.Tensor,
        stoch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        deter_last = deter[-1]  # (batch, hidden_size)
        feat = torch.cat([deter_last, stoch], dim=-1)

        obs_out = self.obs_decoder(feat)
        obs_mean, obs_log_std = obs_out.chunk(2, dim=-1)
        obs_std = F.softplus(obs_log_std) + 1e-5

        priv_mean, priv_std = None, None
        if self.priv_decoder is not None:
            priv_out = self.priv_decoder(feat)
            priv_mean, priv_log_std = priv_out.chunk(2, dim=-1)
            priv_std = F.softplus(priv_log_std) + 1e-5

        return obs_mean, obs_std, priv_mean, priv_std

    def autoregressive_rollout(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        future_actions: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Autoregressive rollout using RSSM.

        First processes history using posterior (teacher-forcing over history),
        then rolls out N steps using prior.

        Args:
            obs_history: (batch, M, obs_size)
            action_history: (batch, M, action_size)
            future_actions: (batch, N, action_size)
            deterministic: unused (RSSM always uses straight-through)

        Returns:
            pred_obs_means: (batch, N, obs_size)
            pred_obs_stds: (batch, N, obs_size)
            pred_priv_means: (batch, N, priv_size) or None
            pred_priv_stds: (batch, N, priv_size) or None
        """
        batch_size = obs_history.shape[0]
        M = obs_history.shape[1]
        N = future_actions.shape[1]
        device = obs_history.device

        state = self.rssm_cell.init_state(batch_size, device)

        # Process history with posterior
        for t in range(M):
            obs_t = obs_history[:, t, :]
            act_t = action_history[:, t, :]
            _, stoch_prior, deter = self.rssm_cell.prior(
                state["stoch"], act_t, state["deter"]
            )
            _, stoch_post = self.rssm_cell.posterior(deter, obs_t)
            state = {"deter": deter, "stoch": stoch_post}

        # Rollout N steps using prior
        pred_obs_means = []
        pred_obs_stds = []
        pred_priv_means = []
        pred_priv_stds = []

        for k in range(N):
            act_k = future_actions[:, k, :]
            _, stoch, deter = self.rssm_cell.prior(state["stoch"], act_k, state["deter"])
            state = {"deter": deter, "stoch": stoch}

            obs_mean, obs_std, priv_mean, priv_std = self._decode(deter, stoch)
            pred_obs_means.append(obs_mean)
            pred_obs_stds.append(obs_std)
            if priv_mean is not None:
                pred_priv_means.append(priv_mean)
                pred_priv_stds.append(priv_std)

        pred_obs_means = torch.stack(pred_obs_means, dim=1)
        pred_obs_stds = torch.stack(pred_obs_stds, dim=1)

        if pred_priv_means:
            pred_priv_means = torch.stack(pred_priv_means, dim=1)
            pred_priv_stds = torch.stack(pred_priv_stds, dim=1)
        else:
            pred_priv_means = None
            pred_priv_stds = None

        return pred_obs_means, pred_obs_stds, pred_priv_means, pred_priv_stds


# ---------------------------------------------------------------------------
# Transformer Baseline
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerWorldModel(nn.Module):
    """
    Decoder-only transformer world model.

    Architecture (Table S8):
      - Type: decoder (causal)
      - Dimension: 64
      - Heads: 8
      - Layers: 2
      - Context length: 32
      - Positional encoding: sinusoidal
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        priv_size: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        context_length: int = 32,
        head_hidden_size: int = 128,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.priv_size = priv_size
        self.d_model = d_model
        self.context_length = context_length

        # Input projection
        input_size = obs_size + action_size
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=context_length + 64)

        # Transformer decoder layers (causal self-attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output heads
        self.obs_head = nn.Sequential(
            nn.Linear(d_model, head_hidden_size),
            nn.ReLU(),
            nn.Linear(head_hidden_size, obs_size * 2),
        )

        if priv_size > 0:
            self.priv_head = nn.Sequential(
                nn.Linear(d_model, head_hidden_size),
                nn.ReLU(),
                nn.Linear(head_hidden_size, priv_size * 2),
            )
        else:
            self.priv_head = None

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper triangular causal mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            obs_history: (batch, seq_len, obs_size)
            action_history: (batch, seq_len, action_size)

        Returns:
            obs_mean: (batch, seq_len, obs_size)
            obs_std: (batch, seq_len, obs_size)
            priv_mean: (batch, seq_len, priv_size) or None
            priv_std: (batch, seq_len, priv_size) or None
        """
        seq_len = obs_history.shape[1]
        x = torch.cat([obs_history, action_history], dim=-1)
        x = self.input_proj(x)
        x = self.pos_enc(x)

        causal_mask = self._causal_mask(seq_len, x.device)

        # Decoder-only: use x as both tgt and memory (self-attention only)
        out = self.transformer(tgt=x, memory=x, tgt_mask=causal_mask, memory_mask=causal_mask)

        obs_out = self.obs_head(out)
        obs_mean, obs_log_std = obs_out.chunk(2, dim=-1)
        obs_std = F.softplus(obs_log_std) + 1e-5

        priv_mean, priv_std = None, None
        if self.priv_head is not None:
            priv_out = self.priv_head(out)
            priv_mean, priv_log_std = priv_out.chunk(2, dim=-1)
            priv_std = F.softplus(priv_log_std) + 1e-5

        return obs_mean, obs_std, priv_mean, priv_std

    def autoregressive_rollout(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        future_actions: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Autoregressive rollout for transformer.

        Args:
            obs_history: (batch, M, obs_size)
            action_history: (batch, M, action_size)
            future_actions: (batch, N, action_size)
            deterministic: use mean if True

        Returns:
            pred_obs_means: (batch, N, obs_size)
            pred_obs_stds: (batch, N, obs_size)
            pred_priv_means: (batch, N, priv_size) or None
            pred_priv_stds: (batch, N, priv_size) or None
        """
        N = future_actions.shape[1]
        M = obs_history.shape[1]

        current_obs_hist = obs_history.clone()
        current_act_hist = action_history.clone()

        pred_obs_means = []
        pred_obs_stds = []
        pred_priv_means = []
        pred_priv_stds = []

        for k in range(N):
            obs_mean, obs_std, priv_mean, priv_std = self.forward(
                current_obs_hist, current_act_hist
            )
            # Take last step prediction
            obs_mean_k = obs_mean[:, -1, :]
            obs_std_k = obs_std[:, -1, :]

            pred_obs_means.append(obs_mean_k)
            pred_obs_stds.append(obs_std_k)

            if priv_mean is not None:
                pred_priv_means.append(priv_mean[:, -1, :])
                pred_priv_stds.append(priv_std[:, -1, :])

            # Slide window
            if deterministic:
                next_obs = obs_mean_k
            else:
                next_obs = obs_mean_k + obs_std_k * torch.randn_like(obs_mean_k)

            next_action = future_actions[:, k:k+1, :]
            current_obs_hist = torch.cat(
                [current_obs_hist[:, 1:, :], next_obs.unsqueeze(1)], dim=1
            )
            current_act_hist = torch.cat(
                [current_act_hist[:, 1:, :], next_action], dim=1
            )

        pred_obs_means = torch.stack(pred_obs_means, dim=1)
        pred_obs_stds = torch.stack(pred_obs_stds, dim=1)

        if pred_priv_means:
            pred_priv_means = torch.stack(pred_priv_means, dim=1)
            pred_priv_stds = torch.stack(pred_priv_stds, dim=1)
        else:
            pred_priv_means = None
            pred_priv_stds = None

        return pred_obs_means, pred_obs_stds, pred_priv_means, pred_priv_stds
