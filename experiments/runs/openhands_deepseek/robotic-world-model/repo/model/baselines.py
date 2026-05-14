"""Baseline models for comparison with RWM.

- MLP baseline: Simple feed-forward with autoregressive training
- RSSM baseline: Recurrent State-Space Model (PlaNet/Dreamer style)
- Transformer baseline: Decoder-only transformer with sinusoidal positional encoding

Architectures follow Table S8.
"""

from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn

from .modules import GaussianHead, SinusoidalPositionalEncoding, RecurrentStateSpaceModel


class MLPBaseline(nn.Module):
    """MLP-based dynamics model trained autoregressively (same M, N as RWM).

    Architecture: 2 hidden layers of 256 with ReLU (Table S8).
    Takes concatenated (obs_history, act_history) → predicts next obs.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        history_horizon: int = 32,
        forecast_horizon: int = 8,
        hidden_shape: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon

        input_dim = history_horizon * (obs_dim + action_dim) + action_dim
        self.predictor = GaussianHead(
            input_dim=input_dim,
            output_dim=obs_dim,
            hidden_dim=hidden_shape[0],
            activation=activation,
        )

        # Build intermediate layers manually for the specified hidden shape
        layers = []
        dims = [input_dim] + list(hidden_shape)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if activation == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.ELU())
        self.shared_backbone = nn.Sequential(*layers)
        self.head = GaussianHead(
            input_dim=hidden_shape[-1],
            output_dim=obs_dim,
            hidden_dim=128,
            activation=activation,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        forecast_horizon: int,
    ) -> Dict[str, torch.Tensor]:
        """Predict next N observations autoregressively.

        Args:
            observations: (B, M, obs_dim)
            actions: (B, M-1+N, action_dim)
            forecast_horizon: N

        Returns:
            dict with obs_means, obs_log_stds, predicted_obs
        """
        batch_size = observations.shape[0]
        device = observations.device
        M = observations.shape[1]

        # Flatten the history
        obs_flat = observations.reshape(batch_size, -1)  # (B, M*obs_dim)
        act_history = actions[:, :M - 1].reshape(batch_size, -1)  # (B, (M-1)*action_dim)

        obs_means = []
        obs_log_stds = []
        predicted_obs = []

        current_obs = observations[:, -1]

        for k in range(forecast_horizon):
            act_next = actions[:, M - 1 + k]  # (B, action_dim)
            inp = torch.cat([obs_flat, act_history, act_next], dim=-1)
            features = self.shared_backbone(inp)
            mean, log_std = self.head(features)
            obs_means.append(mean)
            obs_log_stds.append(log_std)

            std = torch.exp(log_std)
            eps = torch.randn_like(mean)
            sampled = mean + std * eps
            predicted_obs.append(sampled)

            # Update history: shift left, append prediction
            obs_flat = torch.cat([
                obs_flat[:, self.obs_dim:],
                sampled,
            ], dim=-1)

        return {
            "obs_means": torch.stack(obs_means, dim=1),
            "obs_log_stds": torch.stack(obs_log_stds, dim=1),
            "predicted_obs": torch.stack(predicted_obs, dim=1),
        }


class RSSMBaseline(nn.Module):
    """RSSM-based world model baseline.

    Uses teacher-forcing during training (the standard RSSM approach).
    The autoregressive training framework can be applied to RSSM as mentioned in the paper.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        history_horizon: int = 32,
        hidden_size: int = 256,
        num_layers: int = 2,
        latent_dim: int = 64,
        num_categories: int = 32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.history_horizon = history_horizon

        self.rssm = RecurrentStateSpaceModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            latent_dim=latent_dim,
            num_categories=num_categories,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        forecast_horizon: int,
    ) -> Dict[str, torch.Tensor]:
        """Run RSSM over history then forecast.

        Uses teacher-forcing during training (ground truth observations as input).
        """
        batch_size = observations.shape[0]
        device = observations.device
        M = observations.shape[1]
        hidden = torch.zeros(
            self.rssm.num_layers, batch_size, self.rssm.hidden_size, device=device
        )

        # Process history to get initial hidden state and latent
        prev_z = torch.zeros(batch_size, self.rssm.num_categories * self.rssm.latent_dim, device=device)
        for t in range(M - 1):
            _, _, h_flat, hidden = self.rssm.forward_step(hidden, actions[:, t], prev_z)
            posterior_mean, posterior_logit = self.rssm.encode_step(h_flat, observations[:, t + 1])
            prev_z = self.rssm.sample_latent(posterior_mean, posterior_logit)

        obs_means = []
        obs_log_stds = []
        predicted_obs = []

        for k in range(forecast_horizon):
            prior_mean, prior_logit, h_flat, hidden = self.rssm.forward_step(
                hidden, actions[:, M - 1 + k], prev_z
            )
            prior_z = self.rssm.sample_latent(prior_mean, prior_logit)
            obs_mean, obs_log_std = self.rssm.decode_step(h_flat, prior_z)
            obs_means.append(obs_mean)
            obs_log_stds.append(obs_log_std)

            std = torch.exp(obs_log_std)
            eps = torch.randn_like(obs_mean)
            sampled = obs_mean + std * eps
            predicted_obs.append(sampled)

            prev_z = prior_z

        return {
            "obs_means": torch.stack(obs_means, dim=1),
            "obs_log_stds": torch.stack(obs_log_stds, dim=1),
            "predicted_obs": torch.stack(predicted_obs, dim=1),
        }


class TransformerBaseline(nn.Module):
    """Transformer-based world model baseline.

    Architecture: decoder-only transformer with sinusoidal positional encoding.
    Configuration follows Table S8: d_model=64, nhead=8, num_layers=2, context_length=32.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        context_length: int = 32,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_length = context_length
        self.d_model = d_model

        self.obs_embed = nn.Linear(obs_dim, d_model)
        self.action_embed = nn.Linear(action_dim, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=2048)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.obs_head = GaussianHead(
            input_dim=d_model,
            output_dim=obs_dim,
            hidden_dim=128,
            activation="relu",
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        forecast_horizon: int,
    ) -> Dict[str, torch.Tensor]:
        """Process history through transformer and forecast autoregressively."""
        batch_size = observations.shape[0]
        device = observations.device
        M = observations.shape[1]

        # Prepare sequence: interleave obs and action embeddings
        seq_len = M
        obs_emb = self.obs_embed(observations)  # (B, M, d_model)
        act_emb = self.action_embed(actions[:, :M - 1])  # (B, M-1, d_model)

        # Create input sequence: [obs_0, act_0, obs_1, ..., act_{M-2}, obs_{M-1}]
        tokens = []
        for t in range(M - 1):
            tokens.append(obs_emb[:, t:t + 1])
            tokens.append(act_emb[:, t:t + 1])
        tokens.append(obs_emb[:, M - 1:M - 1 + 1])
        seq = torch.cat(tokens, dim=1)  # (B, 2M-1, d_model)
        seq = self.pos_encoding(seq)

        # Decoder self-attention (tgt=seq, memory=seq for causal masking)
        causal_mask = torch.triu(
            torch.ones(seq.shape[1], seq.shape[1], device=device) * float("-inf"),
            diagonal=1,
        )
        hidden = self.transformer(seq, seq, tgt_mask=causal_mask)

        obs_means = []
        obs_log_stds = []
        predicted_obs = []

        current_obs = observations[:, -1]

        for k in range(forecast_horizon):
            # Use last output for prediction
            last_hidden = hidden[:, -1]
            mean, log_std = self.obs_head(last_hidden)
            obs_means.append(mean)
            obs_log_stds.append(log_std)

            std = torch.exp(log_std)
            eps = torch.randn_like(mean)
            sampled = mean + std * eps
            predicted_obs.append(sampled)

            # Append prediction to sequence for next step
            pred_emb = self.obs_embed(sampled.unsqueeze(1))
            seq = torch.cat([seq, pred_emb], dim=1)

            if k < forecast_horizon - 1:
                act_next_idx = M - 1 + k + 1
                if act_next_idx < actions.shape[1]:
                    next_act_emb = self.action_embed(
                        actions[:, act_next_idx:act_next_idx + 1]
                    )
                    seq = torch.cat([seq, next_act_emb], dim=1)

            causal_mask = torch.triu(
                torch.ones(seq.shape[1], seq.shape[1], device=device) * float("-inf"),
                diagonal=1,
            )
            hidden = self.transformer(seq, seq, tgt_mask=causal_mask)

        return {
            "obs_means": torch.stack(obs_means, dim=1),
            "obs_log_stds": torch.stack(obs_log_stds, dim=1),
            "predicted_obs": torch.stack(predicted_obs, dim=1),
        }
