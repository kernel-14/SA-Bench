"""
Diffusion Forcing scheme for conditional flow marching.

Implements the RNN-based latent state evolution for conditioning the
flow marching model on past states with different noise levels.

Based on Diffusion Forcing (Chen et al., 2024) adapted for PDE dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DiffusionForcingRNN(nn.Module):
    """
    RNN-based diffusion forcing for PDE condition propagation.

    Maintains a compressed latent state h_s that summarizes the history
    of past states with potentially different noise levels. This allows
    the model to condition on partially noisy history during training,
    which reduces exposure bias during long-term rollout.

    The latent state evolves as:
        h_s ~ p_phi(h_s | h_{s-1}, x_{s,t_s}^{k_s}, t_s)

    where x_{s,t_s}^{k_s} is the noisy version of state x_s.

    Architecture:
    - GRU cell for state evolution
    - Cross-attention to compress current noisy state to a single token
    - Shared embedding dimension with the SiT transformer
    """

    def __init__(
        self,
        embed_dim: int,
        latent_channels: int = 16,
        latent_size: int = 16,
        num_heads: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_channels = latent_channels
        self.latent_size = latent_size

        # Project noisy latent state to embedding dimension
        # The noisy state x_{t}^{k} has shape (B, latent_channels, latent_size, latent_size)
        # We flatten and project to embed_dim
        self.state_proj = nn.Linear(latent_channels * latent_size * latent_size, embed_dim)

        # Time embedding for t conditioning
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Cross-attention to compress noisy state to single token
        self.cross_attn_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # GRU cell for latent state evolution
        self.gru = nn.GRUCell(embed_dim, embed_dim)

        # Layer norm for stability
        self.norm = nn.LayerNorm(embed_dim)

    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal time embedding.

        Args:
            t: (B,) time values in [0, 1]

        Returns:
            emb: (B, embed_dim) time embeddings
        """
        half_dim = self.embed_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.time_embed(emb)

    def compress_state(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compress noisy state to single token via cross-attention.

        Args:
            x_noisy: (B, latent_channels, H, W) noisy latent state
            t: (B,) time values

        Returns:
            token: (B, embed_dim) compressed state token
        """
        B = x_noisy.shape[0]

        # Flatten and project state
        x_flat = x_noisy.reshape(B, -1)  # (B, C*H*W)
        x_proj = self.state_proj(x_flat)  # (B, embed_dim)

        # Add time embedding
        t_emb = self.get_time_embedding(t)  # (B, embed_dim)
        x_proj = x_proj + t_emb

        # Cross-attention: query is learnable, key/value from projected state
        query = self.cross_attn_query.expand(B, -1, -1)  # (B, 1, embed_dim)
        kv = x_proj.unsqueeze(1)  # (B, 1, embed_dim)

        token, _ = self.cross_attn(query, kv, kv)  # (B, 1, embed_dim)
        return token.squeeze(1)  # (B, embed_dim)

    def forward(
        self,
        h_prev: Optional[torch.Tensor],
        x_noisy: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update latent state given previous state and current noisy observation.

        Args:
            h_prev: (B, embed_dim) previous latent state, or None for initialization
            x_noisy: (B, latent_channels, H, W) noisy current state x_{s,t_s}^{k_s}
            t: (B,) time values t_s

        Returns:
            h_new: (B, embed_dim) updated latent state
        """
        B = x_noisy.shape[0]

        # Initialize hidden state if not provided
        if h_prev is None:
            h_prev = torch.zeros(B, self.embed_dim, device=x_noisy.device, dtype=x_noisy.dtype)

        # Compress current noisy state to token
        state_token = self.compress_state(x_noisy, t)  # (B, embed_dim)

        # Update GRU state
        h_new = self.gru(state_token, h_prev)  # (B, embed_dim)
        h_new = self.norm(h_new)

        return h_new

    def init_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        return torch.zeros(batch_size, self.embed_dim, device=device, dtype=dtype)
