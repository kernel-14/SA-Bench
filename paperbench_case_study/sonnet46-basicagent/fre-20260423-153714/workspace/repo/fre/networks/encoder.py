"""
Functional Reward Encoding (FRE) Encoder.

Architecture details (from paper + addendum):
- State embedding: linear projection to 64-dim
- Reward discretization: 32 bins, mapped to 64-dim learned embedding
- Concatenated token: 128-dim (64 state + 64 reward)
- Transformer: 4 layers, 128-dim residual, MLP expands to 256 then back to 128
- 4 attention heads, no positional encoding, no causal masking
- Output: mean-pool -> two linear heads for mu and log_std of z (128-dim)
"""

import torch
import torch.nn as nn


class RewardDiscretizer(nn.Module):
    """Discretizes scalar rewards into 32 bins and maps to learned embeddings."""

    def __init__(self, num_bins: int = 32, embed_dim: int = 64):
        super().__init__()
        self.num_bins = num_bins
        self.embedding = nn.Embedding(num_bins, embed_dim)

    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rewards: (batch, K) scalar rewards normalized to [0, 1]
        Returns:
            embeddings: (batch, K, embed_dim)
        """
        r_clipped = rewards.clamp(0.0, 1.0)
        bin_ids = (r_clipped * self.num_bins).floor().long().clamp(0, self.num_bins - 1)
        return self.embedding(bin_ids)


class TransformerBlock(nn.Module):
    """Single transformer block: multi-head self-attention + MLP residual."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, mlp_dim: int = 256):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


class FREEncoder(nn.Module):
    """
    Permutation-invariant transformer encoder for FRE.

    Encodes a set of K (state, reward) pairs into a latent distribution
    z ~ N(mu, sigma) of dimension latent_dim (128).
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        state_embed_dim: int = 64,
        reward_embed_dim: int = 64,
        num_reward_bins: int = 32,
        n_layers: int = 4,
        n_heads: int = 4,
        mlp_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        token_dim = state_embed_dim + reward_embed_dim  # 128

        self.state_proj = nn.Linear(state_dim, state_embed_dim)
        self.reward_discretizer = RewardDiscretizer(num_reward_bins, reward_embed_dim)

        self.transformer = nn.ModuleList(
            [TransformerBlock(token_dim, n_heads, mlp_dim) for _ in range(n_layers)]
        )

        self.mu_head = nn.Linear(token_dim, latent_dim)
        self.log_std_head = nn.Linear(token_dim, latent_dim)

    def forward(self, states: torch.Tensor, rewards: torch.Tensor):
        """
        Args:
            states:  (batch, K, state_dim)
            rewards: (batch, K) scalar rewards normalized to [0, 1]
        Returns:
            mu:      (batch, latent_dim)
            log_std: (batch, latent_dim)
        """
        s_emb = self.state_proj(states)           # (B, K, 64)
        r_emb = self.reward_discretizer(rewards)  # (B, K, 64)
        tokens = torch.cat([s_emb, r_emb], dim=-1)  # (B, K, 128)

        x = tokens
        for block in self.transformer:
            x = block(x)

        pooled = x.mean(dim=1)  # (B, 128)
        mu = self.mu_head(pooled)
        log_std = self.log_std_head(pooled)
        return mu, log_std

    def encode(self, states: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Sample z via reparameterization trick."""
        mu, log_std = self.forward(states, rewards)
        std = log_std.exp()
        return mu + std * torch.randn_like(std)

    def encode_deterministic(self, states: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Return the mean of the encoded distribution."""
        mu, _ = self.forward(states, rewards)
        return mu
