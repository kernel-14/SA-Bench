## models/fmt.py
"""
Flow Marching Transformer (FMT) for generative PDE modelling.

This module implements the core FMT model as described in the paper, including:
- A SiT (Scalable Interpolant Transformer) backbone with AdaLN‑Zero conditioning.
- Diffusion Forcing using a GRU to maintain a recurrent dynamics condition.
- Latent temporal pyramids for efficient multi‑scale tokenisation.
- Methods for computing conditional velocity fields and the flow‑marching loss.

The FMT takes a sequence of noisy latent states (already coded by a frozen P2VAE)
and predicts per‑token transport velocities. It can be used both for deterministic
prediction (k=1) and stochastic ensemble generation (k<1).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import (RMSNorm, CrossAttnPool, TransformerBlock)

# -----------------------------------------------------------------------------
# Helper: sinusoidal timestep embedding (following DiT)
# -----------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Embeds scalar timesteps into a vector of dimension `out_dim` using sinusoidal
    frequencies and a small MLP."""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.linear = nn.Sequential(
            nn.Linear(dim, out_dim, bias=True),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or scalar – values in [0,1].
        Returns:
            (B, out_dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # (B, dim)
        return self.linear(embedding)


# -----------------------------------------------------------------------------
# SiT Transformer (stack of TransformerBlocks with final RMSNorm)
# -----------------------------------------------------------------------------

class SiTTransformer(nn.Module):
    """Scalable Interpolant Transformer backbone.

    Stacks `depth` TransformerBlocks, each with its own AdaLN‑Zero modulation.
    Conditioning vectors `c` of shape (B, L, cond_dim) are passed to each block.
    """

    def __init__(self, dim: int, depth: int, heads: int, cond_dim: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, cond_dim) for _ in range(depth)
        ])
        self.final_norm = RMSNorm(dim)

    def forward(self, tokens: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, L, dim) token sequence.
            c: (B, L, cond_dim) per‑token conditioning.
        Returns:
            (B, L, dim) transformed tokens.
        """
        for block in self.blocks:
            tokens = block(tokens, c)
        return self.final_norm(tokens)


# -----------------------------------------------------------------------------
# Diffusion Forcing module (GRU + cross-attention pooling)
# -----------------------------------------------------------------------------

class DiffusionForcing(nn.Module):
    """Maintains a compressed dynamics condition via a GRU that processes
    per‑frame latent tokens summarised by cross‑attention pooling."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.pool = CrossAttnPool(dim, num_heads=8)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.gru.hidden_size, device=self.gru.weight_ih.device)

    def step(self, x_tokens: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_tokens: (B, L, dim) token sequence for the current frame.
            h: (B, dim) hidden state from previous step.
        Returns:
            (new_hidden, pooled) where pooled is the cross‑attention output.
        """
        pooled = self.pool(x_tokens)          # (B, dim)
        h_new = self.gru(pooled, h)           # (B, dim)
        return h_new, pooled

    def forward_sequence(self, x_tokens_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Processes a sequence of token lists (one per frame) and returns the
        GRU hidden states **after** each frame.

        Args:
            x_tokens_list: list of length T, each item (B, N_i, dim).
        Returns:
            list of hidden states of length T (same device as inputs).
        """
        batch_size = x_tokens_list[0].size(0)
        h = self.init_hidden(batch_size)
        hs = []
        for tokens in x_tokens_list:
            h, _ = self.step(tokens, h)
            hs.append(h)
        return hs


# -----------------------------------------------------------------------------
# Main FMT class
# -----------------------------------------------------------------------------

class FMT(nn.Module):
    """Flow Marching Transformer.

    Args:
        dim: Transformer embedding dimension (256, 512, 768 for S, B, L).
        depth: Number of transformer layers.
        heads: Number of attention heads.
        pyramid_factors: List of 4 downsampling factors (e.g. [8,4,2,1]).
        latent_dim: Number of channels in latent grid (16).
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        pyramid_factors: List[int],
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.pyramid_factors = pyramid_factors
        self.latent_dim = latent_dim

        # ===================== Sub‑modules =====================

        # Diffusion forcing (GRU + pool)
        self.diff_forcing = DiffusionForcing(dim)

        # SiT transformer backbone
        self.transformer = SiTTransformer(dim, depth, heads, cond_dim=dim)

        # Time embedding (maps scalar t -> dim)
        self.time_embed = TimestepEmbedding(dim=256, out_dim=dim)

        # Input projection: latent channels -> dim
        self.proj_in = nn.Linear(latent_dim, dim)

        # Velocity output head: dim -> latent_dim
        self.vel_head = nn.Linear(dim, latent_dim)

        # Projection for concatenated [t_emb, hidden] -> cond_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(2 * dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

        # Learned positional embeddings, one per frame (size varies with pyramid)
        self.pos_embeddings = nn.ParameterList()
        for factor in pyramid_factors:
            grid_size = 16 // factor   # latent spatial size after downsampling
            num_tokens = grid_size * grid_size
            emb = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
            self.pos_embeddings.append(emb)

        # Store token counts for each frame (useful for splitting later)
        self.token_counts = []
        for factor in pyramid_factors:
            grid_size = 16 // factor
            self.token_counts.append(grid_size * grid_size)

    # -----------------------------------------------------------------
    # Private helper: downsample latent and project to tokens
    # -----------------------------------------------------------------

    def _downsample_and_tokenize(
        self, y: torch.Tensor, frame_idx: int
    ) -> torch.Tensor:
        """
        Downsamples a latent frame to the resolution determined by
        `pyramid_factors[frame_idx]`, projects to transformer dim,
        and adds positional embeddings.

        Args:
            y: (B, latent_dim, 16, 16)
            frame_idx: index 0..3
        Returns:
            (B, N_i, dim) token sequence.
        """
        factor = self.pyramid_factors[frame_idx]
        # Downsample with average pooling
        y_down = F.avg_pool2d(y, kernel_size=factor, stride=factor)  # (B, C, H, W)
        B, C, H, W = y_down.shape
        tokens = y_down.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B, N_i, C)
        tokens = self.proj_in(tokens)                               # (B, N_i, dim)
        tokens = tokens + self.pos_embeddings[frame_idx]            # broadcast
        return tokens

    # -----------------------------------------------------------------
    # Public method: build full token sequence and per‑token conditioning
    # -----------------------------------------------------------------

    def build_tokens(
        self,
        latents: List[torch.Tensor],
        times: List[torch.Tensor],
        ks: List[float],                # not used for conditioning, kept for API consistency
        h_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assembles the multi‑scale token sequence and per‑token conditioning
        vectors that will be passed to the transformer.

        Args:
            latents: length‑4 list of (B, latent_dim, 16, 16) noisy states.
            times: length‑4 list of scalar tensors (B,) or floats.
            ks: length‑4 list of bridge parameters (float), ignored here.
            h_list: length‑4 list of hidden states (B, dim) – one per frame
                    (h_list[i] is the hidden state used for conditioning frame i).
                    Typically h_list[0] = zeros, h_list[i] = GRU after frame i-1.

        Returns:
            tokens_all: (B, total_tokens, dim)
            cond_all:   (B, total_tokens, dim)
        """
        all_tokens = []
        all_cond = []
        total_tokens = 0
        for i, (y, h) in enumerate(zip(latents, h_list)):
            # Tokenise the latent frame
            tokens = self._downsample_and_tokenize(y, i)  # (B, N_i, dim)
            N_i = tokens.shape[1]

            # Time embedding (handles scalar or batch tensor)
            t_val = times[i]
            if isinstance(t_val, float):
                t_val = torch.tensor(t_val, device=y.device).float()
            if t_val.dim() == 0:
                t_val = t_val.expand(y.shape[0])
            t_emb = self.time_embed(t_val)                # (B, dim)

            # Concatenate [t_emb, h] -> condition vector
            cond = torch.cat([t_emb, h], dim=-1)          # (B, 2*dim)
            cond = self.cond_proj(cond)                   # (B, dim)
            # Expand to per token
            cond_tokens = cond.unsqueeze(1).expand(-1, N_i, -1)

            all_tokens.append(tokens)
            all_cond.append(cond_tokens)
            total_tokens += N_i

        tokens_all = torch.cat(all_tokens, dim=1)        # (B, total_tokens, dim)
        cond_all   = torch.cat(all_cond, dim=1)          # (B, total_tokens, dim)
        return tokens_all, cond_all

    # -----------------------------------------------------------------
    # Forward pass through the transformer backbone
    # -----------------------------------------------------------------

    def forward_tokens(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Pass the full token sequence through the SiT transformer."""
        return self.transformer(tokens, cond)

    # -----------------------------------------------------------------
    # Compute velocity tokens for all transitions
    # -----------------------------------------------------------------

    def compute_velocity(
        self,
        y_seq: List[torch.Tensor],
        ts: List[float],
        ks: List[float],
        h_init: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Computes per‑frame velocity tokens for a 4‑frame input sequence.

        Args:
            y_seq: list of 4 latent tensors (B, latent_dim, 16, 16) – already
                   noised according to ts and ks.
            ts: list of 4 time values (float or tensors).
            ks: list of 4 bridge parameters (float).
            h_init: optional (B, dim) initial GRU hidden state (default zeros).

        Returns:
            velocity_tokens: list of 3 tensors (for transitions 0→1,1→2,2→3),
                             each of shape (B, N_i, latent_dim) at token resolution.
        """
        B = y_seq[0].size(0)

        # 1. Tokenise each frame for the GRU (downsampled + projected)
        gru_tokens = []
        for i in range(4):
            gru_tokens.append(self._downsample_and_tokenize(y_seq[i], i))

        # 2. Unroll diffusion forcing to obtain hidden states
        hs = self.diff_forcing.forward_sequence(gru_tokens)   # length 4

        # 3. Build conditioning list: h_cond[i] = (initial) or hs[i-1]
        if h_init is None:
            h_init = torch.zeros(B, self.dim, device=y_seq[0].device)
        h_cond = [h_init] + hs[:3]   # length 4

        # 4. Construct full transformer input with conditioning
        tokens_all, cond_all = self.build_tokens(y_seq, ts, ks, h_cond)

        # 5. Transformer forward
        out_tokens = self.forward_tokens(tokens_all, cond_all)  # (B, total, dim)

        # 6. Split tokens per frame and extract velocity for frames 0,1,2
        velocities = []
        offset = 0
        for i in range(3):  # only first three frames need velocity output
            n = self.token_counts[i]
            frame_out = out_tokens[:, offset:offset+n, :]       # (B, N_i, dim)
            vel_i = self.vel_head(frame_out)                    # (B, N_i, latent_dim)
            velocities.append(vel_i)
            offset += n
        return velocities

    # -----------------------------------------------------------------
    # Loss function (Eq. 13 in the paper)
    # -----------------------------------------------------------------

    def loss(
        self,
        velocities: List[torch.Tensor],
        targets: List[torch.Tensor],
        ts: List[float],
    ) -> torch.Tensor:
        """
        Computes the conditional flow marching loss.

        Args:
            velocities: list of three predicted velocity tensors
                       (each [B, N_i, latent_dim]).
            targets:   list of three target residual tensors
                       (each [B, N_i, latent_dim], computed as
                       y_{s+1} - y_{s,t_s}^{k_s} at token resolution).
            ts:        list of three time values (float or tensors).

        Returns:
            scalar loss.
        """
        loss = 0.0
        for i, (vel, targ, t) in enumerate(zip(velocities, targets, ts)):
            # Ensure t is a tensor of shape (B,) for scaling
            if isinstance(t, float):
                t = torch.tensor(t, device=vel.device).float()
            if t.dim() == 0:
                t = t.expand(vel.size(0))
            # Preconditioned error: (1-t)*velocity - target
            scaling = (1.0 - t).reshape(-1, 1, 1).to(vel.dtype)
            err = scaling * vel - targ
            loss += 0.5 * err.pow(2).mean()
        return loss


# -----------------------------------------------------------------------------
# Factory functions for standard sizes (optional)
# -----------------------------------------------------------------------------

def make_fmt_small() -> FMT:
    return FMT(
        dim=256,
        depth=12,
        heads=8,
        pyramid_factors=[8, 4, 2, 1],
        latent_dim=16,
    )

def make_fmt_base() -> FMT:
    return FMT(
        dim=512,
        depth=12,
        heads=8,
        pyramid_factors=[8, 4, 2, 1],
        latent_dim=16,
    )

def make_fmt_large() -> FMT:
    return FMT(
        dim=768,
        depth=24,
        heads=12,
        pyramid_factors=[8, 4, 2, 1],
        latent_dim=16,
    )
