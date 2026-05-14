from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Positional Embeddings
# ---------------------------------------------------------------------------

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for spatial or temporal positions (ViT-style)."""

    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (N,) integer tensor of position indices
        Returns:
            embeddings: (N, dim)
        """
        return self.pe[positions]


class SpatialPositionalEmbedding(nn.Module):
    """2D sinusoidal spatial positional embedding for H×W grid."""

    def __init__(self, dim: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        self.h_embed = SinusoidalPositionalEmbedding(half_dim, max_len=max_h)
        self.w_embed = SinusoidalPositionalEmbedding(half_dim, max_len=max_w)

    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Returns:
            spe: (H*W, dim)
        """
        h_pos = torch.arange(h, device=device)
        w_pos = torch.arange(w, device=device)
        h_emb = self.h_embed(h_pos)  # (H, dim/2)
        w_emb = self.w_embed(w_pos)  # (W, dim/2)
        # Broadcast and concatenate
        h_emb = h_emb.unsqueeze(1).expand(-1, w, -1)  # (H, W, dim/2)
        w_emb = w_emb.unsqueeze(0).expand(h, -1, -1)  # (H, W, dim/2)
        spe = torch.cat([h_emb, w_emb], dim=-1)  # (H, W, dim)
        return spe.reshape(h * w, -1)


class TemporalPositionalEmbedding(nn.Module):
    """1D sinusoidal temporal positional embedding."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        self.embed = SinusoidalPositionalEmbedding(dim, max_len=max_len)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (L,) integer tensor of frame indices
        Returns:
            tpe: (L, dim)
        """
        return self.embed(positions)


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by MLP projection."""

    def __init__(self, dim: int, out_dim: Optional[int] = None, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        out_dim = out_dim or dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer or float timestep values
        Returns:
            emb: (B, out_dim)
        """
        emb = self._sinusoidal(t)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Adaptive Layer Normalization (adaLN-Zero, DiT-style)
# ---------------------------------------------------------------------------

class AdaLayerNorm(nn.Module):
    """Adaptive layer normalization conditioned on timestep embedding."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim)
            cond: (B, cond_dim) — broadcast over sequence dims
        Returns:
            normed: (..., dim)
        """
        shift, scale = self.proj(cond).chunk(2, dim=-1)
        # Reshape for broadcasting: cond is (B, dim), x may be (B, L, dim) or (B, HW, dim)
        while shift.dim() < x.dim():
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


class AdaLayerNormZero(nn.Module):
    """adaLN-Zero: returns scale/shift/gate for modulation (DiT-style)."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            normed_x, shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff
        """
        params = self.proj(cond)  # (B, 6*dim)
        while params.dim() < x.dim():
            params = params.unsqueeze(1)
        shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = params.chunk(6, dim=-1)
        normed = self.norm(x) * (1 + scale_attn) + shift_attn
        return normed, shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff


# ---------------------------------------------------------------------------
# Multi-Head Attention (base)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        x = x.reshape(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (B, H, L, d)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Lq, dim)
            key:   (B, Lk, dim)
            value: (B, Lv, dim)
            attn_mask: (Lq, Lk) additive mask, -inf for masked positions
        Returns:
            out: (B, Lq, dim)
        """
        Q = self._split_heads(self.q_proj(query))  # (B, H, Lq, d)
        K = self._split_heads(self.k_proj(key))    # (B, H, Lk, d)
        V = self._split_heads(self.v_proj(value))  # (B, H, Lv, d)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, Lq, Lk)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, Lq, d)
        out = out.permute(0, 2, 1, 3).reshape(query.shape[0], query.shape[1], -1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Causal Temporal Attention
# ---------------------------------------------------------------------------

class CausalTemporalAttention(nn.Module):
    """
    Causal temporal attention (Eq. 3 in paper).

    Each frame can only attend to its preceding frames (lower-triangular mask).
    During inference with KV-cache, the clean cached KV is prepended to the
    current noisy KV before attention computation.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular causal mask: M_{i,j} = -inf if i < j else 0."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def compute_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute key and value projections for caching."""
        return self.k_proj(x), self.v_proj(x)

    def forward(
        self,
        x: torch.Tensor,
        cached_k: Optional[torch.Tensor] = None,
        cached_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B*HW, L, dim) — input sequence (spatial grids as batch)
            cached_k: (B*HW, P_k, dim) — clean cached keys from previous AR steps
            cached_v: (B*HW, P_k, dim) — clean cached values from previous AR steps
        Returns:
            out: (B*HW, L, dim)
            k:   (B*HW, L, dim) — current keys (for cache writing)
            v:   (B*HW, L, dim) — current values (for cache writing)
        """
        B, L, _ = x.shape

        Q = self._split_heads(self.q_proj(x))  # (B, H, L, d)
        K_cur = self.k_proj(x)                 # (B, L, dim)
        V_cur = self.v_proj(x)                 # (B, L, dim)

        if cached_k is not None and cached_v is not None:
            # Concatenate cached clean KV with current noisy KV
            K_full = torch.cat([cached_k, K_cur], dim=1)  # (B, P_k+L, dim)
            V_full = torch.cat([cached_v, V_cur], dim=1)  # (B, P_k+L, dim)
        else:
            K_full = K_cur
            V_full = V_cur

        K = self._split_heads(K_full)  # (B, H, P_k+L, d)
        V = self._split_heads(V_full)  # (B, H, P_k+L, d)

        total_len = K_full.shape[1]
        prefix_len = total_len - L

        # Build causal mask: queries (L) attend to keys (P_k + L)
        # Query i can attend to key j if j <= prefix_len + i (all prefix + causal within chunk)
        # Shape: (L, P_k + L)
        query_pos = torch.arange(L, device=x.device).unsqueeze(1)       # (L, 1)
        key_pos = torch.arange(total_len, device=x.device).unsqueeze(0)  # (1, P_k+L)
        # Allow attending to all prefix keys and causal keys within chunk
        attn_mask = torch.where(
            key_pos <= prefix_len + query_pos,
            torch.zeros(1, device=x.device),
            torch.full((1,), float("-inf"), device=x.device),
        )  # (L, P_k+L)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, L, P_k+L)
        attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, L, d)
        out = out.permute(0, 2, 1, 3).reshape(B, L, -1)
        out = self.out_proj(out)

        return out, K_cur, V_cur


# ---------------------------------------------------------------------------
# Prefix-Enhanced Spatial Attention
# ---------------------------------------------------------------------------

class PrefixEnhancedSpatialAttention(nn.Module):
    """
    Prefix-enhanced spatial attention (Eq. 4 in paper).

    For denoising frames (i >= P): keys/values are formed by concatenating
    P' prefix frames spatially before the current frame's tokens.
    For prefix frames (i < P): keys/values are formed by self-repeating P'+1 times.

    Attention map shape: (HW) x ((P'+1)*HW) per frame.
    """

    def __init__(self, dim: int, num_heads: int, prefix_len: int = 3, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.prefix_len = prefix_len  # P'

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def compute_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute key and value projections for spatial cache writing."""
        return self.k_proj(x), self.v_proj(x)

    def forward(
        self,
        h: torch.Tensor,
        prefix_frames: int,
        spatial_cached_k: Optional[torch.Tensor] = None,
        spatial_cached_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            h: (B, L, HW, dim) — hidden states, L frames, HW spatial tokens
            prefix_frames: P — number of clean prefix frames in h
            spatial_cached_k: (B, P', HW, dim) — cached spatial keys (inference only)
            spatial_cached_v: (B, P', HW, dim) — cached spatial values (inference only)
        Returns:
            out: (B, L, HW, dim)
            new_spatial_k: (B, l, HW, dim) — spatial keys for denoising target (cache writing)
            new_spatial_v: (B, l, HW, dim) — spatial values for denoising target
        """
        B, L, HW, C = h.shape
        P = prefix_frames
        P_prime = self.prefix_len

        outputs = []
        new_spatial_k = None
        new_spatial_v = None

        for i in range(L):
            h_i = h[:, i]  # (B, HW, C)
            Q_i = self.q_proj(h_i)  # (B, HW, C)

            if i >= P:
                # Denoising target frame: enhance with P' prefix frames
                if spatial_cached_k is not None:
                    # Inference: use cached spatial KV
                    # spatial_cached_k: (B, P', HW, C)
                    prefix_k = spatial_cached_k.reshape(B, P_prime * HW, C)
                    prefix_v = spatial_cached_v.reshape(B, P_prime * HW, C)
                    cur_k = self.k_proj(h_i)  # (B, HW, C)
                    cur_v = self.v_proj(h_i)
                    K_i = torch.cat([prefix_k, cur_k], dim=1)  # (B, (P'+1)*HW, C)
                    V_i = torch.cat([prefix_v, cur_v], dim=1)
                else:
                    # Training: use last P' prefix frames from h
                    start = max(0, P - P_prime)
                    prefix_h = h[:, start:P]  # (B, P', HW, C)
                    # Pad if fewer than P' prefix frames available
                    if prefix_h.shape[1] < P_prime:
                        pad = h[:, :1].expand(-1, P_prime - prefix_h.shape[1], -1, -1)
                        prefix_h = torch.cat([pad, prefix_h], dim=1)
                    prefix_k = self.k_proj(prefix_h.reshape(B, P_prime * HW, C))
                    prefix_v = self.v_proj(prefix_h.reshape(B, P_prime * HW, C))
                    cur_k = self.k_proj(h_i)
                    cur_v = self.v_proj(h_i)
                    K_i = torch.cat([prefix_k, cur_k], dim=1)
                    V_i = torch.cat([prefix_v, cur_v], dim=1)
            else:
                # Prefix frame: self-repeat P'+1 times
                repeated = h_i.unsqueeze(1).expand(-1, P_prime + 1, -1, -1)  # (B, P'+1, HW, C)
                repeated_flat = repeated.reshape(B, (P_prime + 1) * HW, C)
                K_i = self.k_proj(repeated_flat)
                V_i = self.v_proj(repeated_flat)

            # Attention for frame i: (B, HW) x (B, (P'+1)*HW)
            Q_mh = Q_i.reshape(B, HW, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            K_mh = K_i.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            V_mh = V_i.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            attn = torch.matmul(Q_mh, K_mh.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out_i = torch.matmul(attn, V_mh)  # (B, H, HW, d)
            out_i = out_i.permute(0, 2, 1, 3).reshape(B, HW, C)
            out_i = self.out_proj(out_i)
            outputs.append(out_i)

        out = torch.stack(outputs, dim=1)  # (B, L, HW, C)

        # Compute spatial KV for denoising target (for cache writing)
        if P < L:
            target_h = h[:, P:]  # (B, l, HW, C)
            l = L - P
            target_flat = target_h.reshape(B, l * HW, C)
            sk = self.k_proj(target_flat).reshape(B, l, HW, C)
            sv = self.v_proj(target_flat).reshape(B, l, HW, C)
            new_spatial_k = sk
            new_spatial_v = sv

        return out, new_spatial_k, new_spatial_v


# ---------------------------------------------------------------------------
# Cross Attention (visual-text)
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Visual-text cross attention for text-conditioned generation."""

    def __init__(self, dim: int, context_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(context_dim, dim)
        self.v_proj = nn.Linear(context_dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, dim) — visual queries
            context: (B, S, context_dim) — text key/value source
        Returns:
            out: (B, L, dim)
        """
        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(context))
        V = self._split_heads(self.v_proj(context))

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """2D patch embedding for spatial frames."""

    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            patches: (B, H/p * W/p, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, embed_dim)
        return x


# ---------------------------------------------------------------------------
# Final Layer (DiT-style)
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """Final layer with adaLN and linear projection to output channels."""

    def __init__(self, dim: int, patch_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.adaLN = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, HW, dim)
            cond: (B, cond_dim)
        Returns:
            out: (B, HW, patch_size^2 * out_channels)
        """
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.proj(x)
