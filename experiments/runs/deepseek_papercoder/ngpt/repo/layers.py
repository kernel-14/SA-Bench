"""
layers.py

Core Transformer layers for both baseline GPT and the normalized Transformer (nGPT).

Includes:
- RMSNorm: standard root-mean-square layer normalization.
- Attention: multi‑head self‑attention with RoPE, optional QK normalisation and scaling.
- MLP: SwiGLU feed‑forward block, optional rescaling and weight normalisation.
- TransformerBlock: one decoder layer combining attention and MLP with the appropriate
  update rule (residual for GPT, hypersphere LERP for nGPT).

All decisions between GPT and nGPT are driven by the `Config` object passed during
construction.  nGPT‑specific scaling parameters and eigen learning rates implement the
“init / scale” trick described in Section 2.5.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from utils import lerp_update


# ---------------------------------------------------------------------------
# Utility: pre‑compute RoPE frequencies
# ---------------------------------------------------------------------------

def _precompute_rope_freqs(
    d_k: int,
    end: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """
    Compute the complex‑valued rotary embeddings for a given maximum sequence length.

    Returns a tensor of shape (end, d_k // 2, 2) representing cos and sin values.
    """
    indices = torch.arange(0, d_k, 2, dtype=torch.float32)
    freqs = 1.0 / (theta ** (indices / d_k))
    positions = torch.arange(end, dtype=torch.float32)
    # Outer product: (end, d_k//2)
    angles = torch.einsum("i,j->ij", positions, freqs)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return torch.stack([cos, sin], dim=-1)   # (end, d_k//2, 2)


def _apply_rope(x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to the last two dimensions of `x`.

    Args:
        x: Input tensor of shape (..., T, d) where d must be even.
        rope_freqs: Precomputed frequencies of shape (T, d//2, 2).

    Returns:
        Tensor of same shape as x with RoPE applied.
    """
    T = x.size(-2)
    d = x.size(-1)
    half_d = d // 2
    # Reshape to complex pairs
    x_complex = torch.view_as_complex(x[..., :].reshape(*x.shape[:-1], half_d, 2).contiguous())
    cos = rope_freqs[:T, :, 0]
    sin = rope_freqs[:T, :, 1]
    freqs = torch.view_as_complex(torch.stack([cos, sin], dim=-1))
    # Broadcast over batch and heads
    while freqs.dim() < x_complex.dim():
        freqs = freqs.unsqueeze(0)
    x_rotated = x_complex * freqs
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.type_as(x)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root‑mean‑square layer normalisation (used in baseline GPT).

    Normalises each hidden dimension vector to unit L2 norm and then scales with a
    learnable vector (initialised to ones).  The scaling emulates the standard
    implementation where the output has norm √d_model on average.
    """
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x / (rms + self.eps) * self.scale


# ---------------------------------------------------------------------------
# Multi‑head Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """
    Multi‑head self‑attention with causal masking and Rotary Position Embeddings.

    For nGPT, all linear weight matrices are normalised on the fly (columns lie on
    the sphere), query and key are additionally normalised after RoPE and scaled
    element‑wise, and the softmax scaling factor is changed to √d_k.
    """
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.n_heads = config.model.n_heads
        self.d_k = config.model.d_k
        self.d_model = config.model.d_model
        self.head_dim_ft = self.n_heads * self.d_k
        self.use_ngpt = config.model.use_ngpt

        # Linear projections
        self.w_q = nn.Linear(self.d_model, self.head_dim_ft, bias=False)
        self.w_k = nn.Linear(self.d_model, self.head_dim_ft, bias=False)
        self.w_v = nn.Linear(self.d_model, self.head_dim_ft, bias=False)
        self.w_o = nn.Linear(self.head_dim_ft, self.d_model, bias=False)

        # RoPE frequencies (pre‑computed up to max_seq_len)
        max_len = config.model.max_seq_len
        self.rope_freqs = _precompute_rope_freqs(self.d_k, max_len, config.model.rope_base)
        # Register as buffer so it moves to the correct device
        self.register_buffer("_rope_freqs", self.rope_freqs, persistent=False)

        # nGPT‑specific QK scaling (per head, per key dimension)
        if self.use_ngpt:
            s_qk_scale = config.ngpt.s_qk_scale
            s_qk_init = config.ngpt.s_qk_init
            self.s_qk_raw = nn.Parameter(
                torch.full((self.n_heads, self.d_k), s_qk_scale)
            )
            self.s_qk_factor = s_qk_init / s_qk_scale
        else:
            self.s_qk_raw = None
            self.s_qk_factor = None

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape

        if self.use_ngpt:
            # Normalise weight matrices along embedding dimension (columns)
            w_q_norm = F.normalize(self.w_q.weight, dim=0)
            w_k_norm = F.normalize(self.w_k.weight, dim=0)
            w_v_norm = F.normalize(self.w_v.weight, dim=0)
            q = F.linear(h, w_q_norm)
            k = F.linear(h, w_k_norm)
            v = F.linear(h, w_v_norm)
        else:
            q = self.w_q(h)
            k = self.w_k(h)
            v = self.w_v(h)

        # Reshape for multi‑head: (B, T, n_heads, d_k) -> (B, n_heads, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Apply Rotary Position Embedding
        q = _apply_rope(q, self._rope_freqs)
        k = _apply_rope(k, self._rope_freqs)

        if self.use_ngpt:
            # Normalise Q and K to unit sphere and apply per‑head scaling
            s_qk = self.s_qk_raw * self.s_qk_factor          # (n_heads, d_k)
            q = F.normalize(q, dim=-1) * s_qk.unsqueeze(0)   # broadcast batch, seq
            k = F.normalize(k, dim=-1) * s_qk.unsqueeze(0)

        # Attention scores
        if self.use_ngpt:
            # scale = sqrt(d_k) because dot product of normalised vectors has variance 1/d_k
            scale = math.sqrt(self.d_k)
        else:
            scale = 1.0 / math.sqrt(self.d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(h.dtype)
        attn_output = torch.matmul(attn_weights, v)   # (B, n_heads, T, d_k)

        # Concatenate heads: (B, T, head_dim_ft)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.head_dim_ft)

        # Final output projection
        if self.use_ngpt:
            w_o_norm = F.normalize(self.w_o.weight, dim=0)
            out = F.linear(attn_output, w_o_norm)
            out = F.normalize(out, dim=-1)            # h_A on the sphere
        else:
            out = self.w_o(attn_output)

        return out


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Gated MLP with SwiGLU activation (SiLU gate).

    For nGPT, the up‑projected vectors `u` and `v` are rescaled element‑wise,
    `v` is additionally multiplied by √d_model to benefit from SiLU non‑linearity,
    and all weight matrices are normalised on the fly.
    """
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.d_model = config.model.d_model
        self.d_mlp = config.model.d_mlp
        self.use_ngpt = config.model.use_ngpt

        self.w_u = nn.Linear(self.d_model, self.d_mlp, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_mlp, bias=False)
        self.w_o_mlp = nn.Linear(self.d_mlp, self.d_model, bias=False)

        if self.use_ngpt:
            # Scaling factors for intermediate states
            s_u_scale = config.ngpt.s_u_scale
            s_u_init = config.ngpt.s_u_init
            s_v_scale = config.ngpt.s_v_scale
            s_v_init = config.ngpt.s_v_init

            self.s_u_raw = nn.Parameter(torch.full((self.d_mlp,), s_u_scale))
            self.s_v_raw = nn.Parameter(torch.full((self.d_mlp,), s_v_scale))
            self.s_u_factor = s_u_init / s_u_scale
            self.s_v_factor = s_v_init / s_v_scale

            # Fixed rescaling for v to activate SiLU non‑linearity (Appendix A.1)
            self.v_rescale = math.sqrt(self.d_model)
        else:
            self.s_u_raw = None
            self.s_v_raw = None
            self.s_u_factor = None
            self.s_v_factor = None
            self.v_rescale = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.use_ngpt:
            w_u_norm = F.normalize(self.w_u.weight, dim=0)
            w_v_norm = F.normalize(self.w_v.weight, dim=0)

            u = F.linear(h, w_u_norm) * (self.s_u_raw * self.s_u_factor)
            v = F.linear(h, w_v_norm) * (self.s_v_raw * self.s_v_factor) * self.v_rescale
        else:
            u = self.w_u(h)
            v = self.w_v(h)

        # Gated activation: SwiGLU( u, v ) = u * SiLU(v)
        activation = u * F.silu(v)

        if self.use_ngpt:
            w_o_norm = F.normalize(self.w_o_mlp.weight, dim=0)
            out = F.linear(activation, w_o_norm)
            out = F.normalize(out, dim=-1)          # h_M on the sphere
        else:
            out = self.w_o_mlp(activation)

        return out


# ---------------------------------------------------------------------------
# Transformer Decoder Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A single decoder layer combining an attention sub‑layer and an MLP sub‑layer.

    For the baseline GPT, it uses pre‑normalisation with RMSNorm and standard
    residual connections:

        h = h + attn( RMSNorm(h) )
        h = h + mlp(  RMSNorm(h) )

    For nGPT, the block outputs are normalised (`h_A`, `h_M`) and updates follow
    spherical linear interpolation with learnable eigen learning rates α:

        h = Norm( h + α_A ⊙ (h_A - h) )
        h = Norm( h + α_M ⊙ (h_M - h) )

    Both α vectors are constrained to be non‑negative via absolute value.
    """
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.use_ngpt = config.model.use_ngpt
        self.attn = Attention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)

        if self.use_ngpt:
            # Eigen learning rates (per‑dimension, non‑negative)
            alpha_A_scale = config.ngpt.eigen_alpha_A_scale
            alpha_A_init = config.ngpt.eigen_alpha_A_init
            alpha_M_scale = config.ngpt.eigen_alpha_M_scale
            alpha_M_init = config.ngpt.eigen_alpha_M_init

            self.alpha_A_raw = nn.Parameter(
                torch.full((config.model.d_model,), alpha_A_scale)
            )
            self.alpha_M_raw = nn.Parameter(
                torch.full((config.model.d_model,), alpha_M_scale)
            )
            self.alpha_A_factor = alpha_A_init / alpha_A_scale
            self.alpha_M_factor = alpha_M_init / alpha_M_scale
        else:
            # GPT uses RMSNorm before each sub‑layer
            self.ln1 = RMSNorm(config.model.d_model)
            self.ln2 = RMSNorm(config.model.d_model)
            self.alpha_A_raw = None
            self.alpha_M_raw = None
            self.alpha_A_factor = None
            self.alpha_M_factor = None

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.use_ngpt:
            # Attention sub‑layer
            h_A = self.attn(h, mask)           # already unit norm
            alpha_A = torch.abs(self.alpha_A_raw) * self.alpha_A_factor
            h = lerp_update(h, h_A, alpha_A)

            # MLP sub‑layer
            h_M = self.mlp(h)                  # unit norm
            alpha_M = torch.abs(self.alpha_M_raw) * self.alpha_M_factor
            h = lerp_update(h, h_M, alpha_M)
        else:
            # GPT: pre‑norm and residual
            h_norm = self.ln1(h)
            h_attn = self.attn(h_norm, mask)
            h = h + h_attn

            h_norm = self.ln2(h)
            h_mlp = self.mlp(h_norm)
            h = h + h_mlp

        return h

