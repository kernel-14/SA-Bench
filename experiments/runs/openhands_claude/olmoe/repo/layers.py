import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Used throughout OLMoE instead of non-parametric LayerNorm.
    Parameters are included in weight decay (paper §4.2.3).
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cosine and sine frequency tensors."""
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply Rotary Position Embedding to query or key tensor.

    Args:
        x: (batch, seq_len, n_heads, head_dim)
        cos: (seq_len, head_dim // 2)
        sin: (seq_len, head_dim // 2)
    """
    bsz, seq_len, n_heads, head_dim = x.shape
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim//2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    """Multi-head self-attention with RoPE and optional QK-Norm.

    OLMoE uses full attention (not MQA/GQA), no biases, and QK-Norm
    (RMSNorm applied to Q and K after projection, §4.2.5).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        use_qk_norm: bool = True,
        norm_eps: float = 1e-5,
        use_bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=use_bias)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)

        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len, theta=rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # (bsz, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        attn = attn + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU activation function (Shazeer, 2020).

    Used in all FFN/expert modules in OLMoE.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)
