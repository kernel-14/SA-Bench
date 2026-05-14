import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network as used in Llama and OLMo."""
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2023)."""
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, freqs)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        q_rot = rotate_half(q)
        k_rot = rotate_half(k)
        q_out = q * cos + q_rot * sin
        k_out = k * cos + k_rot * sin
        return q_out, k_out


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class MultiHeadAttention(nn.Module):
    """Full multi-head attention with RoPE and optional QK-Norm."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=layer_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=layer_norm_eps)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = rope(q, k, T)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :T, :T]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)
