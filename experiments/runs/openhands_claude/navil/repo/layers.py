"""
Primitive layers for NaViL:
  - RMSNorm
  - 1D-RoPE (for LLM causal attention)
  - 2D-RoPE (for visual encoder bidirectional attention)
  - Grouped-Query Attention (causal, for LLM)
  - Bidirectional Multi-Head Attention (for visual encoder)
  - SwiGLU FFN
  - Modality-specific MoE attention expert
  - Modality-specific MoE FFN expert
  - Patch Embedding
  - Pixel Shuffle connector
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── Normalization ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ── 1D Rotary Position Embedding (for LLM) ────────────────────────────────────

def precompute_freqs_1d(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1000000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin for 1D-RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope_1d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 1D-RoPE to query and key tensors.
    q, k: (batch, heads, seq_len, head_dim)
    cos, sin: (max_seq_len, head_dim // 2)
    """
    seq_len = q.shape[2]
    if position_ids is not None:
        cos_pos = cos[position_ids]   # (batch, seq_len, head_dim//2)
        sin_pos = sin[position_ids]
    else:
        cos_pos = cos[:seq_len].unsqueeze(0)   # (1, seq_len, head_dim//2)
        sin_pos = sin[:seq_len].unsqueeze(0)

    # Expand for heads: (batch, 1, seq_len, head_dim//2)
    cos_pos = cos_pos.unsqueeze(1)
    sin_pos = sin_pos.unsqueeze(1)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    q_rot = q * cos_pos.repeat(1, 1, 1, 2) + rotate(q) * sin_pos.repeat(1, 1, 1, 2)
    k_rot = k * cos_pos.repeat(1, 1, 1, 2) + rotate(k) * sin_pos.repeat(1, 1, 1, 2)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


# ── 2D Rotary Position Embedding (for visual encoder) ─────────────────────────

def precompute_freqs_2d(
    head_dim: int,
    max_h: int,
    max_w: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute 2D-RoPE frequencies.
    Half of head_dim is used for height, half for width.
    Returns cos, sin of shape (max_h * max_w, head_dim).
    """
    half_dim = head_dim // 2
    freqs_h = 1.0 / (theta ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))
    freqs_w = 1.0 / (theta ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))

    h_pos = torch.arange(max_h, device=device).float()
    w_pos = torch.arange(max_w, device=device).float()

    freqs_h = torch.outer(h_pos, freqs_h)   # (max_h, half_dim//2)
    freqs_w = torch.outer(w_pos, freqs_w)   # (max_w, half_dim//2)

    # Expand to grid
    freqs_h = freqs_h.unsqueeze(1).expand(-1, max_w, -1)   # (max_h, max_w, half_dim//2)
    freqs_w = freqs_w.unsqueeze(0).expand(max_h, -1, -1)   # (max_h, max_w, half_dim//2)

    freqs = torch.cat([freqs_h, freqs_w], dim=-1)           # (max_h, max_w, half_dim)
    freqs = freqs.reshape(max_h * max_w, half_dim)

    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 2D-RoPE to query and key tensors.
    q, k: (batch, heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim // 2)
    """
    cos_pos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, seq_len, head_dim//2)
    sin_pos = sin.unsqueeze(0).unsqueeze(0)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    q_rot = q * cos_pos.repeat(1, 1, 1, 2) + rotate(q) * sin_pos.repeat(1, 1, 1, 2)
    k_rot = k * cos_pos.repeat(1, 1, 1, 2) + rotate(k) * sin_pos.repeat(1, 1, 1, 2)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


# ── Grouped-Query Attention (causal, for LLM) ─────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    Standard GQA with causal mask and 1D-RoPE.
    Used in the LLM (non-MoE path, or as the shared attention base).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope_1d(q, k, cos, sin, position_ids)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v)

        # Expand KV for GQA
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(
                B, self.num_heads, -1, self.head_dim
            )
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(
                B, self.num_heads, -1, self.head_dim
            )

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(attention_mask is None),
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn), present


# ── Bidirectional Multi-Head Attention (for visual encoder) ───────────────────

class BidirectionalAttention(nn.Module):
    """
    Full bidirectional attention with 2D-RoPE.
    Used in the visual encoder layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope_2d(q, k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn)


# ── SwiGLU FFN ────────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network: FFN(x) = (SiLU(x W_gate) ⊙ x W_up) W_down."""

    def __init__(self, hidden_dim: int, mlp_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, mlp_dim, bias=bias)
        self.up_proj   = nn.Linear(hidden_dim, mlp_dim, bias=bias)
        self.down_proj = nn.Linear(mlp_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Modality-specific MoE Attention Expert ────────────────────────────────────

class ModalityAttentionMoE(nn.Module):
    """
    Modality-specific attention expert (MHA-MMoE).

    Each modality (visual / linguistic) has its own Q, K, V, O projection
    matrices. Global attention is computed over all tokens jointly after
    per-modality projection, so cross-modal interaction is preserved.

    Eq. (3) from the paper:
        MHA-MMoE(x_{i,m}) = softmax(QK^T / sqrt(d)) V  W_O^m
        Q_{i,m} = x_{i,m} W_Q^m,  K_{i,m} = x_{i,m} W_K^m,  V_{i,m} = x_{i,m} W_V^m
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        num_experts: int = 2,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.num_experts = num_experts
        self.dropout = dropout

        # Per-modality projection matrices
        self.q_projs = nn.ModuleList([
            nn.Linear(hidden_dim, num_heads * self.head_dim, bias=bias)
            for _ in range(num_experts)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=bias)
            for _ in range(num_experts)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=bias)
            for _ in range(num_experts)
        ])
        self.o_projs = nn.ModuleList([
            nn.Linear(num_heads * self.head_dim, hidden_dim, bias=bias)
            for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        modality_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        x:             (B, T, D)
        modality_mask: (B, T) with integer expert indices (0=visual, 1=linguistic)
        """
        B, T, D = x.shape

        # Project Q, K, V using per-token modality expert
        q = torch.zeros(B, T, self.num_heads * self.head_dim, device=x.device, dtype=x.dtype)
        k = torch.zeros(B, T, self.num_kv_heads * self.head_dim, device=x.device, dtype=x.dtype)
        v = torch.zeros(B, T, self.num_kv_heads * self.head_dim, device=x.device, dtype=x.dtype)

        for expert_id in range(self.num_experts):
            mask = (modality_mask == expert_id)   # (B, T)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=False)    # (N, 2)
            b_idx, t_idx = idx[:, 0], idx[:, 1]
            tokens = x[b_idx, t_idx]              # (N, D)
            q[b_idx, t_idx] = self.q_projs[expert_id](tokens)
            k[b_idx, t_idx] = self.k_projs[expert_id](tokens)
            v[b_idx, t_idx] = self.v_projs[expert_id](tokens)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope_1d(q, k, cos, sin, position_ids)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v)

        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(
                B, self.num_heads, -1, self.head_dim
            )
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(
                B, self.num_heads, -1, self.head_dim
            )

        # Global attention over all tokens
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(attention_mask is None),
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)   # (B, T, num_heads*head_dim)

        # Per-modality output projection
        out = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
        for expert_id in range(self.num_experts):
            mask = (modality_mask == expert_id)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=False)
            b_idx, t_idx = idx[:, 0], idx[:, 1]
            out[b_idx, t_idx] = self.o_projs[expert_id](attn[b_idx, t_idx])

        return out, present


# ── Modality-specific MoE FFN Expert ──────────────────────────────────────────

class ModalityFFNMoE(nn.Module):
    """
    Modality-specific FFN expert (FFN-MMoE).

    Each modality has its own gate, up, and down projection matrices.

    Eq. (4) from the paper:
        FFN-MMoE(x_{i,m}) = (SiLU(x_{i,m} W_gate^m) ⊙ x_{i,m} W_up^m) W_down^m
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        num_experts: int = 2,
        bias: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts

        self.gate_projs = nn.ModuleList([
            nn.Linear(hidden_dim, mlp_dim, bias=bias) for _ in range(num_experts)
        ])
        self.up_projs = nn.ModuleList([
            nn.Linear(hidden_dim, mlp_dim, bias=bias) for _ in range(num_experts)
        ])
        self.down_projs = nn.ModuleList([
            nn.Linear(mlp_dim, hidden_dim, bias=bias) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        """
        x:             (B, T, D)
        modality_mask: (B, T) with integer expert indices
        """
        B, T, D = x.shape
        out = torch.zeros_like(x)

        for expert_id in range(self.num_experts):
            mask = (modality_mask == expert_id)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=False)
            b_idx, t_idx = idx[:, 0], idx[:, 1]
            tokens = x[b_idx, t_idx]
            gate = F.silu(self.gate_projs[expert_id](tokens))
            up   = self.up_projs[expert_id](tokens)
            out[b_idx, t_idx] = self.down_projs[expert_id](gate * up)

        return out


# ── Patch Embedding ───────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Convert image to patch tokens via a strided convolution.
    Patch size = stride = 16 (from paper: "stride of Patch Embedding layer is set to 16").
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        hidden_dim: int = 1472,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        x: (B, 3, H, W)
        Returns: tokens (B, H/P * W/P, D), num_h, num_w
        """
        B, C, H, W = x.shape
        tokens = self.proj(x)                          # (B, D, H/P, W/P)
        num_h, num_w = tokens.shape[2], tokens.shape[3]
        tokens = tokens.flatten(2).transpose(1, 2)     # (B, N, D)
        return tokens, num_h, num_w


# ── Pixel Shuffle Connector ───────────────────────────────────────────────────

class PixelShuffleConnector(nn.Module):
    """
    Downsample visual tokens via pixel shuffle (inverse of sub-pixel convolution),
    then project to LLM hidden dimension via MLP.

    Pixel shuffle with factor r: (B, H*W, D) -> (B, H/r * W/r, D * r^2)
    Then MLP: D * r^2 -> llm_dim
    """

    def __init__(
        self,
        visual_dim: int,
        llm_dim: int,
        downsample_factor: int = 2,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor
        in_dim = visual_dim * (downsample_factor ** 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, llm_dim, bias=False),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=False),
        )

    def forward(self, x: torch.Tensor, num_h: int, num_w: int) -> Tuple[torch.Tensor, int, int]:
        """
        x:     (B, num_h * num_w, visual_dim)
        Returns: projected tokens (B, new_h * new_w, llm_dim), new_h, new_w
        """
        B, N, D = x.shape
        r = self.downsample_factor
        assert num_h % r == 0 and num_w % r == 0, (
            f"Spatial dims ({num_h}, {num_w}) must be divisible by downsample_factor {r}"
        )
        new_h, new_w = num_h // r, num_w // r

        # Reshape to spatial grid, then pixel-unshuffle (merge r×r blocks into channels)
        x = x.view(B, num_h, num_w, D)
        x = x.view(B, new_h, r, new_w, r, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()   # (B, new_h, new_w, r, r, D)
        x = x.view(B, new_h * new_w, r * r * D)         # (B, new_h*new_w, r^2*D)

        return self.mlp(x), new_h, new_w
