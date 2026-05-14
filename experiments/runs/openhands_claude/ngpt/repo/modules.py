"""
Attention and MLP blocks for both the baseline GPT and nGPT.

Key nGPT differences from baseline GPT (Section 2):
  - No RMSNorm on inputs; all weight matrices are kept unit-norm
  - QK normalization + learnable per-head scaling s_qk
  - Softmax scale is sqrt(d_k) instead of 1/sqrt(d_k)
  - MLP intermediate states scaled by s_u, s_v * sqrt(d_model)
  - Block outputs are normalized before the LERP update
  - LERP update: h <- Norm(h + alpha * (h_block - h))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from layers import RMSNorm, l2_norm, apply_rope


# ---------------------------------------------------------------------------
# Scaling-parameter helper
# ---------------------------------------------------------------------------

class ScaledParameter(nn.Module):
    """Learnable parameter with the nGPT effective-learning-rate trick.

    The raw parameter is stored at `s_scale`; during the forward pass the
    actual value is recovered by multiplying by `s_init / s_scale`.  This
    keeps the Adam effective step-size proportional to `s_scale` while the
    semantic value at initialisation equals `s_init`.

    Section 2.5 of the paper.
    """

    def __init__(self, shape: Tuple[int, ...], s_init: float, s_scale: float):
        super().__init__()
        self.s_init = s_init
        self.s_scale = s_scale
        self.param = nn.Parameter(torch.full(shape, s_scale))

    def forward(self) -> torch.Tensor:
        return self.param * (self.s_init / self.s_scale)


# ---------------------------------------------------------------------------
# Baseline GPT — Self-Attention Block
# ---------------------------------------------------------------------------

class GPTAttention(nn.Module):
    """Multi-head causal self-attention for the baseline GPT (Section 2.3.1)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm = RMSNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = h.shape
        x = self.norm(h)

        q = self.Wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scale = 1.0 / math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.Wo(out)


# ---------------------------------------------------------------------------
# Baseline GPT — MLP Block
# ---------------------------------------------------------------------------

class GPTMLP(nn.Module):
    """SwiGLU MLP for the baseline GPT (Section 2.4.1)."""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.Wu = nn.Linear(d_model, d_mlp, bias=False)
        self.Wv = nn.Linear(d_model, d_mlp, bias=False)
        self.Wo = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.norm(h)
        u = self.Wu(x)
        v = self.Wv(x)
        return self.Wo(u * F.silu(v))


# ---------------------------------------------------------------------------
# nGPT — Self-Attention Block
# ---------------------------------------------------------------------------

class NGPTAttention(nn.Module):
    """Multi-head causal self-attention for nGPT (Section 2.3.2).

    Changes vs. GPT:
      - No RMSNorm on input (h is already unit-norm)
      - Wq, Wk, Wv, Wo rows are kept unit-norm (enforced externally)
      - q and k are normalized then scaled by s_qk after RoPE
      - Softmax scale = sqrt(d_head) (not 1/sqrt(d_head))
      - Returns Norm(ATTN(h)) — the normalized block output h_A
    """

    def __init__(self, d_model: int, n_heads: int, sqk_init: float, sqk_scale: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        # Per-head QK scaling s_qk ∈ R^{d_head} (Section 2.3.2, eq. 15-16)
        # One s_qk vector shared across all heads (same init/scale for each)
        self.sqk = ScaledParameter((self.d_head,), sqk_init, sqk_scale)

    def forward(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = h.shape

        q = self.Wq(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # QK normalization + scaling (eq. 15-16)
        sqk = self.sqk()                          # (d_head,)
        q = l2_norm(q, dim=-1) * sqk
        k = l2_norm(k, dim=-1) * sqk

        # Softmax scale = sqrt(d_head) for normalized vectors (Section 2.3.2)
        scale = math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.Wo(out)

        # Return normalized block output h_A = Norm(ATTN(h))
        return l2_norm(out, dim=-1)


# ---------------------------------------------------------------------------
# nGPT — MLP Block
# ---------------------------------------------------------------------------

class NGPTMLP(nn.Module):
    """SwiGLU MLP for nGPT (Section 2.4.2).

    Changes vs. GPT:
      - No RMSNorm on input
      - Wu, Wv, Wo rows are kept unit-norm (enforced externally)
      - u scaled by s_u; v scaled by s_v * sqrt(d_model) (eq. 20-21)
      - Returns Norm(MLP(h)) — the normalized block output h_M
    """

    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        su_init: float,
        su_scale: float,
        sv_init: float,
        sv_scale: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.Wu = nn.Linear(d_model, d_mlp, bias=False)
        self.Wv = nn.Linear(d_model, d_mlp, bias=False)
        self.Wo = nn.Linear(d_mlp, d_model, bias=False)

        self.su = ScaledParameter((d_mlp,), su_init, su_scale)
        self.sv = ScaledParameter((d_mlp,), sv_init, sv_scale)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = self.Wu(h)
        v = self.Wv(h)

        # Scaling (eq. 20-21); v rescaled by sqrt(d_model) for SiLU non-linearity
        su = self.su()
        sv = self.sv()
        u = u * su
        v = v * sv * math.sqrt(self.d_model)

        out = self.Wo(u * F.silu(v))

        # Return normalized block output h_M = Norm(MLP(h))
        return l2_norm(out, dim=-1)


# ---------------------------------------------------------------------------
# nGPT — Transformer Layer
# ---------------------------------------------------------------------------

class NGPTLayer(nn.Module):
    """Single nGPT layer combining attention and MLP with LERP updates.

    Update equations (Table 1 / eq. 10-11):
        h_A = Norm(ATTN(h))
        h   = Norm(h + alpha_A * (h_A - h))
        h_M = Norm(MLP(h))
        h   = Norm(h + alpha_M * (h_M - h))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        alpha_init: float,
        alpha_scale: float,
        sqk_init: float,
        sqk_scale: float,
        su_init: float,
        su_scale: float,
        sv_init: float,
        sv_scale: float,
    ):
        super().__init__()
        self.attn = NGPTAttention(d_model, n_heads, sqk_init, sqk_scale)
        self.mlp = NGPTMLP(d_model, d_mlp, su_init, su_scale, sv_init, sv_scale)

        # Eigen learning rates alpha_A, alpha_M ∈ R^{d_model} (Section 2.2.2)
        self.alpha_A = ScaledParameter((d_model,), alpha_init, alpha_scale)
        self.alpha_M = ScaledParameter((d_model,), alpha_init, alpha_scale)

    def forward(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention update
        h_A = self.attn(h, cos, sin, mask)
        alpha_A = self.alpha_A().abs()            # constrained positive (Appendix A.2)
        h = l2_norm(h + alpha_A * (h_A - h), dim=-1)

        # MLP update
        h_M = self.mlp(h)
        alpha_M = self.alpha_M().abs()
        h = l2_norm(h + alpha_M * (h_M - h), dim=-1)

        return h


# ---------------------------------------------------------------------------
# Baseline GPT — Transformer Layer
# ---------------------------------------------------------------------------

class GPTLayer(nn.Module):
    """Single baseline GPT layer (pre-norm, eq. 4-5)."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.attn = GPTAttention(d_model, n_heads)
        self.mlp = GPTMLP(d_model, d_mlp)

    def forward(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = h + self.attn(h, cos, sin, mask)
        h = h + self.mlp(h)
        return h
