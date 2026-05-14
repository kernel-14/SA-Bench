import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def l2_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Normalize x to unit norm along the specified dimension (Norm in paper)."""
    return x / (x.norm(dim=dim, keepdim=True).clamp(min=eps))


def normalize_matrix_rows(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize each row of a 2-D weight matrix to unit norm.

    For nn.Linear(in, out), weight shape is (out, in).  Each row is a
    d_model-dimensional vector that is dot-producted with the input, so
    normalizing rows makes every dot product a cosine similarity.
    """
    return w / (w.norm(dim=1, keepdim=True).clamp(min=eps))


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE) — Su et al. (2024)
# ---------------------------------------------------------------------------

def build_rope_cache(
    seq_len: int,
    d_head: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cos/sin tables for RoPE.

    Returns:
        cos, sin: tensors of shape (seq_len, d_head)
    """
    theta = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device, dtype=dtype) / d_head))
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(positions, theta)          # (seq_len, d_head/2)
    emb = torch.cat([freqs, freqs], dim=-1)        # (seq_len, d_head)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to query or key tensor.

    Args:
        x:   (batch, n_heads, seq_len, d_head)
        cos: (seq_len, d_head)  or  (1, 1, seq_len, d_head)
        sin: same shape as cos
    """
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, seq_len, d_head)
        sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# RMSNorm — used only in the baseline GPT
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)
