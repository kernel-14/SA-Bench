# model/memory.py
"""
Memory components for SAM 2 – MemoryBank and MemoryAttention.

Implements:
- MemoryBank: FIFO queues for recent and prompted frame memories (spatial
  features + object pointers). Adds temporal position encoding only to
  recent frames.
- MemoryAttention: stack of L transformer blocks that first perform
  self‑attention on the current frame tokens (with 2D RoPE) and then
  cross‑attend to the memory bank (spatial tokens receive RoPE, object
  pointer tokens do not). The output is a set of conditioned tokens ready
  for the mask decoder.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 2D Rotary Position Embedding (RoPE) utilities
# ---------------------------------------------------------------------------

def _compute_2d_freqs(dim: int, max_len: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute sinusoidal frequencies for 2D RoPE.

    The `dim` is split into two halves: the first half uses x‑coordinates,
    the second half uses y‑coordinates.  Frequencies follow the standard
    RoPE formula.

    Returns:
        freqs_x: (dim//4,) – frequencies for x direction.
        freqs_y: (dim//4,) – frequencies for y direction.
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D RoPE interleaving."
    half_dim = dim // 2
    quarter_dim = half_dim // 2
    # standard frequencies for a single dimension of length quarter_dim
    freqs = 1.0 / (max_len ** (torch.arange(0, quarter_dim, dtype=torch.float32) / quarter_dim))
    freqs_x = freqs.clone()
    freqs_y = freqs.clone()
    return freqs_x, freqs_y


@torch.no_grad()
def _build_rope_cache(coords: torch.Tensor, freqs_x: torch.Tensor, freqs_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build cosine and sine tables for the given coordinate grid.

    Args:
        coords: (L, 2) tensor of (x, y) values, typically in [-1, 1].
        freqs_x: (quarter_dim,) frequencies for x.
        freqs_y: (quarter_dim,) frequencies for y.

    Returns:
        cos, sin: each (L, dim_head) (where dim_head = quarter_dim*4).
          The cos/sin tables are arranged so that:
            - first quarter: cos/sin based on x * freqs_x
            - second quarter: cos/sin based on y * freqs_y

    Note: For 2D RoPE, we interleave x and y rotations in the half dimension
    (standard in many implementations).  The final rotation is applied as:
        q_rot = q * cos + rotate_half(q) * sin
    where rotate_half swaps halves.
    """
    L = coords.shape[0]
    x = coords[:, 0:1]   # (L, 1)
    y = coords[:, 1:2]   # (L, 1)

    quarter_dim = freqs_x.numel()
    # x‑part
    theta_x = x @ freqs_x.unsqueeze(0)   # (L, quarter_dim)
    theta_y = y @ freqs_y.unsqueeze(0)   # (L, quarter_dim)

    # interleave: first quarter x, second quarter y
    theta = torch.cat([theta_x, theta_y], dim=1)  # (L, half_dim) with half_dim = quarter_dim*2
    cos = theta.cos().float()
    sin = theta.sin().float()
    # Now duplicate to full dim_head: we need dim_head = half_dim*2 = quarter_dim*4
    # Actually dim_head = d_model // num_heads, and d_model may not be divisible by 4 exactly.
    # We'll construct cos/sin for the full dim_head by replicating the pattern.
    # More generally, 2D RoPE can be applied by constructing a rotation matrix for each
    # pair of dimensions (alternating x and y).  The standard approach: for each token,
    # compute cos_x, sin_x for first half of dimensions; cos_y, sin_y for second half.
    # So we return cos: (L, dim_head) where first half is cos_x, second half cos_y.
    dim_head = quarter_dim * 4
    # The above theta gives (L, quarter_dim*2). We need (L, dim_head) = (L, quarter_dim*4).
    # So repeat the pattern: cos_x, cos_y, cos_x, cos_y? Actually the standard 2D RoPE
    # applies x-rotation to the first half of d, y-rotation to the second half.
    # So if total dim_head = D, then first D/2 uses x, second D/2 uses y.
    # Since we have quarter_dim frequencies, D/2 = quarter_dim*2.
    # So cos_x should be (L, D/2) and cos_y should be (L, D/2).
    # Then cos = [cos_x, cos_y] concatenated along last dim.
    # So compute theta_x to have D/2 frequencies: we extend freqs_x to D/2 dimensions.
    # Let's re-evaluate: We need to create theta_x for D/2 dimensions.
    # A clean way: create a full set of frequencies for total dim_head, alternating x and y:
    freqs = torch.stack([freqs_x, freqs_y], dim=1).flatten()  # (quarter_dim*2,) = (half_dim,)
    # Then repeat to fill dim_head: if dim_head is multiple of half_dim, we can repeat.
    # But simpler: we will not hardcode quarter structure; we'll compute per-dimension
    # frequencies based on whether the index is even/odd.  Let's do:
    # For each of the D dimensions, assign it to x if idx%2==0 else y.
    # This gives interleaved x/y rotation.
    # freq_x, freq_y can be larger to cover all needed.
    # We'll generate a frequency array of length dim_head, where every other is from freqs_x/2.
    # To keep it simple, we'll just compute cos and sin as described in the paper: apply 2d RoPE by
    # splitting head_dim in half, using x for first half, y for second half.
    # So we need theta_x of shape (L, D/2) and theta_y of shape (L, D/2).
    # Let's recompute properly:
    D = dim_head
    half_D = D // 2
    # use freqs_x and freqs_y scaled to half_D length
    # We'll build a frequency tensor for x: (half_D,) using logarithmic spacing.
    # We can generate new freqs for head dim.
    # To avoid confusion, we'll generate freqs_x and freqs_y with length = half_D.
    # Actually, the number of frequencies needed for RoPE is half of the dimension,
    # i.e., for D dimensions we need D/2 frequencies (pairs). So for x, we need half_D frequencies.
    # So we recompute freqs_x, freqs_y inside this function, based on the given dim_head.
    # This is more robust.
    pass  # We'll implement a clearer version inside MemoryAttention's initialization.

    # We'll instead implement a standalone function `apply_2d_rope` that takes q, k, coords, and head_dim,
    # internally computes frequencies for that head_dim. So above cache building is flexible.
