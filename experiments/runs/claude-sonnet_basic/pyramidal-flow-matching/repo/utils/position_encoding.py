"""
Position encoding utilities for Pyramidal Flow Matching.

Implements:
1. 2D sinusoidal position embeddings for spatial dimensions
   - Extrapolation for spatial pyramid (higher resolution stages)
   - Interpolation for temporal pyramid (history conditions)
2. 1D Rotary Position Embedding (RoPE) for temporal dimension

From Section 3.4 of the paper:
"we extrapolate position encoding in the spatial pyramid for better fine-grained
detail (Yang et al., 2024), while interpolating it in the temporal pyramid input
to spatially align the history conditions."
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def get_2d_sinusoidal_pos_embed(
    embed_dim: int,
    grid_size_h: int,
    grid_size_w: int,
    cls_token: bool = False,
    extrapolate: bool = False,
    base_grid_size_h: Optional[int] = None,
    base_grid_size_w: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate 2D sinusoidal position embeddings.
    
    Supports extrapolation for spatial pyramid (higher resolution stages)
    and interpolation for temporal pyramid (history conditions).
    
    Args:
        embed_dim: Embedding dimension
        grid_size_h: Height of the grid (number of patches)
        grid_size_w: Width of the grid (number of patches)
        cls_token: Whether to include a CLS token position
        extrapolate: Whether to extrapolate positions (for spatial pyramid)
        base_grid_size_h: Base grid height for extrapolation
        base_grid_size_w: Base grid width for extrapolation
    
    Returns:
        Position embeddings (grid_size_h * grid_size_w, embed_dim)
        or (1 + grid_size_h * grid_size_w, embed_dim) if cls_token=True
    """
    if extrapolate and base_grid_size_h is not None:
        # Extrapolate: scale positions beyond base resolution
        # This allows the model to handle higher resolutions than seen during training
        h_positions = torch.arange(grid_size_h).float() * (grid_size_h / base_grid_size_h)
        w_positions = torch.arange(grid_size_w).float() * (grid_size_w / base_grid_size_w)
    else:
        h_positions = torch.arange(grid_size_h).float()
        w_positions = torch.arange(grid_size_w).float()
    
    # Create 2D grid
    grid_h, grid_w = torch.meshgrid(h_positions, w_positions, indexing='ij')
    grid_h = grid_h.flatten()  # (H*W,)
    grid_w = grid_w.flatten()  # (H*W,)
    
    # Compute sinusoidal embeddings for each dimension
    half_dim = embed_dim // 4  # Split between H and W, and sin/cos
    
    emb_h = _get_1d_sinusoidal_emb(grid_h, half_dim * 2)  # (H*W, half_dim*2)
    emb_w = _get_1d_sinusoidal_emb(grid_w, half_dim * 2)  # (H*W, half_dim*2)
    
    # Concatenate H and W embeddings
    pos_emb = torch.cat([emb_h, emb_w], dim=-1)  # (H*W, embed_dim)
    
    if cls_token:
        cls_emb = torch.zeros(1, embed_dim)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=0)
    
    return pos_emb


def get_2d_sinusoidal_pos_embed_interpolated(
    embed_dim: int,
    grid_size_h: int,
    grid_size_w: int,
    target_grid_size_h: int,
    target_grid_size_w: int,
) -> torch.Tensor:
    """
    Generate 2D sinusoidal position embeddings with interpolation.
    
    Used for temporal pyramid history conditions to spatially align
    lower-resolution history with the current generation.
    
    Args:
        embed_dim: Embedding dimension
        grid_size_h: Source grid height
        grid_size_w: Source grid width
        target_grid_size_h: Target grid height
        target_grid_size_w: Target grid width
    
    Returns:
        Interpolated position embeddings (target_H * target_W, embed_dim)
    """
    # Get source embeddings
    src_emb = get_2d_sinusoidal_pos_embed(embed_dim, grid_size_h, grid_size_w)
    
    # Reshape to 2D grid
    src_emb = src_emb.view(grid_size_h, grid_size_w, embed_dim)
    src_emb = src_emb.permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
    
    # Interpolate to target size
    tgt_emb = F.interpolate(
        src_emb,
        size=(target_grid_size_h, target_grid_size_w),
        mode='bilinear',
        align_corners=False,
    )
    
    # Reshape back
    tgt_emb = tgt_emb.squeeze(0).permute(1, 2, 0)  # (H', W', D)
    tgt_emb = tgt_emb.view(target_grid_size_h * target_grid_size_w, embed_dim)
    
    return tgt_emb


def _get_1d_sinusoidal_emb(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute 1D sinusoidal embeddings.
    
    Args:
        positions: Position values (N,)
        dim: Embedding dimension
    
    Returns:
        Embeddings (N, dim)
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=positions.device) * -emb)
    emb = positions[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    return emb


def get_1d_rope_embed(
    dim: int,
    seq_len: int,
    device: torch.device,
    base: int = 10000,
    interpolate_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate 1D Rotary Position Embedding (RoPE) for temporal dimension.
    
    Supports interpolation for temporal pyramid history conditions.
    
    Args:
        dim: Head dimension
        seq_len: Sequence length
        device: Device
        base: Base for frequency computation
        interpolate_factor: Factor for position interpolation (< 1 = compress positions)
    
    Returns:
        Tuple of (cos, sin) tensors of shape (seq_len, dim)
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    # Compute positions with optional interpolation
    t = torch.arange(seq_len, device=device).float()
    if interpolate_factor != 1.0:
        t = t * interpolate_factor
    
    # Compute frequencies
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    
    return emb.cos(), emb.sin()


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding to query and key tensors.
    
    Args:
        q: Query tensor (B, H, L, head_dim)
        k: Key tensor (B, H, L, head_dim)
        cos: Cosine embeddings (L, head_dim)
        sin: Sine embeddings (L, head_dim)
    
    Returns:
        Rotated (q, k) tensors
    """
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    
    return q_rot, k_rot
