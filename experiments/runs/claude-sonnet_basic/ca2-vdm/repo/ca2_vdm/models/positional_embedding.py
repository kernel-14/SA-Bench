"""
Positional Embeddings for Ca2-VDM.

Implements:
  - Sinusoidal Spatial Positional Embeddings (SPEs)
  - Sinusoidal Temporal Positional Embeddings (TPEs)
  - Cyclic-TPEs: Cyclic shift mechanism for TPEs during autoregressive inference
    to handle videos longer than the training length.

From Section 3.3 of Ca2-VDM:
  "sinusoidal spatial and temporal positional embeddings (i.e., SPEs and TPEs)
   are added to the frame sequence following Vision Transformer (ViT)."
  "To ensure TPEs are correctly assigned when the cumulatively generated video
   exceeds the training length, we carefully design a cyclic shift mechanism:
   Cyclic-TPEs."
"""

import math
import torch
import torch.nn as nn
from typing import Optional


def get_sinusoidal_embedding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute sinusoidal positional embeddings for given positions.

    Args:
        positions: 1D tensor of position indices, shape (N,).
        dim: Embedding dimension.

    Returns:
        Embeddings of shape (N, dim).
    """
    assert dim % 2 == 0, "Embedding dimension must be even"
    half_dim = dim // 2
    # Frequencies: 1 / (10000^(2i/dim)) for i in [0, half_dim)
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32, device=positions.device) / half_dim
    )
    # positions: (N,), freq: (half_dim,)
    args = positions.float().unsqueeze(1) * freq.unsqueeze(0)  # (N, half_dim)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (N, dim)
    return emb


class SpatialPositionalEmbedding(nn.Module):
    """
    Sinusoidal 2D spatial positional embedding (SPE).

    Adds separate sinusoidal embeddings for height and width positions,
    following ViT-style positional encoding.

    Args:
        dim: Embedding dimension (must be even, split equally for H and W).
        max_height: Maximum height in patches.
        max_width: Maximum width in patches.
    """

    def __init__(self, dim: int, max_height: int = 32, max_width: int = 32):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for 2D SPE"
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        half_dim = dim // 2

        # Precompute embeddings
        h_pos = torch.arange(max_height)
        w_pos = torch.arange(max_width)
        h_emb = get_sinusoidal_embedding(h_pos, half_dim)  # (H, dim/2)
        w_emb = get_sinusoidal_embedding(w_pos, half_dim)  # (W, dim/2)

        # Create 2D grid: (H, W, dim)
        h_emb_grid = h_emb.unsqueeze(1).expand(-1, max_width, -1)  # (H, W, dim/2)
        w_emb_grid = w_emb.unsqueeze(0).expand(max_height, -1, -1)  # (H, W, dim/2)
        spe = torch.cat([h_emb_grid, w_emb_grid], dim=-1)  # (H, W, dim)
        spe = spe.view(max_height * max_width, dim)  # (H*W, dim)

        self.register_buffer("spe", spe)

    def forward(self, h: int, w: int) -> torch.Tensor:
        """
        Get SPE for a given spatial resolution.

        Args:
            h: Height in patches.
            w: Width in patches.

        Returns:
            SPE of shape (h*w, dim).
        """
        assert h <= self.max_height and w <= self.max_width
        device = self.spe.device
        half_dim = self.dim // 2
        h_pos = torch.arange(h, device=device)
        w_pos = torch.arange(w, device=device)
        h_emb = get_sinusoidal_embedding(h_pos, half_dim)  # (h, dim/2)
        w_emb = get_sinusoidal_embedding(w_pos, half_dim)  # (w, dim/2)
        h_emb_grid = h_emb.unsqueeze(1).expand(-1, w, -1)  # (h, w, dim/2)
        w_emb_grid = w_emb.unsqueeze(0).expand(h, -1, -1)  # (h, w, dim/2)
        spe = torch.cat([h_emb_grid, w_emb_grid], dim=-1)  # (h, w, dim)
        return spe.view(h * w, self.dim)


class TemporalPositionalEmbedding(nn.Module):
    """
    Sinusoidal temporal positional embedding (TPE) with Cyclic-TPE support.

    During training, each sample is assigned a TPE sequence that is cyclically
    shifted with a random offset (to support Cyclic-TPEs at inference time).

    During inference, when the cumulative video length exceeds L_train,
    the denoising target frames are assigned TPEs from the beginning (cyclic shift).

    Args:
        dim: Embedding dimension.
        max_len: Maximum sequence length (L_train = P_max + l).
    """

    def __init__(self, dim: int, max_len: int = 65):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        # Precompute base TPE for positions 0..max_len-1
        positions = torch.arange(max_len)
        tpe = get_sinusoidal_embedding(positions, dim)  # (max_len, dim)
        self.register_buffer("tpe", tpe)

    def get_tpe(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """
        Get TPE for given frame indices (with cyclic wrapping).

        Args:
            frame_indices: 1D tensor of frame indices (can exceed max_len).

        Returns:
            TPE of shape (len(frame_indices), dim).
        """
        # Cyclic wrapping: indices modulo max_len
        cyclic_indices = frame_indices % self.max_len
        return self.tpe[cyclic_indices]

    def get_training_tpe(
        self,
        seq_len: int,
        cyclic_offset: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Get TPE for training with optional cyclic shift.

        During training, each sample is assigned a cyclically shifted TPE
        to support the Cyclic-TPE mechanism at inference time.

        Args:
            seq_len: Length of the sequence (L = P + l).
            cyclic_offset: Random cyclic offset. If None, no shift (offset=0).
            device: Target device.

        Returns:
            TPE of shape (seq_len, dim).
        """
        if device is None:
            device = self.tpe.device
        if cyclic_offset is None:
            cyclic_offset = 0
        indices = torch.arange(seq_len, device=device) + cyclic_offset
        return self.get_tpe(indices)

    def get_inference_tpe(
        self,
        p_k: int,
        chunk_size: int,
        p_max: int,
        ar_step: int,
    ) -> torch.Tensor:
        """
        Get TPE for autoregressive inference with Cyclic-TPE mechanism.

        When P_k < P_max (early AR steps), TPEs are assigned sequentially.
        When P_k >= P_max (cache queue is full), the denoising target frames
        are assigned TPEs cyclically from the beginning.

        Args:
            p_k: Number of clean prefix frames at current AR step.
            chunk_size: l, number of frames in the denoising target.
            p_max: Maximum number of conditional frames.
            ar_step: Current autoregression step (0-indexed).

        Returns:
            TPE for the denoising target of shape (chunk_size, dim).
        """
        device = self.tpe.device

        if p_k < p_max:
            # Early AR steps: sequential TPE assignment
            # Denoising target starts at position p_k
            indices = torch.arange(p_k, p_k + chunk_size, device=device)
        else:
            # Cyclic-TPE: denoising target wraps around
            # The denoising target is assigned TPEs from position 0 cyclically
            # (as if the earliest cached frames were dequeued and TPEs reset)
            start_idx = p_k % self.max_len
            indices = torch.arange(start_idx, start_idx + chunk_size, device=device)

        return self.get_tpe(indices)

    def get_cache_tpe(self, p_k: int, p_max: int) -> torch.Tensor:
        """
        Get TPE for the clean prefix frames in the KV-cache.

        Args:
            p_k: Number of clean prefix frames.
            p_max: Maximum number of conditional frames.

        Returns:
            TPE for the cache of shape (p_k, dim).
        """
        device = self.tpe.device
        if p_k <= p_max:
            indices = torch.arange(p_k, device=device)
        else:
            # Should not happen in normal usage
            indices = torch.arange(p_k, device=device) % self.max_len
        return self.get_tpe(indices)

    def forward(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """
        Get TPE for given frame indices.

        Args:
            frame_indices: 1D tensor of frame indices.

        Returns:
            TPE of shape (len(frame_indices), dim).
        """
        return self.get_tpe(frame_indices)
