## model/embeddings.py
"""
Spatial and cyclic temporal positional embeddings for Ca2‑VDM.

Implements :
- ``SpatialPositionalEmbedding`` : fixed 2D sinusoidal encoding for all spatial patches
  of a frame, shared across all frames.
- ``CyclicTPE`` : cyclic temporal positional embeddings (Cyclic‑TPE) that wrap around
  after ``L_train`` positions, enabling the KV‑cache queue to work with the
  autoregressive generation beyond the training length.

The cyclic design is described in Sec. 3.3 of the paper; training‑time random cyclic
shifts and inference‑time modulo indexing guarantee that the positional encodings
always form a contiguous cyclic segment, matching the training distribution.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

from config import Config
from utils.positional_encodings import get_sinusoidal_encoding


class SpatialPositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal spatial positional embedding for each frame.

    The embedding has shape ``(1, N, D)`` where ``N = H_latent * W_latent``
    and ``D`` is the model hidden dimension.  It is added identically to
    every frame in the sequence.
    """

    def __init__(self, num_patches: int, dim: int) -> None:
        """
        Args:
            num_patches: Number of spatial tokens per frame (e.g. 32*32 = 1024).
            dim: Model hidden dimension.
        """
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {dim}")

        # Create the sinusoidal encoding table of shape (num_patches, dim)
        enc = get_sinusoidal_encoding(num_patches, dim)  # (N, D)
        # Register as a non‑trainable buffer: shape (1, N, D)
        self.register_buffer("encoding", enc.unsqueeze(0))

    def forward(self, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Return the spatial positional embedding, broadcastable to
        ``(batch, frames, N, D)``.

        Args:
            batch_size: Batch size for the leading dimension (default 1).
            device: If given, ensure the returned tensor is on this device.

        Returns:
            Tensor of shape ``(batch_size, 1, N, D)`` ready for broadcasting.
        """
        # encoding shape is (1, 1, N, D) after expansion
        enc = self.encoding  # (1, N, D)
        enc = enc.unsqueeze(1)  # (1, 1, N, D)
        if device is not None and enc.device != device:
            enc = enc.to(device)
        return enc.expand(batch_size, -1, -1, -1)  # (batch, 1, N, D)


class CyclicTPE(nn.Module):
    """
    Cyclic Temporal Positional Embeddings (Cyclic‑TPE).

    Holds a base sinusoidal table of length ``L_train`` (e.g. 65).  During
    training each clip is assigned a randomly shifted block of indices;
    during autoregressive inference, frames receive global indices modulo
    ``L_train``, ensuring the KV‑cache queue remains well‑formed.
    """

    def __init__(self, max_len: int, dim: int) -> None:
        """
        Args:
            max_len: Maximum training clip length (``L_train_max = P_max + l``).
            dim: Model hidden dimension.
        """
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {dim}")

        self.max_len = max_len

        # Base table of sinusoidal encodings: shape (max_len, dim)
        base = get_sinusoidal_encoding(max_len, dim)
        self.register_buffer("base", base)

    # ------------------------------------------------------------------
    # Low‑level index helpers
    # ------------------------------------------------------------------
    def _embed_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Gather embeddings from the base table given a 1‑D index tensor.

        Args:
            indices: 1‑D LongTensor of shape ``(L,)`` containing position indices
                     in ``[0, self.max_len-1]``.

        Returns:
            Tensor of shape ``(L, dim)`` with the corresponding embeddings.
        """
        return self.base[indices]

    # ------------------------------------------------------------------
    # Inference API – used by the autoregressive pipeline
    # ------------------------------------------------------------------
    def get_inference_tpe(self, global_indices: torch.Tensor) -> torch.Tensor:
        """
        Return TPEs for a given list of **global** frame indices.

        Performs ``global_indices % self.max_len`` to wrap cyclically.

        Args:
            global_indices: 1‑D Long/Int tensor of shape ``(L,)``.

        Returns:
            Tensor of shape ``(L, dim)``.
        """
        local_indices = global_indices % self.max_len
        return self._embed_indices(local_indices)

    def get_next_tpe_indices(self, last_global_idx: int, num_frames: int) -> torch.Tensor:
        """
        Helper that returns the global indices for a new chunk of frames,
        starting right after ``last_global_idx``.

        Args:
            last_global_idx: The global index of the last already‑generated frame.
            num_frames: Number of frames in the new chunk.

        Returns:
            1‑D LongTensor of shape ``(num_frames,)``.
        """
        return torch.arange(
            last_global_idx + 1,
            last_global_idx + 1 + num_frames,
            dtype=torch.long,
            device=self.base.device,
        )

    # ------------------------------------------------------------------
    # Training API – used by the dataset / trainer
    # ------------------------------------------------------------------
    @staticmethod
    def sample_shifts(
        num_samples: int,
        max_len: int,
        generator: Optional[torch.Generator] = None,
    ) -> List[int]:
        """
        Sample uniformly random cyclic shifts for ``num_samples`` clips.

        Args:
            num_samples: Number of shifts to generate.
            max_len: ``L_train`` (the number of unique TPE positions).
            generator: Optional ``torch.Generator`` for reproducibility.

        Returns:
            List of ``num_samples`` integers, each in ``[0, max_len-1]``.
        """
        shifts = torch.randint(
            0, max_len, (num_samples,), generator=generator
        )
        return shifts.tolist()

    def get_train_tpe(
        self,
        lengths: List[int],
        shifts: Optional[List[int]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> List[torch.Tensor]:
        """
        Return a **list** of TPE tensors for a batch of training clips.

        Each tensor has shape ``(length_i, dim)``.  If ``shifts`` is not
        provided, random shifts are sampled internally (using ``generator``).

        Args:
            lengths: List of clip lengths (one per sample).
            shifts: Optional list of cyclic shift offsets, same length as ``lengths``.
                     If ``None``, random shifts are sampled uniformly.
            generator: ``torch.Generator`` for the random sampling (ignored if
                       ``shifts`` is given).

        Returns:
            List of ``len(lengths)`` tensors, each ``(length_i, dim)``.
        """
        if shifts is not None and len(shifts) != len(lengths):
            raise ValueError(
                f"Lengths ({len(lengths)}) and shifts ({len(shifts)}) must have the same length."
            )

        # Sample random shifts if not provided
        if shifts is None:
            shifts = self.sample_shifts(len(lengths), self.max_len, generator=generator)

        tpe_list: List[torch.Tensor] = []
        for length, shift in zip(lengths, shifts):
            # local indices within the cyclic block: (shift + 0, shift+1, ...)
            indices = torch.arange(length, dtype=torch.long, device=self.base.device)
            indices = (indices + shift) % self.max_len
            tpe = self._embed_indices(indices)  # (length, dim)
            tpe_list.append(tpe)

        return tpe_list

    def prepare_training_tpe_padded(
        self,
        lengths: List[int],
        shifts: Optional[List[int]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Higher‑level helper that returns **padded** TPEs and a boolean mask.

        This is convenient for batching variable‑length clips: zeros are
        placed for padded positions, and the mask indicates valid frames.

        Args:
            lengths: List of clip lengths (one per sample).
            shifts: Optional cyclic shifts; if ``None``, random shifts are sampled.
            generator: ``torch.Generator`` for shift sampling.

        Returns:
            - ``tpe_padded``: Float (or the base table's dtype) tensor of shape
              ``(batch, max_length, dim)``, where ``max_length`` is the maximum
              value in ``lengths``.
            - ``mask``: Bool tensor of shape ``(batch, max_length)``, ``True``
              for valid frames, ``False`` for padding.
        """
        batch = len(lengths)
        max_len = max(lengths)

        # Get the list of (L_i, D) tensors
        tpe_list = self.get_train_tpe(lengths, shifts=shifts, generator=generator)

        # Create padded tensor and mask
        dtype = self.base.dtype
        device = self.base.device
        tpe_padded = torch.zeros(batch, max_len, self.base.shape[-1], dtype=dtype, device=device)
        mask = torch.zeros(batch, max_len, dtype=torch.bool, device=device)

        for i, (tpe, length) in enumerate(zip(tpe_list, lengths)):
            tpe_padded[i, :length] = tpe
            mask[i, :length] = True

        return tpe_padded, mask

