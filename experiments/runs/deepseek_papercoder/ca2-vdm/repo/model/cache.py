## model/cache.py
"""
KV‑cache manager for Ca2‑VDM's autoregressive inference (Sec. 3.3).

The class is responsible for storing and providing:
- **Temporal KV‑caches** (per `CausalTemporalAttention` layer): a queue of
  clean (timestep‑0) keys and values of all previously generated frames,
  with a maximum length `P_max`.  Each layer's cache is a tensor of shape
  `(N, H, P_k, d_h)` where `N` is the number of spatial tokens per frame
  (1024 for 32×32 latents), `H` is the number of heads, and `d_h` the
  per‑head dimension.
- **Spatial KV‑caches** (per `PrefixEnhancedSpatialAttention` layer): the
  raw spatial keys/values of the most recently generated chunk (length `l`),
  stored as `(B, l, H, N, d_h)`.  Only the last `p_prime` frames are used
  as prefix to enhance the current noisy frames.

The manager is entirely passive: it does not interact with attention
operations directly, but provides tensors that the attention layers
concatenate internally.

All internal tensors reside on the same device as the original input
(to avoid unnecessary data movement).  The manager is intended for
inference only; gradients are never tracked.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch


class CacheManager:
    """
    Manages temporal and spatial KV‑caches across transformer blocks.

    Parameters
    ----------
    max_temporal_length : int
        ``P_max`` – maximum number of conditional frames kept in the
        temporal cache queue (e.g. 49 for T2V with l=16).
    p_prime : int
        ``P'`` – number of frames used for spatial attention prefix
        enhancement (e.g. 3).
    chunk_size : int
        ``l`` – number of frames generated in each autoregressive chunk.
    num_temporal_layers : int
        Number of `CausalTemporalAttention` layers in the model.
    num_spatial_layers : int
        Number of `PrefixEnhancedSpatialAttention` layers in the model.
    device : torch.device
        Device on which the cache tensors will be created (inferred from
        the first addition; used for initial empty state).
    """

    def __init__(
        self,
        max_temporal_length: int,
        p_prime: int,
        chunk_size: int,
        num_temporal_layers: int,
        num_spatial_layers: int,
        device: torch.device,
    ) -> None:
        if max_temporal_length <= 0:
            raise ValueError("max_temporal_length must be positive")
        if p_prime <= 0:
            raise ValueError("p_prime must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.max_temporal_length: int = max_temporal_length
        self.p_prime: int = p_prime
        self.chunk_size: int = chunk_size
        self.num_temporal_layers: int = num_temporal_layers
        self.num_spatial_layers: int = num_spatial_layers
        self.device: torch.device = device

        # Temporal caches start empty (will be populated on first addition)
        self.temporal_k: List[Optional[torch.Tensor]] = [None] * num_temporal_layers
        self.temporal_v: List[Optional[torch.Tensor]] = [None] * num_temporal_layers

        # Spatial caches start empty; populated by update_spatial_cache
        self.spatial_k: List[Optional[torch.Tensor]] = [None] * num_spatial_layers
        self.spatial_v: List[Optional[torch.Tensor]] = [None] * num_spatial_layers

    # ------------------------------------------------------------------
    # Temporal cache management
    # ------------------------------------------------------------------

    def add_to_temporal_cache(
        self,
        k_list: List[torch.Tensor],
        v_list: List[torch.Tensor],
    ) -> None:
        """
        Append the clean temporal keys/values of a newly generated chunk
        and enforce the maximum queue length.

        Parameters
        ----------
        k_list : List of Tensor, length ``num_temporal_layers``
            Each tensor has shape ``(N, H, l, d_h)`` where ``l`` is the
            number of new frames (``chunk_size``).  The tensor must be on
            the same device as the existing cache.
        v_list : List of Tensor, length ``num_temporal_layers``
            Corresponding values, same shape.

        Raises
        ------
        ValueError
            If ``k_list``/``v_list`` length does not match the number of
            temporal layers, or if a layer's shape is inconsistent.
        """
        if len(k_list) != self.num_temporal_layers or len(v_list) != self.num_temporal_layers:
            raise ValueError(
                f"Expected {self.num_temporal_layers} entries per list, got "
                f"{len(k_list)} keys and {len(v_list)} values."
            )

        for i in range(self.num_temporal_layers):
            k_new = k_list[i]
            v_new = v_list[i]
            if k_new.shape != v_new.shape:
                raise ValueError(
                    f"Shape mismatch at layer {i}: k {k_new.shape}, v {v_new.shape}"
                )
            if k_new.dim() != 4:
                raise ValueError(
                    f"Temporal cache tensor at layer {i} must have 4 dimensions "
                    f"(N, H, L, d_h), got {k_new.shape}"
                )

            # Optionally move to the manager's device (should already match)
            if k_new.device != self.device:
                k_new = k_new.to(self.device)
                v_new = v_new.to(self.device)

            if self.temporal_k[i] is None:
                # First chunk
                self.temporal_k[i] = k_new
                self.temporal_v[i] = v_new
            else:
                # Concatenate along the temporal dimension (dim=2)
                self.temporal_k[i] = torch.cat([self.temporal_k[i], k_new], dim=2)
                self.temporal_v[i] = torch.cat([self.temporal_v[i], v_new], dim=2)

            # Enforce maximum length: keep only the last max_temporal_length frames
            cur_len = self.temporal_k[i].shape[2]
            if cur_len > self.max_temporal_length:
                self.temporal_k[i] = self.temporal_k[i][:, :, -self.max_temporal_length:, :]
                self.temporal_v[i] = self.temporal_v[i][:, :, -self.max_temporal_length:, :]

    def get_temporal_cache(self) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """
        Return the current temporal key and value caches for all layers.

        Returns
        -------
        temporal_k : List of Optional[Tensor]
            Each tensor has shape ``(N, H, P_k, d_h)``, where ``P_k`` is
            the current number of conditional frames (0 if no cache yet).
            ``None`` is returned for layers that have never been written.
        temporal_v : List of Optional[Tensor]
            Corresponding values, same shape.
        """
        # Return shallow copies of the lists so that callers cannot
        # inadvertently mutate the manager's internal state.
        return copy.copy(self.temporal_k), copy.copy(self.temporal_v)

    # ------------------------------------------------------------------
    # Spatial cache management
    # ------------------------------------------------------------------

    def update_spatial_cache(
        self,
        k_list: List[torch.Tensor],
        v_list: List[torch.Tensor],
    ) -> None:
        """
        Overwrite the spatial cache with the raw spatial keys/values of
        the most recent clean chunk.

        Parameters
        ----------
        k_list : List of Tensor, length ``num_spatial_layers``
            Each tensor has shape ``(B, l, H, N, d_h)``.  ``B`` is the
            batch size (usually 1 during inference), ``l`` is the chunk
            size, and ``N`` is the number of spatial tokens per frame.
        v_list : List of Tensor, length ``num_spatial_layers``
            Corresponding values, same shape.
        """
        if len(k_list) != self.num_spatial_layers or len(v_list) != self.num_spatial_layers:
            raise ValueError(
                f"Expected {self.num_spatial_layers} entries per list, got "
                f"{len(k_list)} keys and {len(v_list)} values."
            )

        for i in range(self.num_spatial_layers):
            k_new = k_list[i]
            v_new = v_list[i]
            if k_new.shape != v_new.shape:
                raise ValueError(
                    f"Spatial cache shape mismatch at layer {i}: "
                    f"k {k_new.shape}, v {v_new.shape}"
                )
            # Store a detached clone to avoid retaining any graph
            self.spatial_k[i] = k_new.detach().clone()
            self.spatial_v[i] = v_new.detach().clone()

    def get_spatial_cache(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Retrieve the spatial prefix (last ``p_prime`` frames) for the
        prefix‑enhanced spatial attention.

        The method guarantees that for each layer a tensor of shape
        ``(B, p_prime, H, N, d_h)`` is returned, even if the stored cache
        is shorter than ``p_prime`` (in which case the available frames
        are self‑repeated to reach the desired length).

        Returns
        -------
        prefix_k : List of Tensor
            One tensor per spatial attention layer, shape
            ``(B, p_prime, H, N, d_h)``.
        prefix_v : List of Tensor
            Corresponding values, same shape.

        Raises
        ------
        RuntimeError
            If any spatial cache layer has not been initialised (i.e. is
            ``None``).  The inference pipeline must call
            ``update_spatial_cache`` at least once before denoising.
        """
        prefix_k: List[torch.Tensor] = []
        prefix_v: List[torch.Tensor] = []

        for i in range(self.num_spatial_layers):
            if self.spatial_k[i] is None or self.spatial_v[i] is None:
                raise RuntimeError(
                    f"Spatial cache for layer {i} has not been initialised. "
                    "Call update_spatial_cache first."
                )

            k_full: torch.Tensor = self.spatial_k[i]   # (B, L, H, N, d_h)
            v_full: torch.Tensor = self.spatial_v[i]

            cur_len = k_full.shape[1]   # L
            if cur_len >= self.p_prime:
                # Take the last p_prime frames
                k_prefix = k_full[:, -self.p_prime:, :, :, :]
                v_prefix = v_full[:, -self.p_prime:, :, :, :]
            else:
                # Self‑repeat the available frames to form a sequence of
                # exactly p_prime frames, mimicking the training behaviour
                # for clean prefix frames (Eq. 4 of the paper).
                # We repeat the whole tensor along the time axis enough times,
                # then take the last p_prime slices.
                repeat_factor = self.p_prime // cur_len + 1
                k_rep = k_full.repeat(1, repeat_factor, 1, 1, 1)    # (B, R*L, H, N, d_h)
                v_rep = v_full.repeat(1, repeat_factor, 1, 1, 1)
                k_prefix = k_rep[:, -self.p_prime:, :, :, :]
                v_prefix = v_rep[:, -self.p_prime:, :, :, :]

            prefix_k.append(k_prefix)
            prefix_v.append(v_prefix)

        return prefix_k, prefix_v

