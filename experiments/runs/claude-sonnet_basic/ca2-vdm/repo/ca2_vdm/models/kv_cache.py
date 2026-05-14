"""
KV-Cache Queue for Ca2-VDM autoregressive inference.

Implements the temporal KV-cache queue described in Section 3.3.
The queue stores clean KV features from previously generated chunks
and dequeues the oldest entries when the maximum condition length P_max is reached.

Key properties:
  - Cache is shared across all denoising timesteps (cache sharing).
  - Queue structure allows extendable long-term context.
  - Oldest entries are dequeued when P_k reaches P_max.
"""

from typing import Dict, List, Optional, Tuple
import torch


class KVCacheQueue:
    """
    Queue-based KV-cache for temporal attention in Ca2-VDM.

    Stores clean (t=0) key-value pairs from previously generated chunks.
    The cache is shared across all denoising timesteps.

    Args:
        max_frames: P_max, maximum number of conditional frames to keep.
        chunk_size: l, number of frames generated per AR step.
        num_layers: Number of transformer layers (one KV pair per layer).
    """

    def __init__(self, max_frames: int, chunk_size: int, num_layers: int):
        self.max_frames = max_frames
        self.chunk_size = chunk_size
        self.num_layers = num_layers

        # Each layer has a list of (K, V) tensors, one per chunk
        # K, V shape: (B_spatial, l, C) per chunk
        self._k_cache: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_layers)}
        self._v_cache: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_layers)}
        self._num_cached_frames: int = 0

    @property
    def num_cached_frames(self) -> int:
        """Total number of frames currently in the cache."""
        return self._num_cached_frames

    def get_cache(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the concatenated KV cache for a specific layer.

        Returns:
            Tuple (K, V) each of shape (B_spatial, P_k, C), or None if empty.
        """
        if not self._k_cache[layer_idx]:
            return None
        K = torch.cat(self._k_cache[layer_idx], dim=1)  # (B, P_k, C)
        V = torch.cat(self._v_cache[layer_idx], dim=1)
        return K, V

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """
        Add new KV chunk to the cache for a specific layer.
        Dequeues oldest chunk if max_frames is exceeded.

        Args:
            layer_idx: Index of the transformer layer.
            k: Key tensor of shape (B_spatial, l, C).
            v: Value tensor of shape (B_spatial, l, C).
        """
        self._k_cache[layer_idx].append(k)
        self._v_cache[layer_idx].append(v)

    def update_all_layers(self, kvs: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Update all layers with new KV chunks from a cache writing step.
        Dequeues oldest chunk if max_frames is exceeded.

        Args:
            kvs: List of (K, V) tuples, one per layer.
                 Each K, V has shape (B_spatial, l, C).
        """
        assert len(kvs) == self.num_layers, (
            f"Expected {self.num_layers} KV pairs, got {len(kvs)}"
        )

        # Check if we need to dequeue
        if self._num_cached_frames >= self.max_frames:
            # Dequeue oldest chunk from all layers
            for i in range(self.num_layers):
                if self._k_cache[i]:
                    self._k_cache[i].pop(0)
                    self._v_cache[i].pop(0)
            self._num_cached_frames -= self.chunk_size

        # Enqueue new chunk
        for i, (k, v) in enumerate(kvs):
            self._k_cache[i].append(k.detach())
            self._v_cache[i].append(v.detach())

        self._num_cached_frames += self.chunk_size

    def reset(self):
        """Clear all cached KV pairs."""
        self._k_cache = {i: [] for i in range(self.num_layers)}
        self._v_cache = {i: [] for i in range(self.num_layers)}
        self._num_cached_frames = 0

    def __repr__(self) -> str:
        return (
            f"KVCacheQueue(max_frames={self.max_frames}, "
            f"chunk_size={self.chunk_size}, "
            f"num_layers={self.num_layers}, "
            f"cached_frames={self._num_cached_frames})"
        )


class SpatialKVCache:
    """
    Spatial KV-cache for prefix-enhanced spatial attention.

    Unlike the temporal KV-cache queue, the spatial cache only stores
    the most recent chunk (P' frames) and is overwritten at each AR step.

    Args:
        num_layers: Number of transformer layers.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._k_cache: Dict[int, Optional[torch.Tensor]] = {i: None for i in range(num_layers)}
        self._v_cache: Dict[int, Optional[torch.Tensor]] = {i: None for i in range(num_layers)}

    def get_cache(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the spatial KV cache for a specific layer.

        Returns:
            Tuple (K, V) each of shape (P', HW, C), or None if not set.
        """
        k = self._k_cache[layer_idx]
        v = self._v_cache[layer_idx]
        if k is None or v is None:
            return None
        return k, v

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """
        Update the spatial KV cache for a specific layer.

        Args:
            layer_idx: Index of the transformer layer.
            k: Key tensor of shape (P', HW, C).
            v: Value tensor of shape (P', HW, C).
        """
        self._k_cache[layer_idx] = k.detach()
        self._v_cache[layer_idx] = v.detach()

    def update_all_layers(self, kvs: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Update all layers with new spatial KV pairs.

        Args:
            kvs: List of (K, V) tuples, one per layer.
        """
        assert len(kvs) == self.num_layers
        for i, (k, v) in enumerate(kvs):
            self._k_cache[i] = k.detach()
            self._v_cache[i] = v.detach()

    def reset(self):
        """Clear all cached KV pairs."""
        self._k_cache = {i: None for i in range(self.num_layers)}
        self._v_cache = {i: None for i in range(self.num_layers)}
