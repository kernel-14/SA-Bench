"""
KV-Cache queue implementation for Ca2-VDM.

Two types of caches:
1. Temporal KV-Cache Queue: Stores K, V for all prefix frames.
   Managed as a queue with max length P_max.
   When queue is full, earliest frames are dequeued.
   
2. Spatial KV-Cache: Stores K, V for most recent P' frames.
   Overwritten at each AR step (not queued).
"""

import torch
from typing import Optional, Tuple, List
from collections import deque


class TemporalKVCacheQueue:
    """
    Temporal KV-cache queue for causal temporal attention.
    
    Stores (K, V) pairs for each frame across all layers.
    Supports enqueue (cache writing) and dequeue operations.
    
    Shape per layer: K, V each (B*S, nH, L_frame, d)
    where L_frame is number of frames in the cache.
    """
    
    def __init__(self, max_length: int):
        """
        Args:
            max_length: P_max, maximum number of conditional frames
        """
        self.max_length = max_length
        self.num_frames = 0
        # Stores list of (K, V) tuples, one per layer per frame chunk
        self.cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        
    def enqueue(
        self, 
        k: torch.Tensor, 
        v: torch.Tensor
    ):
        """
        Add new KV pairs to the queue.
        
        Args:
            k: keys of shape (*, L, d) where L is chunk length
            v: values of shape (*, L, d)
        """
        self.cache.append((k, v))
        self.num_frames += k.shape[-2]  # L dimension
        
    def dequeue(self, num_frames: int):
        """
        Remove earliest frames from queue.
        
        Args:
            num_frames: number of frames to remove
        """
        frames_to_remove = num_frames
        while frames_to_remove > 0 and len(self.cache) > 0:
            k, v = self.cache[0]
            L = k.shape[-2]
            if L <= frames_to_remove:
                self.cache.pop(0)
                self.num_frames -= L
                frames_to_remove -= L
            else:
                # Partially remove from this chunk
                self.cache[0] = (k[..., L - frames_to_remove:, :], v[..., L - frames_to_remove:, :])
                self.num_frames -= frames_to_remove
                frames_to_remove = 0
    
    def get_kv(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get concatenated K and V for all cached frames.
        
        Returns:
            (K, V) each of shape (*, total_frames, d) or None if empty
        """
        if len(self.cache) == 0:
            return None
        
        k_list = [item[0] for item in self.cache]
        v_list = [item[1] for item in self.cache]
        
        k_cat = torch.cat(k_list, dim=-2)
        v_cat = torch.cat(v_list, dim=-2)
        
        return k_cat, v_cat
    
    def __len__(self):
        return self.num_frames
    
    def is_full(self):
        return self.num_frames >= self.max_length
    
    def clear(self):
        self.cache.clear()
        self.num_frames = 0


class KVCacheManager:
    """
    Manages both temporal and spatial KV-caches across all model layers.
    
    This is the main cache management class used during autoregressive inference.
    Provides cache sharing across denoising timesteps.
    """
    
    def __init__(
        self,
        num_layers: int,
        P_max: int,
        P_prime: int = 3,
    ):
        """
        Args:
            num_layers: number of transformer layers in the model
            P_max: maximum number of conditional frames for temporal cache
            P_prime: number of prefix frames for spatial cache
        """
        self.num_layers = num_layers
        self.P_max = P_max
        self.P_prime = P_prime
        
        # One temporal KV-cache queue per layer
        self.temporal_caches: List[TemporalKVCacheQueue] = [
            TemporalKVCacheQueue(P_max) for _ in range(num_layers)
        ]
        
        # One spatial KV-cache per layer (only most recent P' frames)
        self.spatial_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(num_layers)
        ]
        
    def get_temporal_kv(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get combined temporal KV cache for a layer."""
        return self.temporal_caches[layer_idx].get_kv()
    
    def get_spatial_kv(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get spatial KV cache for a layer."""
        return self.spatial_caches[layer_idx]
    
    def update_temporal(
        self, 
        layer_idx: int, 
        k: torch.Tensor, 
        v: torch.Tensor,
        l: int,
    ):
        """
        Update temporal cache for a layer after generating l new frames.
        
        Args:
            layer_idx: layer index
            k, v: new KV from generated chunk, shape (B*S, nH, l, d)
            l: chunk length
        """
        cache = self.temporal_caches[layer_idx]
        
        # Enqueue new K, V
        cache.enqueue(k, v)
        
        # Dequeue if exceeds P_max
        if cache.num_frames > self.P_max:
            cache.dequeue(cache.num_frames - self.P_max)
    
    def update_spatial(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """
        Update spatial cache for a layer (overwrite).
        
        Args:
            layer_idx: layer index
            k, v: new KV from generated chunk, shape (B, P', S, C)
        """
        self.spatial_caches[layer_idx] = (k, v)
    
    def reset(self):
        """Reset all caches (for new video generation)."""
        for cache in self.temporal_caches:
            cache.clear()
        self.spatial_caches = [None for _ in range(self.num_layers)]
