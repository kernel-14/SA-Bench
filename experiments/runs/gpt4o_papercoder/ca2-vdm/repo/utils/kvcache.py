## utils/kvcache.py
import torch
from torch import Tensor
from typing import Tuple, Optional


class KVCache:
    """
    KVCache manages caching of keys and values for temporal and spatial attention in a queue for 
    temporal frames and an overwrite strategy for spatial frames. This optimization ensures efficient 
    memory and computational management during autoregressive video generation.
    """
    def __init__(self, max_length: int, spatial_size: Tuple[int, int], channels: int) -> None:
        """
        Initialize the KVCache object.
        
        Args:
            max_length (int): Maximum number of cached frames for temporal caching.
            spatial_size (Tuple[int, int]): Spatial resolution (H, W) of the keys/values.
            channels (int): Number of feature channels (e.g., attention head dimensions).
        """
        self.max_length = max_length
        self.spatial_size = spatial_size
        self.channels = channels

        # Temporal cache (FIFO queue)
        self.temporal_keys = []  # List of tensors for storing keys
        self.temporal_values = []  # List of tensors for storing values

        # Spatial cache (overwrite strategy)
        self.spatial_keys: Optional[Tensor] = None  # Tensor for the most recent spatial keys
        self.spatial_values: Optional[Tensor] = None  # Tensor for the most recent spatial values

    def enqueue(self, keys: Tensor, values: Tensor, is_temporal: bool = True) -> None:
        """
        Enqueue new keys and values into the cache. Either a temporal cache (queue) or
        a spatial cache (overwrite) can be updated.

        Args:
            keys (Tensor): New keys to cache. Shape:
                           Temporal: (batch_size, channels) or (sequence_len, channels).
                           Spatial: (batch_size, H, W, channels).
            values (Tensor): New values to cache. Shape matches the keys.
            is_temporal (bool): If True, update the temporal cache. Update the spatial cache otherwise.
        """
        if is_temporal:
            # Temporal cache uses a queue. Append keys and values.
            self.temporal_keys.append(keys)
            self.temporal_values.append(values)

            # Dequeue oldest entries if cache size exceeds max_length.
            if len(self.temporal_keys) > self.max_length:
                self.temporal_keys.pop(0)
                self.temporal_values.pop(0)
        else:
            # Spatial cache uses an overwrite strategy.
            self.spatial_keys = keys
            self.spatial_values = values

    def read_current_cache(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Retrieve the current cached keys and values.

        Returns:
            Tuple[Optional[Tensor], Optional[Tensor]]: Cached keys and values. For temporal cache,
            tensors are concatenated along the first dimension. For spatial cache, directly return
            the stored tensors.
            
            Temporal: Keys shape (total_cached_seq_len, channels), Values shape (total_cached_seq_len, channels).
            Spatial: Keys shape (batch_size, H, W, channels), Values shape (batch_size, H, W, channels).
        """
        if self.temporal_keys:
            stacked_keys = torch.cat(self.temporal_keys, dim=0)
            stacked_values = torch.cat(self.temporal_values, dim=0)
            return stacked_keys, stacked_values
        else:
            return self.spatial_keys, self.spatial_values

    def reset(self) -> None:
        """
        Reset both the temporal and spatial caches, clearing all stored keys and values.
        """
        # Clear temporal cache (FIFO queue).
        self.temporal_keys.clear()
        self.temporal_values.clear()

        # Reset spatial cache (overwrite strategy).
        self.spatial_keys = None
        self.spatial_values = None

    def __len__(self) -> int:
        """
        Return the length of the temporal cache, i.e., the number of sequences currently stored.

        Returns:
            int: The length of the temporal cache.
        """
        return len(self.temporal_keys)
