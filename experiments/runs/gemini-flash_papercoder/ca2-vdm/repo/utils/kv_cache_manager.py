import torch
from collections import deque, defaultdict
from typing import Dict, Any, Tuple, Optional

class KVCacheManager:
    """
    Manages the Key-Value (KV) caches for temporal and spatial attention
    layers during autoregressive video generation.

    Implements a queue for temporal KV-cache and an overwrite mechanism
    for spatial KV-cache, as described in the Ca2-VDM paper.
    """

    def __init__(
        self,
        max_temporal_cache_len: int,
        chunk_len: int,
        temporal_cache_dims: Tuple[int, ...], # Placeholder, actual dims inferred
        spatial_cache_dims: Tuple[int, ...],  # Placeholder, actual dims inferred
        device: torch.device
    ):
        """
        Initializes the KV cache manager.

        Args:
            max_temporal_cache_len (int): The maximum number of frames whose KVs will be stored
                                          in the temporal cache (P_max from the paper).
            chunk_len (int): The number of frames generated/processed in each autoregressive step (l).
            temporal_cache_dims (Tuple[int, ...]): Placeholder for expected temporal K/V tensor dimensions.
                                                   Actual dimensions are inferred dynamically.
            spatial_cache_dims (Tuple[int, ...]): Placeholder for expected spatial K/V tensor dimensions.
                                                  Actual dimensions are inferred dynamically.
            device (torch.device): The computational device (e.g., 'cuda', 'cpu') for storing tensors.
        """
        self.max_temporal_cache_len: int = max_temporal_cache_len
        self.chunk_len: int = chunk_len
        self.device: torch.device = device

        # Temporal KV-cache: stores KVs for historical frames.
        # Uses defaultdict to dynamically create deques for each layer.
        # Each deque stores (K, V) tensors corresponding to a 'chunk_len' segment of frames.
        # The structure for _temporal_kv_cache will be:
        # {
        #   'layer_name_1': {'K': deque([tensor_k_chunk1, tensor_k_chunk2, ...]),
        #                    'V': deque([tensor_v_chunk1, tensor_v_chunk2, ...])},
        #   'layer_name_2': ...
        # }
        self._temporal_kv_cache: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {'K': deque(), 'V': deque()}
        )
        self._current_temporal_frames_count: int = 0

        # Spatial KV-cache: stores KVs for the most recent P' frames for prefix enhancement.
        # This is a simple overwrite mechanism, not a queue.
        # The structure for _spatial_kv_cache will be:
        # {
        #   'layer_name_1': {'K': tensor_k_latest, 'V': tensor_v_latest},
        #   'layer_name_2': ...
        # }
        self._spatial_kv_cache: Dict[str, Dict[str, Optional[torch.Tensor]]] = defaultdict(
            lambda: {'K': None, 'V': None}
        )

        # Storing placeholder dimensions for completeness, as per design.
        # They are not directly used for allocation due to dynamic inference.
        self._temporal_cache_dims: Tuple[int, ...] = temporal_cache_dims
        self._spatial_cache_dims: Tuple[int, ...] = spatial_cache_dims

    def update_temporal_cache(self, new_kvs: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """
        Adds KVs from the newly generated chunk to the temporal cache and manages its size.
        This method is called after a new chunk of frames has been denoised and their
        clean KVs computed.

        Args:
            new_kvs (Dict[str, Dict[str, torch.Tensor]]): A dictionary mapping layer names
                                                          to their K/V tensors for the latest chunk.
                                                          Each K/V tensor should represent KVs for `self.chunk_len` frames.
        """
        if not new_kvs:
            return

        # Enqueue new KVs for each layer
        for layer_name, kv_pair in new_kvs.items():
            k_tensor = kv_pair['K'].to(self.device)
            v_tensor = kv_pair['V'].to(self.device)
            self._temporal_kv_cache[layer_name]['K'].append(k_tensor)
            self._temporal_kv_cache[layer_name]['V'].append(v_tensor)

        # Assume new_kvs represents one chunk of frames added.
        # The `chunk_len` of the new KVs is derived from the second dimension (sequence length)
        # of the K/V tensors from the *first* layer provided in `new_kvs`.
        # This is safer than using self.chunk_len directly if the input `new_kvs`
        # could represent less than self.chunk_len frames, but typically it should align.
        # Here we rely on `self.chunk_len` as per the paper, that each update adds `l` frames.
        self._current_temporal_frames_count += self.chunk_len

        # Dequeue oldest KVs if cache size exceeds max_temporal_cache_len
        # The paper states P_max. If P_k reaches P_max, earliest KVs are dequeued.
        # This means the total frames in cache should not exceed P_max.
        while self._current_temporal_frames_count > self.max_temporal_cache_len:
            for layer_name in self._temporal_kv_cache:
                if self._temporal_kv_cache[layer_name]['K']: # Ensure deque is not empty
                    self._temporal_kv_cache[layer_name]['K'].popleft()
                    self._temporal_kv_cache[layer_name]['V'].popleft()
            self._current_temporal_frames_count -= self.chunk_len # Decrement by chunk_len

    def get_temporal_cache(self) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        """
        Retrieves the concatenated temporal KV cache for all attention layers.
        The KVs for each layer are concatenated along the sequence length dimension.

        Returns:
            Dict[str, Dict[str, Optional[torch.Tensor]]]: A dictionary mapping layer names
                                                          to their concatenated K/V tensors.
                                                          Returns None for K or V if deque is empty.
        """
        concatenated_kvs: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}

        for layer_name, deques in self._temporal_kv_cache.items():
            k_list = list(deques['K'])
            v_list = list(deques['V'])

            concatenated_k = torch.cat(k_list, dim=1) if k_list else None
            concatenated_v = torch.cat(v_list, dim=1) if v_list else None

            concatenated_kvs[layer_name] = {'K': concatenated_k, 'V': concatenated_v}
        return concatenated_kvs

    def update_spatial_cache(self, new_kvs: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """
        Updates the spatial KV cache by overwriting existing entries with KVs
        from the most recent frames (the sub-prefix for enhancement).
        This cache is not a queue; it only holds the KVs for the latest required sub-prefix.

        Args:
            new_kvs (Dict[str, Dict[str, torch.Tensor]]): A dictionary mapping layer names
                                                          to their K/V tensors for the spatial sub-prefix.
        """
        if not new_kvs:
            return

        # Overwrite KVs for each layer
        for layer_name, kv_pair in new_kvs.items():
            self._spatial_kv_cache[layer_name]['K'] = kv_pair['K'].to(self.device)
            self._spatial_kv_cache[layer_name]['V'] = kv_pair['V'].to(self.device)

    def get_spatial_cache(self) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        """
        Retrieves the current spatial KV cache for all attention layers.

        Returns:
            Dict[str, Dict[str, Optional[torch.Tensor]]]: A dictionary mapping layer names
                                                          to their K/V tensors for the spatial sub-prefix.
        """
        # Return a copy to prevent external modification of internal state
        return {layer: {k: v for k, v in kv_pair.items()} for layer, kv_pair in self._spatial_kv_cache.items()}

