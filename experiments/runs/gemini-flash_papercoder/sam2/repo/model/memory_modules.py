import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

# Placeholder for Config type hint to avoid circular import with config.py
# In a real project, this would be 'from config import Config'
Config = Any


class MemoryEncoder(nn.Module):
    """
    Generates a condensed memory representation (memory feature) for a given frame.
    It integrates information from the predicted segmentation mask and the unconditioned
    image embedding from the image encoder.
    """

    def __init__(self, config: Config):
        """
        Initializes the MemoryEncoder.

        Args:
            config (Config): The global configuration object.
        """
        super().__init__()
        self._config = config

        # The dimension of the unconditioned frame embedding from the image encoder's FPN output.
        # This is expected to be the same as the hidden_dim for MemoryAttention.
        self.frame_embedding_dim: int = self._config.get("model.memory_attention.hidden_dim", 256)
        
        # The target channel dimension for the output memory feature.
        self.memory_feature_dim: int = self._config.get("model.memory_bank.memory_feature_dim", 64)

        if self.frame_embedding_dim <= 0 or self.memory_feature_dim <= 0:
            raise ValueError(
                f"frame_embedding_dim ({self.frame_embedding_dim}) and memory_feature_dim "
                f"({self.memory_feature_dim}) must be positive integers."
            )

        # Convolutional module for mask processing:
        # Transforms the single-channel predicted mask into a feature map
        # with the same channel dimension as the frame embedding.
        self.mask_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.frame_embedding_dim,
            kernel_size=3,
            padding=1,
            bias=True # SAM's mask head usually has bias
        )

        # "Light-weight convolutional layers to fuse the information"
        # A small sequential block for fusing the summed feature.
        # Using GroupNorm for normalization, which is common in vision models.
        self.fuse_convs = nn.Sequential(
            nn.Conv2d(self.frame_embedding_dim, self.frame_embedding_dim, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(min(32, self.frame_embedding_dim), self.frame_embedding_dim), # Use min(32, C) for small C
            nn.ReLU(inplace=True),
            nn.Conv2d(self.frame_embedding_dim, self.frame_embedding_dim, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(min(32, self.frame_embedding_dim), self.frame_embedding_dim),
        )

        # Final projection layer to reduce the fused feature to the desired memory_feature_dim.
        self.project_conv = nn.Conv2d(
            in_channels=self.frame_embedding_dim,
            out_channels=self.memory_feature_dim,
            kernel_size=1,
            bias=True
        )

    def forward(
        self, predicted_mask: torch.Tensor, frame_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Processes a predicted mask and frame embedding to create a memory feature.

        Args:
            predicted_mask (torch.Tensor): A binary segmentation mask from the mask decoder.
                                           Expected shape: (B, 1, H_in, W_in).
            frame_embedding (torch.Tensor): The unconditioned feature map for the current frame
                                            from the image encoder's FPN output.
                                            Expected shape: (B, C_feat, H_feat, W_feat).

        Returns:
            torch.Tensor: The generated memory feature. Shape: (B, memory_feature_dim, H_feat, W_feat).
        """
        # 1. Resize predicted_mask to match the spatial dimensions of frame_embedding.
        # Convert to float for interpolation. align_corners=False is typically used for feature maps.
        resized_mask = F.interpolate(
            predicted_mask.float(),  
            size=frame_embedding.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # 2. Apply mask_conv to the resized mask.
        mask_conv_feature = self.mask_conv(resized_mask)

        # 3. Perform element-wise addition of mask_conv_feature and frame_embedding.
        # This fuses the mask information with the visual features.
        fused_feature = mask_conv_feature + frame_embedding

        # 4. Pass the fused_feature through light-weight convolutional layers.
        fused_feature = self.fuse_convs(fused_feature)

        # 5. Apply the final projection layer to get the desired memory feature dimension.
        memory_feature = self.project_conv(fused_feature)

        return memory_feature


class MemoryBank(nn.Module):
    """
    Manages and stores memory features and object pointers for both recent (unprompted)
    frames and explicitly prompted frames. Provides mechanisms for updating, retrieving,
    and resetting these memories, incorporating temporal position and occlusion information.
    """

    def __init__(self, config: Config):
        """
        Initializes the MemoryBank.

        Args:
            config (Config): The global configuration object.
        """
        super().__init__()
        self._config = config

        self.max_recent_frames: int = self._config.get("model.memory_bank.max_recent_frames", 6)
        self.max_prompted_frames: int = self._config.get("model.memory_bank.max_prompted_frames", 2)
        self.memory_feature_dim: int = self._config.get("model.memory_bank.memory_feature_dim", 64)
        self.object_pointer_dim: int = self._config.get("model.memory_bank.object_pointer.dim", 256) # Total dim
        self.object_pointer_token_dim: int = self.object_pointer_dim // 4 # Each of 4 tokens is 64-dim

        if not (self.max_recent_frames >= 0 and self.max_prompted_frames >= 0):
            raise ValueError("max_recent_frames and max_prompted_frames must be non-negative.")
        if self.memory_feature_dim <= 0 or self.object_pointer_dim <= 0:
            raise ValueError("memory_feature_dim and object_pointer_dim must be positive.")
        if self.object_pointer_dim % 4 != 0:
            raise ValueError("object_pointer_dim must be divisible by 4 for splitting into tokens.")

        # FIFO queues for recent (unprompted) memories
        self.recent_memory_features: Deque[torch.Tensor] = deque(maxlen=self.max_recent_frames)
        self.recent_object_pointers: Deque[torch.Tensor] = deque(maxlen=self.max_recent_frames)
        self.recent_frame_indices: Deque[int] = deque(maxlen=self.max_recent_frames) # To track time for temporal embeddings

        # FIFO queues for explicitly prompted memories
        # "Stores information from prompts in a FIFO queue of up to M prompted frames."
        self.prompted_memory_features: Deque[torch.Tensor] = deque(maxlen=self.max_prompted_frames)
        self.prompted_object_pointers: Deque[torch.Tensor] = deque(maxlen=self.max_prompted_frames)
        self.prompted_frame_indices: Deque[int] = deque(maxlen=self.max_prompted_frames) # To maintain order, if needed

        # Learned occlusion embedding, added to memory features if frame is predicted occluded.
        # Shape (1, C, 1, 1) allows broadcasting to (C, H, W)
        self.occlusion_embedding = nn.Parameter(torch.randn(1, self.memory_feature_dim, 1, 1))

    def update(
        self,
        frame_idx: int,
        memory_feature: torch.Tensor,
        object_pointer: torch.Tensor,
        is_prompted: bool,
        is_occluded: bool = False,
    ) -> None:
        """
        Adds a new memory feature and its corresponding object pointer to the memory bank.
        Manages the FIFO queues and applies occlusion embedding if necessary.

        Args:
            frame_idx (int): The absolute index of the current frame.
            memory_feature (torch.Tensor): Memory feature generated by MemoryEncoder.
                                           Expected shape: (B=1, memory_feature_dim, H_feat, W_feat).
            object_pointer (torch.Tensor): Object pointer from MaskDecoder output tokens.
                                           Expected shape: (B=1, object_pointer_dim).
            is_prompted (bool): True if this memory comes from an explicitly prompted frame.
            is_occluded (bool, optional): True if the object is predicted to be occluded in this frame.
                                          Defaults to False.
        """
        # The SAM2Model design indicates that for multiple objects, inference is run
        # independently, meaning MemoryBank operates on a single object at a time.
        # So, the batch dimension B should be 1, or absent for a single object.
        if memory_feature.ndim == 4 and memory_feature.shape[0] == 1:
            mem_feat_squeezed = memory_feature.squeeze(0) # (C_mem, H_feat, W_feat)
        elif memory_feature.ndim == 3:
            mem_feat_squeezed = memory_feature # Already (C_mem, H_feat, W_feat)
        else:
            raise ValueError(f"memory_feature has unexpected dimensions: {memory_feature.shape}")
        
        if object_pointer.ndim == 2 and object_pointer.shape[0] == 1:
            obj_ptr_squeezed = object_pointer.squeeze(0) # (D_op)
        elif object_pointer.ndim == 1:
            obj_ptr_squeezed = object_pointer # Already (D_op)
        else:
            raise ValueError(f"object_pointer has unexpected dimensions: {object_pointer.shape}")


        # Apply occlusion embedding if the object is occluded
        if is_occluded:
            # Add occlusion_embedding (1, C, 1, 1) to mem_feat_squeezed (C, H, W) via broadcasting
            # occlusion_embedding.squeeze(0) makes it (C, 1, 1) for spatial broadcasting
            mem_feat_squeezed = mem_feat_squeezed + self.occlusion_embedding.squeeze(0)

        if is_prompted:
            self.prompted_memory_features.append(mem_feat_squeezed)
            self.prompted_object_pointers.append(obj_ptr_squeezed)
            self.prompted_frame_indices.append(frame_idx)
        else:
            self.recent_memory_features.append(mem_feat_squeezed)
            self.recent_object_pointers.append(obj_ptr_squeezed)
            self.recent_frame_indices.append(frame_idx)

    def get_memories(self, current_frame_idx: int) -> Dict[str, List[torch.Tensor]]:
        """
        Retrieves all stored memories (features and object pointers) and calculates
        relative temporal positions for recent frames, suitable for MemoryAttention.

        Args:
            current_frame_idx (int): The absolute index of the current frame being processed.
                                     Used to calculate relative temporal positions.

        Returns:
            Dict[str, List[torch.Tensor]]: A dictionary containing:
                - 'memory_features': List of all memory feature tensors (C_mem, H_feat, W_feat).
                - 'object_pointers': List of all object pointer tensors, split into 4 tokens (4, D_op/4).
                - 'temporal_pos_embeddings': List of 1-element tensors, each containing the
                                             relative temporal position for its corresponding memory feature.
                                             (0 for prompted frames, current_frame_idx - stored_frame_idx for recent).
        """
        all_memory_features: List[torch.Tensor] = []
        all_object_pointers_split: List[torch.Tensor] = []
        all_temporal_pos_embeddings: List[torch.Tensor] = []

        # Collect recent memories and their temporal embeddings
        for i, (mem_feat, obj_ptr, stored_idx) in enumerate(
            zip(self.recent_memory_features, self.recent_object_pointers, self.recent_frame_indices)
        ):
            all_memory_features.append(mem_feat)
            # Split object pointer into 4 tokens of object_pointer_dim // 4
            all_object_pointers_split.append(obj_ptr.reshape(4, self.object_pointer_token_dim))
            
            # Calculate relative temporal position: current_frame - stored_frame
            relative_pos = current_frame_idx - stored_idx
            # Ensure the tensor is on the correct device, e.g., from mem_feat
            all_temporal_pos_embeddings.append(torch.tensor([relative_pos], dtype=torch.int64, device=mem_feat.device))

        # Collect prompted memories
        # For prompted frames, temporal position information is explicitly NOT embedded.
        # We use a value (e.g., 0) to indicate this, which MemoryAttention can interpret.
        for i, (mem_feat, obj_ptr, _) in enumerate(
            zip(self.prompted_memory_features, self.prompted_object_pointers, self.prompted_frame_indices)
        ):
            all_memory_features.append(mem_feat)
            # Split object pointer into 4 tokens of object_pointer_dim // 4
            all_object_pointers_split.append(obj_ptr.reshape(4, self.object_pointer_token_dim))
            
            # Use 0 to indicate no temporal position embedding for prompted frames.
            all_temporal_pos_embeddings.append(torch.tensor([0], dtype=torch.int64, device=mem_feat.device))

        return {
            'memory_features': all_memory_features,
            'object_pointers': all_object_pointers_split,
            'temporal_pos_embeddings': all_temporal_pos_embeddings,
        }

    def reset(self) -> None:
        """
        Clears all stored memories in the memory bank, preparing it for a new video or object.
        """
        self.recent_memory_features.clear()
        self.recent_object_pointers.clear()
        self.recent_frame_indices.clear()
        self.prompted_memory_features.clear()
        self.prompted_object_pointers.clear()
        self.prompted_frame_indices.clear()

