"""
Memory Bank for SAM 2.

The memory bank retains information about past predictions for the target object:
- FIFO queue of memories of up to N recent frames (default N=6)
- FIFO queue of up to M prompted frames
- Object pointers: lightweight vectors from mask decoder output tokens

Key details (Section 4, Appendix D.1):
- Both sets of memories stored as spatial feature maps
- Temporal position information embedded into N recent frame memories
- NOT embedded into prompted frame memories (sparser training signal)
- Object pointers: 256-dim token split into 4 tokens of 64-dim for cross-attention
- Occlusion embedding: learned embedding added to memory features of frames predicted as occluded
- Memory features stored at dimension 64 (4x smaller than 256, reduces storage)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MemoryBankOutput:
    """Output of the memory bank for use by memory attention."""
    spatial_memory: torch.Tensor  # [B, M_spatial, C] flattened spatial memory features
    object_pointers: torch.Tensor  # [B, M_obj, C] object pointer tokens
    temporal_positions: Optional[torch.Tensor]  # [M_spatial] temporal position indices
    memory_positions: Optional[torch.Tensor]  # [M_spatial, 2] spatial positions for RoPE


class MemoryBank(nn.Module):
    """
    Memory bank: stores spatial memories and object pointers with FIFO queues.
    """

    def __init__(
        self,
        memory_dim: int = 64,
        max_recent_frames: int = 6,
        max_prompted_frames: int = 4,  # M
        object_pointer_dim: int = 256,
        object_pointer_tokens: int = 4,
        feature_map_size: Tuple[int, int] = (64, 64),
    ):
        """
        Args:
            memory_dim: dimension of spatial memory features (64)
            max_recent_frames: max number of recent frames to store (N=6)
            max_prompted_frames: max number of prompted frames to store (M)
            object_pointer_dim: dimension of a single object pointer (256)
            object_pointer_tokens: number of 64-dim tokens to split pointer into (4)
            feature_map_size: spatial size of memory feature maps
        """
        super().__init__()
        self.memory_dim = memory_dim
        self.max_recent_frames = max_recent_frames
        self.max_prompted_frames = max_prompted_frames
        self.object_pointer_dim = object_pointer_dim
        self.object_pointer_tokens = object_pointer_tokens
        self.feature_map_size = feature_map_size
        self.H, self.W = feature_map_size

        # Learned occlusion embedding added to frames predicted as occluded
        self.occlusion_embed = nn.Parameter(torch.zeros(1, memory_dim, self.H, self.W))
        nn.init.normal_(self.occlusion_embed, std=0.02)

        # Projection of object pointer from 256 -> 4x64 tokens
        self.object_pointer_proj = nn.Linear(object_pointer_dim, memory_dim * object_pointer_tokens)

        # Initialize empty memory state
        self.reset_state()

    def reset_state(self):
        """Reset all memory state for a new video/object."""
        self._recent_memories: List[torch.Tensor] = []  # list of [1, C, H, W]
        self._prompted_memories: List[torch.Tensor] = []
        self._object_pointers: List[torch.Tensor] = []  # list of [1, 256]
        self._recent_temporal_ids: List[int] = []
        self._memory_positions: List[torch.Tensor] = []  # list of [H*W, 2]
        self._frame_count: int = 0

    def add_memory(
        self,
        memory_features: torch.Tensor,
        object_pointer: torch.Tensor,
        is_prompted: bool = False,
        is_occluded: bool = False,
    ):
        """
        Add a new memory to the bank.

        Args:
            memory_features: [B, memory_dim, H, W] spatial memory features
            object_pointer: [B, 256] object pointer token from mask decoder
            is_prompted: whether this frame was prompted
            is_occluded: whether this frame was predicted as occluded
        """
        B = memory_features.shape[0]
        self._frame_count += 1

        for b in range(B):
            mem = memory_features[b:b+1]  # [1, C, H, W]
            ptr = object_pointer[b:b+1]  # [1, 256]

            # Add occlusion embedding if occluded
            if is_occluded:
                mem = mem + self.occlusion_embed

            if is_prompted:
                # Add to prompted memories FIFO (no temporal position)
                self._prompted_memories.append(mem)
                if len(self._prompted_memories) > self.max_prompted_frames:
                    self._prompted_memories.pop(0)
            else:
                # Add to recent memories FIFO (with temporal position)
                self._recent_memories.append(mem)
                self._recent_temporal_ids.append(self._frame_count)
                # Generate spatial positions for this memory
                ys = torch.arange(self.H, device=mem.device).float()
                xs = torch.arange(self.W, device=mem.device).float()
                grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
                pos = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
                self._memory_positions.append(pos)
                # FIFO
                if len(self._recent_memories) > self.max_recent_frames:
                    self._recent_memories.pop(0)
                    self._recent_temporal_ids.pop(0)
                    self._memory_positions.pop(0)

            # Add object pointer
            self._object_pointers.append(ptr)
            # Keep only object pointers for frames we still have in memory
            total_mem = len(self._recent_memories) + len(self._prompted_memories)
            if len(self._object_pointers) > total_mem:
                self._object_pointers.pop(0)

    def get_memory(self, device: torch.device) -> MemoryBankOutput:
        """
        Get the current memory contents for memory attention.

        Returns:
            MemoryBankOutput with spatial_memory, object_pointers, temporal_positions, memory_positions
        """
        # Collect spatial memories (recent + prompted)
        all_spatial = []
        all_temporal_ids = []
        all_positions = []

        # Recent memories (with temporal positions)
        for i, mem in enumerate(self._recent_memories):
            all_spatial.append(mem)
            all_temporal_ids.append(self._recent_temporal_ids[i])
            all_positions.append(self._memory_positions[i])

        # Prompted memories (no temporal positions, use position 0)
        for mem in self._prompted_memories:
            all_spatial.append(mem)
            # For prompted frames, use a dummy position
            pos = torch.zeros(self.H * self.W, 2, device=device)
            all_positions.append(pos)
            all_temporal_ids.append(-1)  # -1 indicates no temporal embedding

        # Object pointers
        all_pointers = []
        for ptr in self._object_pointers:
            all_pointers.append(ptr)

        if len(all_spatial) == 0:
            # Return empty memory
            spatial_memory = torch.zeros(1, 0, self.memory_dim, device=device)
            object_pointers = torch.zeros(1, 0, self.memory_dim, device=device)
            return MemoryBankOutput(
                spatial_memory=spatial_memory,
                object_pointers=object_pointers,
                temporal_positions=None,
                memory_positions=None,
            )

        # Stack and flatten spatial memories
        spatial_memory = torch.cat(all_spatial, dim=0)  # [M_spatial, C, H, W]
        M_spatial = spatial_memory.shape[0]
        # Flatten to tokens
        spatial_memory = spatial_memory.reshape(M_spatial, self.memory_dim, self.H * self.W).permute(0, 2, 1).reshape(1, M_spatial * self.H * self.W, self.memory_dim)

        # Stack memory positions
        memory_positions = torch.cat(all_positions, dim=0).unsqueeze(0)  # [1, M_spatial*H*W, 2]
        # Repeat positions across spatial locations
        memory_positions = memory_positions.repeat(1, 1, 1)  # already flattened

        # Process object pointers
        if len(all_pointers) > 0:
            object_pointers_raw = torch.cat(all_pointers, dim=0)  # [M_obj, 256]
            # Project to 4x64 tokens
            object_pointers = self.object_pointer_proj(object_pointers_raw)  # [M_obj, 256]
            object_pointers = object_pointers.reshape(-1, self.object_pointer_tokens, self.memory_dim)  # [M_obj, 4, 64]
            object_pointers = object_pointers.reshape(1, -1, self.memory_dim)  # [1, M_obj*4, 64]
        else:
            object_pointers = torch.zeros(1, 0, self.memory_dim, device=device)

        # Filter temporal positions for non-prompted frames only
        valid_temporal = [t for t in all_temporal_ids if t >= 0]
        if len(valid_temporal) > 0:
            temporal_positions = torch.tensor(valid_temporal, device=device).unsqueeze(0)  # [1, N_recent]
            # Repeat for spatial locations
            temporal_positions = temporal_positions.repeat_interleave(self.H * self.W, dim=1)  # [1, N_recent*H*W]
        else:
            temporal_positions = None

        return MemoryBankOutput(
            spatial_memory=spatial_memory,
            object_pointers=object_pointers,
            temporal_positions=temporal_positions,
            memory_positions=memory_positions.squeeze(0) if memory_positions is not None else None,
        )

    def forward(self) -> MemoryBankOutput:
        """Alias for get_memory."""
        return self.get_memory(torch.device("cpu"))
