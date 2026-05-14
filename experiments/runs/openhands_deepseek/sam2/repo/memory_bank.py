"""Memory bank for SAM 2.

Maintains:
1. FIFO queue of up to N recent frame memories (unprompted frames)
2. FIFO queue of up to M prompted frame memories
3. List of object pointers from mask decoder output tokens

Features:
- Spatial memory features stored as feature maps
- Object pointers as lightweight semantic vectors
- Temporal position information embedded into recent frame memories
- Occlusion embedding for frames predicted to be occluded
- Object pointers split into 4 x 64-dim tokens for cross-attention
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from config import MemoryBankConfig


class MemoryBank(nn.Module):
    """FIFO memory bank storing past predictions and prompt memories.

    For the VOS task (mask only on first frame):
    - Consistently retains first frame memory + up to N recent frame memories
    - No prompted frame memories beyond first frame

    During interactive PVS:
    - Retains prompted frames in the prompted queue
    - Retains N recent unprompted frames
    """
    def __init__(self, config: MemoryBankConfig):
        super().__init__()
        self.config = config
        self.num_recent_frames = config.num_recent_frames
        self.num_prompted_frames = config.num_prompted_frames
        self.memory_dim = config.memory_channel_dim
        self.object_pointer_dim = config.object_pointer_dim

        # Temporal positional embedding for recent frame memories
        # Only added to recent unprompted frames (not prompted frames)
        if config.use_temporal_pos:
            self.temporal_pos_embed = nn.Embedding(config.num_recent_frames, config.memory_channel_dim)
        else:
            self.temporal_pos_embed = None

        # Projection for object pointer tokens from 256-dim to 4 x 64-dim
        self.pointer_proj = nn.Linear(config.object_pointer_dim, config.object_pointer_num_tokens * config.memory_channel_dim)
        self.pointer_num_tokens = config.object_pointer_num_tokens

        # Occlusion embedding
        if config.use_occlusion_embedding:
            self.occlusion_embed = nn.Parameter(torch.zeros(1, 1, config.memory_channel_dim))
        else:
            self.occlusion_embed = None

        # Initialize banks as empty
        self.reset()

    def reset(self):
        """Reset all memory banks."""
        self.recent_memories = []   # List of (features, is_empty_flag)
        self.prompted_memories = []  # List of (features, frame_idx)
        self.object_pointers = []    # List of pointer tokens
        self.frame_count = 0

    def _add_memory(self, features: torch.Tensor, is_prompted: bool, frame_idx: int):
        """Add a memory entry to the appropriate bank.

        Args:
            features: [B, memory_dim, H_mem, W_mem] spatial memory features
            is_prompted: whether this frame received a prompt
            frame_idx: temporal index of this frame
        """
        if is_prompted:
            self.prompted_memories.append((features, frame_idx))
            if len(self.prompted_memories) > self.num_prompted_frames:
                self.prompted_memories.pop(0)
        else:
            self.recent_memories.append((features, frame_idx))
            if len(self.recent_memories) > self.num_recent_frames:
                self.recent_memories.pop(0)

    def add_object_pointer(self, pointer: torch.Tensor, is_occluded: bool = False):
        """Add object pointer token.

        Args:
            pointer: [B, pointer_dim] from mask decoder output token
            is_occluded: whether the object is predicted to be occluded
        """
        if is_occluded and self.occlusion_embed is not None:
            pointer = pointer + self.occlusion_embed.squeeze(0) * 0.1
        self.object_pointers.append(pointer)

    def get_memory_for_attention(self, batch_size: int, device: torch.device) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get all memory features and object pointers for cross-attention.

        Returns:
            memory_features: [B, total_memory_tokens, memory_dim] flattened spatial memories
            object_pointers: [B, total_pointer_tokens, memory_dim] projected pointer tokens
            or None if banks are empty
        """
        all_memories = []

        # Collect prompted frame memories (no temporal position encoding)
        for feats, _ in self.prompted_memories:
            feats = feats.to(device)
            B, C_mem, H_mem, W_mem = feats.shape
            feats_flat = feats.permute(0, 2, 3, 1).reshape(B, H_mem * W_mem, C_mem)
            feats_flat = feats_flat.expand(batch_size, -1, -1)
            all_memories.append(feats_flat)

        # Collect recent frame memories (with temporal position encoding)
        for i, (feats, _) in enumerate(self.recent_memories):
            feats = feats.to(device)
            B, C_mem, H_mem, W_mem = feats.shape
            feats_flat = feats.permute(0, 2, 3, 1).reshape(B, H_mem * W_mem, C_mem)
            feats_flat = feats_flat.expand(batch_size, -1, -1)
            if self.temporal_pos_embed is not None:
                t = min(i, self.num_recent_frames - 1)
                t_embed = self.temporal_pos_embed.weight[t].to(device=device, dtype=feats_flat.dtype)
                feats_flat = feats_flat + t_embed.unsqueeze(0).unsqueeze(0)
            all_memories.append(feats_flat)

        memory_features = torch.cat(all_memories, dim=1) if all_memories else None

        # Collect object pointers
        if self.object_pointers:
            pointers = torch.stack([p.to(device) for p in self.object_pointers], dim=0)
            B = pointers.shape[1]
            pointers = pointers.mean(0)  # [B, pointer_dim]
            pointers = self.pointer_proj(pointers)  # [B, num_tokens * memory_dim]
            pointers = pointers.reshape(B, self.pointer_num_tokens, self.memory_dim)
        else:
            pointers = None

        return memory_features, pointers

    def add_frame_memory(self, features: torch.Tensor, mask: torch.Tensor,
                         is_prompted: bool, is_occluded: bool, object_pointer: torch.Tensor):
        """Add a complete frame's memory: spatial features + object pointer.

        Args:
            features: [B, C, H_mem, W_mem] spatial memory features from memory encoder
            mask: [B, 1, H_img, W_img] predicted mask
            is_prompted: whether this frame received a prompt
            is_occluded: predicted occlusion status
            object_pointer: [B, C] object pointer token from mask decoder
        """
        # Apply occlusion embedding to features
        if is_occluded and self.occlusion_embed is not None:
            B, C, H, W = features.shape
            occ = self.occlusion_embed.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
            features = features + occ

        self._add_memory(features, is_prompted, self.frame_count)
        self.add_object_pointer(object_pointer, is_occluded)
        self.frame_count += 1

    def has_memory(self) -> bool:
        """Check if bank has any memories."""
        return len(self.recent_memories) > 0 or len(self.prompted_memories) > 0

    def set_first_frame_memory(self, features: torch.Tensor, object_pointer: torch.Tensor):
        """Set the first frame memory (persistent in VOS mode).

        In VOS mode, the first frame memory is always retained in the prompted queue.
        """
        self.prompted_memories.append((features, 0))
        if len(self.prompted_memories) > self.num_prompted_frames:
            self.prompted_memories.pop(0)
        self.add_object_pointer(object_pointer)
        self.frame_count = 1
