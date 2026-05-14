## Code: models/memory_bank.py

```python
## models/memory_bank.py
"""Memory bank for SAM 2: temporal state container for streaming video inference.

This module implements the MemoryBank — the core temporal state container in
SAM 2's streaming architecture. It maintains two FIFO queues of spatial memory
features and a list of object pointer tokens, providing temporal context to the
MemoryAttention module for conditioning current frame predictions on past
observations.

When the bank is empty (image mode or first frame), the model behaves like SAM.
When populated (video mode), it provides temporal context about the target object.

Architecture (Section 4, Appendix D.1):
    - Two FIFO queues:
        1. recent_memories: up to N=6 unprompted frames (with temporal PE)
        2. prompted_memories: up to M=2 prompted frames (without temporal PE)
    - object_pointers: list of 256-dim tokens split into 4×64-dim for cross-attention
    - Temporal PE applied only to recent frame memories (not prompted frames)
    - Learned occlusion embedding added to memory features of occluded frames

Config references:
    model.num_recent_memories: 6        → N = 6 recent unprompted frames
    model.memory_feature_dim: 64        → memory_dim = 64
    model.object_pointer_dim: 256       → raw pointer dimension
    model.num_object_pointer_tokens: 4  → split into 4 tokens of 64-dim
    training.max_prompted_frames: 2     → M = 2 prompted frames

Paper references:
    Section 4: "The memory bank retains information about past predictions for
        the target object in the video by maintaining a FIFO queue of memories
        of up to N recent frames and stores information from prompts in a FIFO
        queue of up to M prompted frames."
    Section 4: "We embed temporal position information into the memories of N
        recent frames, allowing the model to represent short-term object motion,
        but not into those of prompted frames."
    Appendix D.1: "we project the memory features in our memory bank to a
        dimension of 64, and split the 256-dim object pointer into 4 tokens of
        64-dim for cross-attention to the memory bank."
    Appendix D.1: "we also add a learned occlusion embedding to the memory
        features of those frames that are predicted to be occluded."
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MemoryBankOutput dataclass
# ---------------------------------------------------------------------------


@dataclass
class MemoryBankOutput:
    """Data container returned by MemoryBank.get_memory_for_attention().

    Holds all memory information needed by MemoryAttention for cross-attention.
    Spatial memories have temporal PE already added for recent frames.
    Object pointers are kept separate because they are excluded from RoPE
    in MemoryAttention (Appendix D.1).

    Attributes:
        spatial_memories: Concatenated spatial feature maps from all memories
            (recent + prompted), with temporal PE already added to recent frames.
            Shape: [B, N_mem_tokens, memory_feature_dim] where
            N_mem_tokens = num_memories * H_mem * W_mem.
            Empty tensor of shape [B, 0, memory_feature_dim] when bank is empty.
        object_pointers: Concatenated object pointer tokens from all frames.
            Each frame contributes num_object_pointer_tokens (4) tokens of
            memory_feature_dim (64) dimensions.
            Shape: [B, N_ptr_tokens, memory_feature_dim] where
            N_ptr_tokens = num_frames * num_object_pointer_tokens.
            Empty tensor of shape [B, 0, memory_feature_dim] when bank is empty.
        temporal_embeddings: Temporal PE vectors for recent frame memories only.
            Stored separately for debugging/inspection; already added to
            spatial_memories before returning.
            Shape: [num_recent_frames, memory_feature_dim].
            Empty tensor of shape [0, memory_feature_dim] when no recent memories.
    """

    spatial_memories: torch.Tensor
    object_pointers: torch.Tensor
    temporal_embeddings: torch.Tensor


# ---------------------------------------------------------------------------
# TemporalPositionalEncoding
# ---------------------------------------------------------------------------


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding over recency-order indices for memory bank.

    Applied ONLY to the N=6 recent unprompted frame memories. NOT applied to
    prompted frame memories.

    From Section 4: "We embed temporal position information into the memories
    of N recent frames, allowing the model to represent short-term object motion,
    but not into those of prompted frames, because the training signal from
    prompted frames is sparser and it is more difficult to generalize to the
    inference setting where prompted frames may come from a very different
    temporal range than seen during training."

    The encoding uses recency order (0 = oldest in queue, N-1 = most recent),
    NOT absolute video frame timestamps. This ensures generalization: during
    inference, absolute frame numbers are arbitrary, but the relative ordering
    within the N-frame window is always 0..N-1.

    Config reference: model.num_recent_memories: 6, model.memory_feature_dim: 64

    Args:
        d_model: Embedding dimension of the memory features. Must match
            model.memory_feature_dim (64 per config). Must be even.
        max_len: Maximum number of positions to precompute. Should be at least
            model.num_recent_memories (6 per config). Defaults to 64 to provide
            headroom for ablations (Table 9c tests up to N=8).

    Example:
        tpe = TemporalPositionalEncoding(d_model=64, max_len=64)
        # Encode 6 recent memories (recency indices 0..5)
        indices = list(range(6))
        temporal_pe = tpe(indices)  # (6, 64)
        # Add to memory features before cross-attention
        memory_features += temporal_pe.unsqueeze(0)  # broadcast over batch
    """

    def __init__(
        self,
        d_model: int = 64,
        max_len: int = 64,
    ) -> None:
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(
                f"TemporalPositionalEncoding requires d_model divisible by 2, "
                f"got {d_model}."
            )

        self.d_model: int = d_model
        self.max_len: int = max_len

        # Precompute sinusoidal PE table of shape (max_len, d_model) in float32.
        # Standard Transformer formula:
        #   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        #   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe: torch.Tensor = self._build_pe_table(max_len, d_model)

        # Register as buffer: persists in state_dict, moves with .to(device),
        # but does NOT receive gradients.
        self.register_buffer("pe", pe)
        self.pe: torch.Tensor  # type annotation for IDE

    @staticmethod
    def _build_pe_table(max_len: int, d_model: int) -> torch.Tensor:
        """Precompute the sinusoidal PE table.

        Args:
            max_len: Number of positions to precompute.
            d_model: Embedding dimension (must be even).

        Returns:
            PE table of shape (max_len, d_model), dtype float32.
        """
        # Position indices: (max_len, 1)
        position: torch.Tensor = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)

        # Frequency divisors: (d_model // 2,)
        # div_term[i] = exp(-2i * log(10000) / d_model) = 1 / 10000^(2i/d_model)
        div_term: torch.Tensor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        # PE table: (max_len, d_model)
        pe: torch.Tensor = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims: cos

        return pe

    def forward(self, frame_indices: List[int]) -> torch.Tensor:
        """Retrieve temporal PE vectors for a list of recency-order indices.

        Args:
            frame_indices: List of integer recency indices. Each index
                represents the recency position of a memory in the FIFO queue:
                0 = oldest recent frame, N-1 = most recent frame.
                Length must be <= max_len.

        Returns:
            Temporal PE tensor of shape (len(frame_indices), d_model),
            dtype float32. Returns empty tensor of shape (0, d_model) if
            frame_indices is empty.

        Raises:
            ValueError: If any index in frame_indices exceeds max_len - 1.
        """
        if not frame_indices:
            return torch.zeros(
                0,
                self.d_model,
                dtype=self.pe.dtype,
                device=self.pe.device,
            )

        max_idx: int = max(frame_indices)
        if max_idx >= self.max_len:
            raise ValueError(
                f"frame_index {max_idx} exceeds max_len {self.max_len} - 1. "
                "Increase max_len in TemporalPositionalEncoding.__init__."
            )

        # Index into the PE table: (len(frame_indices), d_model)
        indices_tensor: torch.Tensor = torch.tensor(
            frame_indices,
            dtype=torch.long,
            device=self.pe.device,
        )
        return self.pe[indices_tensor]  # (N, d_model)


# ---------------------------------------------------------------------------
# MemoryBank
# ---------------------------------------------------------------------------


class MemoryBank(nn.Module):
    """Temporal state container for SAM 2's streaming video inference.

    Maintains two FIFO queues of spatial memory features and a list of object
    pointer tokens. Provides temporal context to MemoryAttention for conditioning
    current frame predictions on past observations.

    Two FIFO queues:
        recent_memories: Up to N=6 unprompted frame memories with temporal PE.
            Oldest entry is automatically evicted when the queue is full.
        prompted_memories: Up to M=2 prompted frame memories without temporal PE.
            Oldest entry is automatically evicted when the queue is full.

    Object pointers:
        One entry per processed frame (both prompted and unprompted).
        Each entry is the 256-dim mask decoder output token split into
        4 tokens of 64-dim for cross-attention (Appendix D.1).
        The list grows with video length and is not capped.

    Occlusion embedding:
        A learned nn.Embedding(1, memory_dim) added to memory features of
        frames predicted as occluded (Appendix D.1).

    Config references:
        model.num_recent_memories: 6        → max_recent_frames
        model.memory_feature_dim: 64        → memory_dim
        model.object_pointer_dim: 256       → raw pointer dimension
        model.num_object_pointer_tokens: 4  → pointer split count
        training.max_prompted_frames: 2     → max_prompted_frames

    Args:
        max_recent_frames: Maximum number of recent unprompted frame memories
            to retain. Defaults to 6 (config.model.num_recent_memories).
        memory_dim: Dimension of stored memory features. Defaults to 64
            (config.model.memory_feature_dim).
        max_prompted_frames: Maximum number of prompted frame memories to retain.
            Defaults to 2 (inferred from config.training.max_prompted_frames).
        object_pointer_dim: Raw dimension of object pointer tokens from the
            mask decoder. Defaults to 256 (config.model.object_pointer_dim).
        num_object_pointer_tokens: Number of tokens to split each pointer into.
            Defaults to 4 (config.model.num_object_pointer_tokens).

    Example:
        bank = MemoryBank(max_recent_frames=6, memory_dim=64)
        # Add a memory for an unprompted frame
        memory_feat = torch.randn(1, 64, 64, 64)   # [B, C, H, W]
        obj_ptr = torch.randn(1, 256)               # [B, 256]
        bank.add_memory(memory_feat, is_prompted=False, object_pointer=obj_ptr,
                        is_occluded=False, frame_idx=0)
        # Retrieve for cross-attention
        output = bank.get_memory_for_attention()
        # output.spatial_memories: [1, H*W, 64]
        # output.object_pointers: [1, 4, 64]
    """

    def __init__(
        self,
        max_recent_frames: int = 6,
        memory_dim: int = 64,
        max_prompted_frames: int = 2,
        object_pointer_dim: int = 256,
        num_object_pointer_tokens: int = 4,
    ) -> None:
        super().__init__()

        # Validate that pointer can be evenly split
        if object_pointer_dim % num_object_pointer_tokens != 0:
            raise ValueError(
                f"object_pointer_dim ({object_pointer_dim}) must be divisible by "
                f"num_object_pointer_tokens ({num_object_pointer_tokens})."
            )

        # Store configuration
        self.max_recent_frames: int = max_recent_frames
        self.memory_dim: int = memory_dim
        self.max_prompted_frames: int = max_prompted_frames
        self.object_pointer_dim: int = object_pointer_dim
        self.num_object_pointer_tokens: int = num_object_pointer_tokens
        self.pointer_token_dim: int = object_pointer_dim // num_object_pointer_tokens

        # Validate pointer token dimension matches memory_dim
        if self.pointer_token_dim != memory_dim:
            logger.warning(
                "pointer_token_dim (%d) != memory_dim (%d). "
                "Object pointer tokens will have different dimension than spatial "
                "memories. MemoryAttention must handle this via projection.",
                self.pointer_token_dim,
                memory_dim,
            )

        # ------------------------------------------------------------------
        # FIFO queue for recent unprompted frame memories
        # Python deque with maxlen automatically evicts oldest entries.
        # Each entry is a dict: {features, frame_idx, is_occluded}
        # features shape: [B, memory_dim, H_mem, W_mem] (stored as-is from MemoryEncoder)
        # ------------------------------------------------------------------
        self.recent_memories: deque = deque(maxlen=max_recent_frames)

        # ------------------------------------------------------------------
        # FIFO queue for prompted frame memories
        # Same entry format as recent_memories.
        # No temporal PE applied to these entries.
        # ------------------------------------------------------------------
        self.prompted_memories: deque = deque(maxlen=max_prompted_frames)

        # ------------------------------------------------------------------
        # Object pointer list
        # Each entry is a tensor of shape [B, num_object_pointer_tokens, pointer_token_dim]
        # representing the split pointer for one frame.
        # Grows unboundedly with video length.
        # ------------------------------------------------------------------
        self.object_pointers: List[torch.Tensor] = []

        # ------------------------------------------------------------------
        # Temporal positional encoding for recent frame memories
        # max_len = max_recent_frames + 10 for headroom during ablations
        # ------------------------------------------------------------------
        self.temporal_pe: TemporalPositionalEncoding = TemporalPositionalEncoding(
            d_model=memory_dim,
            max_len=max_recent_frames + 10,
        )

        # ------------------------------------------------------------------
        # Learned occlusion embedding
        # Shape: [1, memory_dim] via nn.Embedding(1, memory_dim)
        # Added to memory features of frames predicted as occluded.
        # Initialized to zero so it has no effect at training start.
        # ------------------------------------------------------------------
        self.occlusion_embedding: nn.Embedding = nn.Embedding(1, memory_dim)
        nn.init.zeros_(self.occlusion_embedding.weight)

    # ------------------------------------------------------------------
    # Core public methods
    # ------------------------------------------------------------------

    def add_memory(
        self,
        memory: torch.Tensor,
        is_prompted: bool,
        object_pointer: torch.Tensor,
        is_occluded: bool,
        frame_idx: int,
    ) -> None:
        """Add a new memory entry to the appropriate FIFO queue.

        Called by SAM2Model.forward_video_frame() after each processed frame.
        The memory tensor comes from MemoryEncoder.forward() and the
        object_pointer comes from MaskDecoder's output token.

        Processing steps:
            1. Optionally add learned occlusion embedding to memory features
            2. Store memory in the appropriate FIFO queue (recent or prompted)
            3. Split 256-dim object pointer into 4×64-dim tokens
            4. Append split pointer to self.object_pointers

        Args:
            memory: Spatial memory feature from MemoryEncoder, shape
                [B, memory_dim, H_mem, W_mem]. This is the projected 64-dim
                feature map at stride-16 spatial resolution.
            is_prompted: If True, store in prompted_memories queue (no temporal PE).
                If False, store in recent_memories queue (temporal PE applied on read).
            object_pointer: Mask decoder output token, shape [B, object_pointer_dim]
                or [B, 1, object_pointer_dim]. Will be split into
                num_object_pointer_tokens tokens of pointer_token_dim each.
            is_occluded: If True, add the learned occlusion embedding to the
                stored memory features before storing.
            frame_idx: Absolute frame index in the video (0-based). Used for
                ordering and debugging; not used for temporal PE (which uses
                relative recency indices).

        Returns:
            None. Modifies self.recent_memories, self.prompted_memories, and
            self.object_pointers in-place.
        """
        # ------------------------------------------------------------------
        # Step 1: Apply occlusion embedding if frame is predicted as occluded
        #
        # The occlusion embedding is a learned [1, memory_dim] vector that
        # broadcasts over [B, memory_dim, H_mem, W_mem] via unsqueeze.
        # Paper: "we also add a learned occlusion embedding to the memory
        # features of those frames that are predicted to be occluded"
        # ------------------------------------------------------------------
        memory_to_store: torch.Tensor = memory.detach().clone()

        if is_occluded:
            # occlusion_embedding.weight: [1, memory_dim]
            # Reshape to [1, memory_dim, 1, 1] for broadcasting over [B, C, H, W]
            occ_emb: torch.Tensor = self.occlusion_embedding(
                torch.zeros(1, dtype=torch.long, device=memory.device)
            )  # [1, memory_dim]
            occ_emb = occ_emb.view(1, self.memory_dim, 1, 1)  # [1, C, 1, 1]
            memory_to_store = memory_to_store + occ_emb

        # ------------------------------------------------------------------
        # Step 2: Create memory entry dict and store in appropriate queue
        #
        # Python deque(maxlen=N) automatically evicts the oldest entry (left)
        # when a new entry is appended (right) to a full deque.
        # ------------------------------------------------------------------
        memory_entry: Dict = {
            "features": memory_to_store,  # [B, memory_dim, H_mem, W_mem]
            "frame_idx": frame_idx,
            "is_occluded": is_occluded,
        }

        if is_prompted:
            self.prompted_memories.append(memory_entry)
        else:
            self.recent_memories.append(memory_entry)

        # ------------------------------------------------------------------
        # Step 3: Process and store object pointer
        #
        # Input: [B, object_pointer_dim] or [B, 1, object_pointer_dim]
        # Output: [B, num_object_pointer_tokens, pointer_token_dim]
        #
        # Paper: "split the 256-dim object pointer into 4 tokens of 64-dim
        # for cross-attention to the memory bank" (Appendix D.1)
        # This is a reshape/chunk operation, not a learned projection.
        # ------------------------------------------------------------------
        ptr: torch.Tensor = object_pointer.detach().clone()

        # Normalize to [B, object_pointer_dim]
        if ptr.ndim == 3:
            # [B, 1, object_pointer_dim] → [B, object_pointer_dim]
            ptr = ptr.squeeze(1)
        elif ptr.ndim == 1:
            # [object_pointer_dim] → [1, object_pointer_dim]
            ptr = ptr.unsqueeze(0)

        if ptr.ndim != 2:
            raise ValueError(
                f"object_pointer must be 1D, 2D, or 3D tensor, got shape {ptr.shape}."
            )

        B: int = ptr.shape[0]

        # Split: [B, object_pointer_dim] → [B, num_object_pointer_tokens, pointer_token_dim]
        ptr_split: torch.Tensor = ptr.view(
            B,
            self.num_object_pointer_tokens,
            self.pointer_token_dim,
        )

        self.object_pointers.append(ptr_split)

    def get_memory_for_attention(self) -> MemoryBankOutput:
        """Assemble all memory information for cross-attention in MemoryAttention.

        Collects spatial memories from both queues, adds temporal PE to recent
        frame memories, flattens spatial dimensions, and concatenates all
        object pointer tokens.

        Called by MemoryAttention.forward() before each frame's cross-attention.

        Returns:
            MemoryBankOutput with:
                - spatial_memories: [B, N_mem_tokens, memory_dim]
                  where N_mem_tokens = (num_recent + num_prompted) * H_mem * W_mem
                  Empty tensor [B, 0, memory_dim] when bank is empty.
                - object_pointers: [B, N_ptr_tokens, pointer_token_dim]
                  where N_ptr_tokens = num_frames * num_object_pointer_tokens
                  Empty tensor [B, 0, pointer_token_dim] when no pointers.
                - temporal_embeddings: [num_recent_frames, memory_dim]
                  PE vectors for recent frames (already added to spatial_memories).
                  Empty tensor [0, memory_dim] when no recent memories.

        Note:
            When the bank is empty (first frame of a video), returns empty
            tensors. MemoryAttention must handle this by skipping cross-attention
            or using zero context.
        """
        # Determine batch size and device from available memories or pointers
        batch_size: int = 1
        device: torch.device = self.occlusion_embedding.weight.device
        dtype: torch.dtype = torch.float32

        # Try to infer batch size and device from stored memories
        if len(self.recent_memories) > 0:
            sample_feat: torch.Tensor = self.recent_memories[0]["features"]
            batch_size = sample_feat.shape[0]
            device = sample_feat.device
            dtype = sample_feat.dtype
        elif len(self.prompted_memories) > 0:
            sample_feat = self.prompted_memories[0]["features"]
            batch_size = sample_feat.shape[0]
            device = sample_feat.device
            dtype = sample_feat.dtype
        elif len(self.object_pointers) > 0:
            batch_size = self.object_pointers[0].shape[0]
            device = self.object_pointers[0].device
            dtype = self.object_pointers[0].dtype

        # ------------------------------------------------------------------
        # Collect recent memories with temporal PE
        #
        # Temporal PE is indexed by relative recency position:
        #   0 = oldest entry in the deque (leftmost)
        #   N-1 = most recent entry (rightmost)
        #
        # The deque is ordered oldest-to-newest (append adds to right).
        # We iterate left-to-right (oldest to newest) and assign indices 0..N-1.
        # ------------------------------------------------------------------
        recent_memory_tensors: List[torch.Tensor] = []
        num_recent: int = len(self.recent_memories)
        temporal_pe_vectors: torch.Tensor = torch.zeros(
            0, self.memory_dim, dtype=torch.float32
        )

        if num_recent > 0:
            # Recency indices: 0 = oldest, num_recent-1 = most recent
            recency_indices: List[int] = list(range(num_recent))

            # Get temporal PE for all recent frames: [num_recent, memory_dim]
            temporal_pe_vectors = self.temporal_pe(recency_indices)
            # Cast to match memory dtype
            temporal_pe_vectors = temporal_pe_vectors.to(dtype=dtype, device=device)

            for i, entry in enumerate(self.recent_memories):
                feat: torch.Tensor = entry["features"]
                # feat: [B, memory_dim, H_mem, W_mem]

                B_feat, C_feat, H_mem, W_mem = feat.shape

                # Flatten spatial dimensions: [B, memory_dim, H_mem, W_mem]
                # → [B, H_mem*W_mem, memory_dim]
                feat_flat: torch.Tensor = feat.permute(0, 2, 3, 1).reshape(
                    B_feat, H_mem * W_mem, C_feat
                )

                # Add temporal PE: broadcast [1, 1, memory_dim] over spatial tokens
                # temporal_pe_vectors[i]: [memory_dim] → [1, 1, memory_dim]
                tpe: torch.Tensor = temporal_pe_vectors[i].view(1, 1, self.memory_dim)
                feat_flat = feat_flat + tpe

                recent_memory_tensors.append(feat_flat)

        # ------------------------------------------------------------------
        # Collect prompted memories WITHOUT temporal PE
        # ------------------------------------------------------------------
        prompted_memory_tensors: List[torch.Tensor] = []

        for entry in self.prompted_memories:
            feat = entry["features"]
            # feat: [B, memory_dim, H_mem, W_mem]
            B_feat, C_feat, H_mem, W_mem = feat.shape

            # Flatten spatial dimensions: [B, H_mem*W_mem, memory_dim]
            feat_flat = feat.permute(0, 2, 3, 1).reshape(
                B_feat, H_mem * W_mem, C_feat
            )
            prompted_memory_tensors.append(feat_flat)

        # ------------------------------------------------------------------
        # Concatenate all spatial memories along the token dimension
        # ------------------------------------------------------------------
        all_memory_tensors: List[torch.Tensor] = (
            recent_memory_tensors + prompted_memory_tensors
        )

        if len(all_memory_tensors) > 0:
            # Each tensor: [B, H_mem*W_mem, memory_dim]
            # Concatenate along dim=1: [B, total_tokens, memory_dim]
            spatial_memories: torch.Tensor = torch.cat(all_memory_tensors, dim=1)
        else:
            # Empty bank — return zero-length tensor
            spatial_memories = torch.zeros(
                batch_size,
                0,
                self.memory_dim,
                dtype=dtype,
                device=device,
            )

        # ------------------------------------------------------------------
        # Collect and concatenate object pointer tokens
        #
        # Each entry in self.object_pointers:
        #   [B, num_object_pointer_tokens, pointer_token_dim]
        # Concatenate along dim=1 (token dimension):
        #   [B, num_frames * num_object_pointer_tokens, pointer_token_dim]
        # ------------------------------------------------------------------
        if len(self.object_pointers) > 0:
            # Stack along token dimension: [B, N_frames * N_ptr_tokens, ptr_dim]
            object_pointers: torch.Tensor = torch.cat(
                self.object_pointers, dim=1
            )
        else:
            # No pointers yet — return zero-length tensor
            object_pointers = torch.zeros(
                batch_size,
                0,
                self.pointer_token_dim,
                dtype=dtype,
                device=device,
            )

        return MemoryBankOutput(
            spatial_memories=spatial_memories,
            object_pointers=object_pointers,
            temporal_embeddings=temporal_pe_vectors,
        )

    def reset(self) -> None:
        """Clear all memory state between videos.

        Called by SAM2Model.reset_memory() at the start of each new video
        to prevent cross-video contamination. Clears both FIFO queues and
        the object pointer list.

        Returns:
            None. Modifies self.recent_memories, self.prompted_memories, and
            self.object_pointers in-place.
        """
        self.recent_memories.clear()
        self.prompted_memories.clear()
        self.object_pointers = []

        logger.debug(
            "MemoryBank reset: cleared %d recent memories, %d prompted memories, "
            "and all object pointers.",
            0, 0,
        )

    # ------------------------------------------------------------------
    # Utility / inspection methods
    # ------------------------------------------------------------------

    def num_recent_memories_stored(self) -> int:
        """Return the current number of recent frame memories in the queue.

        Returns:
            Integer count in [0, max_recent_frames].
        """
        return len(self.recent_memories)

    def num_prompted_memories_stored(self) -> int:
        """Return the current number of prompted frame memories in the queue.

        Returns:
            Integer count in [0, max_prompted_frames].
        """
        return len(self.prompted_memories)

    def num_object_pointers_stored(self) -> int:
        """Return the total number of frames with stored object pointers.

        Returns:
            Integer count of frames processed so far in the current video.
        """
        return len(self.object_pointers)

    def is_empty(self) -> bool:
        """Check whether the memory bank contains any memories.

        Returns:
            True if both queues are empty and no object pointers are stored.
            This is the state at the start of a new video (image mode).
        """
        return (
            len(self.recent_memories) == 0
            and len(self.prompted_memories) == 0
            and len(self.object_pointers) == 0
        )

    def get_memory_summary(self) -> Dict:
        """Return a summary dict of the current memory bank state for logging.

        Returns:
            Dict with keys:
                - num_recent: Number of recent frame memories stored.
                - num_prompted: Number of prompted frame memories stored.
                - num_pointers: Number of object pointer entries stored.
                - max_recent: Maximum capacity of recent memories queue.
                - max_prompted: Maximum capacity of prompted memories queue.
                - is_empty: Whether the bank is empty.
        """
        return {
            "num_recent": len(self.recent_memories),
            "num_prompted": len(self.prompted_memories),
            "num_pointers": len(self.object_pointers),