"""
SAM 2: Segment Anything Model 2.

Unified model for promptable visual segmentation in images and videos.

Architecture overview (Section 4, Appendix D.1):
  1. Image encoder (Hiera): run once per frame, produces unconditioned features.
  2. Memory attention: conditions current frame features on memory bank.
  3. Prompt encoder: encodes clicks, boxes, or masks.
  4. Mask decoder: predicts masks, IoU scores, occlusion score, object pointer.
  5. Memory encoder: encodes predicted mask + image features → memory feature.
  6. Memory bank: FIFO queues for recent frames (N=6) and prompted frames (M).

Video processing:
  - Frames processed one at a time (streaming).
  - Memory bank updated after each frame.
  - Object pointers (256-dim → 4×64 tokens) stored alongside spatial memories.
  - Temporal position embeddings on recent-frame memories only.

Image processing:
  - Memory bank is empty → model behaves like SAM.
  - No temporal information.

Multi-object handling:
  - Each object processed independently.
  - Image encoder features shared across objects.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .image_encoder import HieraImageEncoder, build_image_encoder
from .mask_decoder import MaskDecoder
from .memory_attention import MemoryAttention
from .memory_encoder import MemoryEncoder
from .prompt_encoder import PromptEncoder


# ---------------------------------------------------------------------------
# Memory bank state (per object)
# ---------------------------------------------------------------------------

@dataclass
class MemoryBankState:
    """Holds the memory bank state for a single tracked object."""
    recent_memories: deque = field(default_factory=lambda: deque(maxlen=6))
    prompted_memories: deque = field(default_factory=lambda: deque(maxlen=16))
    object_pointers: deque = field(default_factory=lambda: deque(maxlen=22))  # N + M

    def clear(self) -> None:
        self.recent_memories.clear()
        self.prompted_memories.clear()
        self.object_pointers.clear()

    def add_recent(self, memory: Tensor, pointer: Tensor) -> None:
        self.recent_memories.append(memory)
        self.object_pointers.append(pointer)

    def add_prompted(self, memory: Tensor, pointer: Tensor) -> None:
        self.prompted_memories.append(memory)
        self.object_pointers.append(pointer)

    def get_recent(self) -> List[Tensor]:
        return list(self.recent_memories)

    def get_prompted(self) -> List[Tensor]:
        return list(self.prompted_memories)

    def get_pointers(self) -> List[Tensor]:
        return list(self.object_pointers)


# ---------------------------------------------------------------------------
# SAM 2 model
# ---------------------------------------------------------------------------

class SAM2(nn.Module):
    """
    Segment Anything Model 2.

    Supports:
      - Image segmentation (memory bank empty, behaves like SAM)
      - Video segmentation (streaming, memory bank updated per frame)
      - Interactive refinement via prompts on any frame
    """

    def __init__(
        self,
        image_encoder: HieraImageEncoder,
        memory_attention: MemoryAttention,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        memory_encoder: MemoryEncoder,
        num_multimask_outputs: int = 3,
        max_recent_frames: int = 6,
        max_prompted_frames: int = 16,
        image_size: int = 1024,
        embed_dim: int = 256,
        memory_dim: int = 64,
        pointer_dim: int = 256,
        num_pointer_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.memory_attention = memory_attention
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.memory_encoder = memory_encoder

        self.num_multimask_outputs = num_multimask_outputs
        self.max_recent_frames = max_recent_frames
        self.max_prompted_frames = max_prompted_frames
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim
        self.pointer_dim = pointer_dim
        self.num_pointer_tokens = num_pointer_tokens

        # Project object pointer (embed_dim) → num_pointer_tokens × (pointer_dim // num_pointer_tokens)
        token_dim = pointer_dim // num_pointer_tokens
        self.pointer_proj = nn.Linear(embed_dim, pointer_dim)

        # Per-object memory banks (managed during inference)
        self._memory_banks: Dict[int, MemoryBankState] = {}

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        frames: Tensor,
        points: Optional[Tuple[Tensor, Tensor]] = None,
        boxes: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
        recent_memories: Optional[List[Tensor]] = None,
        prompted_memories: Optional[List[Tensor]] = None,
        object_pointers: Optional[List[Tensor]] = None,
        multimask_output: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for a single frame (or batch of independent frames).

        Args:
            frames:            (B, 3, H, W)
            points:            (coords (B,N,2), labels (B,N)) or None
            boxes:             (B, 4) xyxy or None
            masks:             (B, 1, H, W) or None
            recent_memories:   list of (B, memory_dim, H_m, W_m)
            prompted_memories: list of (B, memory_dim, H_m, W_m)
            object_pointers:   list of (B, num_pointer_tokens, token_dim)
            multimask_output:  return multiple masks for ambiguous prompts

        Returns dict with:
            masks:       (B, num_masks, H_out, W_out) — logits
            iou_pred:    (B, num_masks)
            occlusion:   (B, 1) — logit (positive = object present)
            mask_token:  (B, embed_dim) — object pointer token
            memory:      (B, memory_dim, H_emb, W_emb) — for memory bank
        """
        recent_memories = recent_memories or []
        prompted_memories = prompted_memories or []
        object_pointers = object_pointers or []

        B = frames.shape[0]

        # 1. Image encoding (unconditioned)
        image_embedding, skip_features = self.image_encoder(frames)
        # image_embedding: (B, embed_dim, H/16, W/16)

        # 2. Memory attention: condition on memory bank
        if recent_memories or prompted_memories or object_pointers:
            conditioned_embedding = self.memory_attention(
                image_embedding, recent_memories, prompted_memories, object_pointers
            )
        else:
            conditioned_embedding = image_embedding

        # 3. Prompt encoding
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points, boxes=boxes, masks=masks
        )
        # Expand to batch size if prompt encoder returned batch size 1
        if sparse_embeddings.shape[0] == 1 and B > 1:
            sparse_embeddings = sparse_embeddings.expand(B, -1, -1)
        if dense_embeddings.shape[0] == 1 and B > 1:
            dense_embeddings = dense_embeddings.expand(B, -1, -1, -1)

        # 4. Mask decoding
        # get_dense_pe returns (1, C, H, W); expand to batch size
        image_pe = self.prompt_encoder.get_dense_pe().expand(B, -1, -1, -1)
        pred_masks, iou_pred, occlusion, mask_token = self.mask_decoder(
            image_embeddings=conditioned_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            skip_features=skip_features,
            multimask_output=multimask_output,
        )

        # 5. Memory encoding (use best mask for memory)
        # Select mask with highest IoU for memory encoding
        best_mask_idx = iou_pred.argmax(dim=1, keepdim=True)  # (B, 1)
        best_mask = pred_masks[
            torch.arange(pred_masks.shape[0], device=pred_masks.device),
            best_mask_idx.squeeze(1),
        ].unsqueeze(1)  # (B, 1, H_out, W_out)

        is_occluded = (torch.sigmoid(occlusion) < 0.5).squeeze(1)  # (B,)
        memory = self.memory_encoder(
            image_embedding=image_embedding,
            mask_logits=best_mask,
            is_occluded=bool(is_occluded.any().item()),
        )

        # 6. Object pointer: project mask token to pointer tokens
        pointer = self.pointer_proj(mask_token)  # (B, pointer_dim)
        token_dim = self.pointer_dim // self.num_pointer_tokens
        pointer_tokens = pointer.view(-1, self.num_pointer_tokens, token_dim)  # (B, 4, 64)

        return {
            "masks": pred_masks,
            "iou_pred": iou_pred,
            "occlusion": occlusion,
            "mask_token": mask_token,
            "memory": memory,
            "pointer_tokens": pointer_tokens,
        }

    # ------------------------------------------------------------------
    # Inference helpers for streaming video
    # ------------------------------------------------------------------

    def init_state(self, object_id: int = 0) -> None:
        """Initialize memory bank for a new object."""
        self._memory_banks[object_id] = MemoryBankState(
            recent_memories=deque(maxlen=self.max_recent_frames),
            prompted_memories=deque(maxlen=self.max_prompted_frames),
            object_pointers=deque(maxlen=self.max_recent_frames + self.max_prompted_frames),
        )

    def reset_state(self, object_id: int = 0) -> None:
        """Clear memory bank for an object."""
        if object_id in self._memory_banks:
            self._memory_banks[object_id].clear()

    @torch.no_grad()
    def predict_frame(
        self,
        frame: Tensor,
        object_id: int = 0,
        points: Optional[Tuple[Tensor, Tensor]] = None,
        boxes: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
        is_prompted: bool = False,
        multimask_output: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Predict segmentation for a single video frame.

        Args:
            frame:        (1, 3, H, W)
            object_id:    which object to segment
            points/boxes/masks: optional prompts for this frame
            is_prompted:  if True, store in prompted memory queue
            multimask_output: return multiple masks

        Returns: same dict as forward()
        """
        if object_id not in self._memory_banks:
            self.init_state(object_id)

        bank = self._memory_banks[object_id]
        has_prompts = points is not None or boxes is not None or masks is not None

        output = self.forward(
            frames=frame,
            points=points,
            boxes=boxes,
            masks=masks,
            recent_memories=bank.get_recent(),
            prompted_memories=bank.get_prompted(),
            object_pointers=bank.get_pointers(),
            multimask_output=multimask_output or has_prompts,
        )

        # Update memory bank
        memory = output["memory"].detach()
        pointer = output["pointer_tokens"].detach()

        if is_prompted or has_prompts:
            bank.add_prompted(memory, pointer)
        else:
            bank.add_recent(memory, pointer)

        return output

    # ------------------------------------------------------------------
    # Convenience: encode image only (for caching)
    # ------------------------------------------------------------------

    def encode_image(self, frame: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Run image encoder only. Returns (embedding, skip_features)."""
        return self.image_encoder(frame)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_sam2(
    variant: str = "B+",
    image_size: int = 1024,
    embed_dim: int = 256,
    memory_dim: int = 64,
    num_memory_attention_layers: int = 4,
    max_recent_frames: int = 6,
    num_multimask_outputs: int = 3,
) -> SAM2:
    """Build SAM 2 model with specified configuration."""
    image_encoder = build_image_encoder(variant=variant, img_size=image_size, out_chans=embed_dim)

    memory_attention = MemoryAttention(
        d_model=embed_dim,
        nhead=8,
        num_layers=num_memory_attention_layers,
        dim_feedforward=2048,
        memory_dim=memory_dim,
        max_recent_frames=max_recent_frames,
    )

    image_embedding_size = image_size // 16
    prompt_encoder = PromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    # Skip connection dims from Hiera encoder
    skip_dim_s4 = embed_dim // 4   # stride-4: out_chans // 4
    skip_dim_s8 = embed_dim // 2   # stride-8: out_chans // 2

    mask_decoder = MaskDecoder(
        transformer_dim=embed_dim,
        transformer_depth=2,
        transformer_num_heads=8,
        transformer_mlp_dim=2048,
        num_multimask_outputs=num_multimask_outputs,
        activation=nn.GELU,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        skip_dim_s4=skip_dim_s4,
        skip_dim_s8=skip_dim_s8,
    )

    memory_encoder = MemoryEncoder(
        embed_dim=embed_dim,
        memory_dim=memory_dim,
        num_fuse_layers=2,
    )

    return SAM2(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        memory_encoder=memory_encoder,
        num_multimask_outputs=num_multimask_outputs,
        max_recent_frames=max_recent_frames,
        image_size=image_size,
        embed_dim=embed_dim,
        memory_dim=memory_dim,
    )
