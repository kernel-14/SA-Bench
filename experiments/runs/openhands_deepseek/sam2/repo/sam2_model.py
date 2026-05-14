"""SAM 2: Segment Anything Model 2 - full model combining all components.

Unified model for promptable video and image segmentation.
Processes video frames one at a time in streaming fashion, equipped with a memory
attention module to attend to previous memories of the target object.
When applied to images, the memory is empty and the model behaves like SAM.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SAM2Config, HieraConfig, MemoryAttentionConfig, MemoryBankConfig, \
    MaskDecoderConfig, PromptEncoderConfig, MemoryEncoderConfig
from image_encoder import ImageEncoder
from memory_attention import MemoryAttention
from memory_bank import MemoryBank
from memory_encoder import MemoryEncoder
from mask_decoder import MaskDecoder
from prompt_encoder import PromptEncoder
from transformer import get_sinusoidal_pos_embed


class SAM2(nn.Module):
    """SAM 2: unified model for promptable visual segmentation in images and videos.

    Streaming architecture:
    1. Image encoder processes each frame once (unconditioned tokens)
    2. Past frames' predictions and prompts stored as memories in memory bank
    3. Current frame tokens conditioned on memories via memory attention
    4. Mask decoder produces segmentation mask for current frame
    5. Memory encoder converts prediction to memory for future frames

    For single-image: memory is empty, behaves like SAM.
    """
    def __init__(self, config: SAM2Config):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        # Sub-modules
        self.image_encoder = ImageEncoder(config.hiera)
        self.memory_attention = MemoryAttention(config.memory_attention)
        self.prompt_encoder = PromptEncoder(config.prompt_encoder)
        self.mask_decoder = MaskDecoder(config.mask_decoder)
        self.memory_encoder = MemoryEncoder(config.memory_encoder)

        # Memory bank is per-object during inference
        # We'll create it dynamically but it needs to be a module for saving/loading
        self._memory_bank = None

        # Image positional encoding (dense PE for mask decoder)
        self._img_pe = None

        # Pre-compute sinusoidal absolute position embeddings for memory attention
        H_enc = W_enc = config.image_size // config.hiera.strides[2]  # stride 16 = 64x64
        self.register_buffer(
            "abs_pos_embed",
            get_sinusoidal_pos_embed(H_enc * W_enc, config.image_encoder_output_dim, torch.device("cpu")),
            persistent=False,
        )
        self.spatial_size = H_enc

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for newly added modules."""
        for p in self.mask_decoder.occlusion_prediction_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @property
    def memory_bank(self) -> MemoryBank:
        if self._memory_bank is None:
            self._memory_bank = MemoryBank(self.config.memory_bank)
        return self._memory_bank

    @memory_bank.setter
    def memory_bank(self, bank: MemoryBank):
        self._memory_bank = bank

    def reset_memory(self):
        """Reset memory bank for a new video or object."""
        if self._memory_bank is not None:
            self._memory_bank.reset()

    def _encode_image(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode a single frame through the image encoder.

        Returns:
            image_embeddings: [B, N, C] FPN-fused features
            high_res_features: List of [B, C, H/4, W/4], [B, C, H/8, W/8]
        """
        return self.image_encoder(x)

    def _prepare_dense_pe(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Get dense positional encoding for mask decoder."""
        if self._img_pe is None or self._img_pe.shape[0] != B:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing="ij",
            )
            coords = torch.stack([grid_x, grid_y], dim=-1)
            coords = coords.unsqueeze(0).expand(B, -1, -1, -1)
            self._img_pe = self.prompt_encoder.pe_layer(
                coords / torch.tensor([W, H], device=device, dtype=torch.float32)
            ).permute(0, 3, 1, 2)
        return self._img_pe

    def _process_prompts(self, image_embeddings: torch.Tensor,
                         coords: Optional[torch.Tensor],
                         labels: Optional[torch.Tensor],
                         boxes: Optional[torch.Tensor],
                         masks: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process prompts through prompt encoder.

        Returns:
            sparse_embeddings: [B, N_p, C]
            dense_embeddings: [B, C, H', W']
        """
        return self.prompt_encoder(coords, labels, boxes, masks)

    def forward(self, x: torch.Tensor,
                coords: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                boxes: Optional[torch.Tensor] = None,
                masks: Optional[torch.Tensor] = None,
                is_first_frame: bool = False,
                multimask_output: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass for a single frame.

        Args:
            x: [B, 3, H, W] input image
            coords: [B, N, 2] click prompts in pixel coordinates
            labels: [B, N] click labels (1=fg, 0=bg, -1=pad)
            boxes: [B, N, 4] box prompts
            masks: [B, 1, H, W] mask prompts
            is_first_frame: if True, reset memory bank
            multimask_output: if True, output multiple masks for ambiguous prompts

        Returns:
            dict with keys:
                - masks: [B, num_multimask_outputs, H, W] predicted masks
                - iou_pred: [B, num_multimask_outputs] predicted IoU scores
                - occlusion_pred: [B, 1] or None
                - object_pointer: [B, C] object pointer token
                - pred_masks: [B, 1, H, W] best mask (highest IoU)
        """
        B, C, H, W = x.shape

        if is_first_frame:
            self.reset_memory()

        # 1. Image encoder (unconditioned features)
        image_embeddings, high_res_features = self._encode_image(x)

        # 2. Memory attention: condition on past memories
        if self.memory_bank.has_memory():
            memory_features, object_pointers = self.memory_bank.get_memory_for_attention(B, x.device)
            # Flatten memory features if needed
            if memory_features is not None:
                mem_shape = memory_features.shape
                if len(mem_shape) == 4:
                    B_m, C_m, H_m, W_m = mem_shape
                    memory_features = memory_features.permute(0, 2, 3, 1).reshape(B_m, H_m * W_m, C_m)

            conditioned_embeddings = self.memory_attention(
                image_embeddings,
                memory_features,
                object_pointers,
                pos_embed=self.abs_pos_embed.to(image_embeddings.dtype),
                spatial_size=self.spatial_size,
            )
        else:
            # No memories yet - just add positional encoding
            conditioned_embeddings = image_embeddings + self.abs_pos_embed.unsqueeze(0).to(image_embeddings.dtype)

        # 3. Prompt encoder
        sparse_embeddings, dense_embeddings = self._process_prompts(
            conditioned_embeddings, coords, labels, boxes, masks
        )

        # 4. Dense positional encoding
        H_enc = W_enc = self.spatial_size
        dense_pe = self._prepare_dense_pe(B, H_enc, W_enc, x.device)

        # 5. Mask decoder
        masks, iou_pred, occlusion_pred, object_pointer = self.mask_decoder(
            conditioned_embeddings, dense_pe,
            sparse_embeddings, dense_embeddings,
            high_res_features,
        )

        # 6. Best mask selection
        best_idx = iou_pred.argmax(dim=1)
        pred_masks = masks[torch.arange(B, device=masks.device), best_idx].unsqueeze(1)

        # 7. Memory encoder: generate memory for this frame
        memory_features = self.memory_encoder(image_embeddings, pred_masks)

        # 8. Store in memory bank
        is_occluded = occlusion_pred is not None and (occlusion_pred < 0.5).any()
        has_prompt = (coords is not None and coords.shape[1] > 0) or \
                     (boxes is not None and boxes.shape[1] > 0) or \
                     (masks is not None)
        self.memory_bank.add_frame_memory(
            memory_features, pred_masks, is_prompted=has_prompt,
            is_occluded=is_occluded, object_pointer=object_pointer,
        )

        output = {
            "masks": masks,
            "iou_pred": iou_pred,
            "occlusion_pred": occlusion_pred,
            "object_pointer": object_pointer,
            "pred_masks": pred_masks,
            "memory_features": memory_features,
        }

        if not multimask_output:
            output["masks"] = pred_masks

        return output

    def forward_video(self, frames: torch.Tensor,
                      first_frame_prompts: Optional[Dict] = None,
                      keyframe_prompts: Optional[Dict[int, Dict]] = None) -> List[Dict[str, torch.Tensor]]:
        """Process an entire video in streaming fashion.

        Args:
            frames: [T, C, H, W] video frames
            first_frame_prompts: dict with coords, labels, boxes, masks for frame 0
            keyframe_prompts: dict mapping frame_idx -> prompt dict for correction frames

        Returns:
            List of output dicts, one per frame
        """
        T = frames.shape[0]
        self.reset_memory()
        outputs = []

        for t in range(T):
            frame = frames[t:t+1]  # [1, C, H, W]

            if t == 0 and first_frame_prompts is not None:
                prompts = first_frame_prompts
            elif keyframe_prompts is not None and t in keyframe_prompts:
                prompts = keyframe_prompts[t]
            else:
                prompts = {}

            output = self.forward(
                frame,
                coords=prompts.get("coords"),
                labels=prompts.get("labels"),
                boxes=prompts.get("boxes"),
                masks=prompts.get("masks"),
                is_first_frame=(t == 0),
                multimask_output=(t == 0),
            )
            outputs.append(output)

        return outputs

    def forward_image(self, x: torch.Tensor,
                      coords: Optional[torch.Tensor] = None,
                      labels: Optional[torch.Tensor] = None,
                      boxes: Optional[torch.Tensor] = None,
                      masks: Optional[torch.Tensor] = None,
                      multimask_output: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass for a single image (no memory)."""
        B, C_img, H_img, W_img = x.shape

        # Image encoder
        image_embeddings, high_res_features = self._encode_image(x)

        # No memory attention - just positional encoding
        image_embeddings = image_embeddings + self.abs_pos_embed.unsqueeze(0).to(image_embeddings.dtype)

        # Prompt encoder
        sparse_embeddings, dense_embeddings = self._process_prompts(
            image_embeddings, coords, labels, boxes, masks
        )

        H_enc = W_enc = self.spatial_size
        dense_pe = self._prepare_dense_pe(B, H_enc, W_enc, x.device)

        # Mask decoder
        masks, iou_pred, occlusion_pred, object_pointer = self.mask_decoder(
            image_embeddings, dense_pe,
            sparse_embeddings, dense_embeddings,
            high_res_features,
        )

        best_idx = iou_pred.argmax(dim=1)
        pred_masks = masks[torch.arange(B, device=masks.device), best_idx].unsqueeze(1)

        output = {
            "masks": masks if multimask_output else pred_masks,
            "iou_pred": iou_pred,
            "occlusion_pred": occlusion_pred,
            "object_pointer": object_pointer,
            "pred_masks": pred_masks,
        }
        return output


def build_sam2(encoder_size: str = "b_plus") -> SAM2:
    """Build SAM 2 model with specified encoder size.

    Args:
        encoder_size: one of "t", "s", "b_plus", "l"
    """
    from config import get_config
    config = get_config(encoder_size)
    return SAM2(config)
