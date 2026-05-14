"""
SAM 2: Segment Anything Model 2

Main model class that integrates all components:
- Image encoder (Hiera)
- Memory attention
- Prompt encoder
- Mask decoder
- Memory encoder
- Memory bank

The model supports both image and video segmentation through a unified interface.
For images, the memory bank is empty and the model behaves like SAM.
For videos, the model uses streaming memory to condition predictions on past frames.
"""

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hiera_image_encoder import HieraImageEncoder, build_hiera_encoder
from .memory_attention import MemoryAttention
from .memory_encoder import MemoryEncoder
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder, PositionEmbeddingRandom


class SAM2Model(nn.Module):
    """
    SAM 2: Segment Anything Model 2.

    A unified model for promptable visual segmentation in images and videos.
    Extends SAM with streaming memory for video processing.

    Architecture:
    1. Image encoder: Hiera hierarchical ViT, processes each frame independently
    2. Memory attention: Conditions frame features on memory bank
    3. Prompt encoder: Encodes point/box/mask prompts
    4. Mask decoder: Predicts segmentation masks
    5. Memory encoder: Creates memory features from predictions
    6. Memory bank: FIFO queue of recent frame memories + prompted frame memories

    For video processing:
    - Frames are processed one at a time (streaming)
    - Memory bank stores features from N recent frames + M prompted frames
    - Object pointers (mask tokens) provide high-level semantic information
    """

    def __init__(
        self,
        image_encoder: HieraImageEncoder,
        memory_attention: MemoryAttention,
        memory_encoder: MemoryEncoder,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        # Memory bank configuration
        num_maskmem: int = 7,  # N recent frames + 1 for current
        image_size: int = 1024,
        # Backbone stride (stride of image encoder output)
        backbone_stride: int = 16,
        # Number of prompted frames to keep in memory
        num_prompted_frames: int = 8,
        # Whether to use object pointers
        use_obj_ptrs_in_encoder: bool = True,
        # Number of object pointer tokens
        num_obj_ptr_tokens: int = 4,
        # Whether to add temporal position encoding to memories
        add_tpos_enc_to_obj_ptrs: bool = False,
        # Sigmoid for IoU predictions
        sigmoid_iou: bool = True,
        # Multimask output for ambiguous prompts
        multimask_output_in_sam: bool = True,
        multimask_min_pt_num: int = 1,
        multimask_max_pt_num: int = 1,
        # Occlusion prediction
        pred_obj_scores: bool = True,
        pred_obj_scores_mlp: bool = True,
        # Training settings
        directly_add_no_mem_embed: bool = False,
        use_high_res_features_in_sam: bool = True,
        # Memory feature dimension
        hidden_dim: int = 256,
        mem_dim: int = 64,
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.memory_attention = memory_attention
        self.memory_encoder = memory_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.num_maskmem = num_maskmem
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.num_prompted_frames = num_prompted_frames
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.num_obj_ptr_tokens = num_obj_ptr_tokens
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        self.sigmoid_iou = sigmoid_iou
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.pred_obj_scores = pred_obj_scores
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim

        # Positional encoding for image features
        self.pe_layer = PositionEmbeddingRandom(hidden_dim // 2)

        # No-memory embedding (used when memory bank is empty)
        self.no_mem_embed = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))

        # No-object pointer embedding
        self.no_obj_ptr = nn.Parameter(torch.zeros(1, hidden_dim))

        # Image feature size at backbone stride
        self.feat_size = image_size // backbone_stride

    def get_image_embedding_size(self) -> Tuple[int, int]:
        """Return the spatial size of image embeddings."""
        return (self.feat_size, self.feat_size)

    def _get_image_feature(self, img: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract image features using the image encoder.

        Args:
            img: [B, 3, H, W] input image

        Returns:
            image_embedding: [B, C, H/16, W/16] main image features
            skip_features: list of high-res features for mask decoder
        """
        image_embedding, skip_features = self.image_encoder(img)
        return image_embedding, skip_features

    def _get_memory_conditioned_features(
        self,
        image_embedding: torch.Tensor,
        memory_bank: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Condition image features on memory bank using memory attention.

        Args:
            image_embedding: [B, C, H, W] raw image features
            memory_bank: dict containing memory features and object pointers

        Returns:
            conditioned_features: [B, C, H, W] memory-conditioned features
        """
        B, C, H, W = image_embedding.shape

        # Get positional encoding for current frame
        curr_pos = self.pe_layer(
            (H, W)
        ).unsqueeze(0).expand(B, -1, -1, -1)

        if memory_bank is None or len(memory_bank.get('recent_feats', [])) == 0:
            # No memory available - add no-memory embedding
            if hasattr(self, 'no_mem_embed'):
                image_embedding = image_embedding + self.no_mem_embed
            return image_embedding

        # Get memory features and positions
        memory_feats = memory_bank.get('recent_feats', []) + memory_bank.get('prompted_feats', [])
        memory_pos = memory_bank.get('recent_pos', []) + memory_bank.get('prompted_pos', [])
        object_ptrs = memory_bank.get('object_ptrs', []) if self.use_obj_ptrs_in_encoder else None

        # Apply memory attention
        conditioned = self.memory_attention(
            curr_feats=image_embedding,
            curr_pos=curr_pos,
            memory_bank_feats=memory_feats,
            memory_bank_pos=memory_pos,
            object_ptrs=object_ptrs,
        )

        return conditioned

    def _predict_masks(
        self,
        conditioned_features: torch.Tensor,
        skip_features: List[torch.Tensor],
        sparse_embeddings: torch.Tensor,
        dense_embeddings: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks using the mask decoder.

        Returns:
            masks: [B, num_masks, H, W] predicted masks (logits)
            iou_pred: [B, num_masks] predicted IoU scores
            mask_tokens: [B, num_masks, C] mask token outputs
            occlusion_pred: [B, 1] occlusion prediction
        """
        B, C, H, W = conditioned_features.shape

        # Get positional encoding
        image_pe = self.pe_layer((H, W)).unsqueeze(0).expand(B, -1, -1, -1)

        masks, iou_pred, mask_tokens, occlusion_pred = self.mask_decoder(
            image_embeddings=conditioned_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            high_res_features=skip_features if self.use_high_res_features_in_sam else None,
        )

        return masks, iou_pred, mask_tokens, occlusion_pred

    def _encode_memory(
        self,
        image_embedding: torch.Tensor,
        pred_mask: torch.Tensor,
        is_occluded: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode current frame prediction into memory features.

        Args:
            image_embedding: [B, C, H, W] image encoder features
            pred_mask: [B, 1, H_full, W_full] predicted mask
            is_occluded: [B] boolean tensor

        Returns:
            memory: [B, mem_dim, H, W] memory features
        """
        memory = self.memory_encoder(
            current_vision_feats=image_embedding,
            pred_masks=pred_mask,
            is_occluded=is_occluded,
        )
        return memory

    def forward_image(
        self,
        img: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for image segmentation (SAM mode).

        Args:
            img: [B, 3, H, W] input image
            points: optional (coords, labels) point prompts
            boxes: optional [B, 4] box prompts
            masks: optional [B, 1, H, W] mask prompts
            multimask_output: whether to output multiple masks

        Returns:
            dict with 'masks', 'iou_predictions', 'low_res_masks'
        """
        # Extract image features
        image_embedding, skip_features = self._get_image_feature(img)

        # No memory for image mode
        conditioned_features = image_embedding

        # Encode prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )

        # Predict masks
        low_res_masks, iou_pred, mask_tokens, occlusion_pred = self._predict_masks(
            conditioned_features=conditioned_features,
            skip_features=skip_features,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale masks to input resolution
        masks = F.interpolate(
            low_res_masks,
            size=(img.shape[2], img.shape[3]),
            mode='bilinear',
            align_corners=False,
        )

        return {
            'masks': masks,
            'iou_predictions': iou_pred,
            'low_res_masks': low_res_masks,
            'mask_tokens': mask_tokens,
            'occlusion_pred': occlusion_pred,
        }

    def forward_video_frame(
        self,
        img: torch.Tensor,
        memory_bank: Optional[Dict] = None,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass for a single video frame.

        Args:
            img: [B, 3, H, W] input frame
            memory_bank: dict containing memory features from past frames
            points: optional point prompts
            boxes: optional box prompts
            masks: optional mask prompts
            multimask_output: whether to output multiple masks

        Returns:
            outputs: dict with predictions
            memory: [B, mem_dim, H, W] memory features for this frame
        """
        # Extract image features
        image_embedding, skip_features = self._get_image_feature(img)

        # Condition on memory
        conditioned_features = self._get_memory_conditioned_features(
            image_embedding=image_embedding,
            memory_bank=memory_bank,
        )

        # Encode prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )

        # Predict masks
        low_res_masks, iou_pred, mask_tokens, occlusion_pred = self._predict_masks(
            conditioned_features=conditioned_features,
            skip_features=skip_features,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale masks
        masks = F.interpolate(
            low_res_masks,
            size=(img.shape[2], img.shape[3]),
            mode='bilinear',
            align_corners=False,
        )

        # Determine if object is occluded
        is_occluded = None
        if self.pred_obj_scores:
            is_occluded = (torch.sigmoid(occlusion_pred) < 0.5).squeeze(-1)

        # Select best mask for memory encoding
        if multimask_output and iou_pred.shape[1] > 1:
            best_mask_idx = iou_pred.argmax(dim=1)
            best_mask = low_res_masks[torch.arange(low_res_masks.shape[0]), best_mask_idx].unsqueeze(1)
        else:
            best_mask = low_res_masks[:, 0:1]

        # Encode memory
        memory = self._encode_memory(
            image_embedding=image_embedding,
            pred_mask=best_mask,
            is_occluded=is_occluded,
        )

        outputs = {
            'masks': masks,
            'iou_predictions': iou_pred,
            'low_res_masks': low_res_masks,
            'mask_tokens': mask_tokens,
            'occlusion_pred': occlusion_pred,
            'image_embedding': image_embedding,
        }

        return outputs, memory

    def forward(
        self,
        imgs: torch.Tensor,
        prompts: Optional[List[Dict]] = None,
        memory_bank: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        General forward pass supporting both image and video modes.

        Args:
            imgs: [B, T, 3, H, W] for video or [B, 3, H, W] for image
            prompts: list of prompt dicts per frame
            memory_bank: optional memory bank for video mode

        Returns:
            dict with predictions
        """
        if imgs.dim() == 4:
            # Image mode
            points = None
            boxes = None
            masks = None
            if prompts is not None and len(prompts) > 0:
                p = prompts[0]
                points = p.get('points', None)
                boxes = p.get('boxes', None)
                masks = p.get('masks', None)
            return self.forward_image(imgs, points=points, boxes=boxes, masks=masks)
        else:
            # Video mode - process frame by frame
            B, T, C, H, W = imgs.shape
            all_outputs = []
            current_memory_bank = memory_bank or {
                'recent_feats': [],
                'recent_pos': [],
                'prompted_feats': [],
                'prompted_pos': [],
                'object_ptrs': [],
            }

            for t in range(T):
                frame = imgs[:, t]
                frame_prompts = prompts[t] if prompts is not None else {}

                outputs, memory = self.forward_video_frame(
                    img=frame,
                    memory_bank=current_memory_bank,
                    points=frame_prompts.get('points', None),
                    boxes=frame_prompts.get('boxes', None),
                    masks=frame_prompts.get('masks', None),
                )

                # Update memory bank
                B_f, C_m, H_m, W_m = memory.shape
                pos = self.pe_layer((H_m, W_m)).unsqueeze(0).expand(B_f, -1, -1, -1)

                current_memory_bank['recent_feats'].append(memory)
                current_memory_bank['recent_pos'].append(pos)

                # Keep only N most recent frames
                if len(current_memory_bank['recent_feats']) > self.num_maskmem:
                    current_memory_bank['recent_feats'].pop(0)
                    current_memory_bank['recent_pos'].pop(0)

                # Store object pointers
                if self.use_obj_ptrs_in_encoder and 'mask_tokens' in outputs:
                    # Use the first mask token as object pointer
                    obj_ptr = outputs['mask_tokens'][:, 0, :]  # B C
                    current_memory_bank['object_ptrs'].append(obj_ptr)
                    if len(current_memory_bank['object_ptrs']) > self.num_maskmem:
                        current_memory_bank['object_ptrs'].pop(0)

                all_outputs.append(outputs)

            return {
                'frame_outputs': all_outputs,
                'memory_bank': current_memory_bank,
            }


def build_sam2(
    model_size: str = "base_plus",
    image_size: int = 1024,
    num_maskmem: int = 7,
) -> SAM2Model:
    """
    Build a SAM 2 model of the specified size.

    Args:
        model_size: one of 'tiny', 'small', 'base_plus', 'large'
        image_size: input image size (default 1024)
        num_maskmem: number of frames to keep in memory bank

    Returns:
        SAM2Model instance
    """
    hidden_dim = 256
    mem_dim = 64
    feat_size = image_size // 16

    # Build image encoder
    image_encoder = build_hiera_encoder(model_size=model_size, img_size=image_size, out_chans=hidden_dim)

    # Build memory attention
    memory_attention = MemoryAttention(
        d_model=hidden_dim,
        num_layers=4,
        memory_dim=mem_dim,
        object_ptr_dim=hidden_dim,
        num_object_ptr_tokens=4,
    )

    # Build memory encoder
    memory_encoder = MemoryEncoder(
        out_dim=mem_dim,
        in_dim=hidden_dim,
    )

    # Build prompt encoder
    prompt_encoder = PromptEncoder(
        embed_dim=hidden_dim,
        image_embedding_size=(feat_size, feat_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    # Build mask decoder
    mask_decoder = MaskDecoder(
        transformer_dim=hidden_dim,
        num_multimask_outputs=3,
        use_high_res_features=True,
        skip_dims=(hidden_dim // 4, hidden_dim // 2),
    )

    model = SAM2Model(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        num_maskmem=num_maskmem,
        image_size=image_size,
        hidden_dim=hidden_dim,
        mem_dim=mem_dim,
    )

    return model
