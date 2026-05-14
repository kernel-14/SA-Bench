"""Mask decoder for SAM 2 based on SAM's two-way transformer architecture.

Key additions over SAM:
- Skip connections from hierarchical image encoder (stride 4, 8) for high-res details
- Occlusion prediction head: additional token to predict object visibility
- Object pointer token: mask output token used for cross-attention in memory
- Multi-mask prediction for ambiguous prompts (like SAM)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MaskDecoderConfig
from transformer import TwoWayTransformer, Mlp


class MaskDecoder(nn.Module):
    """SAM 2 mask decoder with two-way transformer, IoU prediction, occlusion head,
    and multi-mask output for ambiguous prompts.
    """
    def __init__(self, config: MaskDecoderConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_multimask_outputs = config.num_multimask_outputs
        self.occlusion_head_enabled = config.occlusion_head_enabled

        # Transformer
        self.transformer = TwoWayTransformer(
            depth=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )

        # Learnable tokens
        # IoU output token
        self.iou_token = nn.Embedding(1, config.d_model)
        # Mask output tokens (one per multimask output)
        self.mask_tokens = nn.Embedding(config.num_multimask_outputs, config.d_model)
        # Occlusion token (presence prediction)
        if config.occlusion_head_enabled:
            self.occlusion_token = nn.Embedding(1, config.d_model)

        # Output heads
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(config.d_model, config.d_model // 4, kernel_size=2, stride=2),
            nn.LayerNorm([config.d_model // 4]),
            nn.GELU(),
            nn.ConvTranspose2d(config.d_model // 4, config.d_model // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

        # High-res feature fusion (from hierarchical encoder stages 1, 2)
        self.use_high_res_features = config.use_high_res_features

        # Hypernetwork for mask prediction
        self.output_hypernetworks_mlps = nn.ModuleList([
            Mlp(config.d_model, config.d_model, config.d_model // 8, drop=0.0)
            for _ in range(config.num_multimask_outputs)
        ])

        # IoU prediction head
        self.iou_prediction_head = Mlp(
            config.d_model, config.iou_head_hidden_dim, config.num_multimask_outputs, drop=0.0
        )

        # Occlusion prediction head
        if config.occlusion_head_enabled:
            self.occlusion_prediction_head = nn.Sequential(
                Mlp(config.d_model, config.occlusion_head_hidden_dim, 1, drop=0.0),
                nn.Sigmoid(),
            )

        # The object pointer token comes from the mask token of the output mask
        self.object_pointer_token_dim = config.d_model

        # Learned occlusion embedding added to memory features of occluded frames
        self.occlusion_embed = nn.Parameter(torch.zeros(1, 1, config.d_model))

    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor,
                dense_prompt_embeddings: torch.Tensor,
                high_res_features: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of mask decoder.

        Args:
            image_embeddings: [B, H*W, C] from image encoder (FPN output)
            image_pe: [B, C, H', W'] dense positional encoding
            sparse_prompt_embeddings: [B, N_p, C] sparse prompt embeddings
            dense_prompt_embeddings: [B, C, H', W'] dense prompt embeddings
            high_res_features: Optional list of [B, C, H/4, W/4] and [B, C, H/8, W/8]

        Returns:
            masks: [B, num_multimask_outputs, H, W] predicted masks
            iou_pred: [B, num_multimask_outputs] predicted IoU scores
            occlusion_pred: [B, 1] or None, object presence prediction
            object_pointer: [B, C] object pointer token (mask token of highest IoU mask)
        """
        B, N, C = image_embeddings.shape

        # Flatten dense embeddings to token format
        dense_prompt_embeddings_flat = dense_prompt_embeddings.permute(0, 2, 3, 1).reshape(B, -1, C)

        # Concatenate output tokens with prompt embeddings
        output_tokens = torch.cat([
            self.iou_token.weight.unsqueeze(0).expand(B, -1, -1),
            self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1),
        ], dim=1)
        if self.occlusion_head_enabled:
            output_tokens = torch.cat([
                output_tokens,
                self.occlusion_token.weight.unsqueeze(0).expand(B, -1, -1),
            ], dim=1)

        # Image tokens (flat image embeddings + dense prompt embeddings)
        image_tokens = image_embeddings + dense_prompt_embeddings_flat

        # Run two-way transformer
        output_tokens, image_tokens = self.transformer(output_tokens, image_tokens)

        # Extract tokens
        iou_token_out = output_tokens[:, 0:1, :]
        mask_tokens_out = output_tokens[:, 1:1 + self.num_multimask_outputs, :]

        if self.occlusion_head_enabled:
            occlusion_token_out = output_tokens[:, 1 + self.num_multimask_outputs, :]
        else:
            occlusion_token_out = None

        # Upscale image tokens back to spatial format
        H_enc, W_enc = int((N) ** 0.5), int((N) ** 0.5)
        src = image_tokens.reshape(B, H_enc, W_enc, C).permute(0, 3, 1, 2)
        upscaled_embedding = self.output_upscaling(src)

        # Incorporate high-res skip features
        if self.use_high_res_features and high_res_features is not None:
            for hf in high_res_features:
                hf_resized = F.interpolate(hf, size=upscaled_embedding.shape[-2:], mode="bilinear", align_corners=False)
                upscaled_embedding = upscaled_embedding + hf_resized

        # Compute masks via hypernetwork: each mask token produces a weight vector
        # that dot-products with the upscaled feature map
        masks = []
        for i in range(self.num_multimask_outputs):
            weight = self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])  # [B, C/8]
            weight = weight.unsqueeze(2).unsqueeze(3)  # [B, C/8, 1, 1]
            mask = torch.sum(upscaled_embedding * weight, dim=1, keepdim=True)  # [B, 1, H', W']
            masks.append(mask)

        masks = torch.cat(masks, dim=1)  # [B, num_multimask_outputs, H', W']

        # Upsample to original resolution
        H_out, W_out = 4 * H_enc, 4 * W_enc
        masks = F.interpolate(masks, size=(H_out, W_out), mode="bilinear", align_corners=False)

        # IoU prediction
        iou_pred = self.iou_prediction_head(iou_token_out)  # [B, 1, num_multimask_outputs]
        iou_pred = iou_pred.squeeze(1)  # [B, num_multimask_outputs]

        # Occlusion prediction
        if self.occlusion_head_enabled and occlusion_token_out is not None:
            occlusion_pred = self.occlusion_prediction_head(occlusion_token_out.unsqueeze(1))
            occlusion_pred = occlusion_pred.squeeze(-1)  # [B, 1]
        else:
            occlusion_pred = None

        # Object pointer: use the mask token with highest predicted IoU
        best_mask_idx = iou_pred.argmax(dim=1)  # [B]
        object_pointer = mask_tokens_out[torch.arange(B, device=mask_tokens_out.device), best_mask_idx]  # [B, C]

        return masks, iou_pred, occlusion_pred, object_pointer

    def predict_masks(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
                      sparse_prompt_embeddings: torch.Tensor,
                      dense_prompt_embeddings: torch.Tensor,
                      high_res_features: Optional[List[torch.Tensor]] = None,
                      multimask_output: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Predict masks with optional multi-mask output.

        Args:
            multimask_output: If True, return all multimask outputs (for ambiguous prompts).
                              If False, return only the best mask.
        """
        masks, iou_pred, occlusion_pred, object_pointer = self.forward(
            image_embeddings, image_pe, sparse_prompt_embeddings,
            dense_prompt_embeddings, high_res_features
        )

        if multimask_output:
            return masks, iou_pred, occlusion_pred, object_pointer
        else:
            best_idx = iou_pred.argmax(dim=1, keepdim=True)
            best_mask = masks.gather(1, best_idx.unsqueeze(1).unsqueeze(-1).unsqueeze(-1))
            best_iou = iou_pred.gather(1, best_idx)
            return best_mask.squeeze(1).unsqueeze(1), best_iou, occlusion_pred, object_pointer
