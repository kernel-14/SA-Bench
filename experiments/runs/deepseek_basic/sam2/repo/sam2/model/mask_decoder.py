"""
Mask Decoder for SAM 2.

Largely follows SAM's design with key modifications:
1. Skip connections from hierarchical image encoder (stride 4 and 8 features)
   bypass memory attention and feed into upsampling layers
2. Additional occlusion prediction head: predicts whether object is visible
   in the current frame
3. Object pointer token: the mask token corresponding to output mask,
   stored in memory bank for cross-attention
4. Multi-mask prediction for ambiguity handling (like SAM)
5. When no follow-up prompts resolve ambiguity, only propagates the mask
   with the highest predicted IoU

Architecture (Figure 8):
- Stacks "two-way" transformer blocks that update prompt and frame embeddings
- Includes stride 4 and 8 features from image encoder during upsampling
- Produces mask, IoU score, occlusion score, and object pointer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class TwoWayTransformerBlock(nn.Module):
    """Two-way transformer block: self-attention + cross-attention between
    prompt tokens and image tokens."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Self-attention for image tokens
        self.norm1_image = nn.LayerNorm(embed_dim)
        self.self_attn_image_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.self_attn_image_proj = nn.Linear(embed_dim, embed_dim)

        # Self-attention for prompt tokens
        self.norm1_prompt = nn.LayerNorm(embed_dim)
        self.self_attn_prompt_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.self_attn_prompt_proj = nn.Linear(embed_dim, embed_dim)

        # Cross-attention: image -> prompt
        self.norm2_image = nn.LayerNorm(embed_dim)
        self.cross_attn_i2p_q = nn.Linear(embed_dim, embed_dim)
        self.cross_attn_i2p_kv = nn.Linear(embed_dim, embed_dim * 2)
        self.cross_attn_i2p_proj = nn.Linear(embed_dim, embed_dim)

        # Cross-attention: prompt -> image
        self.norm2_prompt = nn.LayerNorm(embed_dim)
        self.cross_attn_p2i_q = nn.Linear(embed_dim, embed_dim)
        self.cross_attn_p2i_kv = nn.Linear(embed_dim, embed_dim * 2)
        self.cross_attn_p2i_proj = nn.Linear(embed_dim, embed_dim)

        # MLP for image tokens
        self.norm3_image = nn.LayerNorm(embed_dim)
        self.mlp_image = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

        # MLP for prompt tokens
        self.norm3_prompt = nn.LayerNorm(embed_dim)
        self.mlp_prompt = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

        self.head_dim = embed_dim // num_heads

    def _self_attention(self, qkv_proj, proj, x, norm):
        B, N, C = x.shape
        x_norm = norm(x)
        qkv = qkv_proj(x_norm)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return proj(out)

    def _cross_attention(self, q_proj, kv_proj, proj, x, context, norm_x, norm_ctx):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = q_proj(norm_x(x))
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        kv = kv_proj(norm_ctx(context))
        kv = kv.reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return proj(out)

    def forward(
        self,
        image_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_tokens: [B, H*W, C]
            prompt_tokens: [B, N_prompt, C]

        Returns:
            updated image_tokens, updated prompt_tokens
        """
        # Self-attention
        image_tokens = image_tokens + self._self_attention(
            self.self_attn_image_qkv, self.self_attn_image_proj,
            image_tokens, self.norm1_image,
        )
        prompt_tokens = prompt_tokens + self._self_attention(
            self.self_attn_prompt_qkv, self.self_attn_prompt_proj,
            prompt_tokens, self.norm1_prompt,
        )

        # Cross-attention: image queries attend to prompt keys
        image_tokens = image_tokens + self._cross_attention(
            self.cross_attn_i2p_q, self.cross_attn_i2p_kv, self.cross_attn_i2p_proj,
            image_tokens, prompt_tokens,
            self.norm2_image, self.norm2_prompt,
        )

        # Cross-attention: prompt queries attend to image keys
        prompt_tokens = prompt_tokens + self._cross_attention(
            self.cross_attn_p2i_q, self.cross_attn_p2i_kv, self.cross_attn_p2i_proj,
            prompt_tokens, image_tokens,
            self.norm2_prompt, self.norm2_image,
        )

        # MLP
        image_tokens = image_tokens + self.mlp_image(self.norm3_image(image_tokens))
        prompt_tokens = prompt_tokens + self.mlp_prompt(self.norm3_prompt(prompt_tokens))

        return image_tokens, prompt_tokens


class MaskDecoder(nn.Module):
    """
    Mask decoder for SAM 2.
    Predicts segmentation masks from conditioned frame embeddings and prompts.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_multimask_outputs: int = 3,
        num_transformer_blocks: int = 2,
        num_heads: int = 8,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ):
        """
        Args:
            embed_dim: embedding dimension
            num_multimask_outputs: number of output masks for ambiguity handling (3)
            num_transformer_blocks: number of two-way transformer blocks (2)
            num_heads: number of attention heads
            iou_head_depth: depth of IoU prediction MLP
            iou_head_hidden_dim: hidden dimension of IoU head
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_multimask_outputs = num_multimask_outputs

        # Iou token and mask tokens (learned)
        self.iou_token = nn.Embedding(1, embed_dim)
        self.mask_tokens = nn.Embedding(num_multimask_outputs + 1, embed_dim)  # +1 for no-mask token
        self.occlusion_token = nn.Embedding(1, embed_dim)  # For occlusion prediction

        # Two-way transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TwoWayTransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_transformer_blocks)
        ])

        # Output upsampling layers with skip connections from image encoder
        # Stride 16 -> 4 using skip features
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),  # 64->128
            nn.LayerNorm([embed_dim // 4, 128, 128]),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),  # 128->256
            nn.GELU(),
        )

        # Conv layers that incorporate skip features from stride 8 and 4
        self.skip_fusion_8 = nn.Conv2d(embed_dim // 4 + embed_dim, embed_dim // 8, kernel_size=3, padding=1)
        self.skip_fusion_4 = nn.Conv2d(embed_dim // 8 + embed_dim, embed_dim // 16, kernel_size=3, padding=1)

        # Final upsampling to image resolution
        self.output_hypernetworks_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim // 16),
            )
            for _ in range(num_multimask_outputs)
        ])

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim // 16, embed_dim // 32, kernel_size=2, stride=2),  # 256->512
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 32, embed_dim // 64, kernel_size=2, stride=2),  # 512->1024
            nn.GELU(),
        )

        # IoU prediction head
        self.iou_prediction_head = nn.Sequential(
            nn.Linear(embed_dim, iou_head_hidden_dim),
            nn.GELU(),
            nn.Linear(iou_head_hidden_dim, iou_head_hidden_dim),
            nn.GELU(),
            nn.Linear(iou_head_hidden_dim, num_multimask_outputs),
        )

        # Occlusion prediction head
        self.occlusion_prediction_head = nn.Sequential(
            nn.Linear(embed_dim, iou_head_hidden_dim),
            nn.GELU(),
            nn.Linear(iou_head_hidden_dim, 1),  # Binary: visible or occluded
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: Optional[torch.Tensor],
        high_res_features: List[torch.Tensor],
        multimask_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image_embeddings: [B, C, H, W] conditioned image features from memory attention
            image_pe: [B, C, H, W] positional encodings for image features
            sparse_prompt_embeddings: [B, N_prompt, C] sparse prompt embeddings
            dense_prompt_embeddings: [B, C, H, W] or None, dense mask prompt embeddings
            high_res_features: list of [B, C, H/4, W/4] and [B, C, H/8, W/8] skip features
            multimask_output: whether to output multiple masks for ambiguity

        Returns:
            masks: [B, num_masks, H_img, W_img]
            iou_pred: [B, num_masks]
            occlusion_pred: [B, 1]
            object_pointer: [B, 256]
        """
        B, C, H, W = image_embeddings.shape

        # Flatten image embeddings
        image_tokens = image_embeddings.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        image_pe_flat = image_pe.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        image_tokens = image_tokens + image_pe_flat

        # Prepare prompt tokens
        iou_token_out = self.iou_token.weight.unsqueeze(0).expand(B, 1, C)
        mask_tokens_out = self.mask_tokens.weight.unsqueeze(0).expand(B, -1, C)
        occlusion_token_out = self.occlusion_token.weight.unsqueeze(0).expand(B, 1, C)

        prompt_tokens = torch.cat([
            iou_token_out,
            mask_tokens_out,
            occlusion_token_out,
        ], dim=1)  # [B, 1 + (3+1) + 1, C]

        if sparse_prompt_embeddings is not None:
            prompt_tokens = torch.cat([sparse_prompt_embeddings, prompt_tokens], dim=1)

        # Add dense prompt embeddings to image tokens
        if dense_prompt_embeddings is not None:
            dense_flat = dense_prompt_embeddings.flatten(2).permute(0, 2, 1)
            image_tokens = image_tokens + dense_flat

        # Pass through two-way transformer blocks
        for block in self.transformer_blocks:
            image_tokens, prompt_tokens = block(image_tokens, prompt_tokens)

        # Extract output tokens
        iou_token_out = prompt_tokens[:, 0:1, :]
        mask_tokens_out = prompt_tokens[:, 1:1+self.num_multimask_outputs+1, :]
        occlusion_token_out = prompt_tokens[:, 1+self.num_multimask_outputs+1, :]

        # Object pointer is the predicted mask token (the one that gets selected for propagation)
        # During training, use all mask tokens; during inference, select highest IoU
        object_pointer = mask_tokens_out[:, 0, :]  # [B, 256] - first mask token as default

        # Upsample image tokens to produce masks
        src = image_tokens.permute(0, 2, 1).reshape(B, C, H, W)

        # Upconv to stride 8
        up = self.output_upscaling[0:3](src)  # [B, C//4, 2H, 2W] -> stride 8
        # Fuse with stride 8 skip feature
        skip8 = high_res_features[1]  # [B, C, 2H, 2W]
        if skip8.shape[-2:] != up.shape[-2:]:
            skip8 = F.interpolate(skip8, size=up.shape[-2:], mode="bilinear", align_corners=False)
        up = torch.cat([up, skip8], dim=1)
        up = self.skip_fusion_8(up)  # [B, C//8, 2H, 2W]

        # Upconv to stride 4
        up2 = F.interpolate(up, scale_factor=2, mode="bilinear", align_corners=False)
        # Fuse with stride 4 skip feature
        skip4 = high_res_features[0]  # [B, C, 4H, 4W]
        if skip4.shape[-2:] != up2.shape[-2:]:
            skip4 = F.interpolate(skip4, size=up2.shape[-2:], mode="bilinear", align_corners=False)
        up2 = torch.cat([up2, skip4], dim=1)
        up2 = self.skip_fusion_4(up2)  # [B, C//16, 4H, 4W]

        # Dynamic mask prediction using hypernetworks
        masks = []
        for i in range(self.num_multimask_outputs):
            hypernet = self.output_hypernetworks_mlps[i]
            weights = hypernet(mask_tokens_out[:, i, :])  # [B, C//16]
            # einsum for dynamic convolution
            mask = torch.einsum("bc,bchw->bhw", weights, up2)
            masks.append(mask)
        masks = torch.stack(masks, dim=1)  # [B, 3, 4H, 4W]

        # Final upsampling to image resolution
        masks = F.interpolate(masks, scale_factor=4, mode="bilinear", align_corners=False)  # [B, 3, 16H, 16W]

        # IoU prediction
        iou_pred = self.iou_prediction_head(iou_token_out)  # [B, 1, 3]
        iou_pred = iou_pred.squeeze(1)  # [B, 3]
        # Apply sigmoid to restrict between 0 and 1 (paper: Section D.2.1)
        iou_pred = torch.sigmoid(iou_pred)

        # Occlusion prediction
        occlusion_pred = self.occlusion_prediction_head(occlusion_token_out)  # [B, 1]
        occlusion_pred = occlusion_pred.squeeze(-1)

        if not multimask_output:
            # Select mask with highest IoU
            best_idx = iou_pred.argmax(dim=1)
            masks = masks[torch.arange(B), best_idx].unsqueeze(1)
            iou_pred = iou_pred[torch.arange(B), best_idx].unsqueeze(1)
            object_pointer = mask_tokens_out[torch.arange(B), best_idx, :]

        return masks, iou_pred, occlusion_pred, object_pointer
