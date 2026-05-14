"""
Mask decoder for SAM 2.

Extends SAM's mask decoder with:
  1. Skip connections from stride-4 and stride-8 Hiera features during upsampling.
  2. Occlusion prediction head: predicts whether the object is visible in the frame.
  3. Object pointer: the mask token output is used as the object pointer stored in
     the memory bank.
  4. Multi-mask prediction for ambiguous prompts (single click); highest-IoU mask
     selected for propagation when ambiguity is unresolved.

Architecture:
  - Two-way transformer blocks update both prompt tokens and image tokens.
  - Output tokens: [iou_token, mask_tokens..., occlusion_token]
  - Upsampling path: 2× bilinear + conv, incorporating stride-8 and stride-4 skips.
  - Per-mask MLP heads produce final mask logits.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import MLP, Attention, LayerNorm2d


# ---------------------------------------------------------------------------
# Two-way attention block
# ---------------------------------------------------------------------------

class TwoWayAttentionBlock(nn.Module):
    """
    Transformer block that performs:
      1. Self-attention on sparse (prompt) tokens.
      2. Cross-attention from sparse tokens to dense (image) tokens.
      3. MLP on sparse tokens.
      4. Cross-attention from dense tokens to sparse tokens.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: type = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_pe: Tensor,
        key_pe: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Self-attention on queries
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q, q, queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross-attention: queries attend to keys
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q, k, keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP on queries
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross-attention: keys attend to queries
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(k, q, queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformer(nn.Module):
    """Stack of two-way attention blocks."""

    def __init__(
        self,
        depth: int = 2,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: type = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0),
            )
            for i in range(depth)
        ])
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        image_embedding: (B, C, H, W)
        image_pe:        (B, C, H, W) — positional encoding for image tokens
        point_embedding: (B, N_tokens, C) — output tokens (serve as both queries and query PE)
        Returns: (queries, keys) — updated token and image embeddings.
        """
        B, C, H, W = image_embedding.shape
        # Expand image_pe to batch size if needed
        if image_pe.shape[0] == 1 and B > 1:
            image_pe = image_pe.expand(B, -1, -1, -1)

        image_embedding_flat = image_embedding.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        image_pe_flat = image_pe.flatten(2).permute(0, 2, 1)                # (B, H*W, C)

        # The initial point_embedding serves as both queries and their positional encoding
        query_pe = point_embedding
        queries = point_embedding
        keys = image_embedding_flat

        for layer in self.layers:
            queries, keys = layer(queries, keys, query_pe, image_pe_flat)

        # Final cross-attention: tokens attend to image
        q = queries + query_pe
        k = keys + image_pe_flat
        attn_out = self.final_attn_token_to_image(q, k, keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


# ---------------------------------------------------------------------------
# Mask decoder
# ---------------------------------------------------------------------------

class MaskDecoder(nn.Module):
    """
    SAM 2 mask decoder.

    Predicts:
      - num_multimask_outputs + 1 masks (1 single-mask + num_multimask for ambiguous prompts)
      - IoU scores for each mask
      - Occlusion score (object presence probability)
      - Object pointer token (mask token used as memory pointer)
    """

    def __init__(
        self,
        transformer_dim: int = 256,
        transformer_depth: int = 2,
        transformer_num_heads: int = 8,
        transformer_mlp_dim: int = 2048,
        num_multimask_outputs: int = 3,
        activation: type = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        # Skip connection channel dims from image encoder
        skip_dim_s4: int = 64,   # stride-4 skip (out_chans // 4)
        skip_dim_s8: int = 128,  # stride-8 skip (out_chans // 2)
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1  # +1 for single-mask output

        # Output tokens: IoU + masks + occlusion
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.occlusion_token = nn.Embedding(1, transformer_dim)

        self.transformer = TwoWayTransformer(
            depth=transformer_depth,
            embedding_dim=transformer_dim,
            num_heads=transformer_num_heads,
            mlp_dim=transformer_mlp_dim,
            activation=activation,
        )

        # Upsampling path with skip connections
        # Input: (B, transformer_dim, H/16, W/16) from memory attention
        # After 2× upsample: (B, transformer_dim//4, H/8, W/8) + stride-8 skip
        # After 2× upsample: (B, transformer_dim//8, H/4, W/4) + stride-4 skip
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # Fusion layers for skip connections
        self.skip_fusion_s8 = nn.Sequential(
            nn.Conv2d(transformer_dim // 4 + skip_dim_s8, transformer_dim // 4, kernel_size=1),
            LayerNorm2d(transformer_dim // 4),
            activation(),
        )
        self.skip_fusion_s4 = nn.Sequential(
            nn.Conv2d(transformer_dim // 8 + skip_dim_s4, transformer_dim // 8, kernel_size=1),
            LayerNorm2d(transformer_dim // 8),
            activation(),
        )

        # Per-mask output heads
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, num_layers=3)
            for _ in range(self.num_mask_tokens)
        ])

        # IoU prediction head
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens,
            num_layers=iou_head_depth, sigmoid_output=True
        )

        # Occlusion prediction head (object presence score)
        self.occlusion_head = MLP(transformer_dim, transformer_dim // 4, 1, num_layers=3)

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        skip_features: List[Tensor],
        multimask_output: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            image_embeddings:       (B, C, H, W) — conditioned frame embedding
            image_pe:               (B, C, H, W) — positional encoding
            sparse_prompt_embeddings: (B, N, C)
            dense_prompt_embeddings:  (B, C, H, W)
            skip_features:          [stride-4 feat, stride-8 feat]
            multimask_output:       if True, return all masks; else return single mask

        Returns:
            masks:       (B, num_masks, H_out, W_out)
            iou_pred:    (B, num_masks)
            occlusion:   (B, 1) — logit for object presence
            mask_token:  (B, transformer_dim) — object pointer
        """
        masks, iou_pred, occlusion, mask_token = self._predict_masks(
            image_embeddings, image_pe, sparse_prompt_embeddings,
            dense_prompt_embeddings, skip_features
        )

        if multimask_output:
            # Return all multi-mask outputs (indices 1..num_mask_tokens)
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        else:
            # Return single-mask output (index 0)
            masks = masks[:, :1, :, :]
            iou_pred = iou_pred[:, :1]

        return masks, iou_pred, occlusion, mask_token

    def _predict_masks(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        skip_features: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B = image_embeddings.shape[0]

        # Concatenate output tokens
        output_tokens = torch.cat([
            self.iou_token.weight,
            self.mask_tokens.weight,
            self.occlusion_token.weight,
        ], dim=0)  # (1 + num_mask_tokens + 1, C)
        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)

        # Add dense prompt to image embedding
        src = image_embeddings + dense_prompt_embeddings

        # Run transformer: tokens attend to image, image attends to tokens
        hs, src_out = self.transformer(src, image_pe, tokens)

        # Parse output tokens
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:1 + self.num_mask_tokens, :]
        occlusion_token_out = hs[:, 1 + self.num_mask_tokens, :]

        # Object pointer: use the first mask token (single-mask output token)
        mask_token = mask_tokens_out[:, 0, :]  # (B, transformer_dim)

        # Reshape updated image tokens back to spatial map
        H, W = image_embeddings.shape[2], image_embeddings.shape[3]
        src_spatial = src_out.transpose(1, 2).view(B, -1, H, W)

        # Upscale with skip connections
        upscaled = self._upscale_with_skips(src_spatial, skip_features)

        # Generate masks via hypernetwork MLPs
        hyper_in_list = [
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            for i in range(self.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)  # (B, num_mask_tokens, C//8)
        B_out, C_out, H_out, W_out = upscaled.shape
        masks = (hyper_in @ upscaled.view(B_out, C_out, H_out * W_out)).view(B_out, -1, H_out, W_out)

        # IoU prediction
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Occlusion prediction
        occlusion = self.occlusion_head(occlusion_token_out)  # (B, 1)

        return masks, iou_pred, occlusion, mask_token

    def _upscale_with_skips(self, src: Tensor, skip_features: List[Tensor]) -> Tensor:
        """
        Upsample src (B, C, H/16, W/16) to (B, C//8, H/4, W/4) using skip connections.
        skip_features[0]: stride-4 (B, C//4, H/4, W/4)
        skip_features[1]: stride-8 (B, C//2, H/8, W/8)
        """
        skip_s4, skip_s8 = skip_features

        # First upsample: H/16 → H/8
        x = self.upsample1(src)  # (B, C//4, H/8, W/8)

        # Fuse with stride-8 skip
        if skip_s8 is not None:
            x = torch.cat([x, skip_s8], dim=1)
            x = self.skip_fusion_s8(x)

        # Second upsample: H/8 → H/4
        x = self.upsample2(x)  # (B, C//8, H/4, W/4)

        # Fuse with stride-4 skip
        if skip_s4 is not None:
            x = torch.cat([x, skip_s4], dim=1)
            x = self.skip_fusion_s4(x)

        return x
