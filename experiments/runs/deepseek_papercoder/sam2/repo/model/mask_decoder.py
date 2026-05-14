"""
SAM 2 Mask Decoder.

Implements the prompt‑aware transformer decoder that predicts object masks,
IoU scores, and an occlusion flag from the conditioned image embedding
(after memory attention) and the user prompts.  It also produces the object
pointer token used by the memory bank.

The decoder closely follows the original SAM two‑way transformer design, with
additions for high‑resolution skip connections (from the Hiera image encoder)
and an occlusion prediction head as described in Sections 4 and D.1 of the
SAM 2 paper.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Two‑Way Attention Block
# ---------------------------------------------------------------------------

class TwoWayAttentionBlock(nn.Module):
    """
    A single block of the two‑way transformer used in the mask decoder.

    The block performs four attention operations:
      1. Self‑attention on the prompt tokens (including mask/IoU/occlusion tokens).
      2. Cross‑attention from prompt tokens to image tokens.
      3. Self‑attention on image tokens, with a 2D positional encoding.
      4. Cross‑attention from image tokens to prompt tokens.

    Each attention is followed by a residual connection and layer norm.  After
    the attention stages, two separate MLPs are applied to the prompt and image
    tokens, again with residuals.

    Args:
        embed_dim: Token embedding dimension (e.g., 256).
        num_heads: Number of attention heads (default 8).
        mlp_ratio: Hidden‑dimension multiplier for the MLPs (default 4.0).
        dropout: Dropout rate used in MLPs (default 0.0).
        attention_dropout: Dropout rate used in attention weights (default 0.0).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        mlp_dim = int(embed_dim * mlp_ratio)

        # Layer norms
        self.norm_p_self = nn.LayerNorm(embed_dim)
        self.norm_p_cross = nn.LayerNorm(embed_dim)
        self.norm_i_self = nn.LayerNorm(embed_dim)
        self.norm_i_cross = nn.LayerNorm(embed_dim)

        # Attention modules
        self.self_attn_prompt = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.cross_attn_prompt = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.self_attn_image = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.cross_attn_image = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True
        )

        # MLP blocks
        self.mlp_prompt = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.mlp_image = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_p_mlp = nn.LayerNorm(embed_dim)
        self.norm_i_mlp = nn.LayerNorm(embed_dim)

    def forward(
        self,
        queries: torch.Tensor,
        image_tokens: torch.Tensor,
        image_pe: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: Prompt tokens (B, N_q, C).
            image_tokens: Image tokens (B, H*W, C).
            image_pe: Positional encoding for image tokens (broadcastable to
                (B, H*W, C)).  If ``None``, no extra positional encoding is
                added during image self‑attention.

        Returns:
            Updated (queries, image_tokens) of the same shapes.
        """

        # ----- 1. Prompt self‑attention -----
        shortcut = queries
        queries = self.norm_p_self(queries)
        queries, _ = self.self_attn_prompt(queries, queries, queries)
        queries = shortcut + queries

        # ----- 2. Prompt cross‑attention to image -----
        shortcut = queries
        queries = self.norm_p_cross(queries)
        queries, _ = self.cross_attn_prompt(queries, image_tokens, image_tokens)
        queries = shortcut + queries

        # ----- 3. Image self‑attention (with positional encoding) -----
        shortcut = image_tokens
        image_tokens_norm = self.norm_i_self(image_tokens)
        if image_pe is not None:
            # Add PE to both query and key (original SAM convention)
            q = image_tokens_norm + image_pe
            k = image_tokens_norm + image_pe
        else:
            q = k = image_tokens_norm
        v = image_tokens_norm
        image_tokens, _ = self.self_attn_image(q, k, v)
        image_tokens = shortcut + image_tokens

        # ----- 4. Image cross‑attention to prompt -----
        shortcut = image_tokens
        image_tokens = self.norm_i_cross(image_tokens)
        image_tokens, _ = self.cross_attn_image(image_tokens, queries, queries)
        image_tokens = shortcut + image_tokens

        # ----- 5. Prompt MLP -----
        shortcut = queries
        queries = self.norm_p_mlp(queries)
        queries = self.mlp_prompt(queries)
        queries = shortcut + queries

        # ----- 6. Image MLP -----
        shortcut = image_tokens
        image_tokens = self.norm_i_mlp(image_tokens)
        image_tokens = self.mlp_image(image_tokens)
        image_tokens = shortcut + image_tokens

        return queries, image_tokens


# ---------------------------------------------------------------------------
#  Two‑Way Transformer
# ---------------------------------------------------------------------------

class TwoWayTransformer(nn.Module):
    """
    Stacks of :class:`TwoWayAttentionBlock` that jointly process prompt and
    image tokens.

    Args:
        depth: Number of blocks (default 2, as in SAM).
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden dimension factor for internal MLPs.
        dropout: General dropout rate.
        attention_dropout: Dropout for attention weights.
    """

    def __init__(
        self,
        depth: int = 2,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TwoWayAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        image_pe: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_tokens: (B, H*W, C) image features.
            prompt_tokens: (B, N_q, C) prompt tokens.
            image_pe: Optional positional encoding (B, H*W, C) or (1, …).

        Returns:
            Updated (prompt_tokens, image_tokens).
        """
        for layer in self.layers:
            prompt_tokens, image_tokens = layer(prompt_tokens, image_tokens, image_pe)
        return prompt_tokens, image_tokens


# ---------------------------------------------------------------------------
#  Mask Decoder
# ---------------------------------------------------------------------------

class MaskDecoder(nn.Module):
    """
    Prompt‑guided mask decoder for SAM 2.

    Produces segmentation masks, per‑mask IoU estimates, an occlusion score,
    and a per‑frame object pointer token (the mask token of the selected output).

    The internal upsampling pathway fuses high‑resolution skip features from
    the image encoder (stages 1 & 2) to recover fine details.

    Configuration dictionary (keys used):
        ``model.fpn_channels``      (int)  – channel dimension of image embeddings (default 256)
        ``model.mask_decoder.multi_mask``   (bool) – enable multi‑mask output (3 masks) at first prompt
        ``model.mask_decoder.occlusion_head`` (bool) – include occlusion prediction

    Additionally, the following can be supplied to override default skip channels
    (if not given, they are guessed from the first batch):
        ``skip_stage2_channels``    (int)  – skip from stride 8 (Hiera stage 2)
        ``skip_stage1_channels``    (int)  – skip from stride 4 (Hiera stage 1)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        if config is None:
            config = {}

        # Extract configuration values with defaults
        model_cfg = config.get("model", {})
        decoder_cfg = model_cfg.get("mask_decoder", {})
        self.fpn_channels = model_cfg.get("fpn_channels", 256)
        self.multi_mask = decoder_cfg.get("multi_mask", True)
        self.occlusion_head = decoder_cfg.get("occlusion_head", True)
        self.num_multimask = 3  # fixed

        # ------------------------------------------------------------------
        #  Learnable tokens
        # ------------------------------------------------------------------
        # Single mask token (used when ambiguity is resolved)
        self.mask_tokens_single = nn.Parameter(
            torch.randn(1, self.fpn_channels, dtype=torch.float32) * 0.02
        )
        # Three mask tokens for ambiguous first prompts
        self.mask_tokens_multi = nn.Parameter(
            torch.randn(self.num_multimask, self.fpn_channels, dtype=torch.float32) * 0.02
        )
        # IoU token that helps predicting mask quality
        self.iou_token = nn.Parameter(
            torch.randn(1, self.fpn_channels, dtype=torch.float32) * 0.02
        )
        # Occlusion token (optional)
        if self.occlusion_head:
            self.occlusion_token = nn.Parameter(
                torch.randn(1, self.fpn_channels, dtype=torch.float32) * 0.02
            )
        else:
            self.occlusion_token = None

        # ------------------------------------------------------------------
        #  Image positional encoding (learnable, for the 64 × 64 grid)
        # ------------------------------------------------------------------
        self.image_pe = nn.Parameter(
            torch.randn(1, 64 * 64, self.fpn_channels, dtype=torch.float32) * 0.02
        )

        # ------------------------------------------------------------------
        #  Two‑way transformer
        # ------------------------------------------------------------------
        self.transformer = TwoWayTransformer(
            depth=2,
            embed_dim=self.fpn_channels,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.0,
            attention_dropout=0.1,
        )

        # ------------------------------------------------------------------
        #  Skip projections (initialised lazily in `_init_skip_proj`)
        # ------------------------------------------------------------------
        self.skip_proj_8to256: Optional[nn.Conv2d] = None
        self.skip_proj_4to256: Optional[nn.Conv2d] = None

        # Store channels given by config, otherwise we infer on first forward
        self.skip_stage2_channels: Optional[int] = config.get("skip_stage2_channels")
        self.skip_stage1_channels: Optional[int] = config.get("skip_stage1_channels")

        # ------------------------------------------------------------------
        #  Upsampling pathway
        # ------------------------------------------------------------------
        # 64 → 128
        self.upsample_64_to_128 = nn.Sequential(
            nn.ConvTranspose2d(
                self.fpn_channels, self.fpn_channels, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True),
        )

        # 128 → 256
        self.upsample_128_to_256 = nn.Sequential(
            nn.ConvTranspose2d(
                self.fpn_channels, self.fpn_channels, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True),
        )

        # 256 → 1024 (final resolution)
        self.upsample_256_to_1024 = nn.Sequential(
            nn.ConvTranspose2d(
                self.fpn_channels, self.fpn_channels, kernel_size=4, stride=4
            ),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        #  Prediction heads
        # ------------------------------------------------------------------
        # Mask logit head: project token to fpn_channels, then pointwise multiply
        # with upsampled features, and sum over channels.
        self.mask_token_proj = nn.Linear(self.fpn_channels, self.fpn_channels)

        # IoU head: MLP that takes concatenated [iou_token_output, mask_token_output]
        self.iou_head = nn.Sequential(
            nn.Linear(self.fpn_channels * 2, self.fpn_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fpn_channels, 1),
        )

        # Occlusion head (if enabled)
        if self.occlusion_head:
            self.occlusion_mlp = nn.Sequential(
                nn.Linear(self.fpn_channels, self.fpn_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.fpn_channels, 1),
            )
        else:
            self.occlusion_mlp = None

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Basic initialization matching the paper's setup."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def _init_skip_proj(self, skip_stage2: torch.Tensor, skip_stage1: torch.Tensor) -> None:
        """
        Lazily build the 1 × 1 convolutions that align skip feature channels to
        ``self.fpn_channels``.  This is called once during the first forward pass
        if the channels were not explicitly provided in the configuration.
        """
        if self.skip_proj_8to256 is None:
            in_ch2 = skip_stage2.shape[1]
            self.skip_stage2_channels = in_ch2
            self.skip_proj_8to256 = nn.Conv2d(in_ch2, self.fpn_channels, kernel_size=1)
            # Apply initialization
            self.skip_proj_8to256.apply(self._init_weights)
            # Move to the same device as the input
            self.skip_proj_8to256 = self.skip_proj_8to256.to(skip_stage2.device)

        if self.skip_proj_4to256 is None:
            in_ch1 = skip_stage1.shape[1]
            self.skip_stage1_channels = in_ch1
            self.skip_proj_4to256 = nn.Conv2d(in_ch1, self.fpn_channels, kernel_size=1)
            self.skip_proj_4to256.apply(self._init_weights)
            self.skip_proj_4to256 = self.skip_proj_4to256.to(skip_stage1.device)

    def forward(
        self,
        image_embed: torch.Tensor,
        prompt_embed: Dict[str, torch.Tensor],
        skip_features: List[torch.Tensor],
        multi_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Produce mask predictions and auxiliary outputs.

        Args:
            image_embed: Conditioned image feature map ``(B, C, 64, 64)``,
                typically from the memory attention module.
            prompt_embed: Dictionary with keys:
                - ``'sparse'`` (optional): ``(B, N_sparse, C)`` sparse prompt tokens.
                - ``'dense'`` (optional): ``(B, C, 64, 64)`` dense mask bias.
            skip_features: List of two high‑resolution tensors from the image encoder:
                - ``skip_stride_8``: ``(B, C_skip8, 128, 128)``
                - ``skip_stride_4``: ``(B, C_skip4, 256, 256)``
            multi_mask: If ``True``, predict 3 candidate masks to handle ambiguity.
                Otherwise predict a single mask.

        Returns:
            Dictionary with keys:
                - ``'masks'``: mask logits ``(B, num_masks, 1024, 1024)``
                - ``'iou_pred'``: IoU logits ``(B, num_masks)``
                - ``'occlusion_logit'``: scalar occlusion logit ``(B, 1)``
                - ``'object_pointer_token'``: ``(B, C)`` the mask token chosen
                  for the memory bank.
        """
        B, C_feat, H_feat, W_feat = image_embed.shape
        assert C_feat == self.fpn_channels, \
            f"Expected image_embed channels {self.fpn_channels}, got {C_feat}"
        assert (H_feat, W_feat) == (64, 64), \
            f"Expected 64×64 image embedding, got {H_feat}×{W_feat}"

        # ---- 1. Add dense prompt (mask bias) ----
        dense_prompt = prompt_embed.get("dense", None)
        if dense_prompt is not None:
            image_embed = image_embed + dense_prompt

        # ---- 2. Flatten & add image positional encoding ----
        # image_embed: (B, C, 64, 64) -> (B, C, 64*64) -> (B, 64*64, C)
        image_tokens = image_embed.flatten(2).transpose(1, 2)  # (B, 4096, C)
        # broadcast image_pe over batch
        image_tokens = image_tokens + self.image_pe

        # ---- 3. Choose mask tokens ----
        if multi_mask:
            mask_tokens = self.mask_tokens_multi.unsqueeze(0).expand(B, -1, -1)  # (B, 3, C)
        else:
            mask_tokens = self.mask_tokens_single.unsqueeze(0).expand(B, -1, -1)  # (B, 1, C)
        num_masks = mask_tokens.shape[1]

        # ---- 4. Prepare prompt tokens ----
        # Sparse tokens
        sparse_prompt = prompt_embed.get("sparse", None)
        if sparse_prompt is not None and sparse_prompt.shape[1] > 0:
            prompt_parts = [sparse_prompt]
        else:
            prompt_parts = []

        # Mask tokens
        prompt_parts.append(mask_tokens)

        # IoU token
        iou_token_expanded = self.iou_token.unsqueeze(0).expand(B, -1, -1)
        prompt_parts.append(iou_token_expanded)

        # Occlusion token
        if self.occlusion_token is not None:
            occlusion_token_expanded = self.occlusion_token.unsqueeze(0).expand(B, -1, -1)
            prompt_parts.append(occlusion_token_expanded)
            has_occlusion = True
        else:
            has_occlusion = False

        prompt_tokens = torch.cat(prompt_parts, dim=1)  # (B, total_tokens, C)

        # ---- 5. Two‑way transformer ----
        prompt_tokens, image_tokens = self.transformer(
            image_tokens, prompt_tokens, image_pe=self.image_pe
        )

        # ---- 6. Split updated prompt tokens ----
        # Determine splitting indices
        idx_mask_end = (
            sparse_prompt.shape[1] if sparse_prompt is not None else 0
        ) + num_masks
        updated_mask_tokens = prompt_tokens[:, :idx_mask_end, :]  # includes sparse!
        # We'll slice pure mask tokens after the sparse part
        sparse_len = sparse_prompt.shape[1] if sparse_prompt is not None else 0
        mask_tokens_updated = prompt_tokens[:, sparse_len:sparse_len + num_masks, :]
        iou_token_updated = prompt_tokens[:, sparse_len + num_masks, :]  # (B, C)
        if has_occlusion:
            occlusion_token_updated = prompt_tokens[:, sparse_len + num_masks + 1, :]  # (B, C)
        else:
            occlusion_token_updated = None

        # ---- 7. Upsample image tokens ----
        # Reshape to (B, C, 64, 64)
        feat = image_tokens.transpose(1, 2).view(B, self.fpn_channels, 64, 64)

        # Unpack skip features
        skip_8 = skip_features[0]  # stride 8, 128×128
        skip_4 = skip_features[1]  # stride 4, 256×256

        # Lazy init of skip projections (first call)
        self._init_skip_proj(skip_8, skip_4)

        # ---- Stage 1: 64 → 128 + skip from stride 8 ----
        feat = self.upsample_64_to_128(feat)  # (B, C, 128, 128)
        skip_8_proj = self.skip_proj_8to256(skip_8)  # align channels
        feat = feat + skip_8_proj
        # (Already includes BN+ReLU+Conv inside upsample_64_to_128)

        # ---- Stage 2: 128 → 256 + skip from stride 4 ----
        feat = self.upsample_128_to_256(feat)  # (B, C, 256, 256)
        skip_4_proj = self.skip_proj_4to256(skip_4)
        feat = feat + skip_4_proj

        # ---- Stage 3: 256 → 1024 ----
        final_features = self.upsample_256_to_1024(feat)  # (B, C, 1024, 1024)

        # ---- 8. Generate masks from mask tokens ----
        # Project mask tokens to channel dimension
        mask_tokens_proj = self.mask_token_proj(mask_tokens_updated)  # (B, num_masks, C)
        mask_tokens_proj = mask_tokens_proj[:, :, :, None, None]  # (B, num_masks, C, 1, 1)

        # final_features: (B, C, 1024, 1024)
        # Add spatial dimension corresponding to masks:
        # Expand final_features to (B, 1, C, 1024, 1024) then multiply
        # or use einsum
        masks = torch.sum(
            final_features.unsqueeze(1) * mask_tokens_proj, dim=2
        )  # (B, num_masks, 1024, 1024)

        # ---- 9. IoU prediction ----
        # Concatenate iou_token_updated with each mask token
        iou_token_broad = iou_token_updated.unsqueeze(1).expand(-1, num_masks, -1)  # (B, num_masks, C)
        iou_input = torch.cat([iou_token_broad, mask_tokens_updated], dim=-1)  # (B, num_masks, 2*C)
        iou_logits = self.iou_head(iou_input).squeeze(-1)  # (B, num_masks)

        # ---- 10. Occlusion prediction ----
        if has_occlusion and self.occlusion_mlp is not None:
            occlusion_logit = self.occlusion_mlp(occlusion_token_updated)  # (B, 1)
        else:
            occlusion_logit = torch.zeros((B, 1), device=image_embed.device, dtype=image_embed.dtype)

        # ---- 11. Object pointer token (selected mask token) ----
        if multi_mask and num_masks > 1:
            # Choose the mask with the highest predicted IoU
            best_idx = torch.argmax(iou_logits, dim=1)  # (B,)
            object_pointer_token = mask_tokens_updated[
                torch.arange(B, device=best_idx.device), best_idx
            ]
        else:
            object_pointer_token = mask_tokens_updated[:, 0, :]  # (B, C)

        return {
            "masks": masks,
            "iou_pred": iou_logits,
            "occlusion_logit": occlusion_logit,
            "object_pointer_token": object_pointer_token,
        }


# ---------------------------------------------------------------------------
# Quick shape check (when executed as a script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple smoke test with random inputs and a dummy config.
    config = {
        "model": {
            "fpn_channels": 256,
            "mask_decoder": {
                "multi_mask": True,
                "occlusion_head": True,
            },
        }
    }
    decoder = MaskDecoder(config)
    B = 2
    image_embed = torch.randn(B, 256, 64, 64)
    # prompt_embed
    sparse = torch.randn(B, 2, 256)  # two clicks
    dense = None  # no dense mask prompt
    prompt_embed = {"sparse": sparse, "dense": dense}
    # skip features from Hiera (simulated)
    skip_stride8 = torch.randn(B, 256, 128, 128)  # Hiera stage2 out
    skip_stride4 = torch.randn(B, 128, 256, 256)  # Hiera stage1 out
    skip_features = [skip_stride8, skip_stride4]

    out = decoder(image_embed, prompt_embed, skip_features, multi_mask=True)
    print("masks shape:", out["masks"].shape)         # (2, 3, 1024, 1024)
    print("iou_pred shape:", out["iou_pred"].shape)   # (2, 3)
    print("occlusion_logit shape:", out["occlusion_logit"].shape)  # (2, 1)
    print("object_pointer_token shape:", out["object_pointer_token"].shape)  # (2, 256)

    # Test with single mask
    out_single = decoder(image_embed, prompt_embed, skip_features, multi_mask=False)
    print("Single mask shape:", out_single["masks"].shape)  # (2, 1, 1024, 1024)

