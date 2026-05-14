## Code: models/image_encoder.py

```python
## models/image_encoder.py
"""Image encoder for SAM 2: Hiera backbone with Feature Pyramid Network.

This module implements the visual backbone of SAM 2. It runs once per frame
in the streaming inference loop and produces:

1. frame_embedding — FPN-fused stride-16 features for memory attention
2. skip_features   — stride-4 and stride-8 features for mask decoder upsampling

Architecture (Appendix D.1):
    - MAE pre-trained Hiera backbone (hierarchical ViT, 4 stages)
    - FPN fusing Stage 3 (stride-16) and Stage 4 (stride-32) → frame embedding
    - Stage 1 (stride-4) and Stage 2 (stride-8) passed as skip connections
    - No RPB (relative positional biases) — enables FlashAttention-2
    - Windowed absolute positional embeddings with global PE interpolation
    - Global attention at specified block indices per encoder variant

Config references:
    model.image_encoder_type: "hiera_b_plus"
    model.fpn_out_channels: 256
    model.global_attn_blocks: {hiera_t: [5,7,9], hiera_s: [7,10,13], ...}
    model.encoder_drop_path_rates: {hiera_t: 0.1, hiera_b_plus: 0.2, ...}
    model.use_rpb: false
    model.use_flash_attention: true
    model.input_resolution: 1024

Paper references:
    Section 4: "We use an MAE pre-trained Hiera image encoder"
    Appendix D.1: "we use a feature pyramid network (Lin et al., 2017) to fuse
        the stride 16 and 32 features from Stages 3 and 4"
    Appendix D.1: "the stride 4 and 8 features from Stages 1 and 2 are not
        used in the memory attention but are added to the upsampling layers
        in the mask decoder"
    Appendix D.1: "removing all RPB from the image encoder ... giving a
        significant speed boost at 1024 resolution"
    Table 10: default configuration uses no RPB, no 2D-RoPE in image encoder
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hiera channel dimensions per encoder variant
# Source: Hiera paper (Ryali et al., 2023) and timm implementation
# ---------------------------------------------------------------------------

# Maps encoder type → (stage1_dim, stage2_dim, stage3_dim, stage4_dim)
HIERA_CHANNEL_DIMS: Dict[str, Tuple[int, int, int, int]] = {
    "hiera_t":      (96,  192,  384,  768),
    "hiera_s":      (96,  192,  384,  768),
    "hiera_b_plus": (112, 224,  448,  896),
    "hiera_l":      (144, 288,  576, 1152),
}

# Maps encoder type → timm model name
HIERA_TIMM_NAMES: Dict[str, str] = {
    "hiera_t":      "hiera_tiny_224",
    "hiera_s":      "hiera_small_224",
    "hiera_b_plus": "hiera_base_plus_224",
    "hiera_l":      "hiera_large_224",
}

# Maps encoder type → total number of transformer blocks
HIERA_TOTAL_BLOCKS: Dict[str, int] = {
    "hiera_t":      10,
    "hiera_s":      14,
    "hiera_b_plus": 21,
    "hiera_l":      44,
}

# Maps encoder type → blocks per stage (for stage boundary detection)
# Hiera stages: [stage1_blocks, stage2_blocks, stage3_blocks, stage4_blocks]
HIERA_BLOCKS_PER_STAGE: Dict[str, List[int]] = {
    "hiera_t":      [1, 2, 7,  2],   # total=12 (with 2 extra for q_pool)
    "hiera_s":      [1, 2, 11, 2],   # total=16
    "hiera_b_plus": [2, 3, 16, 3],   # total=24
    "hiera_l":      [2, 6, 36, 4],   # total=48
}

# Drop path rates per encoder variant (Table 12, config.yaml)
HIERA_DROP_PATH_RATES: Dict[str, float] = {
    "hiera_t":      0.1,
    "hiera_s":      0.1,
    "hiera_b_plus": 0.2,
    "hiera_l":      0.3,
}

# Global attention block indices per encoder variant (config.yaml, Appendix D.1)
HIERA_GLOBAL_ATTN_BLOCKS: Dict[str, List[int]] = {
    "hiera_t":      [5, 7, 9],
    "hiera_s":      [7, 10, 13],
    "hiera_b_plus": [12, 16, 20],
    "hiera_l":      [23, 33, 43],
}


# ---------------------------------------------------------------------------
# FeaturePyramidNetwork
# ---------------------------------------------------------------------------


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network fusing stride-16 and stride-32 Hiera features.

    Implements the standard FPN top-down pathway (Lin et al., 2017) to merge
    Stage 3 (stride-16) and Stage 4 (stride-32) outputs from the Hiera
    backbone into a single frame embedding at stride-16 resolution.

    From Appendix D.1: "we use a feature pyramid network (Lin et al., 2017)
    to fuse the stride 16 and 32 features from Stages 3 and 4 of the Hiera
    image encoder respectively to produce the image embeddings for each frame."

    Architecture:
        lateral_convs[0]: 1×1 conv, stage3_channels → out_channels (stride-16)
        lateral_convs[1]: 1×1 conv, stage4_channels → out_channels (stride-32)
        output_convs[0]:  3×3 conv, out_channels → out_channels (stride-16, final)

    Top-down pathway:
        1. Project stage4 via lateral_convs[1] → p5 (stride-32)
        2. Upsample p5 to stride-16 spatial size
        3. Project stage3 via lateral_convs[0] → p4 (stride-16)
        4. Add: p4 + upsample(p5) → merged (stride-16)
        5. Apply output_convs[0] → frame_embedding (stride-16)

    Args:
        in_channels: List of two integers [stage3_channels, stage4_channels].
            Stage 3 is stride-16, Stage 4 is stride-32.
        out_channels: Output channel dimension. Defaults to 256 per config.yaml
            (model.fpn_out_channels: 256).

    Example:
        fpn = FeaturePyramidNetwork(in_channels=[448, 896], out_channels=256)
        stage3_feat = torch.randn(2, 448, 64, 64)   # stride-16
        stage4_feat = torch.randn(2, 896, 32, 32)   # stride-32
        frame_embed = fpn([stage3_feat, stage4_feat])  # (2, 256, 64, 64)
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
    ) -> None:
        super().__init__()

        if len(in_channels) != 2:
            raise ValueError(
                f"FeaturePyramidNetwork expects exactly 2 input channel dims "
                f"(stage3, stage4), got {len(in_channels)}."
            )

        self.in_channels: List[int] = in_channels
        self.out_channels: int = out_channels

        # Lateral 1×1 convolutions: project each stage to out_channels
        # lateral_convs[0]: stage3 (stride-16) → out_channels
        # lateral_convs[1]: stage4 (stride-32) → out_channels
        self.lateral_convs: nn.ModuleList = nn.ModuleList([
            nn.Conv2d(
                in_ch,
                out_channels,
                kernel_size=1,
                bias=False,
            )
            for in_ch in in_channels
        ])

        # Output 3×3 convolution applied after top-down merging at stride-16
        # Only one output conv needed (we only output the stride-16 level)
        self.output_convs: nn.ModuleList = nn.ModuleList([
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        ])

        # Batch normalization after output conv for training stability
        self.output_norms: nn.ModuleList = nn.ModuleList([
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize FPN weights with Kaiming normal for convolutions."""
        for conv in self.lateral_convs:
            nn.init.kaiming_uniform_(conv.weight, a=1)

        for conv in self.output_convs:
            nn.init.kaiming_uniform_(conv.weight, a=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse stage3 and stage4 features via FPN top-down pathway.

        Args:
            features: List of two tensors:
                - features[0]: Stage 3 output, shape [B, C3, H/16, W/16]
                - features[1]: Stage 4 output, shape [B, C4, H/32, W/32]

        Returns:
            Frame embedding tensor of shape [B, out_channels, H/16, W/16].
            This is the primary input to the memory attention module.

        Raises:
            ValueError: If features does not contain exactly 2 tensors.
        """
        if len(features) != 2:
            raise ValueError(
                f"FeaturePyramidNetwork.forward expects exactly 2 feature maps "
                f"(stage3, stage4), got {len(features)}."
            )

        stage3_feat = features[0]  # [B, C3, H/16, W/16]
        stage4_feat = features[1]  # [B, C4, H/32, W/32]

        # Step 1: Lateral projections
        # p4: [B, out_channels, H/16, W/16]
        # p5: [B, out_channels, H/32, W/32]
        p4 = self.lateral_convs[0](stage3_feat)
        p5 = self.lateral_convs[1](stage4_feat)

        # Step 2: Top-down pathway — upsample p5 to p4's spatial size
        # Use bilinear interpolation for smooth upsampling
        p5_upsampled = F.interpolate(
            p5,
            size=p4.shape[-2:],  # (H/16, W/16)
            mode="bilinear",
            align_corners=False,
        )

        # Step 3: Merge — element-wise addition
        merged = p4 + p5_upsampled  # [B, out_channels, H/16, W/16]

        # Step 4: Output conv + norm → final frame embedding
        frame_embedding = self.output_convs[0](merged)
        frame_embedding = self.output_norms[0](frame_embedding)

        return frame_embedding  # [B, out_channels, H/16, W/16]


# ---------------------------------------------------------------------------
# Hiera building blocks (minimal implementation for SAM 2)
# ---------------------------------------------------------------------------


class WindowedAttention(nn.Module):
    """Multi-head self-attention with optional windowing and global attention.

    Implements the attention mechanism used in Hiera blocks. Supports:
    - Local windowed attention (default for most blocks)
    - Global attention (for blocks in global_attn_blocks list)
    - No RPB (relative positional biases removed per config.yaml: use_rpb: false)
    - FlashAttention-2 via torch.nn.functional.scaled_dot_product_attention

    From Appendix D.1: "We follow Bolya et al. (2023) in using windowed
    absolute positional embeddings in the Hiera image encoder."
    "We do not use any relative positional encoding."

    Args:
        dim: Input and output feature dimension.
        num_heads: Number of attention heads.
        window_size: Spatial window size for local attention. Set to 0 for
            global attention (no windowing).
        use_flash_attention: If True, use scaled_dot_product_attention for
            FlashAttention-2 dispatch. Requires no RPB.
        qkv_bias: If True, add bias to QKV projections. Defaults to True.
        proj_drop: Dropout rate on output projection. Defaults to 0.0.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        use_flash_attention: bool = True,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.dim: int = dim
        self.num_heads: int = num_heads
        self.head_dim: int = dim // num_heads
        self.scale: float = self.head_dim ** -0.5
        self.window_size: int = window_size
        self.use_flash_attention: bool = use_flash_attention

        self.qkv: nn.Linear = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj: nn.Linear = nn.Linear(dim, dim)
        self.proj_drop: nn.Dropout = nn.Dropout(proj_drop)

    def _window_partition(
        self,
        x: torch.Tensor,
        window_size: int,
    ) -> Tuple[torch.Tensor, int, int]:
        """Partition spatial feature map into non-overlapping windows.

        Args:
            x: Feature map of shape [B, H, W, C].
            window_size: Window size (square).

        Returns:
            Tuple of:
                - windows: [B*num_windows, window_size*window_size, C]
                - H_pad: Padded height (multiple of window_size)
                - W_pad: Padded width (multiple of window_size)
        """
        B, H, W, C = x.shape

        # Pad to multiple of window_size
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        H_pad, W_pad = H + pad_h, W + pad_w

        # Reshape into windows: [B, H/ws, ws, W/ws, ws, C]
        x = x.view(
            B,
            H_pad // window_size, window_size,
            W_pad // window_size, window_size,
            C,
        )
        # Permute and reshape: [B * num_windows, ws*ws, C]
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size * window_size, C)

        return windows, H_pad, W_pad

    def _window_unpartition(
        self,
        windows: torch.Tensor,
        window_size: int,
        H_pad: int,
        W_pad: int,
        H_orig: int,
        W_orig: int,
    ) -> torch.Tensor:
        """Reverse window partitioning.

        Args:
            windows: [B*num_windows, window_size*window_size, C]
            window_size: Window size.
            H_pad: Padded height.
            W_pad: Padded width.
            H_orig: Original (unpadded) height.
            W_orig: Original (unpadded) width.

        Returns:
            Feature map of shape [B, H_orig, W_orig, C].
        """
        B = windows.shape[0] // ((H_pad // window_size) * (W_pad // window_size))
        C = windows.shape[-1]

        # Reshape: [B, H_pad/ws, W_pad/ws, ws, ws, C]
        x = windows.view(
            B,
            H_pad // window_size,
            W_pad // window_size,
            window_size,
            window_size,
            C,
        )
        # Permute back: [B, H_pad, W_pad, C]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H_pad, W_pad, C)

        # Remove padding
        if H_pad > H_orig or W_pad > W_orig:
            x = x[:, :H_orig, :W_orig, :].contiguous()

        return x

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Apply windowed or global self-attention.

        Args:
            x: Input tensor of shape [B, H*W, C] (flattened spatial tokens).
            H: Spatial height.
            W: Spatial width.

        Returns:
            Output tensor of shape [B, H*W, C].
        """
        B, L, C = x.shape
        assert L == H * W, f"Expected L={H*W}, got {L}"

        # Reshape to spatial: [B, H, W, C]
        x_spatial = x.view(B, H, W, C)

        use_windows = (self.window_size > 0 and
                       self.window_size < H and
                       self.window_size < W)

        if use_windows:
            # Partition into windows
            x_win, H_pad, W_pad = self._window_partition(x_spatial, self.window_size)
            # x_win: [B*num_windows, ws*ws, C]
            attn_input = x_win
        else:
            # Global attention: flatten all spatial tokens
            attn_input = x_spatial.view(B, H * W, C)
            H_pad, W_pad = H, W

        # QKV projection
        qkv = self.qkv(attn_input)  # [..., L_win, 3*C]
        qkv = qkv.reshape(
            attn_input.shape[0],
            attn_input.shape[1],
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B_win, num_heads, L_win, head_dim]
        q, k, v = qkv.unbind(0)  # each: [B_win, num_heads, L_win, head_dim]

        # Attention computation
        if self.use_flash_attention:
            # PyTorch 2.x scaled_dot_product_attention dispatches to FlashAttn-2
            # No RPB → compatible with flash attention
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,
                is_causal=False,
            )  # [B_win, num_heads, L_win, head_dim]
        else:
            # Standard attention
            attn_weights = (q @ k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_out = attn_weights @ v  # [B_win, num_heads, L_win, head_dim]

        # Reshape: [B_win, L_win, C]
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(attn_input.shape[0], attn_input.shape[1], C)

        # Output projection
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)

        if use_windows:
            # Unpartition windows back to spatial map
            attn_spatial = self._window_unpartition(
                attn_out, self.window_size, H_pad, W_pad, H, W
            )  # [B, H, W, C]
        else:
            attn_spatial = attn_out.view(B, H, W, C)

        # Flatten back to sequence: [B, H*W, C]
        return attn_spatial.view(B, H * W, C)


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization for residual blocks.

    From config.yaml: encoder_drop_path_rates per encoder variant.
    Randomly drops entire residual paths during training.

    Args:
        drop_prob: Probability of dropping a path. 0.0 = no drop.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob: float = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with same shape as input. During eval, returns x unchanged.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        # Create random tensor with shape [B, 1, 1, ...] for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class HieraMLP(nn.Module):
    """MLP block used in Hiera transformer blocks.

    Standard two-layer MLP with GELU activation and optional dropout.

    Args:
        in_features: Input feature dimension.
        hidden_features: Hidden layer dimension. Defaults to 4 * in_features.
        out_features: Output feature dimension. Defaults to in_features.
        drop: Dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features

        self.fc1: nn.Linear = nn.Linear(in_features, hidden_features)
        self.act: nn.GELU = nn.GELU()
        self.fc2: nn.Linear = nn.Linear(hidden_features, out_features)
        self.drop: nn.Dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two-layer MLP.

        Args:
            x: Input tensor of shape [..., in_features].

        Returns:
            Output tensor of shape [..., out_features].
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HieraBlock(nn.Module):
    """Single Hiera transformer block.

    Implements the standard ViT block pattern:
        x = x + drop_path(attn(norm1(x)))
        x = x + drop_path(mlp(norm2(x)))

    Supports both windowed (local) and global attention modes.
    No RPB — uses only absolute positional embeddings (config: use_rpb: false).

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        window_size: Window size for local attention. 0 = global attention.
        mlp_ratio: MLP hidden dim ratio. Defaults to 4.0.
        drop_path: Stochastic depth rate. Defaults to 0.0.
        use_flash_attention: Use FlashAttention-2. Defaults to True.
        norm_layer: Normalization layer class. Defaults to nn.LayerNorm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        use_flash_attention: bool = True,
        norm_layer: type = nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.dim: int = dim
        self.window_size: int = window_size

        self.norm1: nn.Module = norm_layer(dim)
        self.attn: WindowedAttention = WindowedAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            use_flash_attention=use_flash_attention,
        )
        self.norm2: nn.Module = norm_layer(dim)
        self.mlp: HieraMLP = HieraMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
        )
        self.drop_path: DropPath = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Apply Hiera block.

        Args:
            x: Input tensor of shape [B, H*W, C].
            H: Spatial height.
            W: Spatial width.

        Returns:
            Output tensor of shape [B, H*W, C].
        """
        # Self-attention with residual
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HieraPatchEmbed(nn.Module):
    """Patch embedding for Hiera: conv-based tokenization.

    Converts input image to patch tokens using a strided convolution.
    Produces stride-4 tokens (patch_size=4, stride=4).

    Args:
        in_channels: Input image channels. Defaults to 3 (RGB).
        embed_dim: Output embedding dimension (Stage 1 channels).
        patch_size: Patch size. Defaults to 4 (stride-4 output).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        patch_size: int = 4,
    ) -> None:
        super().__init__()

        self.patch_size: int = patch_size
        self.embed_dim: int = embed_dim

        self.proj: nn.Conv2d = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Embed input image into patch tokens.

        Args:
            x: Input image of shape [B, 3, H, W].

        Returns:
            Tuple of:
                - tokens: [B, H/patch_size * W/patch_size, embed_dim]
                - H_out: Output height (H / patch_size)
                - W_out: Output width (W / patch_size)
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H/ps, W/ps]
        H_out, W_out = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, H_out*W_out, embed_dim]
        x = self.norm(x)
        return x, H_out, W_out


class HieraDownsample(nn.Module):
    """Spatial downsampling between Hiera stages.

    Reduces spatial resolution by 2× and doubles channel dimension using
    a strided convolution (q_pool in the original Hiera implementation).

    Args:
        in_dim: Input channel dimension.
        out_dim: Output channel dimension (typically 2 * in_dim).
        stride: Downsampling stride. Defaults to 2.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int = 2,
    ) -> None:
        super().__init__()

        self.stride: int = stride
        self.norm: nn.LayerNorm = nn.LayerNorm(in_dim)
        self.proj: nn.Linear = nn.Linear(in_dim, out_dim, bias=False)
        self.pool: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=stride,
            stride=stride,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, int, int]:
        """Apply spatial downsampling.

        Args:
            x: Input tokens of shape [B, H*W, C].
            H: Current spatial height.
            W: Current spatial width.

        Returns:
            Tuple of:
                - downsampled: [B, (H/stride)*(W/stride), out_dim]
                - H_new: New height (H // stride)
                - W_new: New width (W // stride)
        """
        B, L, C = x.shape
        assert L == H * W

        x = self.norm(x)

        # Reshape to spatial for pooling: [B, C, H, W]
        x_spatial = x.view(B, H, W, C).permute(0, 3, 1, 2)

        # Pool: [B, C, H/stride, W/stride]
        x_pooled = self.pool(x_spatial)
        H_new = x_pooled.shape[2]
        W_new = x_pooled.shape[3]

        # Flatten and project: [B, H_new*W_new, out_dim]
        x_flat = x_pooled.flatten