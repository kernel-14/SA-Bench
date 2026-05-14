## Code: models/unet.py

```python
## models/unet.py
"""U-Net denoising backbone for WDNO (Wavelet Diffusion Neural Operator).

This module implements the U-Net architecture used as the noise prediction
network (eps_theta) in the DDPM/DDIM diffusion model. It supports two modes:
    - mode='1d': 2D convolutions on (B, C, T, X) wavelet coefficient tensors
      from 1D PDE experiments (Burgers', advection, compressible NS).
    - mode='2d': 3D convolutions on (B, C, T, H, W) wavelet coefficient tensors
      from 2D PDE experiments (fluid_2d, era5).

Architecture follows DDPM U-Net (Ho et al. 2020) with:
    - Sinusoidal time embeddings projected through a 2-layer MLP
    - ResNet blocks with GroupNorm and time embedding injection
    - Multi-head self-attention at each resolution level
    - Skip connections between encoder and decoder
    - Classifier-free guidance via channel-wise conditioning concatenation

Paper sources:
    - Architecture: Appendix F.4 (1D), Appendix H.3 (2D)
    - Hyperparameters: Table 18 (1D Burgers'), Table 19 (compressible NS),
      Table 20 (2D fluid)
    - Two-UNet design: Appendix F.4 (eps_phi for force, eps_theta for state)

Config references:
    - unet_1d.init_dim: 128
    - unet_1d.dim_mults: [1, 2, 4, 8]
    - unet_1d.resnet_block_groups: 8
    - unet_1d.attn_heads: 4
    - unet_1d.attn_hidden_dim: 32
    - unet_1d.n_downsample_layers: 4
    - unet_3d.conv3d_kernel: [3, 3, 3]
    - unet_3d.downsample_kernel: [1, 4, 4]
    - unet_3d.downsample_stride: [1, 2, 2]
    - unet_3d.upsample_kernel: [1, 4, 4]
    - unet_3d.upsample_stride: [1, 2, 2]
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute sinusoidal position embeddings for diffusion timesteps.

    Converts integer diffusion timesteps into continuous embedding vectors
    using the sinusoidal encoding from "Attention Is All You Need" (Vaswani
    et al. 2017), adapted for diffusion timesteps.

    Formula for position i in [0, dim/2):
        embedding[2i]   = sin(t / 10000^(2i / dim))
        embedding[2i+1] = cos(t / 10000^(2i / dim))

    Args:
        t: Diffusion timestep tensor of shape (batch,). Values in [1, K]
            where K=1000 (config: diffusion.num_timesteps). Can be float
            or long dtype.
        dim: Embedding dimension. Equals init_dim=128 from config
            (unet_1d.init_dim). Must be even.

    Returns:
        Sinusoidal embedding tensor of shape (batch, dim). dtype=float32.

    Raises:
        ValueError: If dim is odd (sinusoidal embedding requires even dim).
    """
    if dim % 2 != 0:
        raise ValueError(
            f"Sinusoidal embedding dimension must be even, got dim={dim}. "
            "Set unet_1d.init_dim to an even number (default: 128)."
        )

    device: torch.device = t.device
    half_dim: int = dim // 2

    # Frequency array: 10000^(-2i/dim) for i in [0, half_dim)
    # Shape: (half_dim,)
    freqs: torch.Tensor = torch.exp(
        -math.log(10000.0)
        * torch.arange(half_dim, dtype=torch.float32, device=device)
        / half_dim
    )

    # Outer product: t[:, None] * freqs[None, :] → (batch, half_dim)
    t_float: torch.Tensor = t.float()
    args: torch.Tensor = t_float[:, None] * freqs[None, :]  # (batch, half_dim)

    # Concatenate sin and cos: (batch, dim)
    embedding: torch.Tensor = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    return embedding  # (batch, dim)


# ---------------------------------------------------------------------------
# ResnetBlock
# ---------------------------------------------------------------------------


class ResnetBlock(nn.Module):
    """ResNet block with GroupNorm, SiLU activation, and time embedding injection.

    Core building block of the U-Net. Processes feature maps while
    incorporating diffusion timestep information via additive injection
    after the first convolution.

    Supports both 2D convolutions (mode='1d', operating on time×space
    wavelet coefficients) and 3D convolutions (mode='2d', operating on
    time×height×width wavelet coefficients).

    Architecture:
        GroupNorm → SiLU → Conv → (+ time_proj) → GroupNorm → SiLU → Conv
        + residual connection (with 1×1 conv if in_channels != out_channels)

    Attributes:
        norm1: GroupNorm applied before first convolution.
        act1: SiLU activation.
        conv1: First convolution (Conv2d or Conv3d).
        time_proj: Linear projection of time embedding to out_channels.
        time_act: SiLU activation for time embedding.
        norm2: GroupNorm applied before second convolution.
        act2: SiLU activation.
        conv2: Second convolution (Conv2d or Conv3d).
        res_conv: Residual projection (1×1 conv or Identity).
        mode: Convolution mode ('1d' or '2d').
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        groups: int = 8,
        mode: str = "1d",
        conv3d_kernel: Tuple[int, int, int] = (3, 3, 3),
        conv3d_padding: Tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        """Initialize the ResnetBlock.

        Args:
            in_channels: Number of input feature map channels.
            out_channels: Number of output feature map channels.
            time_emb_dim: Dimension of the time embedding vector from the
                MLP. Equals init_dim * 4 = 512 for init_dim=128.
                Config: unet_1d.init_dim=128 → time_emb_dim=512.
            groups: Number of groups for GroupNorm. Must divide both
                in_channels and out_channels. Config:
                unet_1d.resnet_block_groups=8.
            mode: Convolution mode. '1d' uses Conv2d on (B,C,T,X) tensors;
                '2d' uses Conv3d on (B,C,T,H,W) tensors.
                Config: experiment.spatial_dim drives this choice.
            conv3d_kernel: 3D convolution kernel size for mode='2d'.
                Config: unet_3d.conv3d_kernel=[3,3,3].
            conv3d_padding: 3D convolution padding for mode='2d'.
                Config: unet_3d.conv3d_padding=[1,1,1].

        Raises:
            ValueError: If mode is not '1d' or '2d'.
            ValueError: If groups does not divide in_channels or out_channels.
        """
        super().__init__()

        if mode not in ("1d", "2d"):
            raise ValueError(
                f"ResnetBlock mode must be '1d' or '2d', got '{mode}'."
            )
        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by groups ({groups}) "
                "for GroupNorm. Check unet_1d.resnet_block_groups in config."
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by groups ({groups}) "
                "for GroupNorm. Check unet_1d.resnet_block_groups in config."
            )

        self.mode: str = mode

        # --- Normalization and activation ---
        self.norm1: nn.Module = nn.GroupNorm(groups, in_channels)
        self.act1: nn.Module = nn.SiLU()
        self.norm2: nn.Module = nn.GroupNorm(groups, out_channels)
        self.act2: nn.Module = nn.SiLU()

        # --- Convolutions ---
        if mode == "1d":
            # 2D convolutions operating on (B, C, T, X) tensors
            self.conv1: nn.Module = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            )
            self.conv2: nn.Module = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1
            )
            # Residual projection: 1×1 conv if channel count changes
            self.res_conv: nn.Module = (
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )
        else:
            # 3D convolutions operating on (B, C, T, H, W) tensors
            k: Tuple[int, int, int] = tuple(conv3d_kernel)  # type: ignore[assignment]
            p: Tuple[int, int, int] = tuple(conv3d_padding)  # type: ignore[assignment]
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=k, padding=p)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=k, padding=p)
            self.res_conv = (
                nn.Conv3d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )

        # --- Time embedding projection ---
        # Projects time_emb_dim → out_channels for additive injection
        self.time_proj: nn.Module = nn.Linear(time_emb_dim, out_channels)
        self.time_act: nn.Module = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block.

        Args:
            x: Input feature map. Shape (B, in_channels, T, X) for mode='1d'
                or (B, in_channels, T, H, W) for mode='2d'.
            time_emb: Time embedding from the MLP. Shape (B, time_emb_dim).
                Projected to (B, out_channels) and added to feature map.

        Returns:
            Output feature map. Shape (B, out_channels, T, X) for mode='1d'
            or (B, out_channels, T, H, W) for mode='2d'.
        """
        # --- First conv path ---
        h: torch.Tensor = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # --- Time embedding injection ---
        # Project: (B, time_emb_dim) → (B, out_channels)
        t: torch.Tensor = self.time_act(self.time_proj(time_emb))

        # Reshape for broadcasting over spatial dims
        if self.mode == "1d":
            # (B, out_channels) → (B, out_channels, 1, 1)
            t = t.unsqueeze(-1).unsqueeze(-1)
        else:
            # (B, out_channels) → (B, out_channels, 1, 1, 1)
            t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        h = h + t  # broadcast over (T, X) or (T, H, W)

        # --- Second conv path ---
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        # --- Residual connection ---
        return h + self.res_conv(x)


# ---------------------------------------------------------------------------
# AttentionBlock
# ---------------------------------------------------------------------------


class AttentionBlock(nn.Module):
    """Multi-head self-attention block for the U-Net.

    Applied at each resolution level to capture long-range dependencies
    in the wavelet coefficient space. Supports both 2D (mode='1d') and
    3D (mode='2d') feature maps by flattening spatial/temporal dimensions
    into a sequence before applying attention.

    Architecture:
        GroupNorm → reshape to sequence → QKV projection → scaled dot-product
        attention → output projection → reshape back → residual connection

    Attributes:
        norm: GroupNorm (groups=1, equivalent to LayerNorm over channels).
        to_qkv: 1D convolution projecting channels to 3 × inner_dim.
        to_out: 1D convolution projecting inner_dim back to channels.
        heads: Number of attention heads.
        head_dim: Dimension per attention head.
        scale: Attention scale factor (head_dim^{-0.5}).
        mode: Feature map mode ('1d' or '2d').
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        head_dim: int = 32,
        mode: str = "1d",
    ) -> None:
        """Initialize the AttentionBlock.

        Args:
            channels: Number of input/output channels. Must be divisible
                by groups in GroupNorm (groups=1 here, so no constraint).
            heads: Number of attention heads. Config: unet_1d.attn_heads=4.
            head_dim: Hidden dimension per attention head. Config:
                unet_1d.attn_hidden_dim=32.
            mode: Feature map mode. '1d' for (B,C,T,X) tensors;
                '2d' for (B,C,T,H,W) tensors.

        Raises:
            ValueError: If mode is not '1d' or '2d'.
        """
        super().__init__()

        if mode not in ("1d", "2d"):
            raise ValueError(
                f"AttentionBlock mode must be '1d' or '2d', got '{mode}'."
            )

        self.heads: int = heads
        self.head_dim: int = head_dim
        self.scale: float = head_dim ** -0.5
        self.mode: str = mode

        inner_dim: int = heads * head_dim  # 4 * 32 = 128

        # GroupNorm with groups=1 acts as LayerNorm over the channel dimension
        self.norm: nn.Module = nn.GroupNorm(1, channels)

        # QKV projection: (B, channels, seq_len) → (B, inner_dim*3, seq_len)
        self.to_qkv: nn.Module = nn.Conv1d(channels, inner_dim * 3, kernel_size=1, bias=False)

        # Output projection: (B, inner_dim, seq_len) → (B, channels, seq_len)
        self.to_out: nn.Module = nn.Conv1d(inner_dim, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention to the feature map.

        Args:
            x: Input feature map. Shape (B, C, T, X) for mode='1d' or
                (B, C, T, H, W) for mode='2d'.

        Returns:
            Output feature map with same shape as input. The attention
            output is added to the input (residual connection).
        """
        residual: torch.Tensor = x

        # --- Normalize ---
        x = self.norm(x)

        # --- Flatten spatial/temporal dims to sequence ---
        b: int = x.shape[0]
        c: int = x.shape[1]

        if self.mode == "1d":
            # (B, C, T, X) → (B, C, T*X)
            spatial_shape: Tuple[int, ...] = x.shape[2:]  # (T, X)
            x_seq: torch.Tensor = x.reshape(b, c, -1)  # (B, C, T*X)
        else:
            # (B, C, T, H, W) → (B, C, T*H*W)
            spatial_shape = x.shape[2:]  # (T, H, W)
            x_seq = x.reshape(b, c, -1)  # (B, C, T*H*W)

        seq_len: int = x_seq.shape[2]

        # --- QKV projection ---
        # (B, C, seq_len) → (B, inner_dim*3, seq_len)
        qkv: torch.Tensor = self.to_qkv(x_seq)
        inner_dim: int = self.heads * self.head_dim

        # Split into Q, K, V: each (B, inner_dim, seq_len)
        q, k, v = qkv.chunk(3, dim=1)

        # --- Reshape for multi-head attention ---
        # (B, inner_dim, seq_len) → (B*heads, head_dim, seq_len)
        q = q.reshape(b * self.heads, self.head_dim, seq_len)
        k = k.reshape(b * self.heads, self.head_dim, seq_len)
        v = v.reshape(b * self.heads, self.head_dim, seq_len)

        # --- Scaled dot-product attention ---
        # Attention scores: (B*heads, seq_len, seq_len)
        # q: (B*heads, head_dim, seq_len) → transpose → (B*heads, seq_len, head_dim)
        # k: (B*heads, head_dim, seq_len)
        # attn[i,j] = scale * sum_d q[i,d] * k[d,j]
        attn: torch.Tensor = torch.bmm(
            q.permute(0, 2, 1),  # (B*heads, seq_len, head_dim)
            k,                    # (B*heads, head_dim, seq_len)
        ) * self.scale  # (B*heads, seq_len, seq_len)

        attn = F.softmax(attn, dim=-1)  # (B*heads, seq_len, seq_len)

        # Apply attention to values
        # v: (B*heads, head_dim, seq_len)
        # out[i] = sum_j attn[i,j] * v[:,j]
        out: torch.Tensor = torch.bmm(
            v,                    # (B*heads, head_dim, seq_len)
            attn.permute(0, 2, 1),  # (B*heads, seq_len, seq_len) → (B*heads, seq_len, seq_len)
        )  # (B*heads, head_dim, seq_len)

        # --- Reshape back ---
        # (B*heads, head_dim, seq_len) → (B, inner_dim, seq_len)
        out = out.reshape(b, inner_dim, seq_len)

        # --- Output projection ---
        # (B, inner_dim, seq_len) → (B, C, seq_len)
        out = self.to_out(out)

        # --- Reshape to original spatial layout ---
        out = out.reshape(b, c, *spatial_shape)

        # --- Residual connection ---
        return out + residual


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------


class UNet(nn.Module):
    """U-Net denoising backbone for WDNO diffusion models.

    Implements the noise prediction network eps_theta used in DDPM training
    and DDIM sampling. Supports both 1D PDE experiments (2D convolutions on
    time×space wavelet coefficients) and 2D PDE experiments (3D convolutions
    on time×height×width wavelet coefficients).

    Conditioning is injected by channel-wise concatenation of the noisy input
    with the conditioning tensor before the first convolution. This implements
    the classifier-free guidance (CFG) conditioning strategy from the paper
    (Section 3.1): the model receives W_cond as an additional input channel.

    For 1D experiments, two separate UNet instances are used:
        - eps_phi: denoises force/control wavelet coefficients
        - eps_theta: denoises state wavelet coefficients conditioned on force
    (Paper Appendix F.4, config: unet_1d.use_dual_unet=true)

    Architecture:
        init_conv → [ResnetBlock × 2 + AttentionBlock + Downsample] × 4
        → mid_block1 + mid_attn + mid_block2
        → [Upsample + ResnetBlock × 2 + AttentionBlock] × 4
        → final_norm + final_act + final_conv

    Attributes:
        in_channels: Number of noisy input channels (wavelet coefficient sets).
        out_channels: Number of output channels (predicted noise channels).
        cond_channels: Number of conditioning channels.
        init_dim: Initial channel dimension after first conv (128).
        dim_mults: Channel multipliers per level ((1,2,4,8)).
        resnet_groups: GroupNorm groups in ResNet blocks (8).
        attn_heads: Number of attention heads (4).
        attn_head_dim: Per-head attention dimension (32).
        mode: Convolution mode ('1d' or '2d').
        time_mlp: Sinusoidal embedding MLP.
        init_conv: Initial projection convolution.
        downs: Encoder ModuleList.
        mid_block1: Middle ResNet block 1.
        mid_attn: Middle attention block.
        mid_block2: Middle ResNet block 2.
        ups: Decoder ModuleList.
        final_norm: Final GroupNorm.
        final_act: Final SiLU activation.
        final_conv: Final 1×1 convolution to output channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        init_dim: int = 128,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        resnet_groups: int = 8,
        attn_heads: int = 4,
        attn_head_dim: int = 32,
        mode: str = "1d",
        conv3d_kernel: Tuple[int, int, int] = (3, 3, 3),
        conv3d_padding: Tuple[int, int, int] = (1, 1, 1),
        conv3d_stride: Tuple[int, int, int] = (1, 1, 1),
        downsample_kernel: Tuple[int, int, int] = (1, 4, 4),
        downsample_padding: Tuple[int, int, int] = (0, 1, 1),
        downsample_stride: Tuple[int, int, int] = (1, 2, 2),
        upsample_kernel: Tuple[int, int, int] = (1, 4, 4),
        upsample_padding: Tuple[int, int, int] = (0, 1, 1),
        upsample_stride: Tuple[int, int, int] = (1, 2, 2),
    ) -> None:
        """Initialize the UNet.

        Args:
            in_channels: Number of channels in the noisy input tensor
                (wavelet coefficient sets). For 1D Burgers' state: 4
                (one set per 2D DWT output). For 2D fluid: 8.
            out_channels: Number of output channels (predicted noise).
                Typically equals in_channels.
            cond_channels: Number of conditioning channels. Concatenated
                with noisy input before init_conv. Computed from
                WaveletTransform.get_output_shape() for the conditioning
                variables (u0, f, u_star, etc.).
            init_dim: Initial channel dimension after first convolution.
                Config: unet_1d.init_dim=128.
            dim_mults: Channel multipliers for each encoder/decoder level.
                Config: unet_1d.dim_mults=[1,2,4,8]. Produces channel
                counts [128, 256, 512, 1024] for init_dim=128.
            resnet_groups: Number of groups for GroupNorm in ResNet blocks.
                Config: unet_1d.resnet_block_groups=8. All channel counts
                must be divisible by this value.
            attn_heads: Number of attention heads. Config:
                unet_1d.attn_heads=4.
            attn_head_dim: Hidden dimension per attention head. Config:
                unet_1d.attn_hidden_dim=32.
            mode: Convolution mode. '1d' uses Conv2d on (B,C,T,X) tensors
                (1D PDE experiments); '2d' uses Conv3d on (B,C,T,H,W)
                tensors (2D PDE experiments). Config: experiment.spatial_dim.
            conv3d_kernel: 3D conv kernel size for ResNet blocks in mode='2d'.
                Config: unet_3d.conv3d_kernel=[3,3,3].
            conv3d_padding: 3D conv padding for ResNet blocks in mode='2d'.
                Config: unet_3d.conv3d_padding=[1,1,1].
            conv3d_stride: 3D conv stride for ResNet blocks in mode='2d'.
                Config: unet_3d.conv3d_stride=[1,1,1].
            downsample_kernel: Downsampling conv kernel. For mode='2d':
                (1,4,4) preserves T, halves H and W.
                Config: unet_3d.downsample_kernel=[1,4,4].
            downsample_padding: Downsampling conv padding.
                Config: unet_3d.downsample_padding=[0,1,1].
            downsample_stride: Downsampling conv stride.
                Config: unet_3d.downsample_stride=[1,2,2].
            upsample_kernel: Upsampling conv kernel.
                Config: unet_3d.upsample_kernel=[1,4,4].
            upsample_padding: Upsampling conv padding.
                Config: unet_3d.upsample_padding=[0,1,1].
            upsample_stride: Upsampling conv stride.
                Config: unet_3d.upsample_stride=[1,2,2].

        Raises:
            ValueError: If mode is not '1d' or '2d'.
            ValueError: If init_dim is not divisible by resnet_groups.
            ValueError: If any channel count in dim_mults is not divisible
                by resnet_groups.
        """
        super().__init__()

        if mode not in ("1d", "2d"):
            raise ValueError(
                f"UNet mode must be '1d' or '2d', got '{mode}'. "
                "Use '1d' for 1D PDE experiments and '2d' for 2D PDE experiments."
            )

        # Validate GroupNorm divisibility for all channel counts
        dims: List[int] = [init_dim * m for m in dim_mults]
        for d in [init_dim] + dims:
            if d % resnet_groups != 0:
                raise ValueError(
                    f"Channel count {d} is not divisible by resnet_groups={resnet_groups}. "
                    "Adjust init_dim or dim_mults so all channel counts are divisible "
                    "by resnet_groups. Config: unet_1d.resnet_block_groups=8."
                )

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.cond_channels: int = cond_channels
        self.init_dim: int = init_dim
        self.dim_mults: Tuple[int, ...] = dim_mults
        self.resnet_groups: int = resnet_groups
        self.attn_heads: int = attn_heads
        self.attn_head_dim: int = attn_head_dim
        self.mode: str = mode

        # Store 3D conv parameters for mode='2d'
        self._conv3d_kernel: Tuple[int, int, int] = tuple(conv3d_kernel)  # type: ignore[assignment]
        self._conv3d_padding: Tuple[int, int, int] = tuple(conv3d_padding)  # type: ignore[assignment]
        self._downsample_kernel: Tuple[int, int, int] = tuple(downsample_kernel)  # type: ignore[assignment]
        self._downsample_padding: Tuple[int, int, int] = tuple(downsample_padding)  # type: ignore[assignment]
        self._downsample_stride: Tuple[int, int, int] = tuple(downsample_stride)  # type: ignore[assignment]
        self._upsample_kernel: Tuple[int, int, int] = tuple(upsample_kernel)  # type: ignore[assignment]
        self._upsample_padding: Tuple[int, int, int] = tuple(upsample_padding)  # type: ignore[assignment]
        self._upsample_stride: Tuple[int, int, int] = tuple(upsample_stride)  # type: ignore[assignment]

        # Time embedding dimension: init_dim * 4 = 512 for init_dim=128
        time_emb_dim: int = init_dim * 4

        # -----------------------------------------------------------------------
        # Time embedding MLP
        # -----------------------------------------------------------------------
        self.time_mlp: nn.Module = self._build_time_mlp(init_dim)

        # -----------------------------------------------------------------------
        # Initial projection convolution
        # Maps (in_channels + cond_channels) → init_dim
        # -----------------------------------------------------------------------
        total_in_channels: int = in_channels + cond_channels
        self.init_conv: nn.Module = self._make_conv(
            total_in_channels, init_dim, kernel_size=3, padding=1
        )

        # -----------------------------------------------------------------------
        # Encoder (downsampling path)
        # -----------------------------------------------------------------------
        # Channel progression: init_dim → init_dim*2 → init_dim*4 → init_dim*8
        # For init_dim=128: 128 → 256 → 512 → 1024
        n_levels: int = len(dim_mults)
        self.downs: nn.ModuleList = nn.ModuleList()

        prev_dim: int = init_dim
        for i, mult in enumerate(dim_mults):
            cur_dim: int = init_dim * mult
            is_last: bool = (i == n_levels - 1)

            level_modules: nn.Module