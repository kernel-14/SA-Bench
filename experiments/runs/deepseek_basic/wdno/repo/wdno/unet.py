"""
U-Net architecture for WDNO.

Based on the DDPM U-Net (Ho et al., 2020b) with modifications for wavelet domain.
Supports 1D, 2D, and 3D variants as described in the paper.

Architecture details from the paper (Tables 18, 19, 20):
- 1D/2D: 128 initial dim, 4 down/up layers, kernel_size=3, dim_mult=[1,2,4,8], 8 resnet groups
- Attention with hidden_dim=32, heads=4
- 3D: 3D convolutions with kernel_size=(3,3,3), specific downsampling for spatial dims
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Dict
from abc import ABC, abstractmethod


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding as in Transformer."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResnetBlock(nn.Module):
    """
    ResNet block with time embedding conditioning.

    Structure: GroupNorm -> SiLU -> Conv -> time_emb proj -> GroupNorm -> SiLU -> Conv
    With residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        groups: int = 8,
        conv_cls=nn.Conv2d,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.silu1 = nn.SiLU()
        padding = kernel_size // 2
        self.conv1 = conv_cls(in_channels, out_channels, kernel_size, padding=padding)

        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.silu2 = nn.SiLU()
        self.conv2 = conv_cls(out_channels, out_channels, kernel_size, padding=padding)

        if in_channels != out_channels:
            self.shortcut = conv_cls(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.silu1(h)
        h = self.conv1(h)

        # Add time embedding
        time_proj = self.time_emb_proj(time_emb)
        # Reshape time_proj to match spatial dims
        while time_proj.dim() < h.dim():
            time_proj = time_proj.unsqueeze(-1)
        h = h + time_proj

        h = self.norm2(h)
        h = self.silu2(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Multi-head attention block.

    As described in Table 18: hidden_dim=32, heads=4.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_head_channels: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        inner_dim = num_heads * num_head_channels

        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv1d(channels, inner_dim * 3, 1)
        self.proj = nn.Conv1d(inner_dim, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, *spatial = x.shape
        x_flat = x.view(B, C, -1)  # (B, C, N)

        h = self.norm(x.view(B, C, *spatial))
        h = h.view(B, C, -1)

        qkv = self.qkv(h)  # (B, 3*inner_dim, N)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.num_head_channels, -1)
        k = k.view(B, self.num_heads, self.num_head_channels, -1)
        v = v.view(B, self.num_heads, self.num_head_channels, -1)

        scale = self.num_head_channels ** -0.5
        attn = torch.einsum('b h c n, b h c m -> b h n m', q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('b h n m, b h c m -> b h c n', attn, v)
        out = out.reshape(B, -1, out.shape[-1])

        out = self.proj(out)
        return (x_flat + out).view(B, C, *spatial)


class UNet2D(nn.Module):
    """
    2D U-Net architecture for 1D PDE tasks (time × space wavelet domain).

    Hyperparameters from Table 18:
    - Initial dimension: 128
    - Downsampling/Upsampling layers: 4
    - Convolution kernel size: 3
    - Dimension multiplier: [1, 2, 4, 8]
    - Resnet block groups: 8
    - Attention hidden dimension: 32
    - Attention heads: 4
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 0,
        init_dim: int = 128,
        dim_mult: List[int] = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        resnet_groups: int = 8,
        attn_heads: int = 4,
        attn_dim_head: int = 32,
        kernel_size: int = 3,
        time_emb_dim: int = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        total_in_channels = in_channels + cond_channels

        if time_emb_dim is None:
            time_emb_dim = init_dim * 4

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(init_dim),
            nn.Linear(init_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        dims = [init_dim * m for m in dim_mult]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # Initial convolution
        self.init_conv = nn.Conv2d(total_in_channels, init_dim, kernel_size, padding=kernel_size // 2)

        # Downsampling
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, dim_in, time_emb_dim, resnet_groups, nn.Conv2d, kernel_size),
                    ResnetBlock(dim_in, dim_in, time_emb_dim, resnet_groups, nn.Conv2d, kernel_size),
                    AttentionBlock(dim_in, attn_heads, attn_dim_head) if not is_last else nn.Identity(),
                    nn.Conv2d(dim_in, dim_out, 4, stride=2, padding=1) if not is_last else
                    nn.Conv2d(dim_in, dim_out, kernel_size, padding=kernel_size // 2),
                ])
            )

        # Middle
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim, resnet_groups, nn.Conv2d, kernel_size)
        self.mid_attn = AttentionBlock(mid_dim, attn_heads, attn_dim_head)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim, resnet_groups, nn.Conv2d, kernel_size)

        # Upsampling
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList([
                    ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim, resnet_groups, nn.Conv2d, kernel_size),
                    ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim, resnet_groups, nn.Conv2d, kernel_size),
                    AttentionBlock(dim_out, attn_heads, attn_dim_head) if not is_last else nn.Identity(),
                    nn.ConvTranspose2d(dim_out, dim_in, 4, stride=2, padding=1) if not is_last else
                    nn.Conv2d(dim_out, dim_in, kernel_size, padding=kernel_size // 2),
                ])
            )

        self.final_resnet = ResnetBlock(init_dim * 2, init_dim, time_emb_dim, resnet_groups, nn.Conv2d, kernel_size)
        self.final_conv = nn.Conv2d(init_dim, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Noisy wavelet coefficients W_u^{(k)} (B, C, H, W)
            t: Timestep indices (B,)
            cond: Conditioning wavelet coefficients W_a (B, C_cond, H, W)

        Returns:
            Predicted noise epsilon (B, C, H, W)
        """
        # Concatenate conditioning along channel dim
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        time_emb = self.time_mlp(t)

        x = self.init_conv(x)
        r = x.clone()

        hs = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb)
            x = block2(x, time_emb)
            x = attn(x)
            hs.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        for block1, block2, attn, upsample in self.ups:
            h = hs.pop()
            x = torch.cat([x, h], dim=1)
            x = block1(x, time_emb)
            x = block2(x, time_emb)
            x = attn(x)
            x = upsample(x)

        x = torch.cat([x, r], dim=1)
        x = self.final_resnet(x, time_emb)
        return self.final_conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net architecture for 2D PDE tasks (time × height × width wavelet domain).

    Based on the video diffusion model architecture (Ho et al., 2022).
    Uses 3D convolutions with spatial-temporal processing.

    Hyperparameters from Table 20:
    - Number of attention heads: 4
    - Kernel size of conv3d: (3, 3, 3)
    - Padding of conv3d: (1, 1, 1)
    - Stride of conv3d: (1, 1, 1)
    - Downsampling kernel: (1, 4, 4), stride: (0, 1, 1)? Actually from table:
      Kernel size of downsampling: (1, 4, 4)
      Padding of downsampling: (1, 2, 2)
      Stride of downsampling: (0, 1, 1) -- note this seems odd, likely (1,1,1) or (1,2,2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 0,
        init_dim: int = 128,
        dim_mult: List[int] = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        resnet_groups: int = 8,
        attn_heads: int = 4,
        attn_dim_head: int = 32,
        kernel_size: int = 3,
        time_emb_dim: int = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        total_in_channels = in_channels + cond_channels

        if time_emb_dim is None:
            time_emb_dim = init_dim * 4

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(init_dim),
            nn.Linear(init_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        dims = [init_dim * m for m in dim_mult]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # 3D convolution with kernel_size (3,3,3)
        k3d = (3, 3, 3)
        p3d = (1, 1, 1)

        # Initial convolution
        self.init_conv = nn.Conv3d(total_in_channels, init_dim, kernel_size=k3d, padding=p3d)

        # Downsampling
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, dim_in, time_emb_dim, resnet_groups, nn.Conv3d, 3),
                    ResnetBlock(dim_in, dim_in, time_emb_dim, resnet_groups, nn.Conv3d, 3),
                    AttentionBlock3D(dim_in, attn_heads, attn_dim_head) if not is_last else nn.Identity(),
                    # Downsample: kernel (1,4,4), stride (1,2,2) -- only downsample spatial
                    nn.Conv3d(dim_in, dim_out, kernel_size=(1, 4, 4), stride=(1, 2, 2),
                             padding=(0, 1, 1)) if not is_last else
                    nn.Conv3d(dim_in, dim_out, kernel_size=k3d, padding=p3d),
                ])
            )

        # Middle
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim, resnet_groups, nn.Conv3d, 3)
        self.mid_attn = AttentionBlock3D(mid_dim, attn_heads, attn_dim_head)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim, resnet_groups, nn.Conv3d, 3)

        # Upsampling
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList([
                    ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim, resnet_groups, nn.Conv3d, 3),
                    ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim, resnet_groups, nn.Conv3d, 3),
                    AttentionBlock3D(dim_out, attn_heads, attn_dim_head) if not is_last else nn.Identity(),
                    # Upsample: kernel (1,4,4), stride (1,2,2) -- only upsample spatial
                    nn.ConvTranspose3d(dim_out, dim_in, kernel_size=(1, 4, 4), stride=(1, 2, 2),
                                      padding=(0, 1, 1)) if not is_last else
                    nn.Conv3d(dim_out, dim_in, kernel_size=k3d, padding=p3d),
                ])
            )

        self.final_resnet = ResnetBlock(init_dim * 2, init_dim, time_emb_dim, resnet_groups, nn.Conv3d, 3)
        self.final_conv = nn.Conv3d(init_dim, out_channels, kernel_size=k3d, padding=p3d)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Noisy wavelet coefficients (B, C, T, H, W)
            t: Timestep indices (B,)
            cond: Conditioning wavelet coefficients (B, C_cond, T, H, W)

        Returns:
            Predicted noise (B, C, T, H, W)
        """
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        time_emb = self.time_mlp(t)

        x = self.init_conv(x)
        r = x.clone()

        hs = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb)
            x = block2(x, time_emb)
            x = attn(x)
            hs.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        for block1, block2, attn, upsample in self.ups:
            h = hs.pop()
            # Ensure spatial alignment
            if h.shape != x.shape:
                x = F.interpolate(x, size=h.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, h], dim=1)
            x = block1(x, time_emb)
            x = block2(x, time_emb)
            x = attn(x)
            x = upsample(x)

        x = torch.cat([x, r], dim=1)
        x = self.final_resnet(x, time_emb)
        return self.final_conv(x)


class AttentionBlock3D(nn.Module):
    """3D attention block: flattens spatial dims for attention computation."""

    def __init__(self, channels: int, num_heads: int = 4, num_head_channels: int = 32):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        inner_dim = num_heads * num_head_channels

        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv1d(channels, inner_dim * 3, 1)
        self.proj = nn.Conv1d(inner_dim, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x_flat = x.view(B, C, -1)

        h = self.norm(x.view(B, C, T, H, W))
        h = h.view(B, C, -1)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.num_heads, self.num_head_channels, -1)
        k = k.view(B, self.num_heads, self.num_head_channels, -1)
        v = v.view(B, self.num_heads, self.num_head_channels, -1)

        scale = self.num_head_channels ** -0.5
        attn = torch.einsum('b h c n, b h c m -> b h n m', q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('b h n m, b h c m -> b h c n', attn, v)
        out = out.reshape(B, -1, out.shape[-1])

        out = self.proj(out)
        return (x_flat + out).view(B, C, T, H, W)


# Alias for backward compatibility
UNet1D = UNet2D  # For 1D PDEs, we use 2D UNet (time × space)
