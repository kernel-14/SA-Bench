"""Swin Transformer v2 Neural Operator.

As described in Section 3 of the paper:
Swin-v2 transformers (referenced as POSEIDON [12]) use hierarchical vision
transformers with shifted windows, applicable to transfer knowledge across
Euler/Navier-Stokes cases.

The paper uses Swin-v2 as a comparison method in their experiments.
Reference: Liu et al., "Swin Transformer V2: Scaling Up Capacity and Resolution", CVPR 2022.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WindowAttention(nn.Module):
    """Multi-head self attention with shifted windows (Swin-v2 style)."""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Log-scale continuous position bias (Swin-v2)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(1, num_heads, 1, 1)))

        # Relative position bias table
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch * num_windows, window_size^2, dim)
        Returns:
            (batch * num_windows, window_size^2, dim)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Cosine attention (Swin-v2)
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale

        if mask is not None:
            attn = attn + mask.unsqueeze(1).unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with window-based attention and shifted windows."""

    def __init__(self, dim, num_heads, window_size, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def _window_partition(self, x):
        B, H, W, C = x.shape
        ws = self.window_size
        x = x.reshape(B, H // ws, ws, W // ws, ws, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        return windows

    def _window_reverse(self, windows, H, W):
        ws = self.window_size
        B = int(windows.shape[0] / ((H // ws) * (W // ws)))
        x = windows.reshape(B, H // ws, W // ws, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
        return x

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x

        x = self.norm1(x)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = self._window_partition(shifted_x)
        attn_windows = self.attn(x_windows)

        # Reverse cyclic shift
        attn_windows = self._window_reverse(attn_windows, H, W)
        if self.shift_size > 0:
            x = torch.roll(attn_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = attn_windows

        x = shortcut + x

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """Patch merging for hierarchical feature maps."""

    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinStage(nn.Module):
    """One stage of Swin Transformer blocks."""

    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.0, downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                SwinTransformerBlock(dim, num_heads, window_size, shift_size, mlp_ratio)
            )
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        return x, None


class SwinTransformerNO(nn.Module):
    """Swin Transformer v2-based Neural Operator.

    Uses hierarchical Swin Transformer architecture adapted for PDE solution
    mapping. Replaces FNO blocks with Swin Transformer stages while keeping
    the lifting-projection adapter framework.

    This follows the POSEIDON-style approach referenced in the paper [12].
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        hidden_channels: int = 64,
        depths: tuple = (2, 2, 2),
        num_heads: tuple = (4, 8, 16),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels

        # Lifting adapter
        self.lifting = nn.Linear(input_channels, hidden_channels)

        # Swin stages
        self.stages = nn.ModuleList()
        dim = hidden_channels
        for i, (depth, n_heads) in enumerate(zip(depths, num_heads)):
            downsample = (i < len(depths) - 1)
            stage = SwinStage(dim, depth, n_heads, window_size, mlp_ratio, downsample)
            self.stages.append(stage)
            if downsample:
                dim = dim * 2

        self.output_dim = dim

        # Projection adapter
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, output_channels),
        )

    def forward(self, x, grid=None):
        """
        Args:
            x: (batch, spatial_x, spatial_y, input_channels)
        Returns:
            (batch, spatial_x, spatial_y, output_channels)
        """
        B, H, W, C = x.shape

        # Lift
        v = self.lifting(x)  # (B, H, W, hidden_channels)

        # Swin stages (operate on spatial feature map)
        for stage in self.stages:
            v, v_down = stage(v)
            if v_down is not None:
                v = v_down
                H, W = H // 2, W // 2

        # Project back to original resolution via interpolation
        v = F.interpolate(
            v.permute(0, 3, 1, 2),  # (B, C, H, W)
            size=(x.shape[1], x.shape[2]),
            mode='bilinear',
            align_corners=False,
        ).permute(0, 2, 3, 1)  # (B, H, W, C)

        out = self.projection(v)
        return out

    def get_lifting_params(self):
        return list(self.lifting.parameters())

    def get_projection_params(self):
        return list(self.projection.parameters())

    def get_core_params(self):
        return list(self.stages.parameters())
