"""
Swin-v2 based Neural Operator.

The paper mentions "Swin-v2 transformers" as one of the comparison models.
This implements a Swin Transformer V2-based neural operator for 2D PDE problems.

Based on: Liu et al., "Swin Transformer V2: Scaling Up Capacity and Resolution"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def window_partition(x, window_size):
    """
    Partition input into non-overlapping windows.
    x: (batch, H, W, C)
    returns: (num_windows * batch, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.
    windows: (num_windows * batch, window_size, window_size, C)
    returns: (batch, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, -1)
    return x


class WindowAttentionV2(nn.Module):
    """
    Swin Transformer V2 window attention with log-spaced continuous position bias.
    """

    def __init__(self, dim, window_size, n_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.scale = nn.Parameter(torch.ones(n_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Continuous position bias (Swin V2)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, n_heads, bias=False),
        )

        # Generate relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, ws, ws)
        coords_flatten = torch.flatten(coords, 1)  # (2, ws^2)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, ws^2, ws^2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (ws^2, ws^2, 2)
        # Normalize to [-1, 1]
        relative_coords[:, :, 0] /= (window_size - 1)
        relative_coords[:, :, 1] /= (window_size - 1)
        relative_coords = relative_coords * 8  # scale to [-8, 8]
        relative_coords = torch.sign(relative_coords) * torch.log2(
            torch.abs(relative_coords) + 1.0
        ) / math.log2(8)
        self.register_buffer("relative_coords_table", relative_coords)

        # Relative position index
        relative_position_index = relative_coords[:, :, 0] * 0 + relative_coords[:, :, 1] * 0
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        """
        x: (num_windows * batch, window_size^2, dim)
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Cosine attention (Swin V2)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * torch.clamp(self.scale, max=100).exp()

        # Continuous position bias
        relative_position_bias = self.cpb_mlp(self.relative_coords_table)  # (ws^2, ws^2, n_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (n_heads, ws^2, ws^2)
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.n_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1, self.n_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer V2 block."""

    def __init__(self, dim, n_heads, window_size=4, shift_size=0, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttentionV2(dim, window_size, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, H, W):
        """
        x: (batch, H*W, dim)
        """
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = x.reshape(B, H, W, C)

        # Pad if needed
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)

        # Attention
        attn_windows = self.attn(self.norm1(x_windows))

        # Merge windows
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.reshape(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SwinNO2d(nn.Module):
    """
    Swin Transformer V2-based Neural Operator for 2D problems.
    
    Architecture: Lifting -> Swin blocks (alternating window/shifted-window) -> Projection
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        width: int = 96,
        n_layers: int = 4,
        n_heads: int = 8,
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.width = width
        self.window_size = window_size

        # Lifting layer (adapter - problem specific)
        self.lifting = nn.Linear(n_input, width)

        # Swin blocks (shared backbone) - alternating regular and shifted window attention
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=width,
                n_heads=n_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for i in range(n_layers)
        ])

        # Projection layer (adapter - problem specific)
        self.projection = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, n_output),
        )

        self.norm = nn.LayerNorm(width)

    def get_backbone_params(self):
        return list(self.swin_blocks.parameters()) + list(self.norm.parameters())

    def get_adapter_params(self):
        return list(self.lifting.parameters()) + list(self.projection.parameters())

    def forward(self, x):
        # x: (batch, n_input, nx, ny)
        batch, n_input, nx, ny = x.shape

        # Lifting
        x = x.permute(0, 2, 3, 1)  # (batch, nx, ny, n_input)
        x = self.lifting(x)  # (batch, nx, ny, width)

        # Flatten spatial for Swin blocks
        x = x.reshape(batch, nx * ny, self.width)

        for block in self.swin_blocks:
            x = block(x, nx, ny)

        x = self.norm(x)
        x = x.reshape(batch, nx, ny, self.width)

        # Projection
        x = self.projection(x)  # (batch, nx, ny, n_output)
        x = x.permute(0, 3, 1, 2)  # (batch, n_output, nx, ny)
        return x
