"""Codomain Attention Neural Operator (CoDA-NO).

As described in Section 3 of the paper:
CoDA-NO [13] employs codomain attention with function space dot product.
Codomain attention mechanisms are advantageous to conventional transformers
in neural-operator based problems: the dot product detects similarity not
between samples, but between features, mapped with neural operators.

Reference: Rahman et al., "Pretraining Codomain Attention Neural Operators
for Solving Multiphysics PDEs", NeurIPS 2024.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .fno import SpectralConv2d, FNOBlock


class CodomainAttention(nn.Module):
    """Codomain attention: computes similarity between features (not samples).

    Instead of dot product between samples (standard attention), CoDA computes
    similarity in the function space between features using an integral (sum
    over spatial dimensions) of the product of feature maps.
    """

    def __init__(self, hidden_channels, num_heads=4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        assert hidden_channels % num_heads == 0

        # FNO-based projections for Q, K, V (operate on feature dimension)
        self.q_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.k_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.v_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.out_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        Args:
            x: (batch, hidden, nx, ny)
        Returns:
            (batch, hidden, nx, ny)

        Codomain attention computes attention weights between feature channels
        (codomain) by integrating over the spatial domain, capturing global
        function-space relationships.
        """
        batch, hidden, nx, ny = x.shape

        Q = self.q_proj(x)  # (batch, hidden, nx, ny)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head: (batch, heads, head_dim, nx*ny)
        Q = Q.reshape(batch, self.num_heads, self.head_dim, nx * ny)
        K = K.reshape(batch, self.num_heads, self.head_dim, nx * ny)
        V = V.reshape(batch, self.num_heads, self.head_dim, nx * ny)

        # Codomain attention: dot product over spatial (integral over domain)
        # Shape: (batch, heads, head_dim, head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Softmax over feature dimension (codomain)
        attn = F.softmax(attn, dim=-1)

        # Apply attention: (batch, heads, head_dim, nx*ny)
        out = torch.matmul(attn, V)
        out = out.reshape(batch, hidden, nx, ny)

        return self.out_proj(out)


class CoDABlock(nn.Module):
    """A block with codomain attention followed by FNO spectral convolution."""

    def __init__(self, hidden_channels, modes1, modes2, num_heads=4):
        super().__init__()
        self.codomain_attn = CodomainAttention(hidden_channels, num_heads)
        self.spectral = SpectralConv2d(hidden_channels, hidden_channels, modes1, modes2)
        self.linear = nn.Conv2d(hidden_channels, hidden_channels, 1)

        self.norm1 = nn.GroupNorm(1, hidden_channels)
        self.norm2 = nn.GroupNorm(1, hidden_channels)

    def forward(self, x):
        # Codomain attention with residual
        x = x + self.codomain_attn(self.norm1(x))

        # Spectral convolution with residual
        x = x + F.gelu(self.spectral(self.norm2(x)) + self.linear(x))

        return x


class CoDANO(nn.Module):
    """Codomain Attention Neural Operator.

    Combines codomain attention (feature-space dot product) with FNO spectral
    convolutions for multiphysics transfer learning.

    Architecture:
    Input -> Lifting -> CoDA Blocks -> Projection -> Output
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 4,
        modes1: int = 12,
        modes2: int = 12,
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels

        # Lifting adapter
        self.lifting = nn.Linear(input_channels, hidden_channels)

        # CoDA blocks
        self.coda_blocks = nn.ModuleList([
            CoDABlock(hidden_channels, modes1, modes2, num_heads)
            for _ in range(n_layers)
        ])

        # Projection adapter
        self.projection = nn.Linear(hidden_channels, output_channels)

    def forward(self, x, grid=None):
        """
        Args:
            x: (batch, spatial_x, spatial_y, input_channels)
        Returns:
            (batch, spatial_x, spatial_y, output_channels)
        """
        batch, nx, ny, _ = x.shape

        # Lift
        v = self.lifting(x)  # (batch, nx, ny, hidden)
        v = v.permute(0, 3, 1, 2)  # (batch, hidden, nx, ny)

        # CoDA blocks
        for block in self.coda_blocks:
            v = block(v)

        # Project
        v = v.permute(0, 2, 3, 1)  # (batch, nx, ny, hidden)
        out = self.projection(v)

        return out

    def get_lifting_params(self):
        return list(self.lifting.parameters())

    def get_projection_params(self):
        return list(self.projection.parameters())

    def get_core_params(self):
        return list(self.coda_blocks.parameters())
