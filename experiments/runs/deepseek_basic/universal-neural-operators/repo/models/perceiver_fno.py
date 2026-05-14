"""Perceiver IO-based Neural Operator.

As described in Section 3 of the paper:
The Perceiver IO [18] enables encoding of information with a smaller number of
latent feature arrays, operating with more abstract features and maintaining a
limited number of parameters.

The latent variables and input state are combined with cross-attention where
keys and values are obtained from FNO-based mapping from inputs K = FNO_K(X),
V = FNO_V(X), and latent variables are taken as queries Q = L. This is followed
by self-attention between latent representations. The output is constructed
with cross-attention matching queries from inputs with keys/values from
transformed latent representations.

Reference: Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs & Outputs", 2021.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .fno import SpectralConv2d, FNOBlock


class CrossAttention(nn.Module):
    """Cross-attention block with FNO-based key/value mapping.

    As described: K = FNO_K(X), V = FNO_V(X), Q = L (latent).
    """

    def __init__(self, hidden_channels, num_heads=4, head_dim=None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_channels // num_heads)
        self.inner_dim = self.head_dim * num_heads

        # FNO-based key/value projections from inputs (1x1 conv as pointwise)
        self.fno_k = nn.Conv2d(hidden_channels, self.inner_dim, 1)
        self.fno_v = nn.Conv2d(hidden_channels, self.inner_dim, 1)

        # Query projection (from latent)
        self.q_proj = nn.Linear(hidden_channels, self.inner_dim)

        # Output projection
        self.out_proj = nn.Linear(self.inner_dim, hidden_channels)

        self.scale = self.head_dim ** -0.5

    def forward(self, x_input, x_latent):
        """
        Args:
            x_input: (batch, hidden, nx, ny) — input features
            x_latent: (batch, n_latent, hidden) — latent array
        Returns:
            (batch, n_latent, hidden) — updated latent
        """
        batch, hidden, nx, ny = x_input.shape
        n_latent = x_latent.shape[1]

        # Keys and values from input via FNO (1x1 conv)
        K = self.fno_k(x_input)  # (batch, inner_dim, nx, ny)
        V = self.fno_v(x_input)  # (batch, inner_dim, nx, ny)

        # Reshape to sequences: (batch, heads, head_dim, nx*ny)
        K = K.reshape(batch, self.num_heads, self.head_dim, nx * ny)
        V = V.reshape(batch, self.num_heads, self.head_dim, nx * ny)

        # Queries from latent: (batch, n_latent, inner_dim)
        Q = self.q_proj(x_latent)
        Q = Q.reshape(batch, n_latent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q: (batch, heads, n_latent, head_dim)

        # Attention scores: (batch, heads, n_latent, nx*ny)
        attn = torch.matmul(Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum: (batch, heads, n_latent, head_dim)
        out = torch.matmul(attn, V.transpose(-2, -1))
        out = out.permute(0, 2, 1, 3).reshape(batch, n_latent, self.inner_dim)

        return self.out_proj(out)


class SelfAttention(nn.Module):
    """Self-attention between latent representations."""

    def __init__(self, hidden_channels, num_heads=4, head_dim=None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_channels // num_heads)
        self.inner_dim = self.head_dim * num_heads

        self.qkv_proj = nn.Linear(hidden_channels, self.inner_dim * 3)
        self.out_proj = nn.Linear(self.inner_dim, hidden_channels)

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        Args:
            x: (batch, n_latent, hidden)
        Returns:
            (batch, n_latent, hidden)
        """
        batch, n_latent, _ = x.shape

        qkv = self.qkv_proj(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        Q = Q.reshape(batch, n_latent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch, n_latent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch, n_latent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(batch, n_latent, self.inner_dim)

        return self.out_proj(out)


class CrossAttentionOutput(nn.Module):
    """Output cross-attention: queries from input, keys/values from latent.

    Maps back from latent space to the spatial input grid.
    """

    def __init__(self, hidden_channels, num_heads=4, head_dim=None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_channels // num_heads)
        self.inner_dim = self.head_dim * num_heads

        # FNO-based query projection from inputs
        self.fno_q = nn.Conv2d(hidden_channels, self.inner_dim, 1)

        # Key/value from latent
        self.kv_proj = nn.Linear(hidden_channels, self.inner_dim * 2)

        self.out_proj = nn.Conv2d(self.inner_dim, hidden_channels, 1)

        self.scale = self.head_dim ** -0.5

    def forward(self, x_input, x_latent):
        """
        Args:
            x_input: (batch, hidden, nx, ny)
            x_latent: (batch, n_latent, hidden)
        Returns:
            (batch, hidden, nx, ny)
        """
        batch, hidden, nx, ny = x_input.shape
        n_latent = x_latent.shape[1]

        # Queries from input: (batch, inner_dim, nx, ny)
        Q = self.fno_q(x_input)
        # Reshape: (batch, heads, head_dim, nx*ny)
        Q = Q.reshape(batch, self.num_heads, self.head_dim, nx * ny)

        # Keys and values from latent: (batch, n_latent, inner_dim)
        kv = self.kv_proj(x_latent)
        K, V = kv.chunk(2, dim=-1)
        # Reshape: (batch, heads, n_latent, head_dim)
        K = K.reshape(batch, n_latent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch, n_latent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention: Q^T * K -> (batch, heads, nx*ny, n_latent)
        # Q: (batch, heads, head_dim, nx*ny), K: (batch, heads, n_latent, head_dim)
        attn = torch.matmul(Q.transpose(-2, -1), K.transpose(-2, -1)) * self.scale
        # attn: (batch, heads, nx*ny, n_latent)
        attn = F.softmax(attn, dim=-1)

        # Weighted sum: attn @ V -> (batch, heads, nx*ny, head_dim)
        out = torch.matmul(attn, V)
        # out: (batch, heads, nx*ny, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(batch, nx * ny, self.inner_dim)
        out = out.transpose(1, 2).reshape(batch, self.inner_dim, nx, ny)

        return self.out_proj(out)


class PerceiverBlock(nn.Module):
    """Single Perceiver IO block with cross-attention + self-attention + output cross-attention."""

    def __init__(self, hidden_channels, num_heads=4, n_latent=256, head_dim=None):
        super().__init__()
        self.cross_attn_in = CrossAttention(hidden_channels, num_heads, head_dim)
        self.self_attn = SelfAttention(hidden_channels, num_heads, head_dim)
        self.cross_attn_out = CrossAttentionOutput(hidden_channels, num_heads, head_dim)

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        # Feed-forward for latent
        self.ff_latent = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.GELU(),
            nn.Linear(hidden_channels * 4, hidden_channels),
        )
        self.norm_ff = nn.LayerNorm(hidden_channels)

    def forward(self, x, latent):
        """
        Args:
            x: (batch, hidden, nx, ny)
            latent: (batch, n_latent, hidden)
        Returns:
            x_out: (batch, hidden, nx, ny)
            latent_out: (batch, n_latent, hidden)
        """
        # Cross-attention: input -> latent
        latent = latent + self.cross_attn_in(x, self.norm1(latent))

        # Self-attention on latent
        latent = latent + self.self_attn(self.norm2(latent))

        # Feed-forward
        latent = latent + self.ff_latent(self.norm_ff(latent))

        # Output cross-attention: latent -> input
        x_out = x + self.cross_attn_out(x, latent)

        return x_out, latent


class PerceiverFNO(nn.Module):
    """Perceiver IO-based Neural Operator with FNO mappings.

    Architecture (Section 3):
    - Lifting layer
    - Perceiver IO blocks with FNO-based key/value/query mappings
    - Projection layer

    The cross-attention uses FNO-based mapping from inputs for keys and values,
    and latent variables as queries. Self-attention operates on latent representations.
    Output cross-attention maps back to input space.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        hidden_channels: int = 64,
        n_layers: int = 4,
        modes1: int = 12,
        modes2: int = 12,
        n_latent: int = 128,
        num_heads: int = 4,
        use_fno_blocks: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.n_latent = n_latent

        # Lifting adapter
        self.lifting = nn.Linear(input_channels, hidden_channels)

        # Optional FNO blocks before Perceiver
        self.use_fno_blocks = use_fno_blocks
        n_perceiver = max(1, n_layers // 2)
        n_fno = n_layers - n_perceiver

        if use_fno_blocks and n_fno > 0:
            self.fno_blocks = nn.ModuleList([
                FNOBlock(hidden_channels, modes1, modes2)
                for _ in range(n_fno // 2)
            ])

        # Perceiver blocks
        n_perceiver_blocks = max(1, n_layers // 2)
        self.perceiver_blocks = nn.ModuleList([
            PerceiverBlock(hidden_channels, num_heads, n_latent)
            for _ in range(n_perceiver_blocks)
        ])

        # FNO blocks after Perceiver
        if use_fno_blocks and n_fno > 0:
            self.post_fno_blocks = nn.ModuleList([
                FNOBlock(hidden_channels, modes1, modes2)
                for _ in range(n_fno - n_fno // 2)
            ])

        # Projection adapter
        self.projection = nn.Linear(hidden_channels, output_channels)

        # Learned latent array
        self.latent_init = nn.Parameter(
            torch.randn(1, n_latent, hidden_channels) * 0.02
        )

    def forward(self, x, grid=None):
        """
        Args:
            x: (batch, spatial_x, spatial_y, input_channels)
        Returns:
            (batch, spatial_x, spatial_y, output_channels)
        """
        batch, nx, ny, _ = x.shape

        # Lift
        v = self.lifting(x)  # (batch, nx, ny, hidden_channels)
        v = v.permute(0, 3, 1, 2)  # (batch, hidden, nx, ny)

        # FNO blocks before Perceiver
        if self.use_fno_blocks and hasattr(self, 'fno_blocks'):
            for block in self.fno_blocks:
                v = block(v)

        # Perceiver blocks
        latent = self.latent_init.expand(batch, -1, -1)
        for block in self.perceiver_blocks:
            v, latent = block(v, latent)

        # FNO blocks after Perceiver
        if self.use_fno_blocks and hasattr(self, 'post_fno_blocks'):
            for block in self.post_fno_blocks:
                v = block(v)

        # Project
        v = v.permute(0, 2, 3, 1)  # (batch, nx, ny, hidden_channels)
        out = self.projection(v)

        return out

    def get_lifting_params(self):
        params = list(self.lifting.parameters())
        if self.use_fno_blocks and hasattr(self, 'fno_blocks'):
            params += list(self.fno_blocks.parameters())
            if hasattr(self, 'post_fno_blocks'):
                params += list(self.post_fno_blocks.parameters())
        return params

    def get_projection_params(self):
        return list(self.projection.parameters())

    def get_core_params(self):
        params = list(self.perceiver_blocks.parameters())
        params.append(self.latent_init)
        return params
