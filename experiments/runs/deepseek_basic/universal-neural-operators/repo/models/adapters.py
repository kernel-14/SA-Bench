"""Problem-specific adapter layers for multiphysics pretraining.

As described in Section 3 ("Pre-training and fine-tuning"):
The lift and proj blocks are considered as the adapters, representing the
mappings associated with the problem-specific part of dynamics: they are
introduced to contain different cardinality input sets, projecting into the
fixed number of hidden features and contain small number of parameters to
represent limited part of the total model variance.

In the fine-tuning stage, only the new adapter parameters are trained while
the core operator parameters θ_F are fixed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LiftAdapter(nn.Module):
    """Problem-specific lifting adapter.

    Maps from a variable number of input functions to a fixed hidden dimension.
    Different PDE problems have different input sets (e.g., initial conditions,
    forcing terms, coefficients, mesh coordinates), and this adapter handles
    that variability.

    Analogous to LoRA adapters in language models [19].
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        use_positional_encoding: bool = True,
        activation=F.gelu,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        layers = []
        layers.append(nn.Linear(input_channels, hidden_channels))
        layers.append(activation)
        layers.append(nn.Linear(hidden_channels, hidden_channels))

        self.net = nn.Sequential(*layers)
        self.use_positional_encoding = use_positional_encoding

    def forward(self, x, grid=None):
        """
        Args:
            x: (batch, spatial_x, spatial_y, input_channels)
            grid: Optional grid coordinates (batch, spatial_x, spatial_y, 2)
        Returns:
            (batch, spatial_x, spatial_y, hidden_channels)
        """
        if self.use_positional_encoding and grid is not None:
            x = torch.cat([x, grid], dim=-1)
        return self.net(x)


class ProjAdapter(nn.Module):
    """Problem-specific projection adapter.

    Maps from fixed hidden dimension to the required number of output functions.
    Different PDE problems have different numbers of output variables, and this
    adapter handles that.
    """

    def __init__(
        self,
        hidden_channels: int,
        output_channels: int,
        activation=F.gelu,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            activation,
            nn.Linear(hidden_channels, output_channels),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, spatial_x, spatial_y, hidden_channels)
        Returns:
            (batch, spatial_x, spatial_y, output_channels)
        """
        return self.net(x)


class LocalAttnFNO(nn.Module):
    """Post-lifting Local Attention FNO (mentioned as PL LocalAttnFNO in paper).

    This variant uses local windowed attention after lifting instead of the
    Mamba SSM. Provides another way to encode local dependencies.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 4,
        modes1: int = 12,
        modes2: int = 12,
        window_size: int = 8,
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels

        from .fno import FNOBlock

        # Lifting adapter
        self.lifting = LiftAdapter(input_channels, hidden_channels)

        # Local windowed attention
        self.local_attn = LocalWindowAttention(hidden_channels, window_size, num_heads)

        # FNO core
        self.fno_blocks = nn.ModuleList([
            FNOBlock(hidden_channels, modes1, modes2)
            for _ in range(n_layers)
        ])

        # Projection adapter
        self.projection = ProjAdapter(hidden_channels, output_channels)

    def forward(self, x, grid=None):
        batch, nx, ny, _ = x.shape

        v = self.lifting(x, grid)
        v = v.permute(0, 3, 1, 2)  # (batch, hidden, nx, ny)
        v = self.local_attn(v)

        for block in self.fno_blocks:
            v = block(v)

        v = v.permute(0, 2, 3, 1)  # (batch, nx, ny, hidden)
        out = self.projection(v)
        return out

    def get_lifting_params(self):
        return list(self.lifting.parameters()) + list(self.local_attn.parameters())

    def get_projection_params(self):
        return list(self.projection.parameters())

    def get_core_params(self):
        return list(self.fno_blocks.parameters())


class LocalWindowAttention(nn.Module):
    """Local windowed attention for spatial feature maps."""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        """Simple local attention via unfold."""
        B, C, H, W = x.shape

        qkv = self.qkv(x)
        Q, K, V = qkv.chunk(3, dim=1)

        # Reshape to attention format
        Q = Q.reshape(B, self.num_heads, self.head_dim, H, W)
        K = K.reshape(B, self.num_heads, self.head_dim, H, W)
        V = V.reshape(B, self.num_heads, self.head_dim, H, W)

        # Unfold into windows
        pad = self.window_size // 2
        K_unfold = F.unfold(
            K.reshape(B * self.num_heads, self.head_dim, H, W),
            kernel_size=self.window_size,
            padding=pad,
        )
        V_unfold = F.unfold(
            V.reshape(B * self.num_heads, self.head_dim, H, W),
            kernel_size=self.window_size,
            padding=pad,
        )

        K_unfold = K_unfold.reshape(
            B, self.num_heads, self.head_dim * self.window_size * self.window_size, H * W
        )
        V_unfold = V_unfold.reshape(
            B, self.num_heads, self.head_dim * self.window_size * self.window_size, H * W
        )

        Q_flat = Q.reshape(B, self.num_heads, self.head_dim, H * W)

        attn = torch.matmul(
            Q_flat.transpose(-2, -1).unsqueeze(-2),  # (B, heads, HW, 1, head_dim)
            K_unfold.reshape(B, self.num_heads, self.window_size * self.window_size, self.head_dim, H * W)
            .permute(0, 1, 4, 2, 3),  # (B, heads, HW, ws^2, head_dim)
        ).squeeze(-2)  # (B, heads, HW, ws^2)

        attn = attn / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(
            attn.unsqueeze(-2),
            V_unfold.reshape(B, self.num_heads, self.window_size * self.window_size, self.head_dim, H * W)
            .permute(0, 1, 4, 2, 3),
        ).squeeze(-2)  # (B, heads, HW, head_dim)

        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)
        return x + out
