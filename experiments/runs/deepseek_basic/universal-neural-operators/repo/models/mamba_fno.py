"""MambaFNO: Neural operator with post-lifting Mamba SSM module.

As described in Section 3 of the paper:
The Mamba SSM module is inserted after the lifting map to encode long-range
temporal and spatial dependencies directly in the hidden representation.
The Mamba module acts as a latent preconditioner: embeddings are aligned with
dominant dynamical motifs (transport, diffusion, oscillation) common across PDEs.

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .fno import SpectralConv2d, FNOBlock


class MambaSSM(nn.Module):
    """
    Simplified Mamba SSM module for 2D spatial data.

    Processes spatial dimensions as sequences using selective state-space models.
    The module applies S4/Mamba-style recurrence along spatial axes to capture
    long-range dependencies.

    As described in the paper, this acts after the lifting:
    v_tilde_0(x, t) = (M_phi v_0)(x, t) = sum_{tau <= t} K_tau v_0(x, t - tau)
    with learnable convolution kernels K_tau defining the causal recurrence.
    """

    def __init__(
        self,
        hidden_channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.d_state = d_state

        # Expand factor for inner dimension
        d_inner = int(hidden_channels * expand)
        self.d_inner = d_inner
        dt_rank = math.ceil(hidden_channels / 16)

        # Projections
        self.in_proj = nn.Linear(hidden_channels, d_inner * 2)  # x and z

        # 1D convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )

        # SSM parameters: x_proj produces dt, B, C
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize A (state matrix) and D (skip connection)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, hidden_channels)

    def _selective_scan(self, u, delta, A, B, C, D):
        """Selective scan operation (simplified S4 recurrence).

        Args:
            u: (batch, length, d_inner)
            delta: (batch, length, d_inner)
            A: (d_inner, d_state) — state matrix (negative for stability)
            B: (batch, length, d_state)
            C: (batch, length, d_state)
            D: (d_inner,) — skip connection
        Returns:
            y: (batch, length, d_inner)
        """
        batch, length, d_inner = u.shape
        d_state = A.shape[1]

        # Discretize A and B
        delta = F.softplus(delta)  # Ensure positive, (batch, length, d_inner)
        # deltaA: (batch, length, d_inner, d_state)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # deltaB_u: (batch, length, d_inner, d_state)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        # Scan over time dimension
        x = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        outputs = []
        for t in range(length):
            x = deltaA[:, t] * x + deltaB_u[:, t]
            # C[:, t]: (batch, d_state)
            y_t = (x * C[:, t].unsqueeze(1)).sum(dim=-1)  # (batch, d_inner)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)  # (batch, length, d_inner)

        return y + u * D.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x: (batch, spatial_x, spatial_y, hidden_channels)
        Returns:
            Output of same shape
        """
        batch, nx, ny, h = x.shape
        seq_len = nx * ny

        # Flatten spatial dims into sequence
        x_flat = x.reshape(batch, seq_len, h)

        # Input projection: split into projection and gating
        x_and_res = self.in_proj(x_flat)
        x_proj, z = x_and_res.split(self.d_inner, dim=-1)

        # Convolution along sequence dimension
        x_conv = x_proj.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)

        # SSM: produce dt, B, C from the convolved features
        x_proj_out = self.x_proj(x_conv)
        dt, B, C = torch.split(
            x_proj_out,
            [self.dt_proj.in_features, self.d_state, self.d_state],
            dim=-1
        )
        dt = self.dt_proj(dt)  # (batch, seq_len, d_inner)

        # Selective scan
        y = self._selective_scan(
            x_conv, dt,
            -torch.exp(self.A_log.float()),
            B, C, self.D,
        )

        # Gating with z (SiLU activation)
        y = y * F.silu(z)
        y = self.out_proj(y)

        # Reshape back to 2D spatial
        y = y.reshape(batch, nx, ny, h)
        return y


class MambaFNO(nn.Module):
    """MambaFNO: FNO with post-lifting Mamba SSM for improved generalization.

    Architecture (Section 3):
    Input -> Lifting -> Mamba SSM -> FNO Blocks -> Projection -> Output

    The Mamba module acts as a latent preconditioner, aligning embeddings with
    dominant dynamical motifs common across PDEs, enabling more stable training
    and efficient transfer of pretrained representations.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 4,
        modes1: int = 12,
        modes2: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels

        # Lifting adapter
        self.lifting = nn.Linear(input_channels, hidden_channels)

        # Post-lifting Mamba SSM
        self.mamba = MambaSSM(
            hidden_channels=hidden_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # FNO core blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock(hidden_channels, modes1, modes2)
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
        v = self.lifting(x)  # (batch, nx, ny, hidden_channels)

        # Post-lifting Mamba SSM
        v = self.mamba(v)  # (batch, nx, ny, hidden_channels)

        # FNO blocks
        v = v.permute(0, 3, 1, 2)  # (batch, hidden, nx, ny)
        for block in self.fno_blocks:
            v = block(v)
        v = v.permute(0, 2, 3, 1)  # (batch, nx, ny, hidden)

        # Project
        out = self.projection(v)
        return out

    def get_lifting_params(self):
        return list(self.lifting.parameters())

    def get_projection_params(self):
        return list(self.projection.parameters())

    def get_mamba_params(self):
        return list(self.mamba.parameters())

    def get_core_params(self):
        return list(self.fno_blocks.parameters())
