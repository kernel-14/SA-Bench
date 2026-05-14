"""
MambaFNO: FNO with Mamba SSM module inserted after the lifting layer.

From the paper:
"inserting a Mamba-SSM module M_phi after the lifting map L allows the model to encode
long-range temporal and spatial dependencies directly in the hidden representation."

The Mamba module acts as a latent preconditioner, aligning embeddings with dominant
dynamical motifs (transport, diffusion, oscillation) common across PDEs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .fno import SpectralConv1d, SpectralConv2d, FNOBlock1d, FNOBlock2d


class MambaSSM(nn.Module):
    """
    Simplified Mamba-style State Space Model module.
    
    Implements the selective state space model from:
    Gu & Dao, "Mamba: Linear-time sequence modeling with selective state spaces"
    
    The module computes:
        v_tilde(x, t) = sum_{tau <= t} K_tau * v_0(x, t - tau)
    
    with learnable convolution kernels K_tau defining the causal recurrence.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal convolution (implements the K_tau kernels)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Initialize dt_proj bias
        dt_init_std = self.d_inner ** -0.5 * 3
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # A matrix (log parameterization for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def ssm_scan(self, u, delta, A, B, C):
        """
        Selective scan: implements the recurrence
        h_t = A_bar * h_{t-1} + B_bar * u_t
        y_t = C * h_t
        """
        batch, d_inner, seq_len = u.shape
        d_state = A.shape[1]

        # Discretize A and B using ZOH
        # delta: (batch, d_inner, seq_len)
        # A: (d_inner, d_state)
        delta_A = torch.exp(
            torch.einsum('bds,dn->bdsn', delta, A)
        )  # (batch, d_inner, seq_len, d_state)

        delta_B_u = torch.einsum('bds,bns,bds->bdsn', delta, B, u)
        # (batch, d_inner, seq_len, d_state)

        # Sequential scan
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(seq_len):
            h = delta_A[:, :, i, :] * h + delta_B_u[:, :, i, :]
            y = torch.einsum('bdn,bdn->bd', h, C[:, :, i, :])
            ys.append(y)

        y = torch.stack(ys, dim=2)  # (batch, d_inner, seq_len)
        return y

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        batch, seq_len, d_model = x.shape

        # Input projection
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_ssm, z = xz.chunk(2, dim=-1)  # each (batch, seq_len, d_inner)

        # Causal convolution
        x_ssm = x_ssm.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_ssm = self.conv1d(x_ssm)[:, :, :seq_len]  # causal truncation
        x_ssm = F.silu(x_ssm)

        # SSM parameters (input-dependent for selectivity)
        x_dbl = self.x_proj(x_ssm.transpose(1, 2))  # (batch, seq_len, d_state*2 + d_inner)
        dt, B, C = x_dbl.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (batch, seq_len, d_inner)

        # Reshape for scan
        dt = dt.transpose(1, 2)  # (batch, d_inner, seq_len)
        B = B.transpose(1, 2).unsqueeze(1).expand(-1, self.d_inner, -1, -1)
        # (batch, d_inner, seq_len, d_state)
        C = C.transpose(1, 2).unsqueeze(1).expand(-1, self.d_inner, -1, -1)
        # (batch, d_inner, seq_len, d_state)

        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Run SSM
        y = self.ssm_scan(x_ssm, dt, A, B, C)  # (batch, d_inner, seq_len)

        # Add D skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * x_ssm

        # Gate with z
        y = y * F.silu(z.transpose(1, 2))  # (batch, d_inner, seq_len)

        # Output projection
        y = self.out_proj(y.transpose(1, 2))  # (batch, seq_len, d_model)

        return y + residual


class MambaFNO1d(nn.Module):
    """
    1D MambaFNO: FNO with Mamba SSM module inserted after lifting.
    
    Architecture: Lifting -> Mamba SSM -> FNO blocks -> Projection
    
    The Mamba module acts as a latent preconditioner that aligns embeddings
    with dominant dynamical motifs common across PDEs.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        width: int = 64,
        modes: int = 16,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.width = width

        # Lifting layer (adapter - problem specific)
        self.lifting = nn.Linear(n_input, width)

        # Mamba SSM module (post-lifting, part of backbone)
        self.mamba = MambaSSM(
            d_model=width,
            d_state=d_state,
            d_conv=d_conv,
            expand=mamba_expand,
        )

        # FNO blocks (shared backbone)
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(width, modes) for _ in range(n_layers)
        ])

        # Projection layer (adapter - problem specific)
        self.projection = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, n_output),
        )

    def get_backbone_params(self):
        """Return parameters of the shared backbone (Mamba + FNO blocks)."""
        return list(self.mamba.parameters()) + list(self.fno_blocks.parameters())

    def get_adapter_params(self):
        """Return parameters of the problem-specific adapters."""
        return list(self.lifting.parameters()) + list(self.projection.parameters())

    def forward(self, x):
        # x: (batch, n_input, nx)
        batch, n_input, nx = x.shape

        # Lifting: (batch, nx, n_input) -> (batch, nx, width)
        x = x.permute(0, 2, 1)
        x = self.lifting(x)

        # Mamba SSM: (batch, nx, width) -> (batch, nx, width)
        x = self.mamba(x)

        # Back to (batch, width, nx) for FNO blocks
        x = x.permute(0, 2, 1)

        for block in self.fno_blocks:
            x = block(x)

        # Projection: (batch, nx, width) -> (batch, nx, n_output)
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        # Back to (batch, n_output, nx)
        x = x.permute(0, 2, 1)
        return x


class MambaFNO2d(nn.Module):
    """
    2D MambaFNO: FNO with Mamba SSM module inserted after lifting.
    
    For 2D problems, the spatial dimensions are flattened for the Mamba module.
    Architecture: Lifting -> Mamba SSM -> FNO blocks -> Projection
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.width = width

        # Lifting layer (adapter - problem specific)
        self.lifting = nn.Linear(n_input, width)

        # Mamba SSM module (post-lifting, part of backbone)
        self.mamba = MambaSSM(
            d_model=width,
            d_state=d_state,
            d_conv=d_conv,
            expand=mamba_expand,
        )

        # FNO blocks (shared backbone)
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(width, modes1, modes2) for _ in range(n_layers)
        ])

        # Projection layer (adapter - problem specific)
        self.projection = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, n_output),
        )

    def get_backbone_params(self):
        """Return parameters of the shared backbone (Mamba + FNO blocks)."""
        return list(self.mamba.parameters()) + list(self.fno_blocks.parameters())

    def get_adapter_params(self):
        """Return parameters of the problem-specific adapters."""
        return list(self.lifting.parameters()) + list(self.projection.parameters())

    def forward(self, x):
        # x: (batch, n_input, nx, ny)
        batch, n_input, nx, ny = x.shape

        # Lifting: (batch, nx, ny, n_input) -> (batch, nx, ny, width)
        x = x.permute(0, 2, 3, 1)
        x = self.lifting(x)

        # Flatten spatial dims for Mamba: (batch, nx*ny, width)
        x = x.reshape(batch, nx * ny, self.width)
        x = self.mamba(x)
        x = x.reshape(batch, nx, ny, self.width)

        # Back to (batch, width, nx, ny) for FNO blocks
        x = x.permute(0, 3, 1, 2)

        for block in self.fno_blocks:
            x = block(x)

        # Projection: (batch, nx, ny, width) -> (batch, nx, ny, n_output)
        x = x.permute(0, 2, 3, 1)
        x = self.projection(x)
        # Back to (batch, n_output, nx, ny)
        x = x.permute(0, 3, 1, 2)
        return x
