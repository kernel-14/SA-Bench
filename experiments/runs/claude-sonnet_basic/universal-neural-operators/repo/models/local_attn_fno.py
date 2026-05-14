"""
LocalAttnFNO: FNO with local attention blocks (post-lifting).

This implements the "post-lifting (PL) LocalAttnFNO" mentioned in the paper.
Local attention is applied after the lifting layer to capture local spatial dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .fno import SpectralConv1d, SpectralConv2d, FNOBlock1d, FNOBlock2d


class LocalAttention1d(nn.Module):
    """
    Local attention for 1D sequences.
    Attends to a local window around each position.
    """

    def __init__(self, d_model: int, n_heads: int = 8, window_size: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch, d_model, nx)
        """
        batch, d_model, nx = x.shape
        x_seq = x.permute(0, 2, 1)  # (batch, nx, d_model)

        residual = x_seq
        x_norm = self.norm(x_seq)

        Q = self.q_proj(x_norm).reshape(batch, nx, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x_norm).reshape(batch, nx, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x_norm).reshape(batch, nx, self.n_heads, self.d_head).transpose(1, 2)

        # Local attention with window
        w = self.window_size
        scale = math.sqrt(self.d_head)

        # Pad for windowed attention
        pad = w // 2
        K_pad = F.pad(K, (0, 0, pad, pad))
        V_pad = F.pad(V, (0, 0, pad, pad))

        # Gather local windows
        out = torch.zeros_like(Q)
        for i in range(nx):
            k_local = K_pad[:, :, i:i + w, :]  # (batch, n_heads, w, d_head)
            v_local = V_pad[:, :, i:i + w, :]
            q_i = Q[:, :, i:i + 1, :]  # (batch, n_heads, 1, d_head)
            attn = torch.matmul(q_i, k_local.transpose(-2, -1)) / scale
            attn = F.softmax(attn, dim=-1)
            out[:, :, i:i + 1, :] = torch.matmul(attn, v_local)

        out = out.transpose(1, 2).reshape(batch, nx, d_model)
        out = self.out_proj(out)
        out = out + residual

        return out.permute(0, 2, 1)  # (batch, d_model, nx)


class LocalAttention2d(nn.Module):
    """
    Local attention for 2D grids using window-based attention.
    """

    def __init__(self, d_model: int, n_heads: int = 8, window_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch, d_model, nx, ny)
        """
        batch, d_model, nx, ny = x.shape
        w = self.window_size

        # Pad to be divisible by window size
        pad_x = (w - nx % w) % w
        pad_y = (w - ny % w) % w
        x_pad = F.pad(x, (0, pad_y, 0, pad_x))
        _, _, nx_pad, ny_pad = x_pad.shape

        # Reshape to windows: (batch * num_windows, window_size^2, d_model)
        x_seq = x_pad.permute(0, 2, 3, 1)  # (batch, nx_pad, ny_pad, d_model)
        x_seq = x_seq.reshape(
            batch, nx_pad // w, w, ny_pad // w, w, d_model
        ).permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, d_model)

        residual = x_seq
        x_norm = self.norm(x_seq)

        bw = x_seq.shape[0]
        Q = self.q_proj(x_norm).reshape(bw, w * w, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x_norm).reshape(bw, w * w, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x_norm).reshape(bw, w * w, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).reshape(bw, w * w, d_model)
        out = self.out_proj(out)
        out = out + residual

        # Reshape back to spatial
        out = out.reshape(
            batch, nx_pad // w, ny_pad // w, w, w, d_model
        ).permute(0, 1, 3, 2, 4, 5).reshape(batch, nx_pad, ny_pad, d_model)

        # Remove padding
        out = out[:, :nx, :ny, :]
        return out.permute(0, 3, 1, 2)  # (batch, d_model, nx, ny)


class LocalAttnFNO1d(nn.Module):
    """
    1D FNO with local attention after lifting (post-lifting LocalAttnFNO).
    
    Architecture: Lifting -> Local Attention -> FNO blocks -> Projection
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        width: int = 64,
        modes: int = 16,
        n_layers: int = 4,
        n_heads: int = 8,
        window_size: int = 16,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.width = width

        # Lifting layer (adapter - problem specific)
        self.lifting = nn.Linear(n_input, width)

        # Local attention (post-lifting, part of backbone)
        self.local_attn = LocalAttention1d(width, n_heads, window_size)

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
        return list(self.local_attn.parameters()) + list(self.fno_blocks.parameters())

    def get_adapter_params(self):
        return list(self.lifting.parameters()) + list(self.projection.parameters())

    def forward(self, x):
        # x: (batch, n_input, nx)
        x = x.permute(0, 2, 1)
        x = self.lifting(x)
        x = x.permute(0, 2, 1)  # (batch, width, nx)

        # Post-lifting local attention
        x = self.local_attn(x)

        for block in self.fno_blocks:
            x = block(x)

        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = x.permute(0, 2, 1)  # (batch, n_output, nx)
        return x


class LocalAttnFNO2d(nn.Module):
    """
    2D FNO with local attention after lifting (post-lifting LocalAttnFNO).
    
    Architecture: Lifting -> Local Attention -> FNO blocks -> Projection
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 4,
        n_heads: int = 8,
        window_size: int = 4,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.width = width

        # Lifting layer (adapter - problem specific)
        self.lifting = nn.Linear(n_input, width)

        # Local attention (post-lifting, part of backbone)
        self.local_attn = LocalAttention2d(width, n_heads, window_size)

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
        return list(self.local_attn.parameters()) + list(self.fno_blocks.parameters())

    def get_adapter_params(self):
        return list(self.lifting.parameters()) + list(self.projection.parameters())

    def forward(self, x):
        # x: (batch, n_input, nx, ny)
        x = x.permute(0, 2, 3, 1)
        x = self.lifting(x)
        x = x.permute(0, 3, 1, 2)  # (batch, width, nx, ny)

        # Post-lifting local attention
        x = self.local_attn(x)

        for block in self.fno_blocks:
            x = block(x)

        x = x.permute(0, 2, 3, 1)
        x = self.projection(x)
        x = x.permute(0, 3, 1, 2)  # (batch, n_output, nx, ny)
        return x
