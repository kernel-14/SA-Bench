"""
CoDA-NO: Codomain Attention Neural Operator.

Based on: Rahman et al., "Pretraining Codomain Attention Neural Operators for Solving
Multiphysics PDEs" (NeurIPS 2024).

From the paper:
"Codomain Attention Neural Operator (CoDA-NO), designed for multiphysics PDE transfer
learning, employs codomain attention with function space dot product."

"Codomain attention mechanisms, introduced in [13], are advantageous to the conventional
transformers in the neural-operator based problems: the dot product detecting similarity
not between samples, but between features, mapped with neural operators."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .fno import SpectralConv1d, SpectralConv2d


class CodomainAttention1d(nn.Module):
    """
    Codomain attention for 1D problems.
    
    Unlike standard attention that computes similarity between spatial positions,
    codomain attention computes similarity between feature channels (codomains),
    using function space dot products.
    """

    def __init__(self, d_model: int, n_heads: int = 8, modes: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # FNO-based projections for Q, K, V in function space
        self.fno_q = SpectralConv1d(d_model, d_model, modes)
        self.fno_k = SpectralConv1d(d_model, d_model, modes)
        self.fno_v = SpectralConv1d(d_model, d_model, modes)

        self.out_proj = nn.Conv1d(d_model, d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch, d_model, nx)
        
        Codomain attention: attention over channels (codomains) rather than spatial positions.
        The dot product is computed in function space (integrated over spatial domain).
        """
        batch, d_model, nx = x.shape

        # Compute Q, K, V via FNO (function space projections)
        Q = self.fno_q(x)  # (batch, d_model, nx)
        K = self.fno_k(x)  # (batch, d_model, nx)
        V = self.fno_v(x)  # (batch, d_model, nx)

        # Reshape for multi-head: (batch, n_heads, d_head, nx)
        Q = Q.reshape(batch, self.n_heads, self.d_head, nx)
        K = K.reshape(batch, self.n_heads, self.d_head, nx)
        V = V.reshape(batch, self.n_heads, self.d_head, nx)

        # Codomain attention: dot product over spatial dimension (function space inner product)
        # attn[b, h, i, j] = <Q[b,h,i,:], K[b,h,j,:]> / (sqrt(nx) * sqrt(d_head))
        # This computes similarity between feature channels, not spatial positions
        scale = math.sqrt(nx) * math.sqrt(self.d_head)
        attn = torch.einsum('bhix,bhjx->bhij', Q, K) / scale  # (batch, n_heads, d_head, d_head)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.einsum('bhij,bhjx->bhix', attn, V)  # (batch, n_heads, d_head, nx)
        out = out.reshape(batch, d_model, nx)

        out = self.out_proj(out)

        # Residual + norm
        out = out + x
        out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        return out


class CodomainAttention2d(nn.Module):
    """
    Codomain attention for 2D problems.
    """

    def __init__(self, d_model: int, n_heads: int = 8, modes: int = 12):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # FNO-based projections for Q, K, V in function space
        self.fno_q = SpectralConv2d(d_model, d_model, modes, modes)
        self.fno_k = SpectralConv2d(d_model, d_model, modes, modes)
        self.fno_v = SpectralConv2d(d_model, d_model, modes, modes)

        self.out_proj = nn.Conv2d(d_model, d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch, d_model, nx, ny)
        """
        batch, d_model, nx, ny = x.shape
        n_spatial = nx * ny

        # Compute Q, K, V via FNO
        Q = self.fno_q(x)  # (batch, d_model, nx, ny)
        K = self.fno_k(x)
        V = self.fno_v(x)

        # Reshape for multi-head: (batch, n_heads, d_head, nx*ny)
        Q = Q.reshape(batch, self.n_heads, self.d_head, n_spatial)
        K = K.reshape(batch, self.n_heads, self.d_head, n_spatial)
        V = V.reshape(batch, self.n_heads, self.d_head, n_spatial)

        # Codomain attention: dot product over spatial dimension
        scale = math.sqrt(n_spatial) * math.sqrt(self.d_head)
        attn = torch.einsum('bhix,bhjx->bhij', Q, K) / scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bhij,bhjx->bhix', attn, V)
        out = out.reshape(batch, d_model, nx, ny)

        out = self.out_proj(out)
        out = out + x
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out


class CoDANOBlock1d(nn.Module):
    """CoDA-NO block for 1D problems."""

    def __init__(self, width: int, modes: int, n_heads: int = 8):
        super().__init__()
        self.codomain_attn = CodomainAttention1d(width, n_heads, modes)
        self.w = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, x):
        # Codomain attention
        x = self.codomain_attn(x)
        # Local linear
        x = x + self.w(x)
        # FFN
        residual = x
        x = self.norm(x.permute(0, 2, 1))
        x = self.ffn(x)
        x = x.permute(0, 2, 1) + residual
        return x


class CoDANOBlock2d(nn.Module):
    """CoDA-NO block for 2D problems."""

    def __init__(self, width: int, modes: int, n_heads: int = 8):
        super().__init__()
        self.codomain_attn = CodomainAttention2d(width, n_heads, modes)
        self.w = nn.Conv2d(width, width, 1)
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, x):
        # Codomain attention
        x = self.codomain_attn(x)
        # Local linear
        x = x + self.w(x)
        # FFN
        residual = x
        x = self.norm(x.permute(0, 2, 3, 1))
        x = self.ffn(x)
        x = x.permute(0, 3, 1, 2) + residual
        return x


class CoDANO1d(nn.Module):
    """
    1D Codomain Attention Neural Operator.
    
    Architecture: Lifting -> CoDA-NO blocks -> Projection
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        width: int = 64,
        modes: int = 16,
        n_layers: int = 4,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.width = width

        # Lifting layer (adapter - problem specific)
        self.lifting = nn.Linear(n_input, width)

        # CoDA-NO blocks (shared backbone)
        self.coda_blocks = nn.ModuleList([
            CoDANOBlock1d(width, modes, n_heads) for _ in range(n_layers)
        ])

        # Projection layer (adapter - problem specific)
        self.projection = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, n_output),
        )

    def get_backbone_params(self):
        return list(self.coda_blocks.parameters())

    def get_adapter_params(self):
        return list(self.lifting.parameters()) + list(self.projection.parameters())

    def forward(self, x):
        # x: (batch, n_input, nx)
        x = x.permute(0, 2, 1)
        x = self.lifting(x)
        x = x.permute(0, 2, 1)  # (batch, width, nx)

        for block in self.coda_blocks:
            x = block(x)

        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = x.permute(0, 2, 1)  # (batch, n_output, nx)
        return x


class CoDANO2d(nn.Module):
    """
    2D Codomain Attention Neural Operator.
    
    Architecture: Lifting -> CoDA-NO blocks -> Projection
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        width: int = 32,
        modes: int = 12,
        n_layers: int = 4,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.width = width

        # Lifting layer (adapter - problem specific)
        self.lifting = nn.Linear(n_input, width)

        # CoDA-NO blocks (shared backbone)
        self.coda_blocks = nn.ModuleList([
            CoDANOBlock2d(width, modes, n_heads) for _ in range(n_layers)
        ])

        # Projection layer (adapter - problem specific)
        self.projection = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, n_output),
        )

    def get_backbone_params(self):
        return list(self.coda_blocks.parameters())

    def get_adapter_params(self):
        return list(self.lifting.parameters()) + list(self.projection.parameters())

    def forward(self, x):
        # x: (batch, n_input, nx, ny)
        x = x.permute(0, 2, 3, 1)
        x = self.lifting(x)
        x = x.permute(0, 3, 1, 2)  # (batch, width, nx, ny)

        for block in self.coda_blocks:
            x = block(x)

        x = x.permute(0, 2, 3, 1)
        x = self.projection(x)
        x = x.permute(0, 3, 1, 2)  # (batch, n_output, nx, ny)
        return x
