"""
Perceiver IO-based Neural Operator.

From the paper:
"The introduction of Perceiver enabled the encoding of information with a smaller number
of latent feature arrays, internal to operator blocks, thereby operating with more abstract
feature arrays and maintaining a limited number of parameters."

"The latent variables and input state are combined first with the cross-attention block,
where keys and values are obtained from FNO-based mapping from the inputs K1 = FNO_K1(X),
V1 = FNO_V1(X), and latent variables are taken as queries Q1 = L. The cross-attention block
is followed by self-attention between latent representation. The output of the block is
constructed with the cross-attention, matching the queries from the inputs with the keys
and values, taken from the transformed latent representations."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .fno import SpectralConv1d, SpectralConv2d, FNOBlock1d, FNOBlock2d


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        q: (batch, seq_q, d_model)
        k: (batch, seq_k, d_model)
        v: (batch, seq_v, d_model)
        """
        batch = q.shape[0]

        # Project
        q = self.q_proj(q).reshape(batch, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(k).reshape(batch, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(v).reshape(batch, -1, self.n_heads, self.d_head).transpose(1, 2)

        # Attention
        scale = math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, -1, self.d_model)
        out = self.out_proj(out)
        return out


class PerceiverIOBlock(nn.Module):
    """
    Perceiver IO block for neural operators.
    
    Implements the symmetric cross-attention mechanism:
    1. Cross-attention: latent queries attend to FNO-processed input (K, V)
    2. Self-attention: latent representations attend to each other
    3. Cross-attention decode: input queries attend to transformed latent (K, V)
    """

    def __init__(
        self,
        d_model: int,
        n_latent: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        modes: int = 12,
        dim: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latent = n_latent
        self.dim = dim

        # FNO-based key/value projections for input
        if dim == 1:
            self.fno_k = nn.Sequential(
                SpectralConv1d(d_model, d_model, modes),
                nn.Conv1d(d_model, d_model, 1),
            )
            self.fno_v = nn.Sequential(
                SpectralConv1d(d_model, d_model, modes),
                nn.Conv1d(d_model, d_model, 1),
            )
        else:
            self.fno_k = nn.Sequential(
                SpectralConv2d(d_model, d_model, modes, modes),
                nn.Conv2d(d_model, d_model, 1),
            )
            self.fno_v = nn.Sequential(
                SpectralConv2d(d_model, d_model, modes, modes),
                nn.Conv2d(d_model, d_model, 1),
            )

        # Cross-attention: latent queries attend to input K, V
        self.cross_attn_encode = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_encode_q = nn.LayerNorm(d_model)
        self.norm_encode_kv = nn.LayerNorm(d_model)

        # Self-attention on latent
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_self = nn.LayerNorm(d_model)

        # Cross-attention decode: input queries attend to latent K, V
        self.cross_attn_decode = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_decode_q = nn.LayerNorm(d_model)
        self.norm_decode_kv = nn.LayerNorm(d_model)

        # FFN for latent
        self.ffn_latent = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # FFN for output
        self.ffn_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Learnable latent array
        self.latent = nn.Parameter(torch.randn(1, n_latent, d_model) * 0.02)

    def forward(self, x):
        """
        x: (batch, d_model, *spatial) - input features
        returns: (batch, d_model, *spatial) - output features
        """
        if self.dim == 1:
            batch, d_model, nx = x.shape
            # FNO-based K, V from input
            K = self.fno_k(x)  # (batch, d_model, nx)
            V = self.fno_v(x)  # (batch, d_model, nx)
            # Reshape to sequence: (batch, nx, d_model)
            K = K.permute(0, 2, 1)
            V = V.permute(0, 2, 1)
            x_seq = x.permute(0, 2, 1)  # (batch, nx, d_model)
        else:
            batch, d_model, nx, ny = x.shape
            # FNO-based K, V from input
            K = self.fno_k(x)  # (batch, d_model, nx, ny)
            V = self.fno_v(x)  # (batch, d_model, nx, ny)
            # Reshape to sequence: (batch, nx*ny, d_model)
            K = K.permute(0, 2, 3, 1).reshape(batch, nx * ny, d_model)
            V = V.permute(0, 2, 3, 1).reshape(batch, nx * ny, d_model)
            x_seq = x.permute(0, 2, 3, 1).reshape(batch, nx * ny, d_model)

        # Expand latent for batch
        L = self.latent.expand(batch, -1, -1)  # (batch, n_latent, d_model)

        # Encode: cross-attention (latent queries, input K/V)
        L_norm = self.norm_encode_q(L)
        K_norm = self.norm_encode_kv(K)
        L = L + self.cross_attn_encode(L_norm, K_norm, V)

        # Self-attention on latent
        L_norm = self.norm_self(L)
        L = L + self.self_attn(L_norm, L_norm, L_norm)
        L = L + self.ffn_latent(L)

        # Decode: cross-attention (input queries, latent K/V)
        x_norm = self.norm_decode_q(x_seq)
        L_norm = self.norm_decode_kv(L)
        out = x_seq + self.cross_attn_decode(x_norm, L_norm, L_norm)
        out = out + self.ffn_out(out)

        # Reshape back to spatial format
        if self.dim == 1:
            out = out.permute(0, 2, 1)  # (batch, d_model, nx)
        else:
            out = out.reshape(batch, nx, ny, d_model).permute(0, 3, 1, 2)

        return out


class PerceiverFNO1d(nn.Module):
    """
    1D Perceiver IO-based Neural Operator.
    
    Architecture: Lifting -> Perceiver IO blocks -> Projection
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        width: int = 64,
        modes: int = 16,
        n_layers: int = 4,
        n_latent: int = 64,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.width = width

        # Lifting layer (adapter - problem specific)
        self.lifting = nn.Linear(n_input, width)

        # Perceiver IO blocks (shared backbone)
        self.perceiver_blocks = nn.ModuleList([
            PerceiverIOBlock(
                d_model=width,
                n_latent=n_latent,
                n_heads=n_heads,
                dropout=dropout,
                modes=modes,
                dim=1,
            )
            for _ in range(n_layers)
        ])

        # Projection layer (adapter - problem specific)
        self.projection = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, n_output),
        )

    def get_backbone_params(self):
        """Return parameters of the shared backbone."""
        return list(self.perceiver_blocks.parameters())

    def get_adapter_params(self):
        """Return parameters of the problem-specific adapters."""
        return list(self.lifting.parameters()) + list(self.projection.parameters())

    def forward(self, x):
        # x: (batch, n_input, nx)
        x = x.permute(0, 2, 1)
        x = self.lifting(x)
        x = x.permute(0, 2, 1)  # (batch, width, nx)

        for block in self.perceiver_blocks:
            x = block(x)

        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = x.permute(0, 2, 1)  # (batch, n_output, nx)
        return x


class PerceiverFNO2d(nn.Module):
    """
    2D Perceiver IO-based Neural Operator.
    
    Architecture: Lifting -> Perceiver IO blocks -> Projection
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        width: int = 32,
        modes: int = 12,
        n_layers: int = 4,
        n_latent: int = 64,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.width = width

        # Lifting layer (adapter - problem specific)
        self.lifting = nn.Linear(n_input, width)

        # Perceiver IO blocks (shared backbone)
        self.perceiver_blocks = nn.ModuleList([
            PerceiverIOBlock(
                d_model=width,
                n_latent=n_latent,
                n_heads=n_heads,
                dropout=dropout,
                modes=modes,
                dim=2,
            )
            for _ in range(n_layers)
        ])

        # Projection layer (adapter - problem specific)
        self.projection = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, n_output),
        )

    def get_backbone_params(self):
        """Return parameters of the shared backbone."""
        return list(self.perceiver_blocks.parameters())

    def get_adapter_params(self):
        """Return parameters of the problem-specific adapters."""
        return list(self.lifting.parameters()) + list(self.projection.parameters())

    def forward(self, x):
        # x: (batch, n_input, nx, ny)
        x = x.permute(0, 2, 3, 1)
        x = self.lifting(x)
        x = x.permute(0, 3, 1, 2)  # (batch, width, nx, ny)

        for block in self.perceiver_blocks:
            x = block(x)

        x = x.permute(0, 2, 3, 1)
        x = self.projection(x)
        x = x.permute(0, 3, 1, 2)  # (batch, n_output, nx, ny)
        return x
