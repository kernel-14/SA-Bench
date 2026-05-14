"""
Multi-head Fourier Layer for MoE-POT.

Implements the multi-head Fourier layer described in the paper (Section 4).
Each head processes a subset of channels in the frequency domain using a
2-layer MLP with complex weights:

    z_0i^l(x) = F^{-1}[ W_{2,i} * sigma(W_{1,i} * F[z_i^l] + b_{1,i}) + b_{2,i} ](x)

The spatial features are split into h groups along the channel dimension,
and each group is processed independently in the Fourier domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierLayer(nn.Module):
    """
    Multi-head Fourier layer that applies frequency-domain transformations.

    Args:
        dim: Total feature dimension (d_z).
        num_heads: Number of heads (h). Each head processes dim//num_heads channels.
        modes: Number of Fourier modes to keep in each spatial dimension.
        activation: Activation function name ('gelu' or 'relu').
    """

    def __init__(self, dim: int, num_heads: int = 4, modes: int = 16, activation: str = "gelu"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.modes = modes
        self.head_dim = dim // num_heads

        # Per-head frequency-domain MLP weights (complex, stored as real+imag pairs)
        # Shape: (num_heads, modes, modes, head_dim, head_dim)
        scale = 0.02
        self.W1_real = nn.Parameter(torch.randn(num_heads, modes, modes, self.head_dim, self.head_dim) * scale)
        self.W1_imag = nn.Parameter(torch.randn(num_heads, modes, modes, self.head_dim, self.head_dim) * scale)
        self.W2_real = nn.Parameter(torch.randn(num_heads, modes, modes, self.head_dim, self.head_dim) * scale)
        self.W2_imag = nn.Parameter(torch.randn(num_heads, modes, modes, self.head_dim, self.head_dim) * scale)
        self.b1_real = nn.Parameter(torch.zeros(num_heads, modes, modes, self.head_dim))
        self.b1_imag = nn.Parameter(torch.zeros(num_heads, modes, modes, self.head_dim))
        self.b2_real = nn.Parameter(torch.zeros(num_heads, modes, modes, self.head_dim))
        self.b2_imag = nn.Parameter(torch.zeros(num_heads, modes, modes, self.head_dim))

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            self.activation = F.gelu

        # Layer norm applied before the Fourier transform
        self.norm = nn.LayerNorm(dim)

    def _complex_matmul(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
    ):
        """
        Complex matrix multiplication: y = x @ W where x, W are complex.
        (x_real + i*x_imag) @ (w_real + i*w_imag)
        = x_real @ w_real - x_imag @ w_imag + i*(x_real @ w_imag + x_imag @ w_real)

        Args:
            x_real, x_imag: (..., in_dim)
            w_real, w_imag: (..., in_dim, out_dim)

        Returns:
            out_real, out_imag: (..., out_dim)
        """
        out_real = torch.matmul(x_real, w_real) - torch.matmul(x_imag, w_imag)
        out_imag = torch.matmul(x_real, w_imag) + torch.matmul(x_imag, w_real)
        return out_real, out_imag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, H, W, C) where C = dim.

        Returns:
            Output tensor of shape (B, H, W, C).
        """
        B, H, W, C = x.shape
        residual = x
        x = self.norm(x)

        # Split into heads: (B, H, W, num_heads, head_dim)
        x_heads = x.reshape(B, H, W, self.num_heads, self.head_dim)

        # Apply 2D FFT on spatial dimensions (H, W)
        # x_heads: (B, H, W, num_heads, head_dim)
        x_fft = torch.fft.rfft2(x_heads, dim=(1, 2), norm="ortho")
        # x_fft shape: (B, H, W//2+1, num_heads, head_dim)

        # Truncate to modes
        modes_h = min(self.modes, H // 2)
        modes_w = min(self.modes, W // 2 + 1)

        # Extract low-frequency components
        x_low = x_fft[:, :modes_h, :modes_w, :, :]  # (B, modes_h, modes_w, num_heads, head_dim)
        x_low_real = x_low.real
        x_low_imag = x_low.imag

        # Prepare weight matrices (truncated to actual modes)
        # W1: (num_heads, modes_h, modes_w, head_dim, head_dim)
        W1_r = self.W1_real[:, :modes_h, :modes_w, :, :]
        W1_i = self.W1_imag[:, :modes_h, :modes_w, :, :]
        W2_r = self.W2_real[:, :modes_h, :modes_w, :, :]
        W2_i = self.W2_imag[:, :modes_h, :modes_w, :, :]
        b1_r = self.b1_real[:, :modes_h, :modes_w, :]  # (num_heads, modes_h, modes_w, head_dim)
        b1_i = self.b1_imag[:, :modes_h, :modes_w, :]
        b2_r = self.b2_real[:, :modes_h, :modes_w, :]
        b2_i = self.b2_imag[:, :modes_h, :modes_w, :]

        # Permute weights to (modes_h, modes_w, num_heads, head_dim, head_dim)
        # for broadcasting with x_low: (B, modes_h, modes_w, num_heads, head_dim)
        W1_r_p = W1_r.permute(1, 2, 0, 3, 4)  # (modes_h, modes_w, num_heads, head_dim, head_dim)
        W1_i_p = W1_i.permute(1, 2, 0, 3, 4)
        W2_r_p = W2_r.permute(1, 2, 0, 3, 4)
        W2_i_p = W2_i.permute(1, 2, 0, 3, 4)
        b1_r_p = b1_r.permute(1, 2, 0, 3)  # (modes_h, modes_w, num_heads, head_dim)
        b1_i_p = b1_i.permute(1, 2, 0, 3)
        b2_r_p = b2_r.permute(1, 2, 0, 3)
        b2_i_p = b2_i.permute(1, 2, 0, 3)

        # Layer 1: h1 = sigma(W1 * x_low + b1)
        # x_low_real: (B, modes_h, modes_w, num_heads, head_dim)
        # W1_r_p: (modes_h, modes_w, num_heads, head_dim, head_dim)
        # matmul broadcasts: (..., head_dim) @ (..., head_dim, head_dim) -> (..., head_dim)
        h1_real, h1_imag = self._complex_matmul(x_low_real, x_low_imag, W1_r_p, W1_i_p)
        h1_real = h1_real + b1_r_p
        h1_imag = h1_imag + b1_i_p

        # Apply activation to real and imaginary parts separately
        h1_real = self.activation(h1_real)
        h1_imag = self.activation(h1_imag)

        # Layer 2: out = W2 * h1 + b2
        out_real, out_imag = self._complex_matmul(h1_real, h1_imag, W2_r_p, W2_i_p)
        out_real = out_real + b2_r_p
        out_imag = out_imag + b2_i_p

        # Reconstruct complex tensor
        out_low = torch.complex(out_real, out_imag)

        # Place back into full frequency grid (zero-pad high frequencies)
        x_out_fft = torch.zeros_like(x_fft)
        x_out_fft[:, :modes_h, :modes_w, :, :] = out_low

        # Inverse FFT: (B, H, W, num_heads, head_dim)
        x_out = torch.fft.irfft2(x_out_fft, s=(H, W), dim=(1, 2), norm="ortho")

        # Merge heads back: (B, H, W, C)
        x_out = x_out.reshape(B, H, W, C)

        return x_out + residual
