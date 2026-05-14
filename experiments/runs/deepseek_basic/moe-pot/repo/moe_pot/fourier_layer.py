"""
Fourier Layer for MoE-POT.

Multi-head Fourier neural operator layer that learns kernel-based integral 
transformations in the frequency domain.

Based on: AFNO [13], FNO [28], and DPOT [15].

As described in Section 4:
- Divides spatial features into h groups (multi-head)
- Applies Fourier transform, MLP in frequency domain, inverse Fourier transform
- Kernel integral operator: K_φ(z^l)(x) = F^{-1}[R_φ · F[z^l]]

Note: The paper describes the frequency-domain MLP as operating on complex values.
We implement this by applying a real-valued MLP separately to real and imaginary parts,
which is mathematically equivalent to a diagonal complex linear transformation.
"""

import torch
from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F


class ComplexMLP(nn.Module):
    """
    MLP applied element-wise to complex-valued frequency coefficients.
    
    Implements the paper's eq (5):
    z_{0i}^l(x) = F^{-1}[W_{2,i}^l · σ(W_{1,i}^l · F[z_i^l] + b_{1,i}^l) + b_{2,i}^l](x)
    
    For complex inputs, we apply real-valued MLP to real and imaginary parts
    separately (equivalent to block-diagonal complex matrix).
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_real: [N, dim] real part
            x_imag: [N, dim] imaginary part
        
        Returns:
            out_real, out_imag: processed real and imaginary parts
        """
        # Process real and imaginary parts separately through the same MLP
        out_real = self.fc2(self.act(self.fc1(x_real)))
        out_imag = self.fc2(self.act(self.fc1(x_imag)))
        return out_real, out_imag


class FourierLayer(nn.Module):
    """
    Multi-head Fourier layer with learnable frequency-domain transformations.
    
    Architecture (Section 4):
    - Split features into h heads along channel dimension
    - For each head: FFT -> Complex MLP -> IFFT
    - Concatenate head outputs
    
    Parameters:
        dim: feature dimension
        num_heads: number of heads (h)
        mode: number of Fourier modes to keep (truncation)
    """
    
    def __init__(self, dim: int, num_heads: int = 4, mode: int = 32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mode = mode
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # Frequency-domain MLPs: one per head
        # Each operates on head_dim-dimensional frequency coefficients
        self.freq_mlps = nn.ModuleList([
            ComplexMLP(self.head_dim, self.head_dim)
            for _ in range(num_heads)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] spatial feature map
            
        Returns:
            out: [B, C, H, W] output after Fourier-domain processing
        """
        B, C, H, W = x.shape
        
        # Split into heads: [B, C, H, W] -> [B, h, C/h, H, W]
        x_heads = x.view(B, self.num_heads, self.head_dim, H, W)
        
        # Apply 2D real FFT along spatial dimensions
        # rfft2 returns half the last dimension (W//2+1)
        x_fft = torch.fft.rfft2(x_heads, norm='ortho')  # [B, h, d, H, W//2+1]
        
        # Get frequency dimensions
        freq_h = min(self.mode, H)
        freq_w = min(self.mode, x_fft.shape[-1])
        
        out_fft = torch.zeros_like(x_fft)
        
        # Process each head independently
        for head_idx in range(self.num_heads):
            head_fft = x_fft[:, head_idx]  # [B, d, H, W//2+1]
            
            # Truncate to the specified number of Fourier modes
            head_fft_trunc = head_fft[:, :, :freq_h, :freq_w]  # [B, d, freq_h, freq_w]
            
            # Separate real and imaginary parts
            head_fft_real = head_fft_trunc.real
            head_fft_imag = head_fft_trunc.imag
            
            # Reshape to apply MLP: merge batch, freq_h, freq_w into one dimension
            B_d = head_fft_real.shape[0]
            head_fft_real = head_fft_real.permute(0, 2, 3, 1).reshape(-1, self.head_dim)
            head_fft_imag = head_fft_imag.permute(0, 2, 3, 1).reshape(-1, self.head_dim)
            
            # Apply complex MLP
            out_real, out_imag = self.freq_mlps[head_idx](head_fft_real, head_fft_imag)
            
            # Reshape back
            out_real = out_real.view(B_d, freq_h, freq_w, self.head_dim).permute(0, 3, 1, 2)
            out_imag = out_imag.view(B_d, freq_h, freq_w, self.head_dim).permute(0, 3, 1, 2)
            
            # Reconstruct complex tensor
            head_out = torch.complex(out_real, out_imag)
            
            # Place back into the full frequency tensor
            out_fft[:, head_idx, :, :freq_h, :freq_w] = head_out
        
        # Apply inverse FFT
        out = torch.fft.irfft2(out_fft, s=(H, W), norm='ortho')
        
        # Combine heads: [B, h, C/h, H, W] -> [B, C, H, W]
        out = out.view(B, C, H, W)
        
        return out
