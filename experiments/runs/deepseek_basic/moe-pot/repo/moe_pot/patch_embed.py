"""
Patch Embedding and Temporal Aggregation for MoE-POT.

As described in Section 4:
- Patchification layer with positional embeddings (ViT-style)
- Temporal aggregation layer with Fourier features

Input Encoding and Temporal Aggregation:
    Z_p^t = P(u^t + p^t),  t=1,...,T
where P is a convolutional layer, p^t are learnable positional encodings.

    z_agg = Σ_t W_t · z_p^t · e^{-iγt}
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Patchification layer that partitions spatial domain into non-overlapping patches
    and maps to embedding vectors via convolution.
    
    As described in Appendix B.1:
    Conv2D(C -> d, kernel=P, stride=P)
    
    Plus learnable positional encodings inspired by ViT [10].
    
    Args:
        in_channels: number of input channels (C)
        embed_dim: embedding dimension (d)
        patch_size: size of each patch (P)
    """
    
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Convolutional patch embedding
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] single timestep input
            
        Returns:
            out: [B, embed_dim, H/patch_size, W/patch_size]
        """
        return self.proj(x)


class TemporalAggregation(nn.Module):
    """
    Temporal aggregation layer that combines information across T timesteps
    using learnable MLP transformation and Fourier feature constants.
    
    As described in Section 4:
    z_agg = Σ_t W_t · z_p^t · e^{-iγt}
    
    where W_t is a learnable MLP transformation, γ is a Fourier feature constant.
    """
    
    def __init__(self, embed_dim: int, T: int = 10):
        super().__init__()
        self.embed_dim = embed_dim
        self.T = T
        
        # Learnable transformation W_t for each timestep
        # W_t is implemented as a linear projection followed by activation
        self.time_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
            )
            for _ in range(T)
        ])
        
        # Fourier feature constant γ ∈ R^C
        self.gamma = nn.Parameter(torch.randn(embed_dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, embed_dim, H, W] spatiotemporal features after patchification
            
        Returns:
            out: [B, embed_dim, H, W] temporally aggregated features
        """
        B, T, C, H, W = x.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"
        
        out = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
        
        for t in range(T):
            # Get features at timestep t
            z_t = x[:, t]  # [B, C, H, W]
            
            # Apply learnable MLP W_t (per-timestep)
            # We apply pointwise: reshape to [B*H*W, C], apply linear, reshape back
            B_t, C_t, H_t, W_t = z_t.shape
            z_t_flat = z_t.permute(0, 2, 3, 1).reshape(-1, C_t)
            z_t_flat = self.time_weights[t](z_t_flat)
            z_t = z_t_flat.view(B_t, H_t, W_t, C_t).permute(0, 3, 1, 2)
            
            # Fourier feature modulation: e^{-iγt}
            # Using real-valued approximation: cos(γ·t) 
            # The imaginary part e^{-iγt} = cos(γt) - i*sin(γt)
            # Since we work with real features, we use cosine modulation
            fourier_phase = self.gamma * t  # [C]
            modulation = torch.cos(fourier_phase).view(1, C, 1, 1)  # [1, C, 1, 1]
            
            out = out + z_t * modulation
            
        return out
