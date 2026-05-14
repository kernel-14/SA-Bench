import torch
import torch.nn as nn
from normalization import normalize
from ngpt_attention import NGPTAttention
from ngpt_mlp import NGPTMLP

class NGPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int,
                 alpha_A_init: float = 0.05, alpha_A_scale: float = None,
                 alpha_M_init: float = 0.05, alpha_M_scale: float = None):
        super().__init__()
        self.d_model = d_model

        self.attention = NGPTAttention(d_model, n_heads)
        self.mlp = NGPTMLP(d_model, d_mlp)

        if alpha_A_scale is None:
            alpha_A_scale = 1.0 / (d_model**0.5) # Default from paper Section 2.6, point 3
        if alpha_M_scale is None:
            alpha_M_scale = 1.0 / (d_model**0.5) # Default from paper Section 2.6, point 3

        # alpha_A and alpha_M are learnable eigen learning rates (Section 2.2.2, 2.6.3)
        self.alpha_A_unscaled = nn.Parameter(torch.full((d_model,), alpha_A_init)) # Initialize with alpha_A_init
        self.alpha_A_scale_factor = alpha_A_scale
        self.alpha_A = self.alpha_A_unscaled * (alpha_A_init / alpha_A_scale) # Effective alpha_A as per Section 2.5

        self.alpha_M_unscaled = nn.Parameter(torch.full((d_model,), alpha_M_init)) # Initialize with alpha_M_init
        self.alpha_M_scale_factor = alpha_M_scale
        self.alpha_M = self.alpha_M_unscaled * (alpha_M_init / alpha_M_scale) # Effective alpha_M as per Section 2.5

    def forward(self, h: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # No RMSNorm or LayerNorm (Section 2.6, point 1)

        # Attention block
        h_A_out = self.attention(h, mask)
        h_A_normalized = normalize(h_A_out, dim=-1)
        
        # Update equation for attention (Section 2.2.2, Equation 10 and 2.6, point 3)
        h = normalize(h + self.alpha_A * (h_A_normalized - h), dim=-1)

        # MLP block
        h_M_out = self.mlp(h)
        h_M_normalized = normalize(h_M_out, dim=-1)
        
        # Update equation for MLP (Section 2.2.2, Equation 11 and 2.6, point 3)
        h = normalize(h + self.alpha_M * (h_M_normalized - h), dim=-1)

        return h

