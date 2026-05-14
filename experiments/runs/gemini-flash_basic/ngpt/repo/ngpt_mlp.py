import torch
import torch.nn as nn
from normalization import normalize

class NGPTMLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, s_u_init: float = 1.0, s_u_scale: float = 1.0, s_nu_init: float = 1.0, s_nu_scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp # d_mlp is d_model * 4 in standard Transformers

        self.W_u = nn.Parameter(torch.rand(d_model, d_mlp))
        self.W_nu = nn.Parameter(torch.rand(d_model, d_mlp))
        self.W_oMLP = nn.Parameter(torch.rand(d_mlp, d_model))

        # s_u and s_nu are trainable scaling factors for intermediate MLP states (Section 2.4.2, 2.6.5)
        self.s_u_unscaled = nn.Parameter(torch.full((d_mlp,), s_u_init))
        self.s_u_scale_factor = s_u_scale
        self.s_u = self.s_u_unscaled * (s_u_init / s_u_scale) # Effective s_u as per Section 2.5

        self.s_nu_unscaled = nn.Parameter(torch.full((d_mlp,), s_nu_init))
        self.s_nu_scale_factor = s_nu_scale
        self.s_nu = self.s_nu_unscaled * (s_nu_init / s_nu_scale) # Effective s_nu as per Section 2.5

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # 1. Normalize weight matrices (Section 2.6, point 2)
        self.W_u.data = normalize(self.W_u.data, dim=-1)
        self.W_nu.data = normalize(self.W_nu.data, dim=-1)
        self.W_oMLP.data = normalize(self.W_oMLP.data, dim=-1)

        # Linear projections (Section 2.4.1, Equation 17)
        u = torch.matmul(h, self.W_u)
        nu = torch.matmul(h, self.W_nu)

        # Apply scaling factors s_u and s_nu (Section 2.4.2, Equations 20, 21)
        u = u * self.s_u
        nu = nu * self.s_nu * (self.d_model**0.5) # Rescaling by sqrt(d_model) as per Section 2.4.2

        # SwiGLU activation (Section 2.4.1, Equation 18)
        swiglu_output = u * torch.nn.functional.silu(nu)

        # Final linear transformation (Section 2.4.1, Equation 19)
        h_M = torch.matmul(swiglu_output, self.W_oMLP)

        return h_M

