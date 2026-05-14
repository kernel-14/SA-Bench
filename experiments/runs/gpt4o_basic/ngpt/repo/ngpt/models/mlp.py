import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedMLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super(NormalizedMLP, self).__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp

        # Projection matrices
        self.W_u = nn.Parameter(torch.randn(d_model, d_mlp))
        self.W_v = nn.Parameter(torch.randn(d_model, d_mlp))
        self.W_o_mlp = nn.Parameter(torch.randn(d_mlp, d_model))

        # Scaling factors
        self.s_u = nn.Parameter(torch.ones(d_mlp))
        self.s_v = nn.Parameter(torch.ones(d_mlp))

    def forward(self, h):
        # Normalize projection matrices
        W_u_norm = F.normalize(self.W_u, p=2, dim=0)
        W_v_norm = F.normalize(self.W_v, p=2, dim=0)
        W_o_norm = F.normalize(self.W_o_mlp, p=2, dim=0)

        # Compute intermediate vectors
        u = torch.matmul(h, W_u_norm) * self.s_u
        v = torch.matmul(h, W_v_norm) * self.s_v

        # Apply SwiGLU activation
        v_activated = v * torch.sigmoid(v)
        h_mlp = torch.matmul(u * v_activated, W_o_norm)
        return h_mlp
