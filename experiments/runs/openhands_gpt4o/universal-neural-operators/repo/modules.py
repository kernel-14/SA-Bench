# modules.py

import torch
import torch.nn as nn
import torch.fft

class FourierLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(FourierLayer, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x_fft = torch.fft.rfft(x, dim=-1)
        x_fft = torch.fft.irfft(x_fft, n=x.size(-1), dim=-1)
        return x_fft

class MambaSSM(nn.Module):
    def __init__(self, hidden_dim: int):
        super(MambaSSM, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, x):
        return torch.matmul(x, self.kernel)

class PerceiverIO(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_latents: int):
        super(PerceiverIO, self).__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4)
        self.self_attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4)

    def forward(self, x):
        latents = self.latents.unsqueeze(1).repeat(1, x.size(0), 1)
        latents, _ = self.cross_attention(latents, x, x)
        latents, _ = self.self_attention(latents, latents, latents)
        return latents