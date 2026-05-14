
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) as described in Su et al., 2021; Heo et al., 2024.
    Applies 2D RoPE to queries and keys in attention mechanisms.
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0) # 1, seq_len, dim

        cos = emb.cos()
        sin = emb.sin()

        x_rope = (x * cos) + (self._rotate_half(x) * sin)
        return x_rope

