import torch
import torch.nn as nn

from .attention import NormalizedAttention
from .mlp import NormalizedMLP

class NormalizedTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp, n_layers):
        super(NormalizedTransformer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.n_layers = n_layers

        # Embedding normalization
        self.embedding = nn.Parameter(torch.randn(d_model))

        # Layers
        self.attention_blocks = nn.ModuleList([NormalizedAttention(d_model, n_heads) for _ in range(n_layers)])
        self.mlp_blocks = nn.ModuleList([NormalizedMLP(d_model, d_mlp) for _ in range(n_layers)])

        # Learnable eigen learning rates
        self.alpha_attention = nn.Parameter(torch.ones(n_layers, d_model))
        self.alpha_mlp = nn.Parameter(torch.ones(n_layers, d_model))

    def forward(self, x):
        h = F.normalize(x, p=2, dim=-1)  # Normalize input onto hypersphere

        for i in range(self.n_layers):
            # Attention block
            h_att = self.attention_blocks[i](h)
            h = F.normalize(h + self.alpha_attention[i] * (h_att - h), p=2, dim=-1)

            # MLP block
            h_mlp = self.mlp_blocks[i](h)
            h = F.normalize(h + self.alpha_mlp[i] * (h_mlp - h), p=2, dim=-1)

        return h
