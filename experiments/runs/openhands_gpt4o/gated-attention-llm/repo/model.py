import torch
import torch.nn as nn
from layers import GatedAttentionLayer

class GatedAttentionModel(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int, d_ff: int, dropout: float):
        super(GatedAttentionModel, self).__init__()
        self.layers = nn.ModuleList([
            GatedAttentionLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)