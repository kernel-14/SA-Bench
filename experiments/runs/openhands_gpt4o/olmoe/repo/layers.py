# layers.py

import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, moe_layer: nn.Module):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=8)
        self.moe_layer = moe_layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        moe_output = self.moe_layer(x)
        x = self.norm2(x + moe_output)
        return x