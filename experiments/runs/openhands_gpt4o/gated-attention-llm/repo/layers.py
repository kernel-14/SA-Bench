import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super(GatedAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention with gating
        attn_output, _ = self.multihead_attn(x, x, x)
        gated_output = attn_output * self.gate(x)
        x = x + self.dropout(gated_output)
        x = self.norm1(x)

        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x