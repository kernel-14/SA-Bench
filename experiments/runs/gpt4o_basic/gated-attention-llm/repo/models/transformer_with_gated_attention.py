import torch
import torch.nn as nn
from src.gated_attention import GatedAttention

class TransformerWithGatedAttention(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, num_layers):
        super(TransformerWithGatedAttention, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(model_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttentionWithGating(model_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, model_dim)
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        
    def forward(self, x):
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class MultiHeadAttentionWithGating(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttentionWithGating, self).__init__()
        
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads)
        self.gated_attention = GatedAttention(model_dim, num_heads)
        
    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        gated_output = self.gated_attention(attn_output, query)
        return gated_output

