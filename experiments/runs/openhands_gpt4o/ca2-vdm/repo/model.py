import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalTemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CausalTemporalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mask = None

    def forward(self, x):
        if self.mask is None or self.mask.size(0) != x.size(0):
            self.mask = torch.triu(torch.ones(x.size(0), x.size(0)), diagonal=1).bool().to(x.device)
        return self.attention(x, x, x, attn_mask=self.mask)[0]

class PrefixEnhancedSpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, prefix_length, dropout=0.1):
        super(PrefixEnhancedSpatialAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.prefix_length = prefix_length

    def forward(self, x, prefix):
        prefix_repeated = prefix.repeat(self.prefix_length, 1, 1)
        combined = torch.cat([prefix_repeated, x], dim=0)
        return self.attention(combined, combined, combined)[0]

class Ca2VDM(nn.Module):
    def __init__(self, latent_dim, num_layers, num_heads, dropout, causal_attention, prefix_enhanced_attention):
        super(Ca2VDM, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if causal_attention:
                self.layers.append(CausalTemporalAttention(latent_dim, num_heads, dropout))
            if prefix_enhanced_attention:
                self.layers.append(PrefixEnhancedSpatialAttention(latent_dim, num_heads, prefix_length=3, dropout=dropout))

    def forward(self, x, prefix=None):
        for layer in self.layers:
            if isinstance(layer, PrefixEnhancedSpatialAttention):
                x = layer(x, prefix)
            else:
                x = layer(x)
        return x