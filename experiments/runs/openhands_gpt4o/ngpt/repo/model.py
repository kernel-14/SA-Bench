import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, vocab_size):
        super(NormalizedTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=-1)  # Normalize embeddings
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(TransformerLayer, self).__init__()
        self.attention = NormalizedMultiHeadAttention(d_model, n_heads)
        self.mlp = NormalizedMLP(d_model, d_ff)

    def forward(self, x):
        x = self.attention(x)
        x = self.mlp(x)
        return x