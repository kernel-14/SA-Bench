import torch
import torch.nn as nn

class AdaLNZero(nn.Module):
    def __init__(self, hidden_size):
        super(AdaLNZero, self).__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.shift = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + 1e-5)
        return self.scale * normalized + self.shift

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(SinusoidalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, positions):
        half_dim = self.embedding_dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / half_dim))
        emb = positions[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)