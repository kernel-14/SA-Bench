import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super(TimeEmbedding, self).__init__()
        self.linear = nn.Linear(1, embedding_dim)

    def forward(self, t):
        t = t.unsqueeze(-1)  # Ensure time is a column vector
        return self.linear(t)