import torch
import torch.nn as nn

class AdaLNZero(nn.Module):
    def __init__(self, hidden_dim):
        super(AdaLNZero, self).__init__()
        self.hidden_dim = hidden_dim
        self.ln = nn.LayerNorm(hidden_dim)
        self.gamma = nn.Parameter(torch.zeros(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        normalized = self.ln(x)
        return self.gamma * normalized + self.beta

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))