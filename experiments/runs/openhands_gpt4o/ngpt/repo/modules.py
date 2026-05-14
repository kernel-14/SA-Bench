import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(NormalizedMultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        q = self.q_linear(x).view(batch_size, -1, self.n_heads, self.d_k)
        k = self.k_linear(x).view(batch_size, -1, self.n_heads, self.d_k)
        v = self.v_linear(x).view(batch_size, -1, self.n_heads, self.d_k)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k**0.5
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)

        output = output.view(batch_size, -1, self.n_heads * self.d_k)
        return self.out(output)

class NormalizedMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super(NormalizedMLP, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = F.normalize(x, p=2, dim=-1)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=-1)