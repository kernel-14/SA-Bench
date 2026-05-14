import torch
import torch.nn as nn

class CustomLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(CustomLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.linear(x))

class AttentionLayer(nn.Module):
    def __init__(self, dim: int):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_weights = self.softmax(torch.matmul(q, k.transpose(-2, -1)))
        return torch.matmul(attention_weights, v)