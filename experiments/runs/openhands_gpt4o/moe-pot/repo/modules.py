import torch
import torch.nn as nn
import torch.nn.functional as F

class RouterGatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, top_k: int):
        super(RouterGatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.linear = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.linear(x.mean(dim=(1, 2)))  # Global average pooling
        weights = F.softmax(logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        return top_k_weights, top_k_indices

class SharedExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(SharedExpert, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class RoutedExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(RoutedExpert, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.mlp(x)