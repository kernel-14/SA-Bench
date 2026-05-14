"""
MLP model for PINNs.
- tanh activations
- 3 hidden layers
- Xavier normal initialization
- all biases initialized to zero
"""

import torch
import torch.nn as nn
import math


class MLP(nn.Module):
    """
    Multi-layer perceptron with tanh activations.
    Architecture: input_dim -> [width]*3 -> 1
    Initialized with Xavier normal weights and zero biases.
    """

    def __init__(self, input_dim=2, width=100, output_dim=1):
        super().__init__()
        layers = []
        dims = [input_dim] + [width] * 3 + [output_dim]
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            # Xavier normal initialization
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
