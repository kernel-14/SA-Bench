# layers.py

import torch
import torch.nn as nn

class CustomLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(CustomLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))