# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import FourierLayer, MambaSSM, PerceiverIO

class NeuralOperator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super(NeuralOperator, self).__init__()
        self.lifting = nn.Linear(input_dim, hidden_dim)
        self.operator_layers = nn.ModuleList([
            FourierLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.mamba_ssm = MambaSSM(hidden_dim)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.lifting(x))
        x = self.mamba_ssm(x)
        for layer in self.operator_layers:
            x = layer(x)
        x = self.projection(x)
        return x