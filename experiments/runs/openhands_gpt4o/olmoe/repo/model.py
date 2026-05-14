# model.py

import torch
import torch.nn as nn
from modules import MoELayer
from layers import TransformerLayer

class OLMoEModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_experts: int, num_active_experts: int):
        super(OLMoEModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, MoELayer(d_model, num_experts, num_active_experts))
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x