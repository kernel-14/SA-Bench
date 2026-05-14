# modules.py

import torch
import torch.nn as nn

class SpatialPyramid(nn.Module):
    def __init__(self, num_stages: int):
        super(SpatialPyramid, self).__init__()
        self.num_stages = num_stages
        self.downsamplers = nn.ModuleList([
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1) for _ in range(num_stages)
        ])

    def forward(self, x):
        outputs = []
        for downsampler in self.downsamplers:
            x = downsampler(x)
            outputs.append(x)
        return outputs

class TemporalPyramid(nn.Module):
    def __init__(self, num_stages: int):
        super(TemporalPyramid, self).__init__()
        self.num_stages = num_stages
        self.temporal_layers = nn.ModuleList([
            nn.Conv3d(3, 3, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)) for _ in range(num_stages)
        ])

    def forward(self, x):
        outputs = []
        for temporal_layer in self.temporal_layers:
            x = temporal_layer(x)
            outputs.append(x)
        return outputs

class DiffusionTransformer(nn.Module):
    def __init__(self):
        super(DiffusionTransformer, self).__init__()
        self.transformer = nn.Transformer()

    def forward(self, x):
        return self.transformer(x)