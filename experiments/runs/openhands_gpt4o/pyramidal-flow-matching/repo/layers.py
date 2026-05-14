# layers.py

import torch
import torch.nn as nn

class UpsampleLayer(nn.Module):
    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

class DownsampleLayer(nn.Module):
    def __init__(self):
        super(DownsampleLayer, self).__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)