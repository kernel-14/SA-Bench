import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)