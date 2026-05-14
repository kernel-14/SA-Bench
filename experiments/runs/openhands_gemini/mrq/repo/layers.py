
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    "Normalizes inputs over the last dimension. This is often used in transformer networks."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# Helper for LayerNorm + Activation
def ln_activ(layer_norm_module, activation_fn, x):
    x = layer_norm_module(x)
    return activation_fn(x)

