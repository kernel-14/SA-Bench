import torch
import torch.nn as nn
import torch.fft

class FourierLayer(nn.Module):
    def __init__(self, input_dim: int, num_heads: int):
        super(FourierLayer, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = input_dim // num_heads
        self.weights = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim))

    def forward(self, x):
        batch_size, height, width, channels = x.shape
        x = x.view(batch_size, height, width, self.num_heads, self.head_dim)
        x = torch.fft.fft2(x, dim=(1, 2))
        x = torch.einsum('bhwd,hdc->bhwc', x, self.weights)
        x = torch.fft.ifft2(x, dim=(1, 2)).real
        return x.view(batch_size, height, width, channels)