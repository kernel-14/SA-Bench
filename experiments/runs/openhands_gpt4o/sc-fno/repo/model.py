import torch
import torch.nn as nn
import torch.fft

class FourierLayer(nn.Module):
    def __init__(self, modes, width):
        super(FourierLayer, self).__init__()
        self.modes = modes
        self.width = width
        self.weights = nn.Parameter(torch.randn(modes, width, dtype=torch.cfloat))

    def forward(self, x):
        x_ft = torch.fft.fft(x, dim=-1)
        x_ft = x_ft[..., :self.modes] * self.weights
        x = torch.fft.ifft(x_ft, dim=-1).real
        return x

class SCFNO(nn.Module):
    def __init__(self, modes, width, layers):
        super(SCFNO, self).__init__()
        self.layers = nn.ModuleList([FourierLayer(modes, width) for _ in range(layers)])
        self.fc_in = nn.Linear(3, width)  # Assuming 3 inputs: initial condition, spatial, and temporal
        self.fc_out = nn.Linear(width, 1)  # Assuming scalar output

    def forward(self, x):
        x = self.fc_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        return x