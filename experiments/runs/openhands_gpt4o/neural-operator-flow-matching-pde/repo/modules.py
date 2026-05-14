import torch
import torch.nn as nn

class GRUConditioner(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUConditioner, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h):
        out, h_next = self.gru(x, h)
        out = self.fc(out)
        return out, h_next

class TemporalPyramid(nn.Module):
    def __init__(self, input_dim, scales):
        super(TemporalPyramid, self).__init__()
        self.scales = scales
        self.downsamplers = nn.ModuleList([
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=scale, padding=1)
            for scale in scales
        ])

    def forward(self, x):
        outputs = []
        for downsampler in self.downsamplers:
            outputs.append(downsampler(x))
        return outputs