import torch
import torch.nn as nn

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ConditionalDiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, condition):
        """
        Forward pass of the diffusion model.
        :param x: Input wavelet coefficients.
        :param condition: Conditioning input (e.g., low-res wavelet coefficients).
        :return: Predicted coefficients for the next denoising step.
        """
        combined = torch.cat([x, condition], dim=-1)
        hidden = torch.relu(self.fc1(combined))
        return self.fc2(hidden)
