## model.py

import torch
import torch.nn as nn
import torch.fft
from typing import Tuple

class Model(nn.Module):
    """
    Fourier Neural Operator (FNO) and Sensitivity-Constrained FNO (SC-FNO) implementation.
    Handles forward prediction and sensitivity computation for parametric differential equations.
    """

    def __init__(self, input_dims: Tuple[int, ...], output_dims: Tuple[int, ...], fourier_modes: int = 8, width: int = 20):
        """
        Initializes the Fourier Neural Operator model.

        Args:
        - input_dims (Tuple[int, ...]): Shape of input tensor (batch_size, input_features).
        - output_dims (Tuple[int, ...]): Shape of output tensor (batch_size, output_features).
        - fourier_modes (int): Number of Fourier modes for frequency domain encoding. Default is 8.
        - width (int): Number of channels in the hidden layers. Default is 20.
        """
        super(Model, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.fourier_modes = fourier_modes
        self.width = width

        # Lifting layer to map input tensors to higher dimensions
        self.lifting_layer = nn.Linear(input_dims[-1], self.width)

        # Fourier layer setup
        self.fourier_layers = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size=1) for _ in range(fourier_modes)
        ])

        # Output projection layer
        self.projection_layer = nn.Linear(self.width, output_dims[-1])

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FNO model.

        Args:
        - x (torch.Tensor): Input tensor containing parameters, spatial/temporal grids, and initial conditions.
          Shape: (batch_size, input_features).

        Returns:
        - torch.Tensor: Predicted solution tensor across spatial/temporal domains.
          Shape: (batch_size, output_features).
        """
        # Apply lifting layer
        x_lifted = self.lifting_layer(x)  # Shape: (batch_size, width)

        # Apply Fourier transformations across Fourier modes
        for i, layer in enumerate(self.fourier_layers):
            # Frequency space transformation
            x_freq = torch.fft.rfft(x_lifted, norm="ortho")  # Encode into frequency domain
            x_freq = layer(x_freq)  # Learnable filtering in frequency space
            x_lifted = torch.fft.irfft(x_freq, norm="ortho")  # Decode back to spatial/temporal domain

        # Apply activation function
        x_activated = self.activation(x_lifted)

        # Project results to output dimensions
        output = self.projection_layer(x_activated)  # Shape: (batch_size, output_features)

        return output

    def compute_sensitivities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Jacobian of the predicted solutions with respect to input parameters.

        Args:
        - x (torch.Tensor): Input tensor containing parameters, spatial/temporal grids, and initial conditions.
          Shape: (batch_size, input_features).

        Returns:
        - torch.Tensor: Jacobian tensor representing the sensitivities ∂u/∂p.
          Shape: (batch_size, input_features, output_features).
        """
        # Ensure input requires gradients
        x = x.clone().detach().requires_grad_(True)

        # Forward pass to compute predictions
        outputs = self.forward(x)  # Shape: (batch_size, output_features)

        # Compute Jacobians using Automatic Differentiation (AD)
        jacobians = []
        for i in range(outputs.size(1)):  # Loop over output dimensions
            jacobian = torch.autograd.grad(
                outputs[:, i].sum(), x, retain_graph=True, allow_unused=True
            )[0]  # Gradient w.r.t input
            jacobians.append(jacobian.unsqueeze(-1))  # Append per-dimension gradient

        # Concatenate all Jacobians across output dimensions
        return torch.cat(jacobians, dim=-1)  # Shape: (batch_size, input_features, output_features)
