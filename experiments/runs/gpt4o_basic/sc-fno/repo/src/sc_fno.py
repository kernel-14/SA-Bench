import torch
import torch.nn as nn
import torch.fft as fft

class SCFNO(nn.Module):
    def __init__(self, input_dim, output_dim, modes, width):
        super(SCFNO, self).__init__()
        # Store input configurations
        self.input_dim = input_dim  # Dimensions: e.g., for parameters and initial conditions
        self.output_dim = output_dim  # Predicted solution path dimensions
        self.modes = modes  # Fourier domain modes
        self.width = width  # Number of features after lifting

        # Define the lifting layer (reshaping input to higher dimensions)
        self.lifting = nn.Sequential(
            nn.Linear(self.input_dim, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
        )

        # Fourier layers for transforming in frequency domain
        self.fourier_blocks = nn.ModuleList([
            FourierBlock(self.width, self.modes) for _ in range(4)  # Example of stacking 4 Fourier operations
        ])

        # Fully connected layers at output
        self.projection = nn.Linear(self.width, self.output_dim)

    def forward(self, x):
        # Apply input lifting
        lifted = self.lifting(x)  # Shape: (batch, seq, width)

        # Apply Fourier transforms through each Fourier Block
        for fb in self.fourier_blocks:
            lifted = fb(lifted)

        # Final layer projects back to output dimension
        return self.projection(lifted)

class FourierBlock(nn.Module):
    def __init__(self, width, modes):
        super(FourierBlock, self).__init__()
        self.width = width
        self.modes = modes

        # Learnable weights in the Fourier domain
        self.weights_real = nn.Parameter(torch.rand((width, modes)))
        self.weights_imag = nn.Parameter(torch.rand((width, modes)))

    def forward(self, x):
        # Compute Fourier transform
        x_ft = fft.fft(x, dim=-1)

        # Apply learnable weights in Fourier space (real + imaginary composition)
        x_transformed = (
            self.weights_real * x_ft.real - self.weights_imag * x_ft.imag
        ) + 1j * (
            self.weights_real * x_ft.imag + self.weights_imag * x_ft.real
        )

        # Inverse Fourier transform back to the spatial domain
        x_ifft = fft.ifft(x_transformed, dim=-1)
        return x_ifft.real
