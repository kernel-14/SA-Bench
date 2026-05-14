import torch
import torch.nn as nn
import torch.fft
import math
from typing import List, Tuple, Union

from models.base_operator import CoreOperator
from utils import get_activation_fn


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer. It performs a 2D convolution in Fourier space.
    Truncates higher frequencies and learns transformations on lower frequencies.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        """
        Initializes the SpectralConv2d layer.

        Args:
            in_channels (int): Number of input feature channels.
            out_channels (int): Number of output feature channels.
            modes1 (int): Number of Fourier modes to retain along the first spatial dimension (height).
                          This corresponds to `self.modes1` in the internal representation.
            modes2 (int): Number of Fourier modes to retain along the second spatial dimension (width).
                          This corresponds to `self.modes2` in the internal representation, which should
                          be less than or equal to `width // 2 + 1` of the input tensor's width.
        """
        super().__init__()
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError(f"in_channels must be a positive integer, got {in_channels}")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError(f"out_channels must be a positive integer, got {out_channels}")
        if not isinstance(modes1, int) or modes1 <= 0:
            raise ValueError(f"modes1 (number of Fourier modes) must be a positive integer, got {modes1}")
        if not isinstance(modes2, int) or modes2 <= 0:
            raise ValueError(f"modes2 (number of Fourier modes) must be a positive integer, got {modes2}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Configured number of modes for height
        self.modes2 = modes2  # Configured number of modes for width (of rfft2 output)

        # Complex weights for Fourier modes.
        # Initialized to complex numbers for learning in Fourier space.
        # Shape: (out_channels, in_channels, modes1, modes2)
        # We need two sets of weights because rfft2 produces a non-symmetric output (positive freqs)
        # but for inverse, we implicitly consider negative freqs. Here we handle the two "halves"
        # of the Fourier spectrum explicitly, often top-left and bottom-left for 2D.
        
        # Following common initialization in FNOs (e.g., from original FNO paper code)
        scale = 1 / (in_channels * out_channels) # A simple scaling factor
        
        # Initialize real and imaginary parts of complex weights
        self.weights1 = nn.Parameter(
            scale * torch.randn(out_channels, in_channels, modes1, modes2, dtype=torch.complex64)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(out_channels, in_channels, modes1, modes2, dtype=torch.complex64)
        )
        
        # Use Xavier Uniform for complex numbers for a more standard initialization
        # Initialize real and imaginary parts separately
        nn.init.xavier_uniform_(self.weights1.real)
        nn.init.xavier_uniform_(self.weights1.imag)
        nn.init.xavier_uniform_(self.weights2.real)
        nn.init.xavier_uniform_(self.weights2.imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SpectralConv2d layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        batch_size = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]

        # 1. Perform 2D Real-to-Complex FFT
        # Input: (batch_size, in_channels, height, width)
        # Output: (batch_size, in_channels, height, width // 2 + 1)
        fourier_transform = torch.fft.rfft2(x, dim=(-2, -1), norm="forward")

        # Ensure modes do not exceed available dimensions in Fourier space
        # This handles cases where input resolution is smaller than configured modes
        current_modes1 = min(self.modes1, height)
        current_modes2 = min(self.modes2, width // 2 + 1) # rfft2 output width is floor(width/2) + 1

        # 2. Multiply relevant Fourier modes with learned weights
        # Extracting specific modes to be transformed
        # Part 1: Top-left corner (positive frequencies)
        out_ft_1 = fourier_transform[:, :, :current_modes1, :current_modes2]
        # Part 2: Bottom-left corner (negative frequencies in H, positive in W)
        # For rfft2 output, the negative frequencies along the height dimension correspond to
        # indices `height - current_modes1` to `height` (non-inclusive of `height`)
        # in the height dimension of the Fourier transform.
        out_ft_2 = fourier_transform[:, :, height - current_modes1 :, :current_modes2]

        # Perform channel-wise multiplication for selected modes using einsum
        # einsum("bixy,ioxy->boxy") means:
        # b: batch, i: in_channels, o: out_channels, x: modes1, y: modes2
        # Result: (batch_size, out_channels, current_modes1, current_modes2)
        
        # Slice weights to match the actual modes used in this forward pass
        sliced_weights1 = self.weights1[:, :, :current_modes1, :current_modes2]
        sliced_weights2 = self.weights2[:, :, :current_modes1, :current_modes2]

        transformed_ft_1 = torch.einsum("bixy,ioxy->boxy", out_ft_1, sliced_weights1)
        transformed_ft_2 = torch.einsum("bixy,ioxy->boxy", out_ft_2, sliced_weights2)

        # 3. Zero-pad and reconstruct full Fourier transform (for irfft2)
        # Create a new tensor to hold the transformed Fourier modes
        # Shape: (batch_size, out_channels, height, width // 2 + 1)
        out_fourier_transform = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.complex64,
            device=x.device,
        )

        # Place the transformed modes back into the full Fourier tensor
        out_fourier_transform[:, :, :current_modes1, :current_modes2] = transformed_ft_1
        out_fourier_transform[:, :, height - current_modes1 :, :current_modes2] = transformed_ft_2

        # 4. Perform 2D Inverse Complex-to-Real FFT
        # Output: (batch_size, out_channels, height, width)
        output = torch.fft.irfft2(out_fourier_transform, s=(height, width), dim=(-2, -1), norm="forward")

        return output


class FNO(CoreOperator):
    """
    Fourier Neural Operator (FNO) model.

    This model implements the FNO architecture which consists of multiple
    spectral convolution blocks interleaved with pointwise feed-forward networks (MLPs).
    It acts as a CoreOperator within the larger NeuralOperatorModel framework.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_fourier_modes: Union[List[int], Tuple[int, int]],
        num_layers: int,
        mlp_width: int,
        activation: str = 'gelu'
    ):
        """
        Initializes the FNO model.

        Args:
            hidden_dim (int): Dimensionality of the hidden feature representation throughout the FNO blocks.
            num_fourier_modes (Union[List[int], Tuple[int, int]]): Number of Fourier modes to retain
                                                                    along each spatial dimension (e.g., [16, 16] for 2D).
                                                                    Expects [modes_H, modes_W].
            num_layers (int): Number of FNO blocks in the core operator.
            mlp_width (int): Width of the hidden layer in the pointwise FFNs within each block.
            activation (str): Name of the activation function to use (e.g., 'gelu', 'relu').
        """
        super().__init__()
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if not isinstance(num_fourier_modes, (list, tuple)) or len(num_fourier_modes) != 2 or \
           not all(isinstance(m, int) and m > 0 for m in num_fourier_modes):
            raise ValueError(f"num_fourier_modes must be a list/tuple of 2 positive integers, got {num_fourier_modes}")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers}")
        if not isinstance(mlp_width, int) or mlp_width <= 0:
            raise ValueError(f"mlp_width must be a positive integer, got {mlp_width}")

        self.hidden_dim = hidden_dim
        self.num_fourier_modes = num_fourier_modes
        self.num_layers = num_layers
        self.mlp_width = mlp_width
        self.activation = get_activation_fn(activation)

        self.conv_blocks = nn.ModuleList()
        self.mlp_blocks = nn.ModuleList()

        modes1, modes2 = self.num_fourier_modes[0], self.num_fourier_modes[1]

        for _ in range(self.num_layers):
            # Spectral Convolution layer (in_channels, out_channels, modes_H, modes_W)
            # Both in and out channels are hidden_dim for consistency in FNO blocks
            conv_layer = SpectralConv2d(self.hidden_dim, self.hidden_dim, modes1, modes2)
            self.conv_blocks.append(conv_layer)

            # Pointwise Feed-Forward Network (MLP)
            # Input and output dimensions are hidden_dim for residual connection compatibility
            mlp_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.mlp_width),
                self.activation,
                nn.Linear(self.mlp_width, self.hidden_dim)
            )
            self.mlp_blocks.append(mlp_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FNO model.

        Args:
            x (torch.Tensor): Input tensor from the LiftingAdapter.
                              Expected shape: (batch_size, H, W, hidden_dim).

        Returns:
            torch.Tensor: Output tensor representing the transformed hidden features.
                          Expected shape: (batch_size, H, W, hidden_dim).
                          This output will be passed to the ProjectionAdapter.
        """
        # Ensure input dimensions are as expected (B, H, W, C)
        if x.dim() != 4 or x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected input tensor to have shape (batch_size, H, W, hidden_dim={self.hidden_dim}), "
                f"but got {x.shape}"
            )

        for i in range(self.num_layers):
            x_residual = x # Store for residual connection

            # 1. Apply Spectral Convolution
            # SpectralConv2d expects input (B, C, H, W), so permute from (B, H, W, C)
            x_spectral = x.permute(0, 3, 1, 2)
            x_spectral = self.conv_blocks[i](x_spectral)
            # Permute back to (B, H, W, C) for consistency with MLP and residual
            x_spectral = x_spectral.permute(0, 2, 3, 1)

            # 2. Apply Pointwise MLP
            # MLP operates independently on each spatial point (H, W), mapping channel dim (C) to (C)
            # Input x (B, H, W, C) is suitable for Linear layer applied to last dimension
            x_mlp = self.mlp_blocks[i](x)

            # 3. Combine spectral, MLP outputs with residual connection and apply activation
            x = x_spectral + x_mlp + x_residual
            x = self.activation(x)

        return x

