## models/fno.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class SpectralConvND(nn.Module):
    """
    N-dimensional spectral convolution layer used in Fourier Neural Operators.
    Performs Fourier Transform, truncates high-frequency modes, applies learnable
    complex weights, and then performs Inverse Fourier Transform.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: Tuple[int, ...]) -> None:
        """
        Initializes the SpectralConvND layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes (Tuple[int, ...]): Tuple specifying the maximum number of Fourier modes
                                     to retain along each of the N transformed dimensions.
                                     For the last transformed dimension by `torch.fft.rfftn`,
                                     this mode count will be capped by D_last // 2 + 1.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.n_dims = len(modes) # Number of dimensions over which to perform the Fourier transform

        # Scaling factor for weight initialization
        self.scale = 1 / (in_channels * out_channels)

        # Learnable complex weights for spectral multiplication
        # Shape: (in_channels, out_channels, mode_0, mode_1, ..., mode_{n_dims-1})
        # The `modes` tuple defines the *maximum* size of these weight tensors
        # in the Fourier dimensions. Actual application will slice them if x_ft is smaller.
        weights_shape = (self.in_channels, self.out_channels) + self.modes
        self.weights1 = nn.Parameter(self.scale * (torch.randn(*weights_shape) + 1j * torch.randn(*weights_shape)))
        self.weights2 = nn.Parameter(self.scale * (torch.randn(*weights_shape) + 1j * torch.randn(*weights_shape)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SpectralConvND layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, D0, D1, ..., Dn-1, in_channels),
                              where D0 to Dn-1 are the spatial/temporal grid dimensions.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, D0, D1, ..., Dn-1, out_channels).
        """
        batchsize = x.shape[0]
        # Get the spatial/temporal dimensions (D0, ..., Dn-1)
        grid_dims = x.shape[1:-1]

        # 1. Fourier Transform (real-to-complex N-dimensional)
        # Apply rfftn across the `*grid_dims` (dimensions 1 to n_dims).
        # x: (batch_size, D0, ..., Dn-1, in_channels)
        # fft_dims: (1, 2, ..., n_dims)
        fft_dims = tuple(range(1, self.n_dims + 1))
        # x_ft shape: (batch_size, D0_fft, ..., Dn-1_fft, in_channels)
        # Note: D_{n-1}_fft = D_{n-1} // 2 + 1 due to rfftn
        x_ft = torch.fft.rfftn(x, dim=fft_dims, norm='forward')

        # Create output container in Fourier space, initialized to zeros.
        # Its dimensions match x_ft, but `in_channels` is replaced by `out_channels`.
        out_ft_shape = list(x_ft.shape)
        out_ft_shape[-1] = self.out_channels
        out_ft = torch.zeros(out_ft_shape, device=x.device, dtype=torch.cfloat)

        # 2. Spectral Multiplication and Truncation
        # Dynamically build slices for both x_ft and weights based on `self.modes`
        # and the actual sizes of `x_ft` in the Fourier domain.

        x_ft_mode_slices: List[slice] = []
        weights_mode_slices: List[slice] = []

        for dim_idx in range(self.n_dims):
            # For x_ft, the Fourier dimensions are at indices 1 to n_dims (after batch dim 0).
            actual_fft_dim_size = x_ft.shape[dim_idx + 1]
            # Desired maximum modes for this dimension from config.
            desired_modes = self.modes[dim_idx]
            # Number of modes to actually keep/process: min of desired and available.
            num_modes_to_keep = min(desired_modes, actual_fft_dim_size)
            
            x_ft_mode_slices.append(slice(0, num_modes_to_keep))
            weights_mode_slices.append(slice(0, num_modes_to_keep)) # Slice weights to match actual used modes

        # Construct full slice tuples for x_ft and weights
        # x_ft: (batch, D0_fft, ..., Dn-1_fft, in_channels)
        x_ft_full_slice = (slice(None), *x_ft_mode_slices, slice(None))
        x_ft_sliced_part = x_ft[x_ft_full_slice]

        # weights: (in_channels, out_channels, M0_max, ..., Mn-1_max)
        weights_full_slice = (slice(None), slice(None), *weights_mode_slices)
        w1_sliced_part = self.weights1[weights_full_slice]
        w2_sliced_part = self.weights2[weights_full_slice]

        # Perform spectral multiplication for the low-frequency modes
        # einsum string: 'b...i,io...->b...o'
        # 'b': batch_size
        # '...': represents the n_dims Fourier dimensions (D0', ..., Dn-1')
        # 'i': in_channels
        # 'o': out_channels
        op1 = torch.einsum('b...i,io...->b...o', x_ft_sliced_part, w1_sliced_part)
        op2 = torch.einsum('b...i,io...->b...o', x_ft_sliced_part, w2_sliced_part)
        
        # Sum the results and place them into the corresponding slice of `out_ft`
        out_ft_full_slice = (slice(None), *x_ft_mode_slices, slice(None))
        out_ft[out_ft_full_slice] = op1 + op2

        # 3. Inverse Fourier Transform (complex-to-real N-dimensional)
        # `s` argument: specify the original sizes of the dimensions that were transformed.
        # These are `grid_dims` before `rfftn`.
        x = torch.fft.irfftn(out_ft, s=grid_dims, dim=fft_dims, norm='forward')

        return x


class FourierBlock(nn.Module):
    """
    A single Fourier Neural Operator block, combining spectral convolution with
    a linear skip connection and a GELU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: Tuple[int, ...]) -> None:
        """
        Initializes a FourierBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes (Tuple[int, ...]): Tuple of Fourier modes to retain for SpectralConvND.
        """
        super().__init__()
        self.conv = SpectralConvND(in_channels, out_channels, modes)
        self.w = nn.Linear(in_channels, out_channels) # Linear layer for the skip connection
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FourierBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *grid_dims, in_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *grid_dims, out_channels).
        """
        # Output from spectral convolution
        x_conv = self.conv(x)
        
        # Output from linear skip connection.
        # The linear layer operates independently on each spatial/temporal point across the batch.
        x_linear = self.w(x)
        
        # Residual connection and activation
        x = x_conv + x_linear
        x = self.activation(x)
        return x


class FNO(nn.Module):
    """
    Fourier Neural Operator (FNO) model architecture.
    Comprises a lifting layer, multiple Fourier blocks, and projection layers.
    """
    def __init__(self, modes: Tuple[int, ...], width: int, num_fourier_layers: int, input_dim: int, output_dim: int) -> None:
        """
        Initializes the FNO model.

        Args:
            modes (Tuple[int, ...]): Tuple specifying the number of Fourier modes to retain
                                     along each dimension for the spectral convolution layers.
            width (int): The number of channels in the FNO's hidden feature representations.
            num_fourier_layers (int): The number of Fourier blocks in the FNO backbone.
            input_dim (int): The total number of input channels for each grid point
                             (e.g., concatenated initial conditions, coords, parameters).
            output_dim (int): The number of output channels for the predicted solution.
        """
        super().__init__()
        self.modes = modes
        self.width = width
        self.num_fourier_layers = num_fourier_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Lifting Layer: Projects input features to a higher-dimensional latent space
        self.fc0 = nn.Linear(input_dim, width)

        # Fourier Blocks: Core FNO operations
        self.convs = nn.ModuleList([
            FourierBlock(width, width, modes) for _ in range(num_fourier_layers)
        ])

        # Projection Layers: Maps features back to the output dimension
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, output_dim)

        # Activation Function (GELU is common in FNO implementations)
        self.activation = nn.GELU()

    def forward(self, input_func: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FNO model.

        Args:
            input_func (torch.Tensor): The input tensor containing concatenated features
                                       (initial conditions, spatial/temporal coordinates, parameters).
                                       Expected shape: (batch_size, D0, D1, ..., Dn-1, input_dim),
                                       where D0 to Dn-1 are the spatial/temporal grid dimensions.

        Returns:
            torch.Tensor: The predicted solution path.
                          The output shape will be (batch_size, D0, D1, ..., Dn-1, output_dim).
        """
        # 1. Lifting Layer
        x = self.fc0(input_func) # Shape: (batch_size, *grid_dims, width)

        # 2. Fourier Blocks
        for conv_block in self.convs:
            x = conv_block(x) # Shape remains (batch_size, *grid_dims, width)

        # 3. Projection Layers
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x) # Final shape: (batch_size, *grid_dims, output_dim)

        return x

