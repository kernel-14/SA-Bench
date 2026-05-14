## model.py

"""
This module implements neural operator models as part of the reproducibility effort for the paper
"Towards Universal Neural Operators through Multiphysics Pretraining."

Classes:
    - Model: Main class to initialize and use different neural operator architectures, including
             Fourier Neural Operator (FNO), Mamba-SSM, and Perceiver IO.
"""

import torch
import torch.nn as nn
import torch.fft as fft
from typing import Dict, List


class FourierLayer(nn.Module):
    """
    Implements a Fourier Neural Operator layer consisting of:
    - Fourier transforms: For mapping input to frequency domain.
    - Spectral filters: Parameterized kernels in the frequency domain.
    - Nonlinear activation: Operates in hidden spaces.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """
        Initialize the FourierLayer.

        Args:
            in_channels (int): Number of input channels/features.
            out_channels (int): Number of output channels/features.
            modes (int): Number of Fourier modes to preserve.
        """
        super(FourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Parameterized kernel weights for spectral transformation
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass via Fourier transform.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Apply 2D Fourier transform
        x_ft = fft.fft2(x)  # Shape: (batch_size, in_channels, height, width)

        # Truncate to relevant modes
        x_ft = x_ft[:, :, : self.modes, : self.modes]

        # Perform frequency-domain multiplication with weight
        out_ft = torch.einsum("bchw,iohw->bihw", x_ft, self.weight)

        # Perform inverse Fourier transform back to spatial domain
        out = fft.ifft2(out_ft).real  # Take the real part of the inverse transform
        return out


class FourierNeuralOperator(nn.Module):
    """
    Implements the architecture of the Fourier Neural Operator (FNO).
    Includes a lifting layer, multiple Fourier layers, and a projection layer.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int, layers: int):
        """
        Initialize the FourierNeuralOperator.

        Args:
            in_channels (int): Dimensionality of the input functions.
            out_channels (int): Dimensionality of the output functions.
            modes (int): Number of Fourier modes to use in each Fourier layer.
            layers (int): Number of Fourier layers in the architecture.
        """
        super(FourierNeuralOperator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.layers = layers

        # Lifting layer to project input functions to hidden space
        self.lifting = nn.Conv2d(in_channels, 64, kernel_size=1)

        # Fourier layers stack
        self.fourier_layers = nn.ModuleList(
            [FourierLayer(64, 64, modes) for _ in range(layers)]
        )

        # Nonlinear activation between Fourier layers
        self.activation = nn.ReLU()

        # Projection layer to map from hidden space to output space
        self.projection = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the Fourier Neural Operator.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Lift input to hidden space
        x = self.lifting(x)

        # Apply Fourier layers sequentially
        for layer in self.fourier_layers:
            x = self.activation(layer(x))

        # Project back to output space
        x = self.projection(x)
        return x


class MambaSSM(nn.Module):
    """
    Implements the Mamba State-Space Model (SSM) for neural operators.
    Adds causal recurrence via convolution to encode spatio-temporal dynamics.
    """

    def __init__(self, in_channels: int, out_channels: int, layers: int, kernel_size: int):
        """
        Initialize the MambaSSM.

        Args:
            in_channels (int): Dimensionality of the input functions.
            out_channels (int): Dimensionality of the output functions.
            layers (int): Number of layers in the architecture.
            kernel_size (int): Size of the convolutional kernel.
        """
        super(MambaSSM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.kernel_size = kernel_size

        # Lifting layer
        self.lifting = nn.Conv1d(in_channels, 64, kernel_size=1)

        # Causal recurrent convolutions (state-space modeling)
        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(64, 64, kernel_size, padding=(kernel_size // 2)) for _ in range(layers)]
        )

        # Nonlinear activation
        self.activation = nn.GELU()

        # Projection layer
        self.projection = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the Mamba SSM.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, sequence_length).
        """
        # Lift input to hidden space
        x = self.lifting(x)

        # Apply recurrent convolutional layers
        for layer in self.conv_layers:
            x = self.activation(layer(x))

        # Project back to output space
        x = self.projection(x)
        return x


class PerceiverIO(nn.Module):
    """
    Implements the Perceiver IO-based neural operator, leveraging attention for feature mapping.
    Includes cross-attention and self-attention blocks for latent processing.
    """

    def __init__(self, in_channels: int, out_channels: int, latent_size: int, attention_layers: int):
        """
        Initialize the PerceiverIO model.

        Args:
            in_channels (int): Input function dimensionality.
            out_channels (int): Output function dimensionality.
            latent_size (int): Size of the latent feature representation.
            attention_layers (int): Number of cross/self-attention layers.
        """
        super(PerceiverIO, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.attention_layers = attention_layers

        # Lifting layer
        self.lifting = nn.Linear(in_channels, latent_size)

        # Cross-attention and self-attention blocks
        self.cross_attention = nn.MultiheadAttention(latent_size, num_heads=8)
        self.self_attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(latent_size, num_heads=8) for _ in range(attention_layers)]
        )

        # Nonlinear activation
        self.activation = nn.ReLU()

        # Projection layer
        self.projection = nn.Linear(latent_size, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the PerceiverIO model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, in_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, out_channels).
        """
        # Initial mapping to latent space
        latent = self.lifting(x)

        # Apply cross-attention
        latent, _ = self.cross_attention(latent, latent, latent)

        # Apply self-attention layers
        for layer in self.self_attention_layers:
            latent, _ = layer(latent, latent, latent)

        # Project back to output space
        output = self.projection(latent)
        return output


class Model:
    """
    Wrapper class to initialize and manage different neural operator architectures (FNO, Mamba-SSM, Perceiver IO).
    """

    def __init__(self, model_type: str, params: Dict):
        """
        Initialize the Model class.

        Args:
            model_type (str): Type of neural operator ("Fourier Neural Operator", "Mamba-SSM", "Perceiver IO").
            params (dict): Model-specific configuration parameters.
        """
        model_map = {
            "Fourier Neural Operator": FourierNeuralOperator,
            "Mamba-SSM": MambaSSM,
            "Perceiver IO": PerceiverIO,
        }

        if model_type not in model_map:
            raise ValueError(f"Unsupported model type '{model_type}'. Supported types: {list(model_map.keys())}")

        self.model = model_map[model_type](**params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the selected neural operator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def freeze_layers(self, except_layers: List[str]) -> None:
        """
        Freeze all layers except specified adapter layers (e.g., lifting, projection).

        Args:
            except_layers (List[str]): List of layer names to exclude from freezing.
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = any(except_layer in name for except_layer in except_layers)
