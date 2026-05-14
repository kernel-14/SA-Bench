import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from utils import get_activation_fn


class LiftingAdapter(nn.Module):
    """
    Implements the Lifting Layer (adapter) for the Neural Operator.
    This adapter maps raw input functions from their original dimensionality
    to a higher-dimensional hidden representation using a Multi-Layer Perceptron (MLP).
    The MLP operates point-wise across the spatial dimensions.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_mlp_layers: int, activation: str = 'gelu'):
        """
        Initializes the LiftingAdapter.

        Args:
            input_dim (int): The dimensionality of the input features at each spatial point.
                             (e.g., coordinates, initial condition, parameters).
            hidden_dim (int): The target dimensionality of the lifted hidden representation.
                              This becomes the channel dimension for the core operator.
            num_mlp_layers (int): The number of linear layers in the MLP.
                                  Each intermediate linear layer is followed by an activation.
            activation (str): The name of the activation function to use (e.g., 'gelu', 'relu').
        """
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if not isinstance(num_mlp_layers, int) or num_mlp_layers <= 0:
            raise ValueError(f"num_mlp_layers must be a positive integer, got {num_mlp_layers}")

        modules = []
        current_dim = input_dim
        act_fn = get_activation_fn(activation)

        for i in range(num_mlp_layers):
            if i < num_mlp_layers - 1:
                # Intermediate layers, map to hidden_dim
                modules.append(nn.Linear(current_dim, hidden_dim))
                modules.append(act_fn)
                current_dim = hidden_dim
            else:
                # Last layer of the MLP, maps to hidden_dim, followed by activation
                # as per paper's formula for lifting: sigma(A_L a + b_L)
                modules.append(nn.Linear(current_dim, hidden_dim))
                modules.append(act_fn) # Activation after the final lifting layer

        self.mlp = nn.Sequential(*modules)
        self.output_dim = hidden_dim # Store output dim for later reshaping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the lifting operation on the input tensor.

        The input tensor is expected to have shape (batch_size, *spatial_dims, input_dim).
        It is reshaped to (batch_size * product_of_spatial_dims, input_dim) for MLP processing,
        then reshaped back to (batch_size, *spatial_dims, hidden_dim).

        Args:
            x (torch.Tensor): The input tensor from the data,
                              e.g., (batch_size, H, W, input_dim) for 2D spatial data.

        Returns:
            torch.Tensor: The lifted hidden representation,
                          e.g., (batch_size, H, W, hidden_dim).
        """
        original_shape = x.shape
        # Flatten spatial dimensions and batch dimension for point-wise MLP application
        # x_flat shape: (batch_size * H * W, input_dim)
        x_flat = x.view(-1, original_shape[-1])

        # Apply the MLP
        lifted_x_flat = self.mlp(x_flat)

        # Reshape back to (batch_size, *spatial_dims, hidden_dim)
        # new_shape_tuple: (original_shape[0], *original_shape[1:-1], self.output_dim)
        new_shape = (*original_shape[:-1], self.output_dim)
        lifted_x = lifted_x_flat.view(new_shape)

        return lifted_x


class ProjectionAdapter(nn.Module):
    """
    Implements the Projection Layer (adapter) for the Neural Operator.
    This adapter maps the hidden representation from the core operator
    to the problem-specific output function space using an MLP.
    The MLP operates point-wise across the spatial dimensions.
    """

    def __init__(self, hidden_dim: int, output_dim: int, num_mlp_layers: int, activation: str = 'gelu'):
        """
        Initializes the ProjectionAdapter.

        Args:
            hidden_dim (int): The dimensionality of the hidden features from the core operator.
            output_dim (int): The target dimensionality of the output function
                              (e.g., 1 for a scalar field, 2 for a vector field).
            num_mlp_layers (int): The number of linear layers in the MLP.
                                  Intermediate layers are followed by an activation.
            activation (str): The name of the activation function to use for intermediate layers.
        """
        super().__init__()
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        if not isinstance(num_mlp_layers, int) or num_mlp_layers <= 0:
            raise ValueError(f"num_mlp_layers must be a positive integer, got {num_mlp_layers}")

        modules = []
        current_dim = hidden_dim
        act_fn = get_activation_fn(activation)

        for i in range(num_mlp_layers):
            if i < num_mlp_layers - 1:
                # Intermediate layers, map from current_dim to hidden_dim
                modules.append(nn.Linear(current_dim, hidden_dim))
                modules.append(act_fn)
                current_dim = hidden_dim
            else:
                # Last layer of the MLP, maps to output_dim, no activation for regression output
                modules.append(nn.Linear(current_dim, output_dim))

        self.mlp = nn.Sequential(*modules)
        self.output_dim = output_dim # Store output dim for later reshaping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the projection operation on the hidden representation.

        The input tensor is expected to have shape (batch_size, *spatial_dims, hidden_dim).
        It is reshaped to (batch_size * product_of_spatial_dims, hidden_dim) for MLP processing,
        then reshaped back to (batch_size, *spatial_dims, output_dim).

        Args:
            x (torch.Tensor): The hidden representation tensor from the core operator,
                              e.g., (batch_size, H, W, hidden_dim).

        Returns:
            torch.Tensor: The projected output tensor in the original output space,
                          e.g., (batch_size, H, W, output_dim).
        """
        original_shape = x.shape
        # Flatten spatial dimensions and batch dimension for point-wise MLP application
        # x_flat shape: (batch_size * H * W, hidden_dim)
        x_flat = x.view(-1, original_shape[-1])

        # Apply the MLP
        projected_x_flat = self.mlp(x_flat)

        # Reshape back to (batch_size, *spatial_dims, output_dim)
        # new_shape_tuple: (original_shape[0], *original_shape[1:-1], self.output_dim)
        new_shape = (*original_shape[:-1], self.output_dim)
        projected_x = projected_x_flat.view(new_shape)

        return projected_x

