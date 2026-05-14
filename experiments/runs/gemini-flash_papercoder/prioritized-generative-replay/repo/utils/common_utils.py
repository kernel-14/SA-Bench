import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_activation_fn(name: str) -> nn.Module:
    """
    Retrieves a PyTorch activation function module based on its name.

    Args:
        name (str): The name of the activation function (case-insensitive).
                    Supported: "ReLU", "LeakyReLU", "Tanh", "Sigmoid", "GELU", "Identity" (for no activation).

    Returns:
        nn.Module: An instance of the requested activation function module.

    Raises:
        ValueError: If an unsupported activation function name is provided.
    """
    name_lower = name.lower()
    if name_lower == "relu":
        return nn.ReLU()
    elif name_lower == "leakyrelu":
        return nn.LeakyReLU()
    elif name_lower == "tanh":
        return nn.Tanh()
    elif name_lower == "sigmoid":
        return nn.Sigmoid()
    elif name_lower == "gelu":
        return nn.GELU()
    elif name_lower == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {name}")


def get_optimizer(
    name: str, params: Union[torch.Tensor, List[torch.Tensor]], lr: float, **kwargs
) -> optim.Optimizer:
    """
    Retrieves a PyTorch optimizer instance based on its name.

    Args:
        name (str): The name of the optimizer (case-insensitive).
                    Supported: "Adam", "SGD", "RMSprop".
        params (Union[torch.Tensor, List[torch.Tensor]]): Iterable of parameters to optimize or dicts defining
                                                          parameter groups.
        lr (float): Learning rate.
        **kwargs: Additional keyword arguments to pass to the optimizer constructor.

    Returns:
        optim.Optimizer: An instance of the requested optimizer.

    Raises:
        ValueError: If an unsupported optimizer name is provided.
    """
    name_lower = name.lower()
    if name_lower == "adam":
        return optim.Adam(params, lr=lr, **kwargs)
    elif name_lower == "sgd":
        return optim.SGD(params, lr=lr, **kwargs)
    elif name_lower == "rmsprop":
        return optim.RMSprop(params, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def init_weights(m: nn.Module, init_type: str = 'orthogonal', gain: float = math.sqrt(2)):
    """
    Initializes the weights and biases of a PyTorch module.

    Args:
        m (nn.Module): The module to initialize.
        init_type (str): The type of initialization to use.
                         Supported: "xavier_uniform", "kaiming_normal", "orthogonal", "default".
        gain (float): Scaling factor for some initialization methods (e.g., 'orthogonal').
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=gain)
        elif init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, gain=gain)
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight, gain=gain)
        elif init_type == 'default':
            pass  # Use default PyTorch initialization
        else:
            raise ValueError(f"Unsupported initialization type: {init_type}")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class MLPBlock(nn.Module):
    """
    A basic Multi-Layer Perceptron (MLP) block for constructing neural networks.
    Consists of linear layers with activation functions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: int,
        num_hidden_layers: int,
        activation_fn_name: str = "ReLU",
        output_activation_fn_name: Optional[str] = None,
    ):
        """
        Initializes the MLPBlock.

        Args:
            input_dim (int): The dimension of the input features.
            output_dim (int): The dimension of the output features.
            hidden_units (int): The number of units in each hidden layer.
            num_hidden_layers (int): The number of hidden layers.
            activation_fn_name (str): The name of the activation function to use for hidden layers.
                                       Defaults to "ReLU".
            output_activation_fn_name (Optional[str]): The name of the activation function for the output layer.
                                                       If None, no activation is applied to the output.
                                                       Defaults to None.
        """
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.hidden_units: int = hidden_units
        self.num_hidden_layers: int = num_hidden_layers
        self.activation_fn: nn.Module = get_activation_fn(activation_fn_name)
        self.output_activation_fn: Optional[nn.Module] = (
            get_activation_fn(output_activation_fn_name) if output_activation_fn_name else None
        )

        layers: List[nn.Module] = []
        current_dim: int = input_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_units))
            layers.append(self.activation_fn)
            current_dim = hidden_units

        layers.append(nn.Linear(current_dim, output_dim))
        if self.output_activation_fn:
            layers.append(self.output_activation_fn)

        self.net: nn.Sequential = nn.Sequential(*layers)
        self.apply(init_weights)  # Apply default orthogonal initialization for MLPs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor from the MLP.
        """
        return self.net(x)


class CNNEncoder(nn.Module):
    """
    A Convolutional Neural Network (CNN) encoder, suitable for processing pixel-based observations.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (C, H, W)
        output_dim: int,
        num_filters: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        activation_fn_name: str = "ReLU",
    ):
        """
        Initializes the CNNEncoder.

        Args:
            input_shape (Tuple[int, int, int]): The shape of the input image (channels, height, width).
            output_dim (int): The dimension of the flattened output features.
            num_filters (Optional[List[int]]): List of number of filters for each convolutional layer.
                                               Defaults to [32, 32, 32] if None.
            kernel_sizes (Optional[List[int]]): List of kernel sizes for each convolutional layer.
                                                Defaults to [3, 3, 3] if None.
            strides (Optional[List[int]]): List of strides for each convolutional layer.
                                           Defaults to [2, 1, 1] if None.
            activation_fn_name (str): The name of the activation function to use after convolutional layers.
                                       Defaults to "ReLU".
        """
        super().__init__()
        self.input_shape: Tuple[int, int, int] = input_shape
        self.output_dim: int = output_dim
        self.activation_fn: nn.Module = get_activation_fn(activation_fn_name)

        # Default CNN architecture (e.g., from DRQ-V2 or common practice)
        if num_filters is None:
            num_filters = [32, 32, 32, 32]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3, 3]
        if strides is None:
            strides = [2, 1, 1, 1]

        if not (len(num_filters) == len(kernel_sizes) == len(strides)):
            raise ValueError("num_filters, kernel_sizes, and strides must have the same length.")

        layers: List[nn.Module] = []
        current_channels: int = input_shape[0]
        current_height: int = input_shape[1]
        current_width: int = input_shape[2]

        for i in range(len(num_filters)):
            conv_layer = nn.Conv2d(
                in_channels=current_channels,
                out_channels=num_filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=kernel_sizes[i] // 2,  # 'same' padding approximation
            )
            layers.append(conv_layer)
            layers.append(self.activation_fn)
            current_channels = num_filters[i]

            # Calculate output dimensions after convolution
            # O = (I - K + 2P)/S + 1
            padding_val = kernel_sizes[i] // 2
            current_height = math.floor((current_height - kernel_sizes[i] + 2 * padding_val) / strides[i]) + 1
            current_width = math.floor((current_width - kernel_sizes[i] + 2 * padding_val) / strides[i]) + 1

        self.conv_net: nn.Sequential = nn.Sequential(*layers)

        # Calculate the size of the flattened features
        flattened_features_dim: int = current_channels * current_height * current_width
        
        # Final linear layer to project to output_dim
        self.linear: nn.Linear = nn.Linear(flattened_features_dim, output_dim)

        self.apply(init_weights) # Apply orthogonal initialization for CNNs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the CNN encoder.

        Args:
            x (torch.Tensor): The input image tensor (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: The encoded feature vector.
        """
        conv_out: torch.Tensor = self.conv_net(x)
        flattened_out: torch.Tensor = conv_out.reshape(conv_out.size(0), -1)
        output: torch.Tensor = self.linear(flattened_out)
        return output


class NoisyLinear(nn.Module):
    """
    Noisy Linear layer for implicit exploration, as proposed by Fortunato et al. (2018).
    This layer replaces a standard nn.Linear layer.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        Initializes the NoisyLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            std_init (float): Initial standard deviation for noise parameters.
                              Defaults to 0.5 based on common practices.
        """
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.std_init: float = std_init

        # Learnable parameters for mean and std of weights and biases
        self.weight_mu: nn.Parameter = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma: nn.Parameter = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu: nn.Parameter = nn.Parameter(torch.empty(out_features))
        self.bias_sigma: nn.Parameter = nn.Parameter(torch.empty(out_features))

        # Buffers for noise (not learnable parameters)
        self.register_buffer('eps_i', torch.empty(1, in_features))
        self.register_buffer('eps_j', torch.empty(out_features, 1))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Resets the parameters of the layer.
        Initializes mu with Kaiming uniform and sigma based on std_init.
        """
        # Initialize mu for weights and biases
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)

        # Initialize sigma for weights and biases
        # As per paper's recommendations, std_init is typically 0.5 or 0.1
        # The initial noise scale `sigma_0` is `std_init / sqrt(in_features)` or `std_init / sqrt(out_features)`
        # Here we use the fixed `std_init` value across both, which is a common variant.
        self.weight_sigma.data.fill_(self._scale_sigma_initialization(self.in_features))
        self.bias_sigma.data.fill_(self._scale_sigma_initialization(self.in_features))

    def _scale_sigma_initialization(self, dim: int) -> float:
        """Helper to compute initial sigma value based on input dimension."""
        return self.std_init / math.sqrt(dim)


    def _scale_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates factorized noise as described in the paper.
        f(x) = sign(x) * sqrt(abs(x))
        """
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """
        Resamples the noise variables.
        Called at the beginning of each episode for exploration.
        """
        self.eps_i.normal_() # Noise for input features
        self.eps_j.normal_() # Noise for output features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the NoisyLinear layer.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        if self.training:
            # Generate factorized noise
            eps_i_scaled: torch.Tensor = self._scale_noise(self.eps_i)
            eps_j_scaled: torch.Tensor = self._scale_noise(self.eps_j)
            
            # Combine for weight and bias noise
            weight_epsilon: torch.Tensor = eps_j_scaled.mul(eps_i_scaled)
            bias_epsilon: torch.Tensor = eps_j_scaled.squeeze(1) # Squeeze to make it (out_features,)

            # Calculate noisy weights and biases
            noisy_weight: torch.Tensor = self.weight_mu + self.weight_sigma.mul(weight_epsilon)
            noisy_bias: torch.Tensor = self.bias_mu + self.bias_sigma.mul(bias_epsilon)
        else:
            # In evaluation mode, use the mean (deterministic)
            noisy_weight: torch.Tensor = self.weight_mu
            noisy_bias: torch.Tensor = self.bias_mu

        return F.linear(input, noisy_weight, noisy_bias)

