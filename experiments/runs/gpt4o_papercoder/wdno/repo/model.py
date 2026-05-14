# model.py
"""
Model Implementation for Wavelet Diffusion Neural Operator (WDNO).
This module defines the 'Model' class that implements the Denoising Diffusion Probabilistic Model architecture
to operate on wavelet coefficients, enabling simulation and control tasks.
"""

import torch
from torch import nn, Tensor
from typing import Dict, Any, List
import math


class UNet(nn.Module):
    """
    A U-Net architecture for noise prediction in diffusion models.
    This model accepts wavelet-transformed coefficients and outputs noise estimates.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initializes the U-Net model based on the provided parameters.

        Args:
            params (dict): Configuration parameters for the U-Net.
        """
        super(UNet, self).__init__()
        self.layers = nn.ModuleList()
        dims = params["initial_dimension"]
        depth_mult = params["dimension_multiplier"]

        # Downsampling layers
        for i, mult in enumerate(depth_mult):
            self.layers.append(nn.Conv2d(dims, mult * dims, kernel_size=3, stride=1, padding=1))
            dims *= mult
            self.layers.append(nn.ReLU())

        # Bottleneck
        self.layers.append(nn.Conv2d(dims, dims, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.ReLU())

        # Upsampling layers
        for i, mult in reversed(list(enumerate(depth_mult))):
            self.layers.append(nn.ConvTranspose2d(dims, dims // mult, kernel_size=3, stride=1, padding=1))
            dims //= mult
            self.layers.append(nn.ReLU())

        # Output
        self.layers.append(nn.Conv2d(dims, params["output_channels"], kernel_size=3, stride=1, padding=1))

    def forward(self, x: Tensor, conditions: Tensor) -> Tensor:
        """
        Forward pass for the U-Net.

        Args:
            x (Tensor): Input wavelet coefficients [batch_size, channels, height, width].
            conditions (Tensor): Conditioning inputs [batch_size, condition_channels, height, width].

        Returns:
            Tensor: Noise prediction [batch_size, channels, height, width].
        """
        # Combine input with conditions
        x = torch.cat([x, conditions], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


class Model:
    """
    Core model class implementing the WDNO using DDPM-based architecture.
    Handles forward passes during training and sampling during inference.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initializes the WDNO model with DDPM architecture.

        Args:
            params (dict): Model and training configuration parameters.
        """
        self.params = params
        self.num_timesteps = self.params["ddim_steps"]
        self.eta = self.params["ddim_eta"]
        self.guidance_weight = self.params["control_guidance_intensity"]

        # Initialize U-Net model
        self.epsilon_theta = UNet(self.params["unet"])

        # Compute noise schedule
        self.alpha = torch.linspace(0.0001, 0.02, self.num_timesteps)
        self.alpha_bar = torch.cumprod(1.0 - self.alpha, dim=0)
        self.sigma = torch.sqrt((1.0 - self.alpha_bar[1:]) / (1.0 - self.alpha_bar[:-1]) * self.alpha[:-1])

    def forward(self, time_series: Tensor, conditions: Dict[str, Tensor]) -> Tensor:
        """
        Forward pass during training to predict the denoising process.

        Args:
            time_series (Tensor): Input noisy wavelet coefficients [batch_size, timesteps, channels].
            conditions (dict): Conditioning inputs (e.g., initial state and PDE parameters).

        Returns:
            Tensor: Predicted noise for the given inputs.
        """
        conditioning_inputs = torch.cat([conditions[key] for key in ["initial_state", "parameters"]], dim=1)
        noisy_data = time_series

        # Perform noise prediction with U-Net
        noise_prediction = self.epsilon_theta(noisy_data, conditioning_inputs)
        return noise_prediction

    def sample(self, time_steps: int, noise: Tensor, conditions: Dict[str, Tensor]) -> Tensor:
        """
        Generate samples during inference through DDIM-based denoising.

        Args:
            time_steps (int): Number of sampling steps (K).
            noise (Tensor): Initial Gaussian noise [batch_size, channels, height, width].
            conditions (dict): Conditioning inputs (e.g., parameters for control objectives).

        Returns:
            Tensor: Final wavelet coefficients after the sampling process.
        """
        # Prepare conditioning inputs
        conditioning_inputs = torch.cat([conditions[key] for key in ["initial_state", "parameters"]], dim=1)

        x_k = noise.clone()  # Start with Gaussian noise
        for k in range(time_steps - 1, -1, -1):
            sigma_k = self.sigma[k] if k > 0 else 0
            noise_prediction = self.epsilon_theta(x_k, conditioning_inputs)

            # Control Objective Guidance (if provided)
            if "gradient" in conditions:
                grad_guidance = self.guidance_weight * conditions["gradient"]
            else:
                grad_guidance = 0

            x_k = x_k - self.eta * (noise_prediction + grad_guidance)
            x_k = x_k + sigma_k * torch.randn_like(x_k)

        return x_k

    def infer_high_resolution(self, data_low_res: Tensor, conditions: Dict[str, Tensor]) -> Tensor:
        """
        Perform super-resolution inference if conditioned on lower-resolution data.

        Args:
            data_low_res (Tensor): Input low-resolution wavelet coefficients.
            conditions (dict): Conditioning inputs for super-resolution.

        Returns:
            Tensor: High-resolution wavelet coefficients.
        """
        high_res_result = self.sample(
            time_steps=self.params["ddim_steps"], 
            noise=torch.randn_like(data_low_res), 
            conditions=conditions,
        )
        return high_res_result
