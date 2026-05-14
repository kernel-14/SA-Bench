"""
Consistency model wrapper with the parametrization from Song et al. (2023) and Song & Dhariwal (2024).

The model parametrization ensures the boundary condition f(x_0, sigma_0) = x_0:
    f_theta(x_t, sigma_t) = c_skip(sigma_t) * x_t + c_out(sigma_t) * F_theta(x_t, sigma_t)

where:
    c_skip(sigma) = sigma_d^2 / (sigma_d^2 + (sigma - sigma_0)^2)
    c_out(sigma) = sigma_d * (sigma - sigma_0) / sqrt(sigma^2 + sigma_d^2)
"""

import torch
import torch.nn as nn
import math


class ConsistencyModel(nn.Module):
    """
    Consistency model with the parametrization from Song et al. (2023).
    Wraps a neural network backbone (e.g., SongUNet) with the skip/output scaling.
    """
    def __init__(self, network, sigma_data=0.5, sigma_min=0.002, sigma_max=80.0):
        """
        Args:
            network: The backbone neural network (e.g., SongUNet)
            sigma_data: Standard deviation of the data distribution
            sigma_min: Minimum noise level (sigma_0)
            sigma_max: Maximum noise level (sigma_T)
        """
        super().__init__()
        self.network = network
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def c_skip(self, sigma):
        """Skip connection scaling."""
        return self.sigma_data ** 2 / (self.sigma_data ** 2 + (sigma - self.sigma_min) ** 2)

    def c_out(self, sigma):
        """Output scaling."""
        return self.sigma_data * (sigma - self.sigma_min) / torch.sqrt(
            sigma ** 2 + self.sigma_data ** 2
        )

    def c_in(self, sigma):
        """Input scaling (for the network input)."""
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_noise(self, sigma):
        """Noise level conditioning."""
        return 0.25 * torch.log(sigma)

    def forward(self, x, sigma, class_labels=None, augment_labels=None):
        """
        Forward pass of the consistency model.
        
        Args:
            x: Noisy input tensor of shape (B, C, H, W)
            sigma: Noise level tensor of shape (B,) or scalar
            class_labels: Optional class labels for conditional generation
            augment_labels: Optional augmentation labels
            
        Returns:
            Predicted clean data point
        """
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, dtype=x.dtype, device=x.device)
        
        # Reshape sigma for broadcasting
        sigma = sigma.reshape(-1, *([1] * (x.ndim - 1)))
        
        # Compute scaling factors
        c_skip = self.c_skip(sigma)
        c_out = self.c_out(sigma)
        c_in = self.c_in(sigma)
        c_noise = self.c_noise(sigma.reshape(-1))
        
        # Network forward pass
        F_out = self.network(
            c_in * x,
            c_noise,
            class_labels=class_labels,
            augment_labels=augment_labels
        )
        
        # Apply skip connection and output scaling
        return c_skip * x + c_out * F_out

    @torch.no_grad()
    def sample(self, z, num_steps=1, class_labels=None):
        """
        Generate samples from noise using the consistency model.
        
        For one-step generation, simply apply the model to the noise.
        For multi-step generation, apply the model iteratively.
        
        Args:
            z: Noise tensor of shape (B, C, H, W)
            num_steps: Number of sampling steps (1 for one-step generation)
            class_labels: Optional class labels
            
        Returns:
            Generated samples
        """
        sigma_max = torch.tensor(self.sigma_max, dtype=z.dtype, device=z.device)
        x = self.forward(z, sigma_max, class_labels=class_labels)
        
        if num_steps > 1:
            # Multi-step sampling: apply the model at intermediate noise levels
            # Using a simple schedule
            sigmas = torch.linspace(self.sigma_max, self.sigma_min, num_steps + 1, device=z.device)
            for i in range(1, num_steps):
                sigma = sigmas[i]
                # Add noise at the current level
                noise = torch.randn_like(x)
                x_noisy = x + sigma * noise
                x = self.forward(x_noisy, sigma.expand(z.shape[0]), class_labels=class_labels)
        
        return x
