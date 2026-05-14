import math
from typing import Optional

import torch
import torch.nn as nn

from .network import SongUNet


class ConsistencyModel(nn.Module):
    """
    Consistency model f_θ with the parametrization from Song et al. (2023):

        f_θ(x_t, σ_t) = c_skip(σ_t) * x_t + c_out(σ_t) * F_θ(c_in(σ_t) * x_t, σ_t)

    where:
        c_skip(σ) = σ_d² / (σ_d² + (σ - σ_0)²)
        c_out(σ)  = σ_d * (σ - σ_0) / sqrt(σ² + σ_d²)
        c_in(σ)   = 1 / sqrt(σ² + σ_d²)

    This ensures the boundary condition f_θ(x_0, σ_0) = x_0.
    """

    def __init__(
        self,
        network: SongUNet,
        sigma_min: float = 0.002,
        sigma_data: float = 0.5,
    ):
        super().__init__()
        self.network = network
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma_d = self.sigma_data
        sigma_0 = self.sigma_min
        return sigma_d ** 2 / (sigma_d ** 2 + (sigma - sigma_0) ** 2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma_d = self.sigma_data
        sigma_0 = self.sigma_min
        return sigma_d * (sigma - sigma_0) / torch.sqrt(sigma ** 2 + sigma_d ** 2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma_d = self.sigma_data
        return 1.0 / torch.sqrt(sigma ** 2 + sigma_d ** 2)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: noisy input (B, C, H, W)
            sigma: noise level (B,) or scalar

        Returns:
            Predicted clean image (B, C, H, W)
        """
        if sigma.dim() == 0:
            sigma = sigma.expand(x.shape[0])

        sigma_bc = sigma[:, None, None, None]

        c_skip = self.c_skip(sigma_bc)
        c_out = self.c_out(sigma_bc)
        c_in = self.c_in(sigma_bc)

        x_in = c_in * x
        F_out = self.network(x_in, sigma)
        return c_skip * x + c_out * F_out

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        sigma_max: float = 80.0,
        num_steps: int = 1,
        sigmas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One-step or multi-step sampling from the consistency model.

        For one-step generation: directly apply f_θ(z * σ_max, σ_max).
        For multi-step: iteratively apply the model with decreasing noise levels.

        Args:
            noise: Gaussian noise (B, C, H, W)
            sigma_max: maximum noise level
            num_steps: number of sampling steps (1 for one-step generation)
            sigmas: optional sequence of noise levels for multi-step sampling

        Returns:
            Generated samples (B, C, H, W)
        """
        device = noise.device
        B = noise.shape[0]

        if num_steps == 1:
            x = noise * sigma_max
            sigma = torch.full((B,), sigma_max, device=device)
            return self.forward(x, sigma)

        if sigmas is None:
            sigmas = torch.linspace(sigma_max, self.sigma_min, num_steps + 1, device=device)

        x = noise * sigmas[0]
        sigma = torch.full((B,), sigmas[0].item(), device=device)
        x = self.forward(x, sigma)

        for i in range(1, num_steps):
            sigma_i = sigmas[i]
            z = torch.randn_like(x)
            x = x + torch.sqrt(sigma_i ** 2 - self.sigma_min ** 2) * z
            sigma = torch.full((B,), sigma_i.item(), device=device)
            x = self.forward(x, sigma)

        return x


def build_consistency_model(cfg: dict) -> ConsistencyModel:
    """Build a ConsistencyModel from a config dictionary."""
    net_cfg = cfg["network"]
    network = SongUNet(
        img_resolution=cfg["image_resolution"],
        in_channels=cfg["in_channels"],
        out_channels=cfg["in_channels"],
        model_channels=net_cfg["model_channels"],
        channel_mult=net_cfg["channel_mult"],
        num_blocks=net_cfg["num_blocks"],
        attn_resolutions=net_cfg.get("attn_resolutions", []),
        dropout=net_cfg.get("dropout", 0.0),
        embedding_type=net_cfg.get("embedding_type", "positional"),
    )
    model = ConsistencyModel(
        network=network,
        sigma_min=cfg["sigma_min"],
        sigma_data=cfg.get("sigma_data", 0.5),
    )
    return model
