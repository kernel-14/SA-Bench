"""
DDPM baseline (without wavelet transform).

This is the plain DDPM model that operates directly in the space-time domain,
used as a key comparison in the paper to demonstrate the benefit of the
wavelet transform.

The architecture is identical to WDNO but without the wavelet transform:
  - Input: raw state/force trajectories [batch, C, T, X] or [batch, C, T, H, W]
  - No wavelet transform applied
  - Same U-Net architecture as WDNO

This baseline is used in:
  - Table 1: simulation results (DDPM vs WDNO)
  - Table 2: control results
  - Figure 2a, 7: visualization showing DDPM struggles with abrupt changes
  - Section 4.7: ablation study
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet_1d import UNet1D
from models.unet_3d import UNet3D
from models.diffusion import GaussianDiffusion, cosine_guidance_schedule


class DDPM1D(nn.Module):
    """
    Plain DDPM for 1D PDE data (no wavelet transform).

    Operates directly on [batch, C, T, X] data.
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        diffusion_kwargs: Optional[Dict] = None,
        unet_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        unet_kw = unet_kwargs or {}
        model = UNet1D(
            in_channels=in_channels,
            cond_channels=cond_channels,
            **unet_kw,
        )
        diff_kw = diffusion_kwargs or {}
        self.diffusion = GaussianDiffusion(model=model, **diff_kw)
        self.in_channels = in_channels
        self.cond_channels = cond_channels

    def training_step(self, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target: [batch, C, T, X] raw state/force trajectory
            cond: [batch, C_cond, T, X] raw condition
        """
        return self.diffusion(target, cond)

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple,
        cond: torch.Tensor,
        guidance_fn: Optional[Callable] = None,
        guidance_weight: float = 0.0,
        ddim_steps: int = 50,
        eta: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = cond.device

        def guidance_schedule(k, K_total):
            return cosine_guidance_schedule(k, K_total, guidance_weight)

        return self.diffusion.sample(
            shape=shape,
            cond=cond,
            guidance_fn=guidance_fn,
            guidance_weight=guidance_weight,
            guidance_schedule=guidance_schedule if guidance_fn is not None else None,
            ddim_steps=ddim_steps,
            eta=eta,
            device=device,
        )


class DDPM2D(nn.Module):
    """
    Plain DDPM for 2D PDE data (no wavelet transform).

    Operates directly on [batch, C, T, H, W] data.
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        diffusion_kwargs: Optional[Dict] = None,
        unet_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        unet_kw = unet_kwargs or {}
        model = UNet3D(
            in_channels=in_channels,
            cond_channels=cond_channels,
            **unet_kw,
        )
        diff_kw = diffusion_kwargs or {}
        self.diffusion = GaussianDiffusion(model=model, **diff_kw)
        self.in_channels = in_channels
        self.cond_channels = cond_channels

    def training_step(self, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.diffusion(target, cond)

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple,
        cond: torch.Tensor,
        guidance_fn: Optional[Callable] = None,
        guidance_weight: float = 0.0,
        ddim_steps: int = 100,
        eta: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = cond.device

        def guidance_schedule(k, K_total):
            return cosine_guidance_schedule(k, K_total, guidance_weight)

        return self.diffusion.sample(
            shape=shape,
            cond=cond,
            guidance_fn=guidance_fn,
            guidance_weight=guidance_weight,
            guidance_schedule=guidance_schedule if guidance_fn is not None else None,
            ddim_steps=ddim_steps,
            eta=eta,
            device=device,
        )
