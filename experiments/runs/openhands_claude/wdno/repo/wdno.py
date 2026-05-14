"""
WDNO: Wavelet Diffusion Neural Operator

Implements the full WDNO framework including:
  - Base-Resolution Model (BRM): diffusion in wavelet domain at training resolution
  - Super-Resolution Model (SRM): conditional diffusion for zero-shot super-resolution

The BRM and SRM share the same architecture but differ in their conditioning:
  - BRM: conditioned on W_a (wavelet transform of equation parameters)
  - SRM: conditioned on W_a_h (high-res params) and W_l (low-res BRM output)

Simulation inference:
  1. Apply wavelet transform to condition a → W_a
  2. Run BRM DDIM sampling to get W_u^(0)
  3. Apply inverse wavelet transform → u

Control inference:
  1. Apply wavelet transform to condition a → W_a
  2. Run BRM DDIM sampling with guidance ∇_W I(Ŵ) → W_f^(0)
  3. Apply inverse wavelet transform → f

Super-resolution inference:
  1. Get BRM output at base resolution
  2. Run SRM conditioned on (W_l, W_a_h) to get W_h^(0)
  3. Repeat for each super-resolution level
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion import GaussianDiffusion, cosine_guidance_schedule
from models.unet_1d import UNet1D
from models.unet_3d import UNet3D
from wavelet.transforms import (
    apply_wavelet_2d,
    inverse_wavelet_2d,
    apply_wavelet_3d,
    inverse_wavelet_3d,
    apply_wavelet_1d,
    inverse_wavelet_1d,
    pad_to_match,
)


# ---------------------------------------------------------------------------
# WDNO for 1D PDE experiments
# ---------------------------------------------------------------------------

class WDNO1D(nn.Module):
    """
    WDNO for 1D PDE data (Burgers', advection, compressible NS).

    Data shape: [batch, T, X] (e.g. [batch, 81, 120] for Burgers')
    Wavelet: bior2.4, periodization, 2D DWT over (T, X)
    After transform: [batch, 4, T', X'] where T'≈T/2, X'≈X/2

    For simulation:
      - Target: u_{[0,T]} (state trajectory)
      - Condition: u_0 (initial condition) + f (force)

    For control:
      - Target: f_{[0,T]} (control sequence)
      - Condition: u_0 (initial condition) + u_T (target state)
      - Guidance: I(u, f) = ∫|u(T,x) - u*(x)|² dx + α ∫|f|² dt dx

    Args:
        mode: 'simulation' or 'control'
        wavelet: wavelet basis name
        wt_mode: wavelet transform padding mode
        n_state_channels: number of state variable channels (1 for scalar PDE)
        n_force_channels: number of force channels
        n_cond_channels: number of conditioning channels (after wavelet)
        diffusion_kwargs: kwargs for GaussianDiffusion
        unet_kwargs: kwargs for UNet1D
    """

    def __init__(
        self,
        mode: str = "simulation",
        wavelet: str = "bior2.4",
        wt_mode: str = "periodization",
        n_state_channels: int = 1,
        n_force_channels: int = 1,
        diffusion_kwargs: Optional[Dict] = None,
        unet_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.mode = mode
        self.wavelet = wavelet
        self.wt_mode = wt_mode
        self.n_state_channels = n_state_channels
        self.n_force_channels = n_force_channels

        # After 2D DWT: 4 subbands per channel
        n_wt_subbands = 4

        if mode == "simulation":
            # Target: u_{[0,T]} → 4 * n_state_channels wavelet channels
            # Condition: u_0 (1D → repeated) + f → 4 * (n_state_channels + n_force_channels)
            in_channels = n_wt_subbands * n_state_channels
            cond_channels = n_wt_subbands * (n_state_channels + n_force_channels)
        else:  # control
            # Target: f_{[0,T]} → 4 * n_force_channels
            # Condition: u_0 + u_T → 4 * 2 * n_state_channels
            in_channels = n_wt_subbands * n_force_channels
            cond_channels = n_wt_subbands * 2 * n_state_channels

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

    def _wavelet_transform_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D DWT to [batch, T, X] data.
        Returns [batch, 4, T', X'].
        """
        # Add channel dim: [batch, 1, T, X]
        x_c = x.unsqueeze(1)
        return apply_wavelet_2d(x_c, self.wavelet, self.wt_mode)

    def _wavelet_transform_1d_to_2d(self, x: torch.Tensor, target_shape: Tuple) -> torch.Tensor:
        """
        Apply 1D DWT to [batch, X] data, repeat to match 2D wavelet shape.
        Returns [batch, 4, T', X'] by repeating along T' dimension.
        """
        from pytorch_wavelets import DWT1DForward
        dwt1d = DWT1DForward(J=1, wave=self.wavelet, mode=self.wt_mode).to(x.device)
        x_c = x.unsqueeze(1)  # [batch, 1, X]
        yl, yh = dwt1d(x_c)   # yl: [batch, 1, X'], yh[0]: [batch, 1, 1, X']
        cD = yh[0][:, :, 0]   # [batch, 1, X']
        # Stack: [batch, 4, X'] (LL, LH, HL, HH approximated from 1D)
        # For 1D data, we have only 2 subbands; pad to 4 by repeating
        coeffs_1d = torch.cat([yl, cD, yl, cD], dim=1)  # [batch, 4, X']
        # Repeat along T' dimension
        T_prime = target_shape[-2]
        coeffs_2d = coeffs_1d.unsqueeze(2).expand(-1, -1, T_prime, -1)  # [batch, 4, T', X']
        return coeffs_2d

    def _inverse_wavelet_2d(self, w: torch.Tensor, n_channels: int = 1) -> torch.Tensor:
        """
        Apply inverse 2D DWT to [batch, 4*C, T', X'].
        Returns [batch, C, T, X].
        """
        return inverse_wavelet_2d(w, self.wavelet, self.wt_mode, n_channels)

    def prepare_condition_simulation(
        self,
        u0: torch.Tensor,
        f: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare conditioning for simulation.

        Args:
            u0: initial condition [batch, X]
            f: force trajectory [batch, T_f, X]
        Returns:
            cond: [batch, 4*(1+1), T', X']
        """
        # Wavelet transform of f: [batch, 4, T', X']
        w_f = self._wavelet_transform_2d(f)
        # Wavelet transform of u0 (1D → 2D): [batch, 4, T', X']
        w_u0 = self._wavelet_transform_1d_to_2d(u0, w_f.shape)
        return torch.cat([w_u0, w_f], dim=1)  # [batch, 8, T', X']

    def prepare_condition_control(
        self,
        u0: torch.Tensor,
        u_target: torch.Tensor,
        T_prime: int,
        X_prime: int,
    ) -> torch.Tensor:
        """
        Prepare conditioning for control.

        Args:
            u0: initial condition [batch, X]
            u_target: target state [batch, X]
            T_prime, X_prime: wavelet coefficient spatial dims
        Returns:
            cond: [batch, 8, T', X']
        """
        target_shape = (T_prime, X_prime)
        w_u0 = self._wavelet_transform_1d_to_2d(u0, (None, None, T_prime, X_prime))
        w_ut = self._wavelet_transform_1d_to_2d(u_target, (None, None, T_prime, X_prime))
        return torch.cat([w_u0, w_ut], dim=1)

    def training_step(
        self,
        target: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            target: wavelet coefficients of target [batch, 4*C, T', X']
            cond: wavelet coefficients of condition [batch, 4*C_cond, T', X']
        Returns:
            loss scalar
        """
        return self.diffusion(target, cond)

    @torch.no_grad()
    def simulate(
        self,
        cond: torch.Tensor,
        ddim_steps: int = 50,
        eta: float = 1.0,
        cfg_weight: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Run simulation inference.

        Args:
            cond: conditioning wavelet coefficients [batch, C_cond, T', X']
            ddim_steps: DDIM sampling steps
            eta: DDIM η
            cfg_weight: classifier-free guidance weight ω
        Returns:
            u_pred: predicted state trajectory [batch, T, X]
        """
        if device is None:
            device = cond.device
        batch = cond.shape[0]
        T_prime, X_prime = cond.shape[-2], cond.shape[-1]
        shape = (batch, self.in_channels, T_prime, X_prime)

        w_pred = self.diffusion.sample(
            shape=shape,
            cond=cond,
            cfg_weight=cfg_weight,
            ddim_steps=ddim_steps,
            eta=eta,
            device=device,
        )

        # Inverse wavelet transform
        u_pred = self._inverse_wavelet_2d(w_pred, self.n_state_channels)
        return u_pred.squeeze(1)  # [batch, T, X]

    def control(
        self,
        cond: torch.Tensor,
        guidance_fn: Callable,
        guidance_weight: float = 120000.0,
        ddim_steps: int = 50,
        eta: float = 1.0,
        cfg_weight: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Run control inference with guidance.

        Args:
            cond: conditioning wavelet coefficients [batch, C_cond, T', X']
            guidance_fn: callable(w_hat_0) → scalar loss I
            guidance_weight: λ (e.g. 120000 for 1D Burgers')
            ddim_steps: DDIM sampling steps
            eta: DDIM η
            cfg_weight: classifier-free guidance weight ω
        Returns:
            f_pred: predicted control sequence [batch, T_f, X]
        """
        if device is None:
            device = cond.device
        batch = cond.shape[0]
        T_prime, X_prime = cond.shape[-2], cond.shape[-1]
        shape = (batch, self.in_channels, T_prime, X_prime)

        K = self.diffusion.timesteps

        def guidance_schedule(k, K_total):
            return cosine_guidance_schedule(k, K_total, guidance_weight)

        w_pred = self.diffusion.sample(
            shape=shape,
            cond=cond,
            guidance_fn=guidance_fn,
            guidance_weight=guidance_weight,
            guidance_schedule=guidance_schedule,
            cfg_weight=cfg_weight,
            ddim_steps=ddim_steps,
            eta=eta,
            device=device,
        )

        # Inverse wavelet transform
        f_pred = self._inverse_wavelet_2d(w_pred, self.n_force_channels)
        return f_pred.squeeze(1)  # [batch, T_f, X]


# ---------------------------------------------------------------------------
# WDNO for 2D PDE experiments
# ---------------------------------------------------------------------------

class WDNO2D(nn.Module):
    """
    WDNO for 2D PDE data (incompressible fluid, ERA5).

    Data shape: [batch, T, H, W] (e.g. [batch, 32, 64, 64] for fluid)
    Wavelet: bior1.3, zero mode, 3D DWT over (T, H, W)
    After transform: [batch, 8, T', H', W'] where T'≈T/2, H'≈H/2, W'≈W/2

    For simulation:
      - Target: density + velocity + smoke_pct trajectory
      - Condition: initial density + control sequence

    For control:
      - Target: control sequence f_{[0,T]}
      - Condition: initial density
      - Guidance: percentage of smoke NOT passing through target bucket

    Args:
        mode: 'simulation' or 'control'
        wavelet: wavelet basis name
        wt_mode: wavelet transform padding mode
        n_target_channels: channels of target (e.g. 4 for density+vx+vy+smoke_pct)
        n_cond_channels: channels of condition (after wavelet)
        diffusion_kwargs: kwargs for GaussianDiffusion
        unet_kwargs: kwargs for UNet3D
    """

    def __init__(
        self,
        mode: str = "simulation",
        wavelet: str = "bior1.3",
        wt_mode: str = "zero",
        n_target_channels: int = 4,
        n_cond_channels: int = 8,
        diffusion_kwargs: Optional[Dict] = None,
        unet_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.mode = mode
        self.wavelet = wavelet
        self.wt_mode = wt_mode
        self.n_target_channels = n_target_channels
        self.n_cond_channels = n_cond_channels

        n_wt_subbands = 8  # 3D DWT produces 8 subbands

        in_channels = n_wt_subbands * n_target_channels
        cond_channels = n_wt_subbands * n_cond_channels

        unet_kw = unet_kwargs or {}
        model = UNet3D(
            in_channels=in_channels,
            cond_channels=cond_channels,
            **unet_kw,
        )

        diff_kw = diffusion_kwargs or {}
        self.diffusion = GaussianDiffusion(model=model, **diff_kw)

        self.in_channels = in_channels
        self.cond_channels_total = cond_channels

    def _wavelet_transform_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 3D DWT to [batch, C, T, H, W] data.
        Returns [batch, 8*C, T', H', W'].
        """
        return apply_wavelet_3d(x, self.wavelet, self.wt_mode)

    def _inverse_wavelet_3d(self, w: torch.Tensor, n_channels: int) -> torch.Tensor:
        """
        Apply inverse 3D DWT to [batch, 8*C, T', H', W'].
        Returns [batch, C, T, H, W].
        """
        return inverse_wavelet_3d(w, self.wavelet, self.wt_mode, n_channels)

    def training_step(
        self,
        target: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            target: wavelet coefficients of target [batch, 8*C, T', H', W']
            cond: wavelet coefficients of condition [batch, 8*C_cond, T', H', W']
        Returns:
            loss scalar
        """
        return self.diffusion(target, cond)

    @torch.no_grad()
    def simulate(
        self,
        cond: torch.Tensor,
        ddim_steps: int = 100,
        eta: float = 1.0,
        cfg_weight: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Run simulation inference.

        Returns:
            pred: predicted trajectory [batch, C, T, H, W]
        """
        if device is None:
            device = cond.device
        batch = cond.shape[0]
        T_prime, H_prime, W_prime = cond.shape[-3], cond.shape[-2], cond.shape[-1]
        shape = (batch, self.in_channels, T_prime, H_prime, W_prime)

        w_pred = self.diffusion.sample(
            shape=shape,
            cond=cond,
            cfg_weight=cfg_weight,
            ddim_steps=ddim_steps,
            eta=eta,
            device=device,
        )

        return self._inverse_wavelet_3d(w_pred, self.n_target_channels)

    def control(
        self,
        cond: torch.Tensor,
        guidance_fn: Callable,
        guidance_weight: float = 11500.0,
        ddim_steps: int = 100,
        eta: float = 1.0,
        cfg_weight: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Run control inference with guidance.

        Returns:
            f_pred: predicted control sequence [batch, C_f, T, H, W]
        """
        if device is None:
            device = cond.device
        batch = cond.shape[0]
        T_prime, H_prime, W_prime = cond.shape[-3], cond.shape[-2], cond.shape[-1]
        shape = (batch, self.in_channels, T_prime, H_prime, W_prime)

        def guidance_schedule(k, K_total):
            return cosine_guidance_schedule(k, K_total, guidance_weight)

        w_pred = self.diffusion.sample(
            shape=shape,
            cond=cond,
            guidance_fn=guidance_fn,
            guidance_weight=guidance_weight,
            guidance_schedule=guidance_schedule,
            cfg_weight=cfg_weight,
            ddim_steps=ddim_steps,
            eta=eta,
            device=device,
        )

        return self._inverse_wavelet_3d(w_pred, self.n_target_channels)


# ---------------------------------------------------------------------------
# Super-Resolution Model (SRM)
# ---------------------------------------------------------------------------

class SuperResolutionModel(nn.Module):
    """
    Super-Resolution Model (SRM) for zero-shot super-resolution.

    Models p(W_h | W_l, W_a_h) where:
      - W_h: high-resolution wavelet coefficients (target)
      - W_l: low-resolution wavelet coefficients (from BRM, upsampled to match W_h size)
      - W_a_h: high-resolution equation parameter wavelet coefficients

    The SRM is trained on pairs (W_l, W_h) obtained by downsampling the training data.
    During inference, it iteratively upsamples from base resolution.

    Args:
        base_model: WDNO1D or WDNO2D instance (determines architecture)
        is_3d: True for 2D PDE data (3D wavelet), False for 1D PDE data (2D wavelet)
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        is_3d: bool = False,
        wavelet: str = "bior2.4",
        wt_mode: str = "periodization",
        diffusion_kwargs: Optional[Dict] = None,
        unet_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.is_3d = is_3d
        self.wavelet = wavelet
        self.wt_mode = wt_mode
        self.in_channels = in_channels

        # SRM conditions on: W_l (upsampled) + W_a_h
        # So cond_channels = in_channels (for W_l) + original cond_channels (for W_a_h)
        srm_cond_channels = in_channels + cond_channels

        unet_kw = unet_kwargs or {}
        if is_3d:
            model = UNet3D(
                in_channels=in_channels,
                cond_channels=srm_cond_channels,
                **unet_kw,
            )
        else:
            model = UNet1D(
                in_channels=in_channels,
                cond_channels=srm_cond_channels,
                **unet_kw,
            )

        diff_kw = diffusion_kwargs or {}
        self.diffusion = GaussianDiffusion(model=model, **diff_kw)

    def training_step(
        self,
        w_high: torch.Tensor,
        w_low_upsampled: torch.Tensor,
        w_a_high: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SRM training loss.

        Args:
            w_high: high-res wavelet coefficients [batch, C, ...]
            w_low_upsampled: low-res coefficients upsampled to high-res shape [batch, C, ...]
            w_a_high: high-res equation parameter wavelet coefficients [batch, C_a, ...]
        Returns:
            loss scalar
        """
        cond = torch.cat([w_low_upsampled, w_a_high], dim=1)
        return self.diffusion(w_high, cond)

    @torch.no_grad()
    def upsample(
        self,
        w_low: torch.Tensor,
        w_a_high: torch.Tensor,
        ddim_steps: int = 50,
        eta: float = 1.0,
        cfg_weight: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate high-resolution wavelet coefficients from low-resolution ones.

        Args:
            w_low: low-res wavelet coefficients [batch, C, ...]
            w_a_high: high-res equation parameter wavelet coefficients [batch, C_a, ...]
            ddim_steps: DDIM sampling steps
            eta: DDIM η
        Returns:
            w_high: high-res wavelet coefficients [batch, C, ...]
        """
        if device is None:
            device = w_low.device

        # Upsample w_low to match w_a_high spatial dimensions
        w_low_up = pad_to_match(w_low, w_a_high)

        cond = torch.cat([w_low_up, w_a_high], dim=1)
        shape = (w_a_high.shape[0], self.in_channels, *w_a_high.shape[2:])

        return self.diffusion.sample(
            shape=shape,
            cond=cond,
            cfg_weight=cfg_weight,
            ddim_steps=ddim_steps,
            eta=eta,
            device=device,
        )
