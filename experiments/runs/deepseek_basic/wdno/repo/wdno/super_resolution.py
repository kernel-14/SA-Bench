"""
Super-Resolution Model (SRM) for WDNO.

Implements the multi-resolution training framework described in Section 3.2.

Key ideas:
1. Approximate scale invariance: PDE dynamics follow approximately the same
   pattern across resolutions after appropriate rescaling.
2. Multi-resolution training: Train on pairs of (low-res, high-res) data
   created by downsampling the original dataset.
3. Zero-shot super-resolution: At inference, iteratively apply SRM to reach
   resolutions beyond what was seen during training.

The SRM is a conditional diffusion model that learns:
    p(W_h | W_l, W_{a_h})

where:
- W_h: Wavelet coefficients of high-resolution data
- W_l: Wavelet coefficients of low-resolution data (duplicated to match size)
- W_{a_h}: Wavelet coefficients of high-resolution equation parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional

from .wdno_base import WDNO
from .wavelet_transform import duplicate_low_res_to_match


class SuperResolutionModel(WDNO):
    """
    Super-Resolution Model for zero-shot super-resolution.

    Learns to map from low-resolution wavelet coefficients to high-resolution
    wavelet coefficients, conditioned on high-resolution equation parameters.

    During training:
        - Data pairs (lo_res, hi_res) from multi-resolution dataset
        - Low-res data is duplicated (repeated) to match high-res spatial dims
        - Model learns p(W_h | W_l, W_{a_h})

    During inference:
        - Takes base-resolution output from BRM
        - Conditions on duplicated low-res wavelet coeffs + hi-res parameters
        - Generates high-resolution result
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_multi_res_batch(
        self,
        hi_data: torch.Tensor,
        lo_data: torch.Tensor,
        hi_cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare a multi-resolution training batch.

        Following Section 3.2 Training:
        - Apply wavelet transform to hi_data, lo_data, hi_cond
        - Duplicate lo_data wavelet coeffs to match hi_data shape
        - Concatenate lo coefficients with hi condition as the full condition

        Args:
            hi_data: High-resolution trajectory (B, C, H, W) or (B, C, T, H, W)
            lo_data: Low-resolution trajectory (B, C, H/2, W/2) or ...
            hi_cond: High-resolution condition

        Returns:
            Batch dict with 'data' and 'condition' keys
        """
        # Wavelet encode
        W_hi = self.wavelet_encode(hi_data)
        W_lo = self.wavelet_encode(lo_data)
        W_cond = self.prepare_conditioning(hi_cond)

        # Duplicate low-res to match high-res shape
        W_lo_dup = duplicate_low_res_to_match(W_lo, W_hi.shape)

        # Concatenate: condition = [W_lo_dup, W_cond]
        W_full_cond = torch.cat([W_lo_dup, W_cond], dim=1)

        return {
            'data': W_hi,
            'condition': W_full_cond,
        }

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Training step for super-resolution.

        Expects batch with 'hi_data', 'lo_data', 'hi_cond' keys.
        """
        hi_data = batch['hi_data']
        lo_data = batch['lo_data']
        hi_cond = batch['hi_cond']

        processed = self.prepare_multi_res_batch(hi_data, lo_data, hi_cond)

        # Note: we need to handle the increased conditioning channels
        # The UNet was built with self.wavelet_cond_channels, but now
        # we have extra channels from lo_data
        # We adjust by building a separate UNet or dynamically handling it

        # For now, we use the base diffusion with the expanded conditioning
        W_data = processed['data']
        W_cond = processed['condition']

        # Since cond channels differ, we use a modified forward
        b = W_data.shape[0]
        t = torch.randint(1, self.diffusion.timesteps, (b,), device=W_data.device)
        noise = torch.randn_like(W_data)

        x_t = self.diffusion.q_sample(x_start=W_data, t=t, noise=noise)

        # Concatenate input with condition
        x_input = torch.cat([x_t, W_cond], dim=1)
        predicted_noise = self.denoise_fn(x_input, t)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def super_resolve(
        self,
        lo_data: torch.Tensor,
        hi_condition: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform super-resolution inference.

        Args:
            lo_data: Low-resolution data (from BRM output)
            hi_condition: High-resolution equation parameters

        Returns:
            High-resolution data
        """
        # Wavelet encode
        W_lo = self.wavelet_encode(lo_data)
        W_hi_cond = self.prepare_conditioning(hi_condition)

        # Compute target shape for high-res wavelet coefficients
        # Each SR step doubles spatial resolution
        hi_shape = list(W_lo.shape)
        if self.is_3d:
            hi_shape[2] *= 2  # T
            hi_shape[3] *= 2  # H
            hi_shape[4] *= 2  # W
        else:
            hi_shape[2] *= 2  # H
            hi_shape[3] *= 2  # W

        # Duplicate W_lo to match hi_shape
        W_lo_dup = duplicate_low_res_to_match(W_lo, tuple(hi_shape))

        # Full condition
        W_full_cond = torch.cat([W_lo_dup, W_hi_cond], dim=1)

        # Sample from diffusion
        W_sample = torch.randn(tuple(hi_shape), device=lo_data.device)

        # DDIM sampling with expanded conditioning
        b = W_sample.shape[0]
        for i in range(self.ddim_sampler.ddim_timesteps - 1, -1, -1):
            t = torch.full((b,), self.ddim_sampler.ddim_timesteps_tensor[i].item(),
                          device=W_sample.device, dtype=torch.long)
            t_prev = torch.full((b,), self.ddim_sampler.ddim_timesteps_prev_tensor[i].item(),
                               device=W_sample.device, dtype=torch.long)

            x_input = torch.cat([W_sample, W_full_cond], dim=1)
            noise_pred = self.denoise_fn(x_input, t)

            noise = torch.randn_like(W_sample) if i > 0 else torch.zeros_like(W_sample)
            W_sample = self.ddim_sampler.ddim_step(W_sample, t, t_prev, noise_pred, noise)

        # Inverse wavelet transform
        return self.wavelet_decode(W_sample)


def create_multi_resolution_dataset(
    data: torch.Tensor,
    n_levels: int = 3,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create multi-resolution training dataset through downsampling.

    Following Section 3.2:
    Given data at resolution N×M, create pairs:
    - (N×M, N/2×M/2)
    - (N/2×M/2, N/4×M/4)
    - (N/4×M/4, N/8×M/8)
    etc.

    Args:
        data: Original data at highest resolution
        n_levels: Number of resolution levels

    Returns:
        List of (hi_res, lo_res) tuples
    """
    pairs = []
    current = data
    for level in range(n_levels):
        if current.shape[-1] < 4 or current.shape[-2] < 4:
            break

        # Downsample spatial dimensions by factor 2
        if current.dim() == 4:  # (N, C, H, W)
            lo_res = F.interpolate(current, scale_factor=0.5, mode='bilinear',
                                   align_corners=False)
        elif current.dim() == 5:  # (N, C, T, H, W)
            lo_res = F.interpolate(current, scale_factor=0.5, mode='trilinear',
                                   align_corners=False)
        else:
            raise ValueError(f"Unsupported data dim: {current.dim()}")

        pairs.append((current, lo_res))
        current = lo_res

    return pairs
