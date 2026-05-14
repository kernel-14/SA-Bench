"""
WDNO for Simulation Tasks.

For simulation, the objective is to learn a mapping from the equation parameter
function a (e.g., initial condition u_0 and force f) to the solution function u_{[0,T]}.

We learn the conditional probability p(W_u | W_a) in the wavelet domain
using classifier-free conditioning.

Section 3.1, Eq. 3: The denoising update for simulation:
    W_u^{(k-1)} = W_u^{(k)} - η * ε_θ(W_u^{(k)}, W_a, k) + ξ
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional

from .wdno_base import WDNO


class WDNOSimulation(WDNO):
    """
    WDNO for PDE simulation.

    Learns p(W_{u_{[0,T]}} | W_a) where:
    - W_u: Wavelet coefficients of the solution trajectory
    - W_a: Wavelet coefficients of the parameter function (e.g., initial condition + force)

    The model conditions on the equation parameters a and generates the full trajectory
    u_{[0,T]} in one shot, leveraging the diffusion model's ability to capture
    long-term dependencies.

    Additional losses on initial condition matching can be applied during
    training to better satisfy initial conditions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Training step for simulation.

        batch contains:
            - 'data': Full trajectory u_{[0,T]} of shape (B, C, T, X) or (B, C, T, H, W)
            - 'condition': Equation parameters a of shape (B, C_cond, T, X) or similar

        Returns:
            Diffusion loss
        """
        return super().training_step(batch)

    def forward(
        self,
        condition: torch.Tensor,
        guidance_weight: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Simulate the PDE trajectory given equation parameters.

        Args:
            condition: Equation parameters a (wavelet-transformed internally)
            guidance_weight: Optional classifier-free guidance weight

        Returns:
            Simulated trajectory u_{[0,T]} in original space
        """
        return super().forward(condition, guidance_weight=guidance_weight, **kwargs)

    def simulate_with_super_resolution(
        self,
        condition: torch.Tensor,
        super_res_model: 'SuperResolutionModel',
        n_sr_steps: int = 1,
    ) -> torch.Tensor:
        """
        Perform simulation with zero-shot super-resolution.

        Following Section 3.2 inference procedure:
        1. Downsample condition to base resolution
        2. Generate base-resolution trajectory using BRM
        3. Iteratively apply SRM to reach target resolution

        Args:
            condition: High-resolution equation parameters
            super_res_model: Super-Resolution Model
            n_sr_steps: Number of super-resolution steps

        Returns:
            Simulated trajectory at target resolution
        """
        # Step 1: Downsample to base resolution
        lo_condition = self._downsample_condition(condition)

        # Step 2: Generate base-resolution trajectory
        lo_trajectory = self.forward(lo_condition)

        # Step 3: Iteratively super-resolve
        hi_trajectory = lo_trajectory
        for step in range(n_sr_steps):
            hi_trajectory = super_res_model.super_resolve(
                lo_data=hi_trajectory,
                hi_condition=condition,  # or appropriately scaled condition
            )
            # Update condition scaling for next step if needed

        return hi_trajectory

    def _downsample_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """Downsample condition to base resolution."""
        # Simple average pooling or interpolation
        if condition.dim() == 4:  # (B, C, H, W)
            return torch.nn.functional.interpolate(
                condition, scale_factor=0.5, mode='bilinear', align_corners=False
            )
        elif condition.dim() == 5:  # (B, C, T, H, W)
            return torch.nn.functional.interpolate(
                condition, scale_factor=0.5, mode='trilinear', align_corners=False
            )
        return condition
