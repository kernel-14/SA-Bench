"""
WDNO for Control Tasks.

For control, we want to find the optimal external force f_{[0,T]} that minimizes
some objective I(u, f), given an environment determined by parameter a.

We learn p(W_f | W_a) and during inference, we use guidance to steer
generation towards smaller I.

Section 3.1, Eq. 4-5: The denoising update for control:
    W_f^{(k-1)} = W_f^{(k)} - η(ε_θ(W_f^{(k)}, W_a, k) + λ ∇I(Ŵ_f^{(k)})) + ξ

where Ŵ_f^{(k)} is the predicted clean wavelet coefficients:
    Ŵ_f^{(k)} = (W_f^{(k)} - √(1-ᾱ_k) ε_θ(W_f^{(k)}, W_a, k)) / √ᾱ_k
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional, Callable

from .wdno_base import WDNO
from .diffusion import extract


class WDNOControl(WDNO):
    """
    WDNO for PDE control.

    Energy-guided diffusion in wavelet domain for control tasks.

    During training: learns p(W_f | W_a) from data.
    During inference: applies guidance ∇I to steer generation towards optimal control.

    The control objective I (Eq. 6 for 1D Burgers):
        I = ∫_D |u(T,x) - u*(x)|² dx + α ∫_{[0,T]×D} |f(t,x)|² dt dx

    Args:
        objective_fn: Function computing control objective I
        guidance_weight: Weight λ for guidance gradient
        solver: Optional ground-truth solver for evaluating I
    """

    def __init__(
        self,
        objective_fn: Optional[Callable] = None,
        guidance_weight: float = 120000.0,
        solver: Optional[Callable] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.objective_fn = objective_fn
        self.guidance_weight = guidance_weight
        self.solver = solver

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Training step for control.

        batch contains:
            - 'data': Control sequence f_{[0,T]}
            - 'condition': Environment parameters a (initial condition u_0 + target u_T)

        Returns:
            Diffusion loss
        """
        return super().training_step(batch)

    def compute_guidance(
        self,
        w_f_pred: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the guidance gradient for control optimization.

        Following Eq. 4-5: compute ∇_{W_f} I(Ŵ_f^{(k)}) where Ŵ_f is the
        predicted clean wavelet coefficients.

        Args:
            w_f_pred: Predicted clean wavelet coefficients Ŵ_f^{(k)}
            condition: Conditioning information

        Returns:
            Gradient tensor of same shape as w_f_pred
        """
        if self.objective_fn is None:
            return torch.zeros_like(w_f_pred)

        # Decode wavelet to original space for objective computation
        f_pred = self.wavelet_decode(w_f_pred)

        # Compute objective
        # For 1D Burgers control: I = ∫|u(T,x) - u*(x)|²dx + α∫|f|²dtdx
        if self.solver is not None:
            # Use solver to compute final state
            u_T = self.solver(f_pred, condition)
            I_val = self.objective_fn(u_T, f_pred, condition)
        else:
            I_val = self.objective_fn(f_pred, condition)

        return I_val

    def forward(
        self,
        condition: torch.Tensor,
        guidance_weight: Optional[float] = None,
        return_wavelet: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate optimal control sequence.

        Uses energy-guided diffusion: combines the learned denoising
        with gradient guidance from the objective function.

        Args:
            condition: Environment parameters a
            guidance_weight: Override for guidance weight λ
            return_wavelet: If True, return wavelet coefficients instead of original space

        Returns:
            Control sequence f_{[0,T]} in original space (or wavelet space)
        """
        if guidance_weight is None:
            guidance_weight = self.guidance_weight

        W_cond = self.prepare_conditioning(condition)

        # Create guidance function
        def guidance_fn(w_f_pred):
            # w_f_pred is the predicted clean wavelet coefficients
            return self.compute_guidance(w_f_pred, condition)

        # Sample with guidance
        W_sample = self.ddim_sampler.sample(
            shape=(condition.shape[0],) + self.wavelet_shape,
            conditioning={'cond': W_cond},
            device=condition.device,
            guidance_fn=guidance_fn,
            guidance_weight=guidance_weight,
            **kwargs
        )

        if return_wavelet:
            return W_sample

        # Inverse wavelet transform
        return self.wavelet_decode(W_sample)

    def set_objective(self, objective_fn: Callable):
        """Set or update the objective function."""
        self.objective_fn = objective_fn

    def set_guidance_weight(self, weight: float):
        """Set the guidance weight λ."""
        self.guidance_weight = weight
