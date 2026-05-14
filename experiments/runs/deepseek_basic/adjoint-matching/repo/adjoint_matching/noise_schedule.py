"""
Memoryless noise schedule for fine-tuning dynamical generative models.

This module implements the core theoretical contribution from Section 4.3:
Proposition 1 and Theorem 1 of the paper establish that the memoryless noise
schedule σ(t) = √(2η_t) is both sufficient and necessary for removing the
initial value function bias, ensuring convergence to the tilted distribution.

For Flow Matching:
    α_t = t, β_t = 1 - t
    η_t = β_t (α̇_t/α_t · β_t - β̇_t) = (1-t)((1-t)/t + 1) = (1-t)/t
    σ(t) = √(2η_t) = √(2(1-t)/t)

For DDIM / Diffusion:
    κ_t = α̇_t / (2ᾱ_t)
    η_t = α̇_t / (2ᾱ_t)
    σ(t) = √(2η_t) = √(α̇_t / ᾱ_t)

The DDIM memoryless noise schedule recovers the continuous-time limit of DDPM.
"""

import torch
import torch.nn as nn
import math


class NoiseSchedule:
    """Base class for noise schedules used in fine-tuning."""

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient σ(t)."""
        raise NotImplementedError

    def kappa(self, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient κ_t = α̇_t / α_t (FM) or ᾱ̇_t / (2ᾱ_t) (DDIM)."""
        raise NotImplementedError

    def eta(self, t: torch.Tensor) -> torch.Tensor:
        """Coefficient η_t = β_t (α̇_t/α_t · β_t - β̇_t) for FM, ᾱ̇_t/(2ᾱ_t) for DDIM."""
        raise NotImplementedError

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Interpolation coefficient α_t."""
        raise NotImplementedError

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Interpolation coefficient β_t."""
        raise NotImplementedError

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative of α_t."""
        raise NotImplementedError

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative of β_t."""
        raise NotImplementedError


class FlowMatchingNoiseSchedule(NoiseSchedule):
    """
    Noise schedule for Flow Matching models with α_t = t, β_t = 1 - t.

    The memoryless noise schedule is:
        σ(t) = √(2η_t) = √(2(1-t)/t)

    In practice, we use a numerically stable offset version:
        σ(t) = √(2(1-t+h)/(t+h))
    where h is the step size.
    """

    def __init__(self, num_steps: int = 40, offset: bool = True):
        """
        Args:
            num_steps: Number of discretization steps K.
            offset: Whether to use the numerically stable offset (Appendix G.1).
        """
        self.num_steps = num_steps
        self.h = 1.0 / num_steps
        self.offset = offset

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

    def kappa(self, t: torch.Tensor) -> torch.Tensor:
        # κ_t = α̇_t / α_t = 1/t
        return 1.0 / t

    def eta(self, t: torch.Tensor) -> torch.Tensor:
        # η_t = β_t (α̇_t/α_t · β_t - β̇_t)
        #     = (1-t)((1-t)/t + 1) = (1-t)/t
        return (1.0 - t) / t

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Memoryless noise schedule for Flow Matching.
        σ(t) = √(2η_t) = √(2(1-t)/t)

        With numerical offset (Appendix G.1):
        σ(t) = √(2(1-t+h)/(t+h))
        """
        if self.offset:
            # Numerically stable offset version from Appendix G.1
            t_safe = t + self.h
            one_minus_t_safe = 1.0 - t + self.h
        else:
            t_safe = t
            one_minus_t_safe = 1.0 - t

        return torch.sqrt(2.0 * one_minus_t_safe / t_safe)


class DDIMMemorylessNoiseSchedule(NoiseSchedule):
    """
    Noise schedule for DDIM/Diffusion models.

    In the DDIM framework:
        κ_t = ᾱ̇_t / (2ᾱ_t)
        η_t = ᾱ̇_t / (2ᾱ_t)
        σ(t) = √(2η_t) = √(ᾱ̇_t / ᾱ_t)

    With the standard cosine schedule for ᾱ_t.
    The memoryless noise schedule σ(t) = √(2η_t) recovers the
    continuous-time limit of DDPM (see Section 4.3, Table 1).
    """

    def __init__(
        self,
        num_steps: int = 40,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "linear",
    ):
        """
        Args:
            num_steps: Number of discretization steps K.
            beta_start: Starting value for β in the noise schedule.
            beta_end: Ending value for β in the noise schedule.
            schedule_type: Type of schedule ('linear' or 'cosine').
        """
        self.num_steps = num_steps

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule_type == "cosine":
            # Cosine schedule as in Nichol & Dhariwal (2021)
            steps = num_steps + 1
            s = 0.008
            t = torch.linspace(0, num_steps, steps)
            alphas_cumprod = torch.cos(
                ((t / num_steps) + s) / (1 + s) * math.pi * 0.5
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = torch.clamp(betas, max=0.999)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        # Register buffers
        self.register_buffer = lambda name, val: setattr(self, name, val)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = alphas_cumprod
        self.betas = betas

    def get_alpha_bar(self, t_idx: torch.Tensor) -> torch.Tensor:
        """Get ᾱ_t for discrete time indices."""
        return self.alphas_cumprod[t_idx.long()]

    def sigma(self, t_idx: torch.Tensor) -> torch.Tensor:
        """
        Memoryless noise schedule for DDIM.
        σ(t) = √(ᾱ̇_t / ᾱ_t)
        Approximates √(2η_t) for the DDIM framework.
        """
        alpha_bar = self.get_alpha_bar(t_idx)
        # Finite difference approximation of ᾱ̇_t / ᾱ_t
        # For DDPM: σ(t) = √(ᾱ̇_t / ᾱ_t)
        # This is the continuous-time limit of DDPM
        return torch.sqrt(self.betas[t_idx.long()] / (1.0 - alpha_bar + 1e-8))


def get_memoryless_noise_schedule(
    model_type: str = "flow_matching",
    num_steps: int = 40,
    offset: bool = True,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> NoiseSchedule:
    """
    Factory function to get the appropriate noise schedule.

    Args:
        model_type: "flow_matching" or "ddim"
        num_steps: Number of discretization steps
        offset: Whether to use numerical offset (FM only)
        beta_start, beta_end: Beta schedule parameters (DDIM only)

    Returns:
        Appropriate NoiseSchedule instance.
    """
    if model_type == "flow_matching":
        return FlowMatchingNoiseSchedule(num_steps=num_steps, offset=offset)
    elif model_type == "ddim":
        return DDIMMemorylessNoiseSchedule(
            num_steps=num_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
