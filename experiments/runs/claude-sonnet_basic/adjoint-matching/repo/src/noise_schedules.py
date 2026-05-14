"""
Noise schedules for Flow Matching and Diffusion models.

This module implements the noise schedules described in the paper:
"Adjoint Matching: Fine-tuning Flow and Diffusion Models with Memoryless SOC"

Key schedules:
- Flow Matching (ODE): sigma(t) = 0
- Memoryless Flow Matching: sigma(t) = sqrt(2 * eta_t) where eta_t = beta_t * (alpha_dot_t/alpha_t * beta_t - beta_dot_t)
- DDIM: arbitrary sigma(t)
- DDPM: sigma(t) = sqrt(2 * eta_t) (same as memoryless)

For the linear interpolant (alpha_t = t, beta_t = 1-t):
  eta_t = (1-t) * ((1-t)/t + 1) = (1-t)^2/t + (1-t) = (1-t)/t
  sigma(t) = sqrt(2*(1-t)/t)

With the offset trick from Appendix G.1:
  sigma(t) = sqrt(2*(1-t+h)/(t+h))
"""

import torch
import math


class FlowMatchingSchedule:
    """
    Linear interpolant schedule: alpha_t = t, beta_t = 1 - t.
    
    Reference flow: X_bar_t = beta_t * X_0 + alpha_t * X_1
    where X_0 ~ N(0, I) and X_1 ~ p_data.
    """
    
    def __init__(self, num_steps: int = 40):
        self.num_steps = num_steps
        self.h = 1.0 / num_steps
        # Timesteps: 0, h, 2h, ..., (K-1)*h
        self.timesteps = torch.linspace(0, 1 - self.h, num_steps)
    
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """alpha_t = t"""
        return t
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """beta_t = 1 - t"""
        return 1.0 - t
    
    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        """d/dt alpha_t = 1"""
        return torch.ones_like(t)
    
    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        """d/dt beta_t = -1"""
        return -torch.ones_like(t)
    
    def kappa(self, t: torch.Tensor) -> torch.Tensor:
        """kappa_t = alpha_dot_t / alpha_t = 1/t"""
        return self.alpha_dot(t) / (self.alpha(t) + 1e-8)
    
    def eta(self, t: torch.Tensor) -> torch.Tensor:
        """
        eta_t = beta_t * (alpha_dot_t/alpha_t * beta_t - beta_dot_t)
        For linear: eta_t = (1-t) * ((1-t)/t + 1) = (1-t)/t
        """
        alpha = self.alpha(t)
        beta = self.beta(t)
        alpha_dot = self.alpha_dot(t)
        beta_dot = self.beta_dot(t)
        return beta * (alpha_dot / (alpha + 1e-8) * beta - beta_dot)
    
    def sigma_memoryless(self, t: torch.Tensor, offset: bool = True) -> torch.Tensor:
        """
        Memoryless noise schedule: sigma(t) = sqrt(2 * eta_t)
        
        For linear interpolant: sigma(t) = sqrt(2*(1-t)/t)
        
        With offset (Appendix G.1): sigma(t) = sqrt(2*(1-t+h)/(t+h))
        """
        h = self.h if offset else 0.0
        numerator = 2.0 * (1.0 - t + h)
        denominator = t + h
        return torch.sqrt(numerator / denominator)
    
    def sigma_ode(self, t: torch.Tensor) -> torch.Tensor:
        """ODE (noiseless) schedule: sigma(t) = 0"""
        return torch.zeros_like(t)
    
    def sigma_constant(self, t: torch.Tensor, value: float = 1.0) -> torch.Tensor:
        """Constant noise schedule (for ablation)"""
        return torch.full_like(t, value)


class DDIMSchedule:
    """
    DDIM/DDPM noise schedule.
    
    Uses alpha_bar_t (cumulative product of alphas).
    For DDPM: sigma(t) = sqrt(alpha_bar_dot_t / alpha_bar_t)
    """
    
    def __init__(self, num_steps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_steps = num_steps
        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)
        # Prepend 1.0 for alpha_bar_0
        self.alpha_bar = torch.cat([torch.tensor([1.0]), self.alpha_bar])
    
    def get_alpha_bar(self, k: int) -> torch.Tensor:
        """Get alpha_bar at step k"""
        return self.alpha_bar[k]
    
    def sigma_ddpm(self, k: int) -> torch.Tensor:
        """DDPM noise schedule at step k"""
        alpha_bar_k = self.alpha_bar[k]
        alpha_bar_k1 = self.alpha_bar[k + 1] if k + 1 < len(self.alpha_bar) else self.alpha_bar[k]
        # sigma_k = sqrt((1 - alpha_bar_{k+1}) / (1 - alpha_bar_k) * (1 - alpha_bar_k / alpha_bar_{k+1}))
        return torch.sqrt(
            (1 - alpha_bar_k1) / (1 - alpha_bar_k) * (1 - alpha_bar_k / alpha_bar_k1)
        )


def get_sigma_memoryless_fm(t: torch.Tensor, h: float = 0.025) -> torch.Tensor:
    """
    Compute the memoryless noise schedule for Flow Matching.
    
    For linear interpolant (alpha_t = t, beta_t = 1-t):
    sigma(t) = sqrt(2*(1-t+h)/(t+h))
    
    This is the key schedule from Proposition 1 and Theorem 1.
    
    Args:
        t: Time tensor in [0, 1]
        h: Step size offset (default 1/40 = 0.025)
    
    Returns:
        sigma(t) values
    """
    numerator = 2.0 * (1.0 - t + h)
    denominator = t + h
    return torch.sqrt(numerator / denominator)


def get_eta_fm(t: torch.Tensor) -> torch.Tensor:
    """
    Compute eta_t for Flow Matching with linear interpolant.
    
    eta_t = beta_t * (alpha_dot_t/alpha_t * beta_t - beta_dot_t)
    For alpha_t = t, beta_t = 1-t:
    eta_t = (1-t) * ((1-t)/t + 1) = (1-t)/t
    
    Args:
        t: Time tensor in [0, 1]
    
    Returns:
        eta_t values
    """
    return (1.0 - t) / (t + 1e-8)


def get_kappa_fm(t: torch.Tensor) -> torch.Tensor:
    """
    Compute kappa_t for Flow Matching with linear interpolant.
    
    kappa_t = alpha_dot_t / alpha_t = 1/t
    
    Args:
        t: Time tensor in [0, 1]
    
    Returns:
        kappa_t values
    """
    return 1.0 / (t + 1e-8)
