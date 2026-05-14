"""
Score-based SDE implementations for the Simformer.

Implements VESDE and VPSDE as described in Song et al. (2021b):
"Score-Based Generative Modeling through Stochastic Differential Equations"

Parameters from the paper:
  sigma_max=15, sigma_min=0.0001 (VESDE)
  beta_min=0.01, beta_max=10 (VPSDE)
  Time interval: [1e-5, 1.0]
"""

import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod


class SDE(ABC):
    """Abstract base class for SDEs."""

    @abstractmethod
    def drift(self, x, t):
        """Drift coefficient f(x, t)."""
        pass

    @abstractmethod
    def diffusion(self, t):
        """Diffusion coefficient g(t)."""
        pass

    @abstractmethod
    def marginal_params(self, t):
        """Returns (mean_coeff, std) such that x_t = mean_coeff * x_0 + std * eps."""
        pass

    def marginal_score(self, x_t, x_0, t):
        """Analytical score of p_t(x_t | x_0)."""
        mu, sigma = self.marginal_params(t)
        return -(x_t - mu * x_0) / (sigma ** 2)

    def prior_sample(self, key, shape):
        """Sample from the prior distribution p_T."""
        mu_T, sigma_T = self.marginal_params(self.T)
        return jax.random.normal(key, shape) * sigma_T

    @property
    def T(self):
        return 1.0

    @property
    def t_min(self):
        return 1e-5


class VESDE(SDE):
    """
    Variance Exploding SDE.

    f(x, t) = 0
    g(t) = sigma_min * (sigma_max / sigma_min)^t * sqrt(2 * log(sigma_max / sigma_min))

    Marginal: x_t = x_0 + sigma(t) * eps
    where sigma(t) = sigma_min * (sigma_max / sigma_min)^t
    """

    def __init__(self, sigma_min: float = 0.0001, sigma_max: float = 15.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def drift(self, x, t):
        return jnp.zeros_like(x)

    def diffusion(self, t):
        return self.sigma(t) * jnp.sqrt(2 * jnp.log(self.sigma_max / self.sigma_min))

    def marginal_params(self, t):
        """x_t = x_0 + sigma(t) * eps, so mean_coeff=1, std=sigma(t)."""
        mu = jnp.ones_like(t) if hasattr(t, 'shape') else 1.0
        std = self.sigma(t)
        return mu, std

    def prior_sample(self, key, shape):
        sigma_T = self.sigma(self.T)
        return jax.random.normal(key, shape) * sigma_T


class VPSDE(SDE):
    """
    Variance Preserving SDE.

    f(x, t) = -0.5 * beta(t) * x
    g(t) = sqrt(beta(t))
    beta(t) = beta_min + t * (beta_max - beta_min)

    Marginal: x_t = sqrt(alpha_bar(t)) * x_0 + sqrt(1 - alpha_bar(t)) * eps
    where alpha_bar(t) = exp(-0.5 * integral_0^t beta(s) ds)
    """

    def __init__(self, beta_min: float = 0.01, beta_max: float = 10.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def log_alpha_bar(self, t):
        """log alpha_bar(t) = -0.5 * integral_0^t beta(s) ds"""
        return -0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2)

    def drift(self, x, t):
        return -0.5 * self.beta(t) * x

    def diffusion(self, t):
        return jnp.sqrt(self.beta(t))

    def marginal_params(self, t):
        """x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps"""
        log_ab = self.log_alpha_bar(t)
        alpha_bar = jnp.exp(log_ab)
        mu = jnp.sqrt(alpha_bar)
        std = jnp.sqrt(1.0 - alpha_bar)
        return mu, std

    def prior_sample(self, key, shape):
        return jax.random.normal(key, shape)


def get_sde(sde_type: str = "vesde", **kwargs) -> SDE:
    """Factory function to get an SDE by name."""
    if sde_type.lower() == "vesde":
        return VESDE(**kwargs)
    elif sde_type.lower() == "vpsde":
        return VPSDE(**kwargs)
    else:
        raise ValueError(f"Unknown SDE type: {sde_type}")
