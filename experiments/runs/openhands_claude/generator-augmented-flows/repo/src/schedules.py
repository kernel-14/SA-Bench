import math
from typing import Tuple

import torch
import numpy as np
from scipy.special import erf


class NoiseSchedule:
    """
    EDM-style noise schedule from Karras et al. (2022).

    σ_i = (σ_0^(1/ρ) + i/N * (σ_N^(1/ρ) - σ_0^(1/ρ)))^ρ

    with ρ=7, σ_0=0.002, σ_N=80.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(self, N: int, device: torch.device = None) -> torch.Tensor:
        """
        Compute N+1 noise levels σ_0, ..., σ_N.

        Args:
            N: number of intervals (N+1 noise levels)
            device: target device

        Returns:
            Tensor of shape (N+1,) with noise levels
        """
        i = torch.arange(N + 1, dtype=torch.float64)
        sigma_min_inv_rho = self.sigma_min ** (1.0 / self.rho)
        sigma_max_inv_rho = self.sigma_max ** (1.0 / self.rho)
        sigmas = (sigma_max_inv_rho + i / N * (sigma_min_inv_rho - sigma_max_inv_rho)) ** self.rho
        # Reverse so sigmas[0] = sigma_min, sigmas[N] = sigma_max
        sigmas = sigmas.flip(0)
        if device is not None:
            sigmas = sigmas.to(device)
        return sigmas.float()


class TimestepSchedule:
    """
    Exponential timestep schedule from Song and Dhariwal (2024).

    N(k) = min(s_0 * 2^floor(k/K'), s_1) + 1

    where K' = floor(K / (log2(s_1/s_0) + 1))
    """

    def __init__(
        self,
        total_steps: int,
        s0: int = 10,
        s1: int = 1280,
    ):
        self.total_steps = total_steps
        self.s0 = s0
        self.s1 = s1
        self.K_prime = math.floor(total_steps / (math.log2(s1 / s0) + 1))

    def get_N(self, step: int) -> int:
        """Get the number of timestep intervals at training step k."""
        return min(self.s0 * 2 ** math.floor(step / self.K_prime), self.s1) + 1


class TimestepSampler:
    """
    Discrete timestep sampling distribution from Song and Dhariwal (2024).

    p(σ_i) ∝ erf((log(σ_{i+1}) - P_mean) / (√2 * P_std))
              - erf((log(σ_i) - P_mean) / (√2 * P_std))

    This mimics the continuous lognormal distribution recommended by Karras et al. (2022).
    """

    def __init__(
        self,
        p_mean: float = -1.1,
        p_std: float = 2.0,
    ):
        self.p_mean = p_mean
        self.p_std = p_std

    def get_weights(self, sigmas: torch.Tensor) -> torch.Tensor:
        """
        Compute sampling weights for each timestep interval.

        Args:
            sigmas: noise levels of shape (N+1,), from sigma_min to sigma_max

        Returns:
            Normalized weights of shape (N,) for intervals [σ_i, σ_{i+1}]
        """
        log_sigmas = torch.log(sigmas.cpu().float()).numpy()
        sqrt2 = math.sqrt(2)
        weights = np.array([
            erf((log_sigmas[i + 1] - self.p_mean) / (sqrt2 * self.p_std))
            - erf((log_sigmas[i] - self.p_mean) / (sqrt2 * self.p_std))
            for i in range(len(sigmas) - 1)
        ])
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        return torch.from_numpy(weights).float()

    def sample_indices(
        self,
        sigmas: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample timestep indices according to the distribution.

        Args:
            sigmas: noise levels (N+1,)
            batch_size: number of indices to sample
            device: target device

        Returns:
            Sampled indices of shape (batch_size,), each in [0, N-1]
        """
        weights = self.get_weights(sigmas)
        indices = torch.multinomial(weights, batch_size, replacement=True)
        return indices.to(device)


class LossWeighting:
    """
    Loss weighting function λ(σ_i) = 1 / (σ_{i+1} - σ_i).

    Combined with the noise schedule, this emphasizes consistency at low noise levels.
    """

    @staticmethod
    def get_weights(sigmas: torch.Tensor) -> torch.Tensor:
        """
        Compute loss weights for each timestep interval.

        Args:
            sigmas: noise levels (N+1,) on any device

        Returns:
            Weights (N,) on the same device as sigmas, where weights[i] = 1 / (σ_{i+1} - σ_i)
        """
        return 1.0 / (sigmas[1:] - sigmas[:-1])
