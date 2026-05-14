"""
Scheduling functions for consistency model training.

Based on:
- Karras et al. (2022) EDM noise schedule
- Song and Dhariwal (2024) improved techniques for consistency models

Contains:
1. Noise schedule: σ_i = (σ_0^(1/ρ) + (i/N) * (σ_N^(1/ρ) - σ_0^(1/ρ)))^ρ
2. Weighting function: λ(σ_i) = 1 / (σ_{i+1} - σ_i)
3. Discretization schedule (exponential): N(k) = min(s_0 * 2^{⌊k/K'⌋}, s_1) + 1
4. Timestep sampling distribution (log-normal CDF difference)
"""

import torch
import numpy as np
from scipy.special import erf


def noise_schedule_karras(
    N: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> torch.Tensor:
    """
    Karras et al. (2022) noise schedule.

    σ_i = (σ_0^(1/ρ) + (i/N) * (σ_N^(1/ρ) - σ_0^(1/ρ)))^ρ

    Args:
        N: Number of noise levels (N+1 points from 0 to N)
        sigma_min: Minimum noise level σ_0
        sigma_max: Maximum noise level σ_N
        rho: Exponent parameter (default 7)

    Returns:
        Tensor of shape (N+1,) with noise levels σ_0, ..., σ_N
    """
    sigma_min_rho = sigma_min ** (1.0 / rho)
    sigma_max_rho = sigma_max ** (1.0 / rho)
    i = torch.arange(N + 1, dtype=torch.float32)
    sigmas = (sigma_min_rho + (i / N) * (sigma_max_rho - sigma_min_rho)) ** rho
    return sigmas


def weighting_function(sigmas: torch.Tensor) -> torch.Tensor:
    """
    Weighting function for the consistency loss.

    λ(σ_i) = 1 / (σ_{i+1} - σ_i)

    Args:
        sigmas: Noise levels σ_0, ..., σ_N of shape (N+1,)

    Returns:
        Weights λ_i for i = 0, ..., N-1 of shape (N,)
    """
    diffs = sigmas[1:] - sigmas[:-1]
    return 1.0 / diffs


def discretization_schedule(
    k: int,
    K: int,
    s0: int = 10,
    s1: int = 1280,
) -> int:
    """
    Exponential discretization schedule for the number of timesteps.

    N(k) = min(s_0 * 2^{⌊k/K'⌋}, s_1) + 1

    where K' = ⌊K / (log_2(s_1 / s_0) + 1)⌋

    Args:
        k: Current training step
        K: Total number of training steps
        s0: Initial number of timesteps
        s1: Final number of timesteps

    Returns:
        Number of timesteps N for step k
    """
    K_prime = int(K / (np.log2(s1 / s0) + 1))
    exponent = k // K_prime
    N = min(s0 * (2 ** exponent), s1) + 1
    return N


def timestep_sampling_distribution(
    sigmas: torch.Tensor,
    P_mean: float = -1.1,
    P_std: float = 2.0,
) -> torch.Tensor:
    """
    Timestep sampling distribution based on log-normal CDF.

    p(σ_i) ∝ erf((log(σ_{i+1}) - P_mean) / (√2 * P_std))
             - erf((log(σ_i) - P_mean) / (√2 * P_std))

    This distribution emphasizes intermediate timesteps with moderate noise levels.

    Args:
        sigmas: Noise levels of shape (N+1,)
        P_mean: Mean of the log-normal distribution
        P_std: Standard deviation of the log-normal distribution

    Returns:
        Probability distribution over timestep indices (0 to N-1) of shape (N,)
    """
    log_sigmas = torch.log(sigmas)
    log_prev = log_sigmas[:-1]
    log_next = log_sigmas[1:]

    z_prev = (log_prev - P_mean) / (np.sqrt(2) * P_std)
    z_next = (log_next - P_mean) / (np.sqrt(2) * P_std)

    # Erf-based CDF difference
    probs = torch.erf(z_next) - torch.erf(z_prev)
    probs = torch.clamp(probs, min=1e-12)  # Avoid zero probabilities
    probs = probs / probs.sum()

    return probs


def sample_timesteps(
    batch_size: int,
    sigmas: torch.Tensor,
    P_mean: float = -1.1,
    P_std: float = 2.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Sample timestep indices according to the log-normal distribution.

    Args:
        batch_size: Number of indices to sample
        sigmas: Noise levels
        P_mean: Mean parameter
        P_std: Std parameter
        device: Device for the output tensor

    Returns:
        Sampled indices of shape (batch_size,)
    """
    probs = timestep_sampling_distribution(sigmas, P_mean, P_std)
    N = len(probs)
    if device is not None:
        probs = probs.to(device)
    indices = torch.multinomial(probs, batch_size, replacement=True)
    return indices


def get_sigmas_for_indices(
    sigmas: torch.Tensor,
    indices: torch.Tensor,
) -> tuple:
    """
    Get the sigma values for given timestep indices.

    Args:
        sigmas: Full noise schedule of shape (N+1,)
        indices: Timestep indices i of shape (B,)

    Returns:
        (sigma_i, sigma_{i+1}) both of shape (B,)
    """
    sigma_i = sigmas[indices]  # σ_{t_i}
    sigma_next = sigmas[indices + 1]  # σ_{t_{i+1}}
    return sigma_i, sigma_next
