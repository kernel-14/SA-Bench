"""
schedules.py – Noise schedule generation and timestep discretization for 
improved consistency training (iCT).

This module implements the Schedules class described in the paper's appendix,
which provides:

- EDM‑style discrete noise levels (get_discrete_sigma).
- Log‑normal probability distribution over timestep intervals 
  (timestep_sampling_distribution).
- Exponential schedule for gradually increasing the number of discrete 
  timesteps during training (current_N).

All formulas follow Karras et al. (2022) and Song & Dhariwal (2024).
"""

import math
import torch
from torch import Tensor
from typing import Optional

# Fallback defaults – the actual values are expected to be passed via the
# configuration file (config.yaml) or constructor arguments.
# These match the paper’s standard settings.
from utils import P_MEAN, P_STD


class Schedules:
    """
    Manages noise schedule discretisation and timestep sampling.

    Constructor parameters (from config.yaml):
        sigma_min : minimum noise level σ₀ (default 0.002).
        sigma_max : maximum noise level σ_T (default 80).
        rho       : exponent controlling spacing (default 7).
        s0        : initial number of discrete intervals (default 10).
        s1        : final number of discrete intervals (default 1280).
        total_steps : total training iterations.
        p_mean    : mean of log‑normal timestep distribution (default from utils).
        p_std     : std of log‑normal timestep distribution (default from utils).

    The class pre‑computes K' (K_prime) which controls how frequently the
    number of timesteps doubles.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        s0: int = 10,
        s1: int = 1280,
        total_steps: int = 100_000,
        p_mean: Optional[float] = None,
        p_std: Optional[float] = None,
    ) -> None:
        # Core schedule parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.s0 = s0
        self.s1 = s1
        self.total_steps = total_steps

        # Timestep sampling distribution parameters
        self.p_mean = p_mean if p_mean is not None else P_MEAN
        self.p_std = p_std if p_std is not None else P_STD

        # Pre‑compute K' for the exponential discretisation schedule
        # K' = floor( total_steps / ( log2(s1/s0) + 1 ) )
        self.K_prime: int = int(
            math.floor(total_steps / (math.log2(s1 / s0) + 1))
        )

    def get_discrete_sigma(self, N: int) -> Tensor:
        """
        Build the discrete noise schedule σ₀, σ₁, …, σ_N.

        The levels are evenly spaced in a 1/ρ‑transformed space, giving more
        resolution at low noise levels – as recommended by Karras et al.

        Args:
            N : number of intervals (the returned tensor has length N + 1).

        Returns:
            sigmas : 1D tensor of shape (N+1,) containing the noise levels.
        """
        # Pre‑compute powers
        sigma_min_pow = self.sigma_min ** (1.0 / self.rho)
        sigma_max_pow = self.sigma_max ** (1.0 / self.rho)

        # Indices 0 .. N as float32
        indices = torch.arange(0, N + 1, dtype=torch.float32)

        # Interpolate in the transformed space
        step = (sigma_max_pow - sigma_min_pow) / N
        vals = sigma_min_pow + indices * step

        # Return to original domain
        sigmas = vals.pow(self.rho)
        return sigmas

    def timestep_sampling_distribution(self, N: int) -> Tensor:
        """
        Compute a categorical distribution over consecutive timestep pairs.

        The probabilities are proportional to:
            erf( (log σ_{i+1} - P_mean) / (√2 P_std) )
          - erf( (log σ_i     - P_mean) / (√2 P_std) )

        as in the continuous‑time weighting of EDM.

        Args:
            N : number of intervals (i.e., the schedule has N+1 sigma values).

        Returns:
            probs : 1D tensor of shape (N,) summing to 1.
        """
        sigmas = self.get_discrete_sigma(N)          # length N+1
        sigma_i = sigmas[:-1]                        # σ_0 … σ_{N-1}
        sigma_ip1 = sigmas[1:]                       # σ_1 … σ_N

        # Logarithmic terms
        log_i = torch.log(sigma_i)
        log_ip1 = torch.log(sigma_ip1)

        # Normalised arguments for the error function
        sqrt2 = math.sqrt(2.0)
        term_ip1 = (log_ip1 - self.p_mean) / (sqrt2 * self.p_std)
        term_i   = (log_i   - self.p_mean) / (sqrt2 * self.p_std)

        # Difference of error functions
        p = torch.erf(term_ip1) - torch.erf(term_i)

        # Numerical safety: clamp small negative values to zero
        p = torch.clamp(p, min=0.0)

        # Normalise to a valid probability distribution
        probs = p / p.sum()
        return probs

    def current_N(self, step: int) -> int:
        """
        Determine the number of discrete timesteps N for the given training step.

        The schedule grows exponentially:
            N(step) = min( s0 * 2^{⌊step / K'⌋}, s1 ) + 1

        Args:
            step : current training iteration (0‑based).

        Returns:
            N : integer number of intervals (so the sigma array will have N+1
                entries).
        """
        # Exponent index
        idx = step // self.K_prime

        # Raw number of intervals (capped at s1)
        N_raw = min(self.s0 * (2 ** idx), self.s1)

        # Return number of intervals, ensuring at least 1
        return N_raw + 1
