"""Configuration and hyperparameters for the diffusion model sampler.

Based on: "Instance-dependent Convergence Theory for Diffusion Models"
by Yuchen Jiao and Gen Li (2025).
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplerConfig:
    """Configuration for the randomized midpoint sampler."""

    # Number of total iterations T. The total iteration complexity is KN = 2T.
    T: int = 2000

    # Number of rounds K. Each round has N = 2T/K steps.
    K: int = 10

    # Constants for the randomized schedule (Eq. 8).
    # c_0 and c_1 are sufficiently large constants with c_1 / c_0 sufficiently large.
    c_0: float = 15.0
    c_1: float = 75.0

    # Whether to use exact score functions (no estimation error).
    use_exact_score: bool = True

    @property
    def N(self) -> int:
        """Number of steps per round. N = 2T / K."""
        return 2 * self.T // self.K

    @property
    def total_score_evals(self) -> int:
        """Total number of score evaluations = K * N = 2T."""
        return self.K * self.N


@dataclass
class DataConfig:
    """Configuration for the target distribution (Gaussian)."""

    # Data dimension.
    d: int = 100

    # Number of non-zero variance components.
    k: int = 10

    # Variances for the first k components are drawn from Uniform[0, sigma_max].
    sigma_max: float = 10.0

    # Random seed for reproducibility.
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Configuration for the numerical experiment."""

    # Range of T values to test.
    T_values: tuple = (500, 1000, 2000, 4000, 8000)

    # Number of Monte Carlo samples for estimating the KL divergence.
    num_mc_samples: int = 10000

    # Data dimensions to test.
    d_values: tuple = (10, 100, 500)

    # Number of non-zero variance components for each dimension.
    k_values: tuple = (10, 10, 100)

    # Number of rounds K.
    K: int = 10


DEFAULT_SAMPLER_CONFIG = SamplerConfig()
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()
