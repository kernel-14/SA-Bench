"""
Configuration for instance-dependent convergence experiments.

All hyperparameters are grounded in the paper:
  "Instance-dependent Convergence Theory for Diffusion Models"
  Yuchen Jiao, Gen Li (2025)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScheduleConfig:
    """Learning rate schedule parameters (Section 2.2, Eq. 8)."""
    c0: float = 5.0       # exponent for initial alpha_hat: alpha_hat_{T+1} = 1/T^{c0}
    c1: float = 50.0      # step-size coefficient; ratio c1/c0 must be sufficiently large
    c_R: float = 2.0      # moment bound exponent: E[||X_0||^2] < T^{c_R} (Assumption 1)


@dataclass
class SamplerConfig:
    """Randomized midpoint sampler parameters (Section 2.2)."""
    T: int = 1000          # total number of score evaluations (iteration complexity)
    K: int = 10            # number of rounds
    # N = 2T/K steps per round (derived)

    @property
    def N(self) -> int:
        return 2 * self.T // self.K


@dataclass
class ParallelSamplerConfig:
    """Parallel sampler parameters (Section 3.3, Theorem 2)."""
    N_parallel: int = 100  # number of parallel processors
    M: int = 20            # parallel iterations per round (M << N)
    K: int = 10            # number of rounds
    # T = K * N_parallel / 2 (derived)

    @property
    def T(self) -> int:
        return self.K * self.N_parallel // 2


@dataclass
class GaussianDataConfig:
    """Gaussian target distribution for numerical experiments (Appendix A)."""
    d: int = 10            # data dimension
    k: int = 10            # number of non-zero variance components
    sigma_min: float = 0.0 # minimum variance (zero for degenerate dims)
    sigma_max: float = 10.0 # maximum variance (uniform in [0, sigma_max])
    seed: int = 42


@dataclass
class GMMDataConfig:
    """Gaussian mixture model target distribution (Example 2)."""
    d: int = 10            # data dimension
    H: int = 5             # number of components
    sigma: float = 1.0     # component standard deviation
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    parallel: ParallelSamplerConfig = field(default_factory=ParallelSamplerConfig)
    gaussian_data: GaussianDataConfig = field(default_factory=GaussianDataConfig)
    gmm_data: GMMDataConfig = field(default_factory=GMMDataConfig)

    # Experiment settings
    T_values: list = field(default_factory=lambda: [50, 100, 200, 500, 1000, 2000, 5000])
    n_trials: int = 10     # number of independent trials for averaging
    output_dir: str = "results"
    seed: int = 0


# Preset configurations matching paper experiments (Appendix A, Figure 2)
EXPERIMENT_CONFIGS = {
    "fig2a": ExperimentConfig(
        gaussian_data=GaussianDataConfig(d=10, k=10),
        sampler=SamplerConfig(K=10),
        T_values=[50, 100, 200, 500, 1000, 2000, 5000],
    ),
    "fig2b": ExperimentConfig(
        gaussian_data=GaussianDataConfig(d=100, k=10),
        sampler=SamplerConfig(K=10),
        T_values=[50, 100, 200, 500, 1000, 2000, 5000],
    ),
    "fig2c": ExperimentConfig(
        gaussian_data=GaussianDataConfig(d=500, k=100),
        sampler=SamplerConfig(K=10),
        T_values=[100, 200, 500, 1000, 2000, 5000, 10000],
    ),
}

# Complexity bound comparison configurations (Figure 1, Figure 3)
COMPLEXITY_CONFIGS = {
    "fig1_left": {
        "epsilon": 1.0,
        "d_values": [10, 50, 100, 500],
        "L_range": (0.1, 1e6),
        "n_L_points": 200,
    },
    "fig1_right": {
        "L": float("inf"),
        "d": 100,
        "epsilon_range": (1e-3, 1.0),
        "n_eps_points": 100,
    },
    "fig3": {
        "d": 100,
        "T_values": ["O(d)", "O(d^1.5)", "O(d^2)"],
        "L_range": (0.1, 1e6),
        "n_L_points": 200,
    },
}
