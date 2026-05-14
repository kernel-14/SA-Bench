"""
Hyperparameters and configuration for all experiments.

All values are taken directly from the paper (Section 4, Appendix C).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Experiment1Config:
    """
    Experiment 1: Convergence with different state/action space sizes.
    Section 4 / Appendix C.1.
    """
    sizes: List[Tuple[int, int]] = field(
        default_factory=lambda: [(3, 3), (9, 9), (81, 81)]
    )
    n_iterations: int = 2000
    eta: float = 0.01          # step size (chosen to be < 1/L_2^Π)
    seed: int = 0


@dataclass
class Experiment2Config:
    """
    Experiment 2: Convergence with different reward variances.
    Section 4 / Appendix C.2.
    """
    S: int = 16
    A: int = 16
    n_iterations: int = 2000
    eta: float = 0.01
    seed: int = 42             # seed for random transition kernel


@dataclass
class Experiment3Config:
    """
    Experiment 3: Convergence with different transition kernels.
    Section 4 / Appendix C.3.
    """
    S: int = 16
    A: int = 16
    n_iterations: int = 3000
    eta: float = 0.01
    seed: int = 42


@dataclass
class GlobalConfig:
    """Top-level configuration."""
    exp1: Experiment1Config = field(default_factory=Experiment1Config)
    exp2: Experiment2Config = field(default_factory=Experiment2Config)
    exp3: Experiment3Config = field(default_factory=Experiment3Config)

    # Step-size safety factor: η = safety_factor / L_2^Π
    # Paper requires η < 1/L_2^Π; we use 0.5 as a conservative choice.
    step_size_safety_factor: float = 0.5

    # Number of random policies sampled when computing complexity constants
    n_complexity_samples: int = 100

    # Output directory for figures
    output_dir: str = "figures"

    # Random seed for initial policy
    init_seed: int = 0


CONFIG = GlobalConfig()
