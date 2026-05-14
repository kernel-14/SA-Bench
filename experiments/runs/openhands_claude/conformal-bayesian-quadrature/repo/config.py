"""
All hyperparameters and configuration for reproducing the experiments from
"Conformal Prediction as Bayesian Quadrature" (Snell & Griffiths).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SyntheticBinomialConfig:
    """Section 5.1: Synthetic Binomial Data experiment."""

    # Number of calibration samples
    n: int = 10
    # Number of Bernoulli trials per loss evaluation
    K: int = 4
    # Target risk level
    alpha: float = 0.4
    # Upper bound on losses
    B: float = 1.0
    # Number of random trials
    M: int = 10_000
    # Confidence level for BQ-HPD
    beta: float = 0.95
    # Number of Dirichlet samples for BQ-HPD decision rule
    n_dirichlet_decision: int = 1000
    # Number of Dirichlet samples for L+ histogram (Figure 4)
    n_dirichlet_histogram: int = 100_000
    # Lambda values for L+ histogram (Figure 4)
    lambda_histogram: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9])
    # Lambda grid for searching decision rules
    lambda_min: float = 0.0
    lambda_max: float = 1.0
    lambda_steps: int = 1001
    # Random seed
    seed: int = 42


@dataclass
class SyntheticHeteroskedasticConfig:
    """Section 5.2: Synthetic Heteroskedastic Data experiment."""

    # Number of calibration samples
    n: int = 200
    # Target risk level (10% miscoverage = 90% coverage)
    alpha: float = 0.1
    # Upper bound on losses (binary miscoverage)
    B: float = 1.0
    # Number of random trials
    M: int = 10_000
    # Confidence level for BQ-HPD
    beta: float = 0.95
    # Number of Dirichlet samples for BQ-HPD decision rule
    n_dirichlet_decision: int = 1000
    # X distribution: X ~ Uniform(x_min, x_max)
    x_min: float = 0.0
    x_max: float = 4.0
    # Lambda grid for searching decision rules
    lambda_min: float = 0.0
    lambda_max: float = 20.0
    lambda_steps: int = 2001
    # Random seed
    seed: int = 42


@dataclass
class COCOConfig:
    """Section 5.3: False Negative Rate on MS-COCO experiment."""

    # Number of calibration examples per split
    n_calib: int = 1000
    # Number of test examples per split
    n_test: int = 3952
    # Target risk level (FNR)
    alpha: float = 0.1
    # Upper bound on losses (FNR in [0,1])
    B: float = 1.0
    # Number of random trials
    M: int = 10_000
    # Confidence level for BQ-HPD
    beta: float = 0.95
    # Number of Dirichlet samples for BQ-HPD decision rule
    n_dirichlet_decision: int = 1000
    # Lambda grid for searching decision rules (threshold on 1 - score)
    lambda_min: float = 0.0
    lambda_max: float = 1.0
    lambda_steps: int = 1001
    # Path to pre-computed COCO scores and labels
    data_path: str = "data/coco_scores.npz"
    # Random seed
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Top-level configuration."""

    synthetic_binomial: SyntheticBinomialConfig = field(
        default_factory=SyntheticBinomialConfig
    )
    synthetic_heteroskedastic: SyntheticHeteroskedasticConfig = field(
        default_factory=SyntheticHeteroskedasticConfig
    )
    coco: COCOConfig = field(default_factory=COCOConfig)
    # Which experiments to run
    run_synthetic_binomial: bool = True
    run_synthetic_heteroskedastic: bool = True
    run_coco: bool = True
    # Output directory for figures and results
    output_dir: str = "results"


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()
