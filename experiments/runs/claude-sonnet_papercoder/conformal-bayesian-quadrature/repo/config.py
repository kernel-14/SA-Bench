## config.py
"""Configuration dataclasses for reproducing 'Conformal Prediction as Bayesian Quadrature'.

All hyperparameter values are sourced directly from the paper or config.yaml.
This file has no dependencies on other project files.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExperimentConfig:
    """Base configuration shared across all experiments.

    Attributes:
        exp_name: Identifier string for the experiment.
        M: Number of random trials (paper Section 5: M = 10,000).
        n_cal: Number of calibration samples per trial.
        alpha: Target risk level.
        beta: Confidence level for the CBQ-HPD decision rule (paper Section 5: β = 0.95).
        B: Upper bound on individual losses (all experiments use B = 1.0).
        n_mc_samples: Monte Carlo samples for L+ estimation in decision rule
            (paper Section 5: 1000 samples).
        n_mc_figure: Monte Carlo samples for Figure 4 density plots
            (paper Section 5.1: 100,000 Dirichlet samples).
        seed: Base random seed for reproducibility (config.yaml: 42).
        lambda_grid: Sorted grid of lambda values for the infimum search.
        n_jobs: Number of parallel jobs for joblib (-1 uses all cores).
    """

    exp_name: str = ""
    M: int = 10000
    n_cal: int = 10
    alpha: float = 0.1
    beta: float = 0.95
    B: float = 1.0
    n_mc_samples: int = 1000
    n_mc_figure: int = 100000
    seed: int = 42
    lambda_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(0.0, 1.0, 500)
    )
    n_jobs: int = -1


@dataclass
class BinomialConfig(ExperimentConfig):
    """Configuration for Experiment 1: Synthetic Binomial Data (Section 5.1).

    The loss is ell(z_i, lambda) = (1/K) * sum_k 1{V_ik > lambda} where
    V_ik ~ Uniform(0, 1). The true expected loss is 1 - lambda, so risk
    exceeds alpha = 0.4 iff lambda < 0.6.

    Attributes:
        exp_name: Fixed identifier 'synthetic_binomial'.
        n_cal: Calibration set size (paper Section 5.1: n = 10).
        alpha: Target risk level (paper Section 5.1: α = 0.4).
        K: Binomial averaging parameter (paper Section 5.1: K = 4).
        lambda_grid: Grid over [0, 1] with 500 points (config.yaml).
    """

    exp_name: str = "synthetic_binomial"
    n_cal: int = 10
    alpha: float = 0.4
    K: int = 4
    lambda_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(0.0, 1.0, 500)
    )


@dataclass
class HeteroskedasticConfig(ExperimentConfig):
    """Configuration for Experiment 2: Synthetic Heteroskedastic Data (Section 5.2).

    Data: X ~ Uniform[x_low, x_high], Y | X ~ N(0, X^2).
    Prediction intervals: [-lambda, lambda].
    Loss: miscoverage loss = 1{|Y| > lambda}.
    Target: 90% coverage, i.e. alpha = 0.1.

    Attributes:
        exp_name: Fixed identifier 'synthetic_heteroskedastic'.
        n_cal: Calibration set size (paper Section 5.2: n = 200).
        alpha: Target miscoverage level (paper Section 5.2: α = 0.1).
        x_low: Lower bound of X distribution (paper Section 5.2: 0.0).
        x_high: Upper bound of X distribution (paper Section 5.2: 4.0).
        n_quad_true_risk: Quadrature points for numerical integration of true
            risk E_X[2*Phi(-lambda/X)] (config.yaml: 1000).
        lambda_grid: Grid over [0, 20] with 1000 points. Must extend beyond
            the RCPS solution (~7.15 based on Table 2 mean PI length / 2).
    """

    exp_name: str = "synthetic_heteroskedastic"
    n_cal: int = 200
    alpha: float = 0.1
    x_low: float = 0.0
    x_high: float = 4.0
    n_quad_true_risk: int = 1000
    lambda_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(0.0, 20.0, 1000)
    )


@dataclass
class MSCOCOConfig(ExperimentConfig):
    """Configuration for Experiment 3: False Negative Rate on MS-COCO (Section 5.3).

    Mirrors the setup from Angelopoulos & Bates (2023, Section 5.1).
    Uses precomputed softmax scores from the conformal-risk-control GitHub repo:
    https://github.com/aangelopoulos/conformal-risk-control

    Each random split contains n_cal calibration examples and n_test test examples.
    The loss is the per-sample false negative rate (FNR) for multilabel classification.

    Attributes:
        exp_name: Fixed identifier 'mscoco'.
        n_cal: Calibration examples per split (paper Section 5.3: 1000).
        n_test: Test examples per split (paper Section 5.3: 3952).
        alpha: Target FNR level (config.yaml: 0.1, mirrors Angelopoulos & Bates 2023).
        scores_path: Path to precomputed softmax scores .npy file, shape (N, C).
        labels_path: Path to binary ground-truth labels .npy file, shape (N, C).
        lambda_grid: Grid over [0, 1] with 500 points. Lambda is a score threshold;
            higher lambda yields smaller prediction sets and higher FNR.
    """

    exp_name: str = "mscoco"
    n_cal: int = 1000
    n_test: int = 3952
    alpha: float = 0.1
    scores_path: str = "data/mscoco/scores.npy"
    labels_path: str = "data/mscoco/labels.npy"
    lambda_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(0.0, 1.0, 500)
    )
