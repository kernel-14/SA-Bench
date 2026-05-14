# utils.py
"""
Low‑level statistical utilities for the conformal prediction experiments.

The module provides:

* Dirichlet sampling for Bayesian quadrature (L^+ distribution).
* A uniform λ‑grid generator.
* Exact Clopper‑Pearson confidence intervals for binomial proportions.

All functions are stateless, deterministic given a seed, and rely only on
numpy and scipy.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats


def dirichlet_sampling(
    n_losses: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw `n_samples` samples from a Dir(1, 1, ..., 1) distribution of
    dimension `n_losses + 1`.

    This function implements the random quantile spacings used in Theorem 4.3
    and Corollary 4.4 of the paper.  The returned array can be multiplied by
    the ordered loss vector (augmented with the upper bound B) to obtain
    Monte Carlo realisations of L^+.

    Args:
        n_losses: Number of observed calibration losses, n.
        n_samples: Number of Monte Carlo draws (from config:
            `bayesian_quadrature.num_dirichlet_samples`, default 1000).
        rng: A seeded numpy random generator instance.

    Returns:
        A 2D numpy array of shape (n_samples, n_losses + 1), where each
        row sums to 1.
    """
    alpha = np.ones(n_losses + 1, dtype=np.float64)
    return rng.dirichlet(alpha, n_samples)


def build_lambda_grid(
    min_val: float,
    max_val: float,
    size: int,
) -> np.ndarray:
    """
    Create a uniformly spaced grid of candidate λ values.

    All decision rules (CRC, RCPS, BQC) use the same grid to search for the
    optimal λ̂.

    Args:
        min_val: Lower bound of the grid (usually 0.0).
        max_val: Upper bound of the grid (usually 1.0).
        size: Number of grid points (from config: `lambda_grid.size`,
            default 1001).

    Returns:
        1D numpy array of length `size` linearly spaced from `min_val` to
        `max_val` inclusive.
    """
    return np.linspace(min_val, max_val, size, dtype=np.float64)


def clopper_pearson_ci(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute an exact Clopper‑Pearson confidence interval for a binomial
    proportion.

    This is used to produce the 95% confidence intervals reported in the
    tables (Tables 5.1, 5.2, 5.3) for the relative frequency of trials
    exceeding the target risk.

    The interval is computed via the beta distribution quantiles:

        lower = Beta⁻¹(α/2; k, n - k + 1)
        upper = Beta⁻¹(1 - α/2; k + 1, n - k)

    where α = 1 - confidence, k = successes, n = trials.

    Args:
        successes: Number of observed successes (trials where risk > α).
        trials: Total number of trials.
        confidence: Desired confidence level (default 0.95 for 95% CI).

    Returns:
        A tuple (lower, upper) containing the confidence interval bounds.
    """
    alpha = 1.0 - confidence

    if successes == 0:
        lower = 0.0
    else:
        lower = stats.beta.ppf(alpha / 2.0, successes, trials - successes + 1)

    if successes == trials:
        upper = 1.0
    else:
        upper = stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)

    return float(lower), float(upper)
