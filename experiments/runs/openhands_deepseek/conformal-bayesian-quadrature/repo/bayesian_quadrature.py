"""Bayesian Quadrature approach for conformal prediction (Sections 4, 4.1-4.6).

Core algorithm: Given observed calibration losses, construct L⁺ as a
Dirichlet-weighted sum of order statistics. L⁺ stochastically dominates
the posterior risk, enabling guaranteed upper confidence bounds.
"""

import numpy as np
from typing import Optional


def compute_lplus_distribution(
    losses: np.ndarray,
    B: float,
    n_samples: int = 100000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute Monte Carlo samples of L⁺ (Theorem 4.3, Eq 27).

    L⁺ = Σ_{i=1}^{n+1} U_i * ℓ_{(i)}
    where U ~ Dir(1, ..., 1) and ℓ_{(n+1)} = B.

    Args:
        losses: Array of n observed calibration losses (any order).
        B: Upper bound on losses.
        n_samples: Number of Dirichlet samples to draw.
        rng: Random number generator.

    Returns:
        Array of shape (n_samples,) containing L⁺ samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(losses)
    sorted_losses = np.sort(losses)
    extended_losses = np.concatenate([sorted_losses, [B]])  # length n+1

    alpha = np.ones(n + 1)  # Dir(1, ..., 1)
    dirichlet_samples = rng.dirichlet(alpha, size=n_samples)  # (n_samples, n+1)

    lplus_samples = dirichlet_samples @ extended_losses  # (n_samples,)
    return lplus_samples


def compute_critical_value(
    lplus_samples: np.ndarray,
    beta: float,
) -> float:
    """Compute b_β* = inf{b : Pr(L⁺ ≤ b) ≥ β} (Corollary 4.4, Eq 29).

    This is the empirical quantile of the L⁺ samples at level β.

    Args:
        lplus_samples: Monte Carlo samples of L⁺.
        beta: Desired confidence level (e.g., 0.95).

    Returns:
        Critical value b_β*.
    """
    return float(np.quantile(lplus_samples, beta))


def select_lambda_hpd(
    losses_by_lambda: dict,
    B: float,
    alpha: float,
    beta: float,
    n_samples: int = 100000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Select λ via HPD decision rule (Eq 31).

    λ_hpd^β = inf_λ { λ : Pr(L⁺(λ) ≤ α | ℓ_{1:n}) ≥ β }

    For each λ, compute L⁺(λ) using the corresponding losses,
    then estimate Pr(L⁺ ≤ α) via Monte Carlo.

    Args:
        losses_by_lambda: Dict mapping λ to array of n calibration losses.
        B: Upper bound on individual losses.
        alpha: Target risk level.
        beta: Confidence level for HPD interval.
        n_samples: Number of Dirichlet samples per λ.
        rng: Random number generator.

    Returns:
        Selected λ value (float), or inf if no λ satisfies the criterion.
    """
    if rng is None:
        rng = np.random.default_rng()

    sorted_lambdas = sorted(losses_by_lambda.keys())

    for lam in sorted_lambdas:
        losses = losses_by_lambda[lam]
        lplus_samples = compute_lplus_distribution(
            losses, B=B, n_samples=n_samples, rng=rng
        )
        prob_leq_alpha = np.mean(lplus_samples <= alpha)
        if prob_leq_alpha >= beta:
            return lam

    return float("inf")


def compute_lplus_for_lambda(
    losses: np.ndarray,
    B: float,
    n_samples: int = 100000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Get L⁺ samples and summary stats for a single λ setting.

    Returns:
        Tuple of (lplus_samples, mean_lplus, prob_leq_alpha).
    """
    lplus_samples = compute_lplus_distribution(
        losses, B=B, n_samples=n_samples, rng=rng
    )
    return lplus_samples


def evaluate_lplus_statistics(
    losses: np.ndarray,
    B: float,
    alpha: float,
    n_samples: int = 100000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Compute full diagnostic statistics for L⁺ at a given λ.

    Args:
        losses: Calibration loss values.
        B: Loss upper bound.
        alpha: Target risk threshold.
        n_samples: Monte Carlo samples.
        rng: Random generator.

    Returns:
        Dict with keys: mean, std, prob_leq_alpha, critical_value_095,
        critical_value_090, critical_value_099.
    """
    if rng is None:
        rng = np.random.default_rng()

    lplus = compute_lplus_distribution(losses, B=B, n_samples=n_samples, rng=rng)

    return {
        "mean": float(np.mean(lplus)),
        "std": float(np.std(lplus)),
        "prob_leq_alpha": float(np.mean(lplus <= alpha)),
        "critical_value_095": float(np.quantile(lplus, 0.95)),
        "critical_value_090": float(np.quantile(lplus, 0.90)),
        "critical_value_099": float(np.quantile(lplus, 0.99)),
    }
