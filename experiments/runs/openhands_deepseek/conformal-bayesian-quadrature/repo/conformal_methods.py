"""Conformal Risk Control and Split Conformal Prediction baselines (Sections 2.1, 3.1, 3.2, 4.6)."""

import numpy as np
from typing import Optional


def select_lambda_crc(
    losses_by_lambda: dict,
    B: float,
    alpha: float,
) -> float:
    """Conformal Risk Control decision rule (Proposition 3.2, Eq 15/43).

    λ_crc = inf{λ : (1/(n+1)) (Σ ℓᵢ(λ) + B) ≤ α}

    This is equivalent to selecting λ where E[L⁺] ≤ α (Section 4.6).

    Args:
        losses_by_lambda: Dict mapping λ to array of n calibration losses.
        B: Upper bound on individual losses.
        alpha: Target risk level.

    Returns:
        Selected λ value, or inf if no λ satisfies the criterion.
    """
    sorted_lambdas = sorted(losses_by_lambda.keys())
    n = len(next(iter(losses_by_lambda.values())))

    for lam in sorted_lambdas:
        losses = losses_by_lambda[lam]
        mean_lplus = (np.sum(losses) + B) / (n + 1)
        if mean_lplus <= alpha:
            return lam

    return float("inf")


def select_lambda_scp(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """Split Conformal Prediction decision rule (Proposition 3.1, Eq 12).

    λ_scp = s_{(⌈(n+1)(1-α)⌉)} if ⌈(n+1)(1-α)⌉ ≤ n, else ∞.

    This uses miscoverage loss: ℓᵢ = 1{sᵢ > λ}.
    Recovered as E[L⁺] ≤ α implies k ≥ ⌈(n+1)(1-α)⌉ (Section 4.6).

    Args:
        scores: Array of n calibration nonconformity scores.
        alpha: Target miscoverage rate.

    Returns:
        Threshold λ value (quantile of scores).
    """
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")

    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1])  # 1-indexed: k-th order statistic


def select_lambda_scp_risk_control(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """SCP formulated as a risk control problem (Section 3.1).

    Uses individual loss ℓᵢ = 1{sᵢ > λ} with B = 1.
    Equivalent to select_lambda_scp.

    Args:
        scores: Array of n calibration nonconformity scores.
        alpha: Target miscoverage rate.

    Returns:
        Threshold λ.
    """
    return select_lambda_scp(scores, alpha)


def compute_miscoverage_losses(scores: np.ndarray, lambdas: np.ndarray) -> dict:
    """Compute miscoverage losses ℓᵢ = 1{sᵢ > λ} at each λ.

    Args:
        scores: Array of n calibration scores.
        lambdas: Array of λ values to evaluate.

    Returns:
        Dict mapping λ to array of n binary loss values.
    """
    result = {}
    for lam in lambdas:
        result[lam] = (scores > lam).astype(np.float64)
    return result


def compute_empirical_risk(
    losses: np.ndarray,
    B: float,
) -> float:
    """Compute the CRC empirical risk bound (1/(n+1))(Σ ℓᵢ + B).

    Args:
        losses: Array of n calibration losses.
        B: Upper bound on losses.

    Returns:
        Empirical risk estimate.
    """
    n = len(losses)
    return (np.sum(losses) + B) / (n + 1)
