"""Risk-Controlling Prediction Sets (RCPS) baseline with Hoeffding bound (Section 5).

Implements RCPS as described in Bates et al. (2021) with the Hoeffding
upper confidence bound for risk estimation.
"""

import numpy as np
from typing import Optional


def hoeffding_ucb(
    losses: np.ndarray,
    B: float,
    delta: float,
) -> float:
    """Compute the Hoeffding upper confidence bound on the risk.

    UCB = (1/n) Σ ℓᵢ + B * sqrt(log(1/δ) / (2n))

    Args:
        losses: Array of n calibration losses.
        B: Upper bound on individual losses.
        delta: Confidence level (failure probability).

    Returns:
        Upper confidence bound on the risk.
    """
    n = len(losses)
    if n == 0:
        return float("inf")
    empirical_risk = np.mean(losses)
    hoeffding_term = B * np.sqrt(np.log(1.0 / delta) / (2.0 * n))
    return float(empirical_risk + hoeffding_term)


def select_lambda_rcps(
    losses_by_lambda: dict,
    B: float,
    alpha: float,
    delta: float = 0.05,
) -> float:
    """RCPS decision rule with Hoeffding bound (Bates et al., 2021).

    λ_rcps = inf{λ : Hoeffding_UCB(ℓ(λ)) ≤ α}

    Args:
        losses_by_lambda: Dict mapping λ to array of n calibration losses.
        B: Upper bound on individual losses.
        alpha: Target risk level.
        delta: Confidence level (1 - delta C.I.).

    Returns:
        Selected λ value, or inf if no λ satisfies the criterion.
    """
    sorted_lambdas = sorted(losses_by_lambda.keys())

    for lam in sorted_lambdas:
        losses = losses_by_lambda[lam]
        ucb = hoeffding_ucb(losses, B=B, delta=delta)
        if ucb <= alpha:
            return lam

    return float("inf")


def select_lambda_rcps_bernstein(
    losses_by_lambda: dict,
    B: float,
    alpha: float,
    delta: float = 0.05,
) -> float:
    """RCPS with Bernstein bound (alternative to Hoeffding, more data-adaptive).

    Args:
        losses_by_lambda: Dict mapping λ to array of n calibration losses.
        B: Upper bound on individual losses.
        alpha: Target risk level.
        delta: Confidence level.

    Returns:
        Selected λ value, or inf if no λ satisfies the criterion.
    """
    sorted_lambdas = sorted(losses_by_lambda.keys())

    for lam in sorted_lambdas:
        losses = losses_by_lambda[lam]
        n = len(losses)
        if n == 0:
            continue
        empirical_risk = np.mean(losses)
        variance = np.var(losses)
        bernstein_term = np.sqrt(2.0 * variance * np.log(1.0 / delta) / n) + (7.0 * B * np.log(1.0 / delta) / (3.0 * n))
        ucb = empirical_risk + bernstein_term
        if ucb <= alpha:
            return lam

    return float("inf")
