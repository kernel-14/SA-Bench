"""
Core methods: Conformal Risk Control (CRC), Risk-Controlling Prediction Sets (RCPS),
and the proposed Bayesian Quadrature HPD method (BQ-HPD).

All three methods select a threshold lambda from a grid such that the expected
loss is controlled at level alpha with confidence beta.

References:
  - CRC: Angelopoulos et al. (2024), Conformal Risk Control, ICLR 2024
  - RCPS: Bates et al. (2021), Distribution-Free, Risk-Controlling Prediction Sets, JACM
  - BQ-HPD: Snell & Griffiths (this paper)
"""

from __future__ import annotations

import numpy as np
from typing import Callable


def _dirichlet_samples(n_components: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample from Dir(1, ..., 1) (symmetric Dirichlet / uniform on simplex).

    Equivalent to normalizing n_components independent Gamma(1,1) = Exp(1) samples.

    Returns shape (n_samples, n_components).
    """
    gammas = rng.exponential(1.0, size=(n_samples, n_components))
    return gammas / gammas.sum(axis=1, keepdims=True)


def compute_L_plus_samples(
    losses: np.ndarray,
    B: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute Monte Carlo samples of L+ (Theorem 4.3).

    L+ = sum_{i=1}^{n+1} U_i * ell_(i)

    where ell_(1) <= ... <= ell_(n) are the sorted calibration losses,
    ell_(n+1) = B, and (U_1, ..., U_{n+1}) ~ Dir(1, ..., 1).

    Args:
        losses: 1-D array of calibration losses, shape (n,).
        B: upper bound on losses.
        n_samples: number of Monte Carlo samples.
        rng: numpy random generator.

    Returns:
        1-D array of L+ samples, shape (n_samples,).
    """
    sorted_losses = np.sort(losses)
    all_losses = np.append(sorted_losses, B)  # shape (n+1,)
    n_components = len(all_losses)
    U = _dirichlet_samples(n_components, n_samples, rng)  # (n_samples, n+1)
    return U @ all_losses  # (n_samples,)


def pr_L_plus_leq_alpha(
    losses: np.ndarray,
    B: float,
    alpha: float,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """
    Estimate Pr(L+ <= alpha) via Monte Carlo (equation 29 / Corollary 4.4).
    """
    L_plus = compute_L_plus_samples(losses, B, n_samples, rng)
    return float(np.mean(L_plus <= alpha))


def lambda_bq_hpd(
    loss_fn: Callable[[float], np.ndarray],
    lambda_grid: np.ndarray,
    alpha: float,
    beta: float,
    B: float,
    n_dirichlet: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Bayesian Quadrature HPD decision rule (equation 31).

    lambda_hpd^beta = inf{lambda : Pr(L+ <= alpha | ell_{1:n}) >= beta}

    Searches lambda_grid in ascending order and returns the first lambda
    for which the Monte Carlo estimate of Pr(L+ <= alpha) >= beta.

    Args:
        loss_fn: callable mapping lambda -> 1-D array of calibration losses.
        lambda_grid: sorted ascending array of candidate lambda values.
        alpha: target risk level.
        beta: desired confidence level (e.g. 0.95).
        B: upper bound on losses.
        n_dirichlet: number of Dirichlet samples for Monte Carlo estimation.
        rng: numpy random generator (created if None).

    Returns:
        Selected lambda value.
    """
    if rng is None:
        rng = np.random.default_rng()

    for lam in lambda_grid:
        losses = loss_fn(lam)
        prob = pr_L_plus_leq_alpha(losses, B, alpha, n_dirichlet, rng)
        if prob >= beta:
            return float(lam)

    return float(lambda_grid[-1])


def lambda_crc(
    loss_fn: Callable[[float], np.ndarray],
    lambda_grid: np.ndarray,
    alpha: float,
    B: float,
) -> float:
    """
    Conformal Risk Control decision rule (equation 15 / Proposition 3.2).

    lambda_crc = inf{lambda : (1/(n+1)) * (sum_i ell(z_i, lambda) + B) <= alpha}

    This is equivalent to taking E(L+) <= alpha (Section 4.6).

    Args:
        loss_fn: callable mapping lambda -> 1-D array of calibration losses.
        lambda_grid: sorted ascending array of candidate lambda values.
        alpha: target risk level.
        B: upper bound on losses.

    Returns:
        Selected lambda value.
    """
    for lam in lambda_grid:
        losses = loss_fn(lam)
        n = len(losses)
        if (np.sum(losses) + B) / (n + 1) <= alpha:
            return float(lam)

    return float(lambda_grid[-1])


def lambda_rcps_hoeffding(
    loss_fn: Callable[[float], np.ndarray],
    lambda_grid: np.ndarray,
    alpha: float,
    beta: float,
    B: float,
) -> float:
    """
    Risk-Controlling Prediction Sets with Hoeffding upper confidence bound
    (Bates et al., 2021).

    lambda_rcps = inf{lambda : R_hat_n(lambda) + B * sqrt(log(1/delta) / (2n)) <= alpha}

    where delta = 1 - beta is the failure probability.

    Args:
        loss_fn: callable mapping lambda -> 1-D array of calibration losses.
        lambda_grid: sorted ascending array of candidate lambda values.
        alpha: target risk level.
        beta: desired confidence level (e.g. 0.95).
        B: upper bound on losses.

    Returns:
        Selected lambda value.
    """
    delta = 1.0 - beta
    # Compute n from the first evaluation (assumed constant across lambda)
    n = len(loss_fn(lambda_grid[0]))
    hoeffding_slack = B * np.sqrt(np.log(1.0 / delta) / (2.0 * n))

    for lam in lambda_grid:
        losses = loss_fn(lam)
        R_hat = float(np.mean(losses))
        if R_hat + hoeffding_slack <= alpha:
            return float(lam)

    return float(lambda_grid[-1])


def lambda_scp(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """
    Split Conformal Prediction threshold (Proposition 3.1 / equation 12).

    Returns the ceil((n+1)(1-alpha))-th order statistic of the nonconformity
    scores, or infinity if that index exceeds n.

    Args:
        scores: 1-D array of nonconformity scores s_1, ..., s_n.
        alpha: miscoverage level.

    Returns:
        Threshold lambda_scp.
    """
    n = len(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1])  # 0-indexed
