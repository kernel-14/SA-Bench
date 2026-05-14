# methods.py
"""
Conformal decision rules for "Conformal Prediction as Bayesian Quadrature".

This module implements the three methods compared in the paper:

* CRCMethod   : Conformal Risk Control (Angelopoulos et al., 2024, Prop. 3.2)
* RCPSMethod  : Risk‑Controlling Prediction Sets (Bates et al., 2021) with
                Hoeffding bound
* BQCMethod   : Bayesian Quadrature‑Based Conformal (Theorem 4.3, Corollary 4.4)

All classes are callable with a common signature:
    (losses_2d: np.ndarray, lambda_grid: np.ndarray) -> Tuple[float, np.ndarray]
where `losses_2d` has shape `(n_calibration, n_lambda)` and `lambda_grid` is
a 1‑D array of sorted λ values.  The returned tuple contains the chosen
parameter λ̂ and a diagnostic curve (risk values or posterior probability).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# local utility import (we will write it in utils.py)
from utils import dirichlet_sampling


# ---------------------------------------------------------------------------
# Helper: find the infimum λ index (first index where condition is True)
# ---------------------------------------------------------------------------

def _infimum_lambda(condition_curve: np.ndarray) -> int:
    """
    Return the index of the first True element in `condition_curve`.
    Falls back to the last index if none are True (should not happen in
    these experiments because λ_max yields zero loss).
    """
    indices = np.nonzero(condition_curve)[0]
    if len(indices) == 0:
        return len(condition_curve) - 1  # fallback
    return indices[0]


# ============================================================================
# Conformal Risk Control (CRC)
# ============================================================================

class CRCMethod:
    """
    Conformal Risk Control decision rule (Proposition 3.2 / Angelopoulos et al.).

    Uses the empirical risk of the n calibration losses and a single upper
    bound B to compute a tightened risk estimate:

        risk_crc(λ) = (1/(n+1)) * (∑_i ℓ(z_i, λ) + B)

    The chosen λ is the smallest value for which risk_crc(λ) ≤ α.

    Parameters
    ----------
    alpha : float
        Target risk level.
    B : float
        Upper bound on the loss (must be ≥ alpha).
    rng : numpy.random.Generator, optional
        Not used internally, but accepted for uniform interface.
    """

    def __init__(
        self,
        alpha: float,
        B: float,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if not 0 < alpha <= B:
            raise ValueError(f"alpha must be in (0, B] ({B}), got {alpha}")
        self.alpha = alpha
        self.B = B

    def __call__(
        self,
        losses_2d: np.ndarray,
        lambda_grid: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        n_cal = losses_2d.shape[0]
        # Empirical sum over n points, shape (n_lambda,)
        sum_losses = losses_2d.sum(axis=0)
        # CRC criterion: (sum + B) / (n+1)
        risk_crc = (sum_losses + self.B) / (n_cal + 1)

        # Condition vector: risk_crc ≤ alpha
        condition = risk_crc <= self.alpha
        idx = _infimum_lambda(condition)
        lambda_hat = lambda_grid[idx]

        return lambda_hat, risk_crc


# ============================================================================
# Risk‑Controlling Prediction Sets (RCPS) with Hoeffding bound
# ============================================================================

class RCPSMethod:
    """
    RCPS decision rule using Hoeffding's inequality (Bates et al., 2021).

    Computes an upper confidence bound:

        ucb(λ) = emp_risk(λ) + B * sqrt( log(1/δ) / (2 n) )

    and selects the smallest λ such that ucb(λ) ≤ α.

    Parameters
    ----------
    alpha : float
        Target risk level.
    B : float
        Upper bound on the loss.
    delta : float
        Failure probability for the Hoeffding bound (e.g., 0.05).
    rng : numpy.random.Generator, optional
        Not used internally, but accepted for uniform interface.
    """

    def __init__(
        self,
        alpha: float,
        B: float,
        delta: float,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if not 0 < alpha <= B:
            raise ValueError(f"alpha must be in (0, B] ({B}), got {alpha}")
        if not 0 < delta < 0.5:
            raise ValueError(f"delta must be in (0, 0.5), got {delta}")
        self.alpha = alpha
        self.B = B
        self.delta = delta
        self._hoeffding_term = np.sqrt(np.log(1.0 / delta) / 2.0)

    def __call__(
        self,
        losses_2d: np.ndarray,
        lambda_grid: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        n_cal = losses_2d.shape[0]
        emp_risk = losses_2d.mean(axis=0)  # shape (n_lambda,)
        ucb = emp_risk + self.B * (self._hoeffding_term / np.sqrt(n_cal))

        condition = ucb <= self.alpha
        idx = _infimum_lambda(condition)
        lambda_hat = lambda_grid[idx]

        return lambda_hat, ucb


# ============================================================================
# Bayesian Quadrature‑Based Conformal (BQC) – the novel method
# ============================================================================

class BQCMethod:
    """
    Bayesian Quadrature decision rule (Theorem 4.3, Corollary 4.4).

    For each candidate λ, the method constructs the random variable L⁺
    (weighted sum of ordered losses augmented by B, with Dirichlet weights)
    and tests whether

        P( L⁺ ≤ α ) ≥ β.

    The chosen λ is the smallest value satisfying this posterior probability
    condition.

    To speed up the search (important for large n), a binary search over the
    λ grid is performed, and Dirichlet samples are drawn only when a new λ
    is evaluated (memoised).

    Parameters
    ----------
    alpha : float
        Target risk level.
    beta : float
        Confidence level for the HPD interval (e.g., 0.95).
    B : float
        Upper bound on the loss (appended as ℓ_{(n+1)}).
    n_dir_samples : int, optional
        Number of Dirichlet Monte Carlo samples per λ evaluation.
        Default 1000 (from paper / config).
    rng : numpy.random.Generator, optional
        Seeded random generator for Dirichlet sampling.  If None, a new
        generator is created.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        B: float,
        n_dir_samples: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if not 0 < alpha <= B:
            raise ValueError(f"alpha must be in (0, B] ({B}), got {alpha}")
        if not 0 < beta <= 1:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if n_dir_samples <= 0:
            raise ValueError("n_dir_samples must be positive")

        self.alpha = alpha
        self.beta = beta
        self.B = B
        self.n_dir_samples = n_dir_samples
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        losses_2d: np.ndarray,
        lambda_grid: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        n_cal = losses_2d.shape[0]
        n_lambda = len(lambda_grid)

        # Memoisation array for probabilities; NaN means "not yet evaluated"
        prob_curve = np.full(n_lambda, np.nan, dtype=np.float64)

        # Binary search for the smallest λ with prob >= beta
        low = 0
        high = n_lambda - 1
        ans = n_lambda  # out‑of‑range sentinel

        while low <= high:
            mid = (low + high) // 2
            prob_mid = prob_curve[mid]
            if np.isnan(prob_mid):
                prob_mid = self._evaluate_prob(mid, losses_2d, n_cal)
                prob_curve[mid] = prob_mid

            if prob_mid >= self.beta:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1

        # Fallback: if no λ is acceptable, use the largest λ (shouldn't happen)
        if ans == n_lambda:
            lambda_hat = lambda_grid[-1]
        else:
            lambda_hat = lambda_grid[ans]

        return lambda_hat, prob_curve

    def _evaluate_prob(
        self,
        col_idx: int,
        losses_2d: np.ndarray,
        n_cal: int,
    ) -> float:
        """
        For the λ at position `col_idx`, draw Dirichlet samples and compute
        P(L⁺ ≤ α).  The result is memoised externally.
        """
        losses = losses_2d[:, col_idx]                     # shape (n_cal,)
        l_sorted = np.sort(losses)                         # ascending
        l_aug = np.append(l_sorted, self.B)                # (n_cal+1,)

        # Dirichlet weights (n_dir_samples, n_cal+1)
        dir_samples = dirichlet_sampling(
            n_cal + 1,
            self.n_dir_samples,
            self.rng,
        )
        L_plus = dir_samples @ l_aug                       # (n_dir_samples,)
        prob = np.mean(L_plus <= self.alpha, dtype=np.float64)
        return prob


# ---------------------------------------------------------------------------
# For convenience: alias the three methods under a unified name
# ---------------------------------------------------------------------------

# Additional type hints for users
Method = CRCMethod  # any of the classes, but for typing we can use object
