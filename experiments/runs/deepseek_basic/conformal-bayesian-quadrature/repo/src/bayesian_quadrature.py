"""Core implementation of Conformal Prediction as Bayesian Quadrature.

This module implements the framework described in:
  "Conformal Prediction as Bayesian Quadrature"
  by Jake C. Snell and Thomas L. Griffiths.

The key idea is to reinterpret conformal prediction through the lens of
Bayesian quadrature. By placing a prior over quantile functions of the loss
distribution and using properties of quantile spacings (which follow a
Dirichlet(1,...,1) distribution), we obtain a richer posterior characterization
of the expected loss.

Key theoretical results implemented:
  - Theorem 4.1: Upper bound on posterior expected loss via quantile spacings
  - Lemma 4.2: Distribution of quantile spacings is Dirichlet(1,...,1)
  - Theorem 4.3: L^+ stochastically dominates the posterior risk
  - Corollary 4.4: Upper confidence bounds using L^+
  - Section 4.6: Recovery of Split Conformal Prediction and Conformal Risk Control
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Callable


def L_plus_random_variable(
    sorted_losses: np.ndarray,
    B: float,
    n_dirichlet_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate samples of the random variable L^+ defined in Theorem 4.3.

    L^+ = sum_{i=1}^{n+1} U_i * l_{(i)}

    where:
      - l_{(1)} <= ... <= l_{(n)} are the order statistics of observed losses
      - l_{(n+1)} = B (the maximum possible loss)
      - U_1, ..., U_{n+1} ~ Dirichlet(1, ..., 1)

    Args:
        sorted_losses: 1D array of sorted observed losses l_{(1)} <= ... <= l_{(n)}.
        B: Upper bound on losses (scalar).
        n_dirichlet_samples: Number of Dirichlet samples to generate.
        rng: Optional numpy random Generator.

    Returns:
        1D array of L^+ samples of length n_dirichlet_samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(sorted_losses)
    # Append B as the (n+1)-th order statistic
    all_losses = np.append(sorted_losses, B)  # shape (n+1,)

    # Generate Dirichlet(1,...,1) samples
    # Dirichlet(1,...,1) is equivalent to: generate n+1 i.i.d. Exp(1) and normalize
    alpha = np.ones(n + 1)
    dirichlet_samples = rng.dirichlet(alpha, size=n_dirichlet_samples)  # (n_samples, n+1)

    # L^+ = weighted sum
    L_plus = dirichlet_samples @ all_losses  # (n_samples,)

    return L_plus


def compute_L_plus_distribution(
    losses: np.ndarray,
    B: float,
    n_dirichlet_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute samples from the L^+ distribution for given (unsorted) losses.

    This is a convenience wrapper that sorts the losses first.

    Args:
        losses: 1D array of observed individual losses l_1, ..., l_n.
        B: Upper bound on losses.
        n_dirichlet_samples: Number of Dirichlet samples.
        rng: Optional random generator.

    Returns:
        1D array of L^+ samples.
    """
    sorted_losses = np.sort(losses)
    return L_plus_random_variable(
        sorted_losses=sorted_losses,
        B=B,
        n_dirichlet_samples=n_dirichlet_samples,
        rng=rng,
    )


def compute_hpd_critical_value(
    L_plus_samples: np.ndarray,
    beta: float,
) -> float:
    """Compute the critical value b_beta^* from Corollary 4.4.

    b_beta^* = inf { b : Pr(L^+ <= b | l_{1:n}) >= beta }

    This is simply the beta-quantile of the L^+ samples.

    Args:
        L_plus_samples: 1D array of L^+ Monte Carlo samples.
        beta: Desired confidence level in (0, 1).

    Returns:
        Critical value b_beta^*.
    """
    return np.quantile(L_plus_samples, beta)


def compute_hpd_lambda(
    loss_fn: Callable[[float], np.ndarray],
    alpha: float,
    B: float,
    beta: float = 0.95,
    lambda_grid: Optional[np.ndarray] = None,
    lambda_min: float = 0.0,
    lambda_max: float = 1.0,
    n_lambda: int = 200,
    n_dirichlet_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, dict]:
    """Compute lambda_hpd^beta defined in Eq. (31) of the paper.

    lambda_hpd^beta = inf { lambda : Pr(L^+ <= alpha | l_{1:n}) >= beta }

    This is the main decision rule proposed in the paper. It selects lambda
    such that the posterior probability that the expected loss is at most alpha
    is at least beta.

    Args:
        loss_fn: Function mapping lambda -> array of n individual losses.
                 Must be monotonically non-increasing in lambda.
        alpha: Target risk threshold.
        B: Upper bound on individual losses.
        beta: Confidence level for the HPD interval (default 0.95).
        lambda_grid: Optional explicit grid of lambda values.
        lambda_min: Minimum lambda value for grid search.
        lambda_max: Maximum lambda value for grid search.
        n_lambda: Number of lambda values in grid.
        n_dirichlet_samples: Number of Dirichlet MC samples.
        rng: Optional random generator.

    Returns:
        Tuple of (lambda_hpd, info_dict) where info_dict contains diagnostic
        information about the search.
    """
    if rng is None:
        rng = np.random.default_rng()

    if lambda_grid is None:
        lambda_grid = np.linspace(lambda_min, lambda_max, n_lambda)

    # For each lambda, compute the probability that L^+ <= alpha
    prob_le_alpha = np.zeros(len(lambda_grid))

    for i, lam in enumerate(lambda_grid):
        losses = loss_fn(lam)
        L_plus_samples = compute_L_plus_distribution(
            losses=losses,
            B=B,
            n_dirichlet_samples=n_dirichlet_samples,
            rng=rng,
        )
        prob_le_alpha[i] = np.mean(L_plus_samples <= alpha)

    # Find the smallest lambda such that Pr(L^+ <= alpha) >= beta
    # Since losses are non-increasing in lambda, L^+ is also non-increasing in lambda,
    # so Pr(L^+ <= alpha) is non-decreasing in lambda.
    feasible = prob_le_alpha >= beta

    if not np.any(feasible):
        # No lambda satisfies the constraint; return lambda_max
        lambda_hpd = lambda_max
    else:
        idx = np.argmax(feasible)  # first feasible index
        lambda_hpd = lambda_grid[idx]

    info = {
        "lambda_grid": lambda_grid,
        "prob_le_alpha": prob_le_alpha,
        "beta": beta,
        "alpha": alpha,
    }

    return lambda_hpd, info


def compute_conformal_risk_control_lambda(
    losses: np.ndarray,
    alpha: float,
    B: float,
) -> float:
    """Compute lambda_crc according to Conformal Risk Control (Proposition 3.2).

    lambda_crc = inf { lambda : (1/(n+1)) * (sum_i l_i(lambda) + B) <= alpha }

    This is the standard CRC decision rule, which corresponds to taking the
    expected value of L^+ as shown in Section 4.6.

    Args:
        losses: 1D array of n individual losses l_1, ..., l_n for a candidate lambda.
        alpha: Target risk threshold.
        B: Upper bound on individual losses.

    Returns:
        Boolean indicating whether the CRC condition is satisfied for these losses.
    """
    n = len(losses)
    E_L_plus = (np.sum(losses) + B) / (n + 1)
    return E_L_plus <= alpha


def compute_split_conformal_lambda(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """Compute lambda_scp according to Split Conformal Prediction (Proposition 3.1).

    lambda_scp = s_{(ceil((n+1)*(1-alpha)))}

    Args:
        scores: 1D array of n nonconformity scores s_1, ..., s_n.
        alpha: Target miscoverage rate.

    Returns:
        Threshold lambda_scp.
    """
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    sorted_scores = np.sort(scores)
    return sorted_scores[k - 1]  # 0-indexed


def compute_crc_decision_rule(
    loss_fn: Callable[[float], np.ndarray],
    alpha: float,
    B: float,
    lambda_grid: Optional[np.ndarray] = None,
    lambda_min: float = 0.0,
    lambda_max: float = 1.0,
    n_lambda: int = 200,
) -> Tuple[float, dict]:
    """Compute lambda_crc by grid search over lambda values.

    lambda_crc = inf { lambda : (1/(n+1)) * (sum_i l_i(lambda) + B) <= alpha }

    Args:
        loss_fn: Function lambda -> array of n losses.
        alpha: Target risk.
        B: Maximum loss.
        lambda_grid: Optional explicit lambda values.
        lambda_min: Minimum lambda for grid.
        lambda_max: Maximum lambda for grid.
        n_lambda: Number of grid points.

    Returns:
        Tuple of (lambda_crc, info).
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(lambda_min, lambda_max, n_lambda)

    mean_risk = np.zeros(len(lambda_grid))
    for i, lam in enumerate(lambda_grid):
        losses = loss_fn(lam)
        n = len(losses)
        mean_risk[i] = (np.sum(losses) + B) / (n + 1)

    feasible = mean_risk <= alpha
    if not np.any(feasible):
        lambda_crc = lambda_max
    else:
        idx = np.argmax(feasible)
        lambda_crc = lambda_grid[idx]

    info = {
        "lambda_grid": lambda_grid,
        "mean_risk": mean_risk,
        "alpha": alpha,
    }

    return lambda_crc, info


def compute_rcps_hoeffding_lambda(
    loss_fn: Callable[[float], np.ndarray],
    alpha: float,
    B: float,
    delta: float = 0.05,
    lambda_grid: Optional[np.ndarray] = None,
    lambda_min: float = 0.0,
    lambda_max: float = 1.0,
    n_lambda: int = 200,
) -> Tuple[float, dict]:
    """Compute lambda using RCPS with Hoeffding UCB (Bates et al., 2021).

    This is used as a baseline in the experiments (Section 5).

    The Hoeffding bound for each lambda is:
        UCB(lambda) = mean_loss(lambda) + B * sqrt(log(1/delta) / (2*n))

    Args:
        loss_fn: Function lambda -> array of n losses.
        alpha: Target risk.
        B: Maximum loss.
        delta: Confidence level for the UCB (default 0.05).
        lambda_grid: Optional lambda values.
        lambda_min: Min lambda.
        lambda_max: Max lambda.
        n_lambda: Number of grid points.

    Returns:
        Tuple of (lambda_rcps, info).
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(lambda_min, lambda_max, n_lambda)

    n = len(loss_fn(lambda_grid[0]))
    hoeffding_correction = B * np.sqrt(np.log(1.0 / delta) / (2 * n))

    ucb = np.zeros(len(lambda_grid))
    for i, lam in enumerate(lambda_grid):
        losses = loss_fn(lam)
        ucb[i] = np.mean(losses) + hoeffding_correction

    feasible = ucb <= alpha
    if not np.any(feasible):
        lambda_rcps = lambda_max
    else:
        idx = np.argmax(feasible)
        lambda_rcps = lambda_grid[idx]

    info = {
        "lambda_grid": lambda_grid,
        "ucb": ucb,
        "alpha": alpha,
        "delta": delta,
    }

    return lambda_rcps, info
