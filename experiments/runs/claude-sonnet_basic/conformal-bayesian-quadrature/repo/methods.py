"""
Core methods for Conformal Prediction as Bayesian Quadrature.

This module implements:
1. Split Conformal Prediction (SCP)
2. Conformal Risk Control (CRC)
3. Risk-Controlling Prediction Sets (RCPS) with Hoeffding UCB
4. Bayesian Quadrature (BQ) method - the main contribution of the paper

The key insight is that the quantile spacings of calibration losses follow a
Dirichlet(1,...,1) distribution, which allows us to construct a random variable
L+ that stochastically dominates the posterior expected loss.
"""

import numpy as np
from scipy import stats


# ============================================================
# Split Conformal Prediction (SCP)
# ============================================================

def split_conformal_prediction(scores, alpha):
    """
    Split Conformal Prediction decision rule.
    
    Returns the threshold lambda_scp such that the miscoverage rate is <= alpha.
    
    Parameters
    ----------
    scores : array-like of shape (n,)
        Nonconformity scores on calibration set.
    alpha : float
        Target miscoverage level (e.g., 0.1 for 90% coverage).
    
    Returns
    -------
    lambda_scp : float
        Threshold for prediction set construction.
    """
    scores = np.sort(scores)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k <= n:
        return scores[k - 1]  # 1-indexed -> 0-indexed
    else:
        return np.inf


# ============================================================
# Conformal Risk Control (CRC)
# ============================================================

def conformal_risk_control(losses_fn, lambda_grid, alpha, B=1.0):
    """
    Conformal Risk Control decision rule.
    
    Finds the infimum lambda such that the empirical risk upper bound <= alpha.
    
    Parameters
    ----------
    losses_fn : callable
        Function that takes lambda and returns array of individual losses.
    lambda_grid : array-like
        Grid of lambda values to search over.
    alpha : float
        Target risk level.
    B : float
        Upper bound on individual losses.
    
    Returns
    -------
    lambda_crc : float
        Selected threshold.
    """
    lambda_grid = np.sort(lambda_grid)
    for lam in lambda_grid:
        losses = losses_fn(lam)
        n = len(losses)
        empirical_risk = (np.sum(losses) + B) / (n + 1)
        if empirical_risk <= alpha:
            return lam
    return lambda_grid[-1]


def conformal_risk_control_from_losses(cal_losses, alpha, B=1.0):
    """
    CRC when losses are already computed for each lambda on a grid.
    
    Parameters
    ----------
    cal_losses : array-like of shape (n,)
        Individual losses at a specific lambda value.
    alpha : float
        Target risk level.
    B : float
        Upper bound on individual losses.
    
    Returns
    -------
    bool : whether the CRC condition is satisfied at this lambda.
    """
    n = len(cal_losses)
    empirical_risk = (np.sum(cal_losses) + B) / (n + 1)
    return empirical_risk <= alpha


# ============================================================
# Risk-Controlling Prediction Sets (RCPS) with Hoeffding UCB
# ============================================================

def hoeffding_ucb(losses, delta, B=1.0):
    """
    Hoeffding upper confidence bound on the expected loss.
    
    Parameters
    ----------
    losses : array-like of shape (n,)
        Individual losses.
    delta : float
        Confidence level (1 - delta is the confidence).
    B : float
        Upper bound on individual losses.
    
    Returns
    -------
    ucb : float
        Upper confidence bound on expected loss.
    """
    losses = np.asarray(losses)
    n = len(losses)
    mean_loss = np.mean(losses)
    # Hoeffding's inequality: UCB = mean + B * sqrt(log(1/delta) / (2n))
    ucb = mean_loss + B * np.sqrt(np.log(1.0 / delta) / (2 * n))
    return ucb


def rcps_hoeffding(losses_fn, lambda_grid, alpha, delta=0.05, B=1.0):
    """
    Risk-Controlling Prediction Sets with Hoeffding UCB.
    
    Parameters
    ----------
    losses_fn : callable
        Function that takes lambda and returns array of individual losses.
    lambda_grid : array-like
        Grid of lambda values to search over.
    alpha : float
        Target risk level.
    delta : float
        Failure probability (1 - beta).
    B : float
        Upper bound on individual losses.
    
    Returns
    -------
    lambda_rcps : float
        Selected threshold.
    """
    lambda_grid = np.sort(lambda_grid)
    for lam in lambda_grid:
        losses = losses_fn(lam)
        ucb = hoeffding_ucb(losses, delta, B)
        if ucb <= alpha:
            return lam
    return lambda_grid[-1]


# ============================================================
# Bayesian Quadrature (BQ) - Main Contribution
# ============================================================

def sample_L_plus(losses, B=1.0, n_samples=1000, rng=None):
    """
    Sample from the distribution of L+, the random variable that
    stochastically dominates the posterior expected loss.
    
    L+ = sum_{i=1}^{n+1} U_i * ell_{(i)}
    where (U_1, ..., U_{n+1}) ~ Dir(1, ..., 1)
    and ell_{(1)} <= ... <= ell_{(n)} are the order statistics of the losses,
    with ell_{(n+1)} = B.
    
    Parameters
    ----------
    losses : array-like of shape (n,)
        Individual losses on calibration set.
    B : float
        Upper bound on individual losses.
    n_samples : int
        Number of Monte Carlo samples.
    rng : np.random.Generator, optional
        Random number generator.
    
    Returns
    -------
    L_plus_samples : array of shape (n_samples,)
        Samples from the distribution of L+.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    losses = np.asarray(losses)
    n = len(losses)
    
    # Order statistics + B as the (n+1)-th value
    sorted_losses = np.sort(losses)
    ell = np.append(sorted_losses, B)  # shape (n+1,)
    
    # Sample Dirichlet(1, ..., 1) with n+1 components
    # Dir(1,...,1) is equivalent to uniform on the simplex
    # Can be sampled as normalized Gamma(1) = Exponential(1) samples
    gamma_samples = rng.exponential(scale=1.0, size=(n_samples, n + 1))
    dirichlet_samples = gamma_samples / gamma_samples.sum(axis=1, keepdims=True)
    
    # L+ = sum_i U_i * ell_{(i)}
    L_plus_samples = dirichlet_samples @ ell  # shape (n_samples,)
    
    return L_plus_samples


def compute_L_plus_quantile(losses, beta, B=1.0, n_samples=1000, rng=None):
    """
    Compute the beta-quantile of L+.
    
    This is b_beta* = inf{b : Pr(L+ <= b) >= beta}.
    
    Parameters
    ----------
    losses : array-like of shape (n,)
        Individual losses on calibration set.
    beta : float
        Desired confidence level.
    B : float
        Upper bound on individual losses.
    n_samples : int
        Number of Monte Carlo samples.
    rng : np.random.Generator, optional
        Random number generator.
    
    Returns
    -------
    b_beta_star : float
        The beta-quantile of L+.
    """
    L_plus_samples = sample_L_plus(losses, B=B, n_samples=n_samples, rng=rng)
    return np.quantile(L_plus_samples, beta)


def bayesian_quadrature_decision_rule(losses_fn, lambda_grid, alpha, beta=0.95,
                                       B=1.0, n_samples=1000, rng=None):
    """
    Bayesian Quadrature decision rule based on HPD interval.
    
    Finds the infimum lambda such that Pr(L+ <= alpha | ell_{1:n}) >= beta.
    
    This is lambda_hpd^beta = inf{lambda : Pr(L+ <= alpha | ell_{1:n}) >= beta}.
    
    Parameters
    ----------
    losses_fn : callable
        Function that takes lambda and returns array of individual losses.
    lambda_grid : array-like
        Grid of lambda values to search over.
    alpha : float
        Target risk level.
    beta : float
        Desired confidence level (e.g., 0.95).
    B : float
        Upper bound on individual losses.
    n_samples : int
        Number of Monte Carlo samples for L+.
    rng : np.random.Generator, optional
        Random number generator.
    
    Returns
    -------
    lambda_bq : float
        Selected threshold.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    lambda_grid = np.sort(lambda_grid)
    for lam in lambda_grid:
        losses = losses_fn(lam)
        # Compute Pr(L+ <= alpha)
        L_plus_samples = sample_L_plus(losses, B=B, n_samples=n_samples, rng=rng)
        prob = np.mean(L_plus_samples <= alpha)
        if prob >= beta:
            return lam
    return lambda_grid[-1]


def bayesian_quadrature_from_losses(losses, alpha, beta=0.95, B=1.0,
                                     n_samples=1000, rng=None):
    """
    Check if the BQ condition is satisfied for given losses.
    
    Returns True if Pr(L+ <= alpha | ell_{1:n}) >= beta.
    
    Parameters
    ----------
    losses : array-like of shape (n,)
        Individual losses at a specific lambda value.
    alpha : float
        Target risk level.
    beta : float
        Desired confidence level.
    B : float
        Upper bound on individual losses.
    n_samples : int
        Number of Monte Carlo samples.
    rng : np.random.Generator, optional
        Random number generator.
    
    Returns
    -------
    bool : whether the BQ condition is satisfied.
    float : estimated probability Pr(L+ <= alpha).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    L_plus_samples = sample_L_plus(losses, B=B, n_samples=n_samples, rng=rng)
    prob = np.mean(L_plus_samples <= alpha)
    return prob >= beta, prob


def compute_expected_L_plus(losses, B=1.0):
    """
    Compute the expected value of L+ analytically.
    
    E[L+] = (1/(n+1)) * (sum_{i=1}^n ell_i + B)
    
    This recovers the CRC decision rule.
    
    Parameters
    ----------
    losses : array-like of shape (n,)
        Individual losses.
    B : float
        Upper bound on individual losses.
    
    Returns
    -------
    expected_L_plus : float
        Expected value of L+.
    """
    losses = np.asarray(losses)
    n = len(losses)
    return (np.sum(losses) + B) / (n + 1)
