"""
Evaluation metrics for uncertainty quantification.

Implements the metrics from Appendix D.4:
  - RMSE: Root Mean Squared Error
  - NLL: Marginal Negative Log-Likelihood
  - chi2: Chi-squared statistic
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


def compute_rmse(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
) -> float:
    """Compute Root Mean Squared Error.
    
    RMSE = sqrt(1/n * sum_i (y_i - y_hat_i)^2)
    
    From Appendix D.4.1.
    
    Args:
        y_true: Ground truth values, shape (n,) or (n, d)
        y_pred: Predicted mean values, shape (n,) or (n, d)
    
    Returns:
        RMSE scalar
    """
    return float(jnp.sqrt(jnp.mean((y_true - y_pred) ** 2)))


def compute_nll(
    y_true: jnp.ndarray,
    y_pred_mean: jnp.ndarray,
    y_pred_std: jnp.ndarray,
    eps: float = 1e-8,
) -> float:
    """Compute Marginal Negative Log-Likelihood.
    
    NLL = -sum_i log(N(y_i; y_hat_i, sigma_i^2))
        = sum_i [0.5 * log(2*pi*sigma_i^2) + (y_i - y_hat_i)^2 / (2*sigma_i^2)]
    
    From Appendix D.4.2.
    
    Args:
        y_true: Ground truth values, shape (n,) or (n, d)
        y_pred_mean: Predicted mean values, shape (n,) or (n, d)
        y_pred_std: Predicted standard deviations, shape (n,) or (n, d)
        eps: Small value for numerical stability
    
    Returns:
        NLL scalar (averaged over all elements)
    """
    sigma2 = jnp.maximum(y_pred_std ** 2, eps)
    nll = 0.5 * jnp.log(2 * jnp.pi * sigma2) + (y_true - y_pred_mean) ** 2 / (2 * sigma2)
    return float(jnp.mean(nll))


def compute_chi2(
    y_true: jnp.ndarray,
    y_pred_mean: jnp.ndarray,
    y_pred_std: jnp.ndarray,
    eps: float = 1e-8,
) -> float:
    """Compute Chi-squared statistic.
    
    chi2 = 1/n * sum_i (y_i - y_hat_i)^2 / sigma_i^2
    
    A value close to 1 indicates well-calibrated uncertainty.
    Values > 1 indicate overconfidence, values < 1 indicate underconfidence.
    
    From Appendix D.4.3.
    
    Args:
        y_true: Ground truth values, shape (n,) or (n, d)
        y_pred_mean: Predicted mean values, shape (n,) or (n, d)
        y_pred_std: Predicted standard deviations, shape (n,) or (n, d)
        eps: Small value for numerical stability
    
    Returns:
        chi2 scalar
    """
    sigma2 = jnp.maximum(y_pred_std ** 2, eps)
    chi2 = jnp.mean((y_true - y_pred_mean) ** 2 / sigma2)
    return float(chi2)


def evaluate_predictions(
    y_true: jnp.ndarray,
    y_pred_mean: jnp.ndarray,
    y_pred_std: jnp.ndarray,
    eps: float = 1e-8,
) -> dict:
    """Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred_mean: Predicted mean values
        y_pred_std: Predicted standard deviations
        eps: Small value for numerical stability
    
    Returns:
        Dictionary with 'rmse', 'nll', 'chi2' keys
    """
    return {
        'rmse': compute_rmse(y_true, y_pred_mean),
        'nll': compute_nll(y_true, y_pred_mean, y_pred_std, eps),
        'chi2': compute_chi2(y_true, y_pred_mean, y_pred_std, eps),
    }


def calibrate_sigma(
    y_true: jnp.ndarray,
    y_pred_mean: jnp.ndarray,
    y_pred_std_unnormalized: jnp.ndarray,
    n_grid: int = 500,
    sigma_range: tuple = (1e-4, 1e2),
) -> float:
    """Calibrate the sigma^2 scaling parameter by minimizing NLL on validation data.
    
    From Appendix D.5: hyperparameters are calibrated using 250 input-output pairs
    of the validation set to minimize the marginal NLL, using grid search over
    a logarithmically spaced grid with 500 points.
    
    Args:
        y_true: Ground truth values
        y_pred_mean: Predicted mean values
        y_pred_std_unnormalized: Predicted std with sigma=1 (to be scaled)
        n_grid: Number of grid points for search
        sigma_range: (min, max) range for sigma search
    
    Returns:
        Optimal sigma value
    """
    sigmas = jnp.logspace(
        jnp.log10(sigma_range[0]),
        jnp.log10(sigma_range[1]),
        n_grid
    )

    best_sigma = sigmas[0]
    best_nll = float('inf')

    for sigma in sigmas:
        scaled_std = y_pred_std_unnormalized * sigma
        nll = compute_nll(y_true, y_pred_mean, scaled_std)
        if nll < best_nll:
            best_nll = nll
            best_sigma = float(sigma)

    return best_sigma
