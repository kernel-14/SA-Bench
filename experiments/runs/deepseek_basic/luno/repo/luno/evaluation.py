"""Evaluation metrics for LUNO uncertainty quantification.

Implements the metrics described in Section D.4:
1. Root Mean Squared Error (RMSE)
2. Marginal Negative Log-Likelihood (NLL)
3. χ²-statistic

And calibration utilities from Section D.5.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional


def compute_rmse(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
) -> float:
    """Compute Root Mean Squared Error.
    
    RMSE = sqrt( (1/n) Σ_i (y_i - ŷ_i)² )
    
    Args:
        predictions: Predicted means ŷ_i, shape (n,) or (n, d)
        targets: Ground truth y_i, same shape
    
    Returns:
        RMSE value
    """
    squared_errors = (predictions - targets) ** 2
    mse = jnp.mean(squared_errors)
    return jnp.sqrt(mse)


def compute_marginal_nll(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    variances: jnp.ndarray,
) -> float:
    """Compute marginal Negative Log-Likelihood.
    
    NLL = -Σ_i log( (1/√(2πσ_i²)) exp(-(y_i - ŷ_i)² / (2σ_i²)) )
        = (1/2) Σ_i [log(2π) + log(σ_i²) + (y_i - ŷ_i)² / σ_i²]
    
    Lower values indicate better calibration.
    
    Args:
        predictions: Predicted means ŷ_i, shape (n,) or (n, d)
        targets: Ground truth y_i
        variances: Predictive variances σ_i², same shape
    
    Returns:
        Marginal NLL value (expected over test samples)
    """
    n = predictions.size
    log_var = jnp.log(jnp.maximum(variances, 1e-12))
    squared_error = (predictions - targets) ** 2
    nll_per_point = 0.5 * (jnp.log(2 * jnp.pi) + log_var + squared_error / jnp.maximum(variances, 1e-12))
    return jnp.mean(nll_per_point)


def compute_chi2_statistic(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    variances: jnp.ndarray,
) -> float:
    """Compute χ²-statistic.
    
    χ² = (1/n) Σ_i (y_i - ŷ_i)² / σ_i²
    
    Values close to 1 indicate well-calibrated uncertainty.
    > 1: overconfident
    < 1: underconfident
    
    Args:
        predictions: Predicted means ŷ_i
        targets: Ground truth y_i
        variances: Predictive variances σ_i²
    
    Returns:
        χ²-statistic value
    """
    squared_error = (predictions - targets) ** 2
    normalized_error = squared_error / jnp.maximum(variances, 1e-12)
    return jnp.mean(normalized_error)


def evaluate_all_metrics(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    variances: jnp.ndarray,
) -> dict:
    """Compute all evaluation metrics.
    
    Args:
        predictions: Predicted means, shape (n_test, n_x, d_out)
        targets: Ground truth, same shape
        variances: Predictive variances, same shape
    
    Returns:
        Dictionary with keys 'rmse', 'nll', 'chi2'
    """
    n_test = predictions.shape[0]
    
    total_rmse = 0.0
    total_nll = 0.0
    total_chi2 = 0.0
    
    for i in range(n_test):
        total_rmse += compute_rmse(predictions[i], targets[i])
        total_nll += compute_marginal_nll(predictions[i], targets[i], variances[i])
        total_chi2 += compute_chi2_statistic(predictions[i], targets[i], variances[i])
    
    return {
        'rmse': total_rmse / n_test,
        'nll': total_nll / n_test,
        'chi2': total_chi2 / n_test,
    }


def calibrate_hyperparameter(
    predictions_fn,
    targets: jnp.ndarray,
    param_grid: jnp.ndarray,
) -> Tuple[float, float]:
    """Calibrate hyperparameter by grid search to minimize NLL.
    
    Following Section D.5: calibrate σ² via grid search over a 
    logarithmically spaced grid with 500 points centered around 
    the relevant value.
    
    Args:
        predictions_fn: Function(param) -> (predictions, variances)
        targets: Validation targets
        param_grid: Grid of hyperparameter values to search
    
    Returns:
        Tuple (best_param, best_nll)
    """
    best_nll = float('inf')
    best_param = param_grid[0]
    
    for param in param_grid:
        preds, varis = predictions_fn(param)
        nll = compute_marginal_nll(preds, targets, varis)
        
        if nll < best_nll:
            best_nll = nll
            best_param = param
    
    return best_param, best_nll


def compute_log_spaced_grid(
    center: float,
    n_points: int = 500,
    span: float = 4.0,
) -> jnp.ndarray:
    """Create logarithmically spaced grid around center.
    
    Args:
        center: Center value
        n_points: Number of grid points (500 in paper)
        span: Logarithmic span around center
    
    Returns:
        Grid of parameter values
    """
    log_center = jnp.log(center)
    log_min = log_center - span / 2
    log_max = log_center + span / 2
    return jnp.exp(jnp.linspace(log_min, log_max, n_points))
