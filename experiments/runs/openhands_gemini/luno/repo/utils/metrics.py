
import jax.numpy as jnp
from typing import Tuple

def rmse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Root Mean Squared Error (RMSE).
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted mean values.
        
    Returns:
        The RMSE value.
    """
    return jnp.sqrt(jnp.mean(jnp.square(y_true - y_pred)))

def nll(y_true: jnp.ndarray, y_pred_mean: jnp.ndarray, y_pred_std: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Negative Log-Likelihood (NLL) assuming Gaussian uncertainty.
    
    Args:
        y_true: Ground truth values.
        y_pred_mean: Predicted mean values.
        y_pred_std: Predicted standard deviation values.
        
    Returns:
        The NLL value.
    """
    # Ensure std is positive for log and division
    y_pred_std = jnp.maximum(y_pred_std, 1e-6)
    
    # NLL = 0.5 * log(2 * pi * sigma^2) + 0.5 * (y - mu)^2 / sigma^2
    term1 = 0.5 * jnp.log(2 * jnp.pi * jnp.square(y_pred_std))
    term2 = 0.5 * jnp.square(y_true - y_pred_mean) / jnp.square(y_pred_std)
    
    return jnp.mean(term1 + term2)

def chi_squared_statistic(y_true: jnp.ndarray, y_pred_mean: jnp.ndarray, y_pred_std: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Chi-squared statistic.
    
    Args:
        y_true: Ground truth values.
        y_pred_mean: Predicted mean values.
        y_pred_std: Predicted standard deviation values.
        
    Returns:
        The Chi-squared statistic.
    """
    # Ensure std is positive for division
    y_pred_std = jnp.maximum(y_pred_std, 1e-6)
    
    return jnp.mean(jnp.square(y_true - y_pred_mean) / jnp.square(y_pred_std))

def compute_metrics(
    y_true: jnp.ndarray, 
    y_pred_mean: jnp.ndarray, 
    y_pred_cov: jnp.ndarray
) -> dict:
    """
    Computes RMSE, NLL, and Chi-squared statistics from predictions.
    
    Args:
        y_true: Ground truth values (batch, spatial_res, output_dim).
        y_pred_mean: Predicted mean values (batch, spatial_res, output_dim).
        y_pred_cov: Predicted covariance matrix (batch, spatial_res, output_dim, spatial_res, output_dim).
                    For NLL and Chi2, we need the marginal variance (diagonal of covariance).
                    If output_dim > 1, we assume independent channels for NLL/Chi2 for simplicity,
                    taking the diagonal of the output_dim x output_dim covariance block.
                    The paper often reports marginal NLL, which implies taking individual variances.
    Returns:
        A dictionary of computed metrics.
    """
    metrics = {}
    
    # RMSE
    metrics['RMSE'] = rmse(y_true, y_pred_mean)

    # Extract marginal standard deviations for NLL and Chi-squared
    # y_pred_cov is (batch, spatial_res, output_dim, spatial_res, output_dim)
    # We need sigma_i for each (batch_idx, spatial_idx, output_dim_idx)
    
    # Get the diagonal elements of the covariance matrix for each (batch, spatial_res)
    # This will give the variance for each output_dim channel.
    # The paper's K_a(x1, x2) refers to the covariance function for the output function F(a).
    # If we need the marginal variance at a point (x, channel), it's K_a(x, x)[channel, channel]
    
    marginal_variances = jnp.zeros_like(y_pred_mean) # (batch, spatial_res, output_dim)
    for b in range(y_pred_mean.shape[0]):
        for s in range(y_pred_mean.shape[1]):
            for o in range(y_pred_mean.shape[2]):
                marginal_variances = marginal_variances.at[b, s, o].set(y_pred_cov[b, s, o, s, o])
    
    y_pred_std = jnp.sqrt(marginal_variances)
    
    # NLL
    metrics['NLL'] = nll(y_true, y_pred_mean, y_pred_std)
    
    # Chi-squared
    metrics['Chi2'] = chi_squared_statistic(y_true, y_pred_mean, y_pred_std)
    
    return metrics
