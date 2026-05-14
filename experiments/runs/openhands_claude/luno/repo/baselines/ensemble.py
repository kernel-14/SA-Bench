"""
Deep ensemble baseline for uncertainty quantification.

Trains n_ensemble (default 10) independent FNOs with different random seeds.
Predictive distribution is approximated by the empirical distribution over ensemble members.

From Lakshminarayanan et al. (2017): "Simple and Scalable Predictive Uncertainty
Estimation using Deep Ensembles".
"""

from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import nnx


def ensemble_mean_std(
    models: List[nnx.Module],
    a: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Compute predictive mean and std from an ensemble of models.

    Args:
        models: list of trained FNO models
        a: input function
    Returns:
        mean: ensemble mean, shape matching model output
        std: ensemble std, shape matching model output
    """
    predictions = jnp.stack([model(a) for model in models], axis=0)  # (n_ensemble, *out_shape)
    mean = jnp.mean(predictions, axis=0)
    std = jnp.std(predictions, axis=0)
    return mean, std


def ensemble_predictions(
    models: List[nnx.Module],
    a: jax.Array,
) -> jax.Array:
    """
    Get all ensemble member predictions.

    Args:
        models: list of trained FNO models
        a: input function
    Returns:
        predictions: (n_ensemble, *out_shape)
    """
    return jnp.stack([model(a) for model in models], axis=0)


def ensemble_covariance(
    models: List[nnx.Module],
    a: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Compute empirical mean and covariance from ensemble predictions.

    Note: The ensemble covariance is rank-deficient (rank ≤ n_ensemble - 1),
    which is a fundamental limitation discussed in the paper.

    Args:
        models: list of trained FNO models
        a: input function
    Returns:
        mean: (n_out,) empirical mean
        cov: (n_out, n_out) empirical covariance (rank-deficient)
    """
    preds = ensemble_predictions(models, a)  # (n_ensemble, *out_shape)
    out_shape = preds.shape[1:]
    n_out = int(jnp.prod(jnp.array(out_shape)))

    preds_flat = preds.reshape(preds.shape[0], -1)  # (n_ensemble, n_out)
    mean = jnp.mean(preds_flat, axis=0)
    centered = preds_flat - mean[None, :]
    n = preds_flat.shape[0]
    cov = (centered.T @ centered) / (n - 1)
    return mean, cov
