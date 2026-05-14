"""
Sample-based pushforward for uncertainty quantification.

Sample-* methods draw samples from the weight-space belief, map them through
the (nonlinear) neural operator, and compute a function-valued GP belief via
moment matching of the empirical mean and covariance.

This is in contrast to LUNO-* which uses the linearized (analytic) pushforward.
"""

from typing import Union

import jax
import jax.numpy as jnp
from flax import nnx

from luno.weight_uncertainty import (
    IsotropicGaussian,
    LowRankLaplace,
    model_fn_flat,
    sample_weights,
    set_flat_params,
)


def sample_pushforward_mean_std(
    model: nnx.Module,
    weight_uncertainty: Union[IsotropicGaussian, LowRankLaplace],
    a: jax.Array,
    n_samples: int,
    key: jax.Array,
) -> tuple:
    """
    Compute empirical mean and std via sample-based pushforward.

    Draws n_samples weight vectors, evaluates the model at each, and
    computes the empirical mean and standard deviation.

    Args:
        model: trained FNO
        weight_uncertainty: Gaussian weight-space belief
        a: input function
        n_samples: number of weight samples
        key: JAX random key
    Returns:
        mean: empirical mean, shape matching model output
        std: empirical std, shape matching model output
    """
    w_samples = sample_weights(weight_uncertainty, n_samples, key)  # (n_samples, p)

    def eval_sample(w):
        return model_fn_flat(model, w, a)

    predictions = jax.vmap(eval_sample)(w_samples)  # (n_samples, *out_shape)

    mean = jnp.mean(predictions, axis=0)
    std = jnp.std(predictions, axis=0)
    return mean, std


def sample_pushforward_covariance(
    model: nnx.Module,
    weight_uncertainty: Union[IsotropicGaussian, LowRankLaplace],
    a: jax.Array,
    n_samples: int,
    key: jax.Array,
) -> tuple:
    """
    Compute empirical mean and full covariance via sample-based pushforward.

    Args:
        model: trained FNO
        weight_uncertainty: Gaussian weight-space belief
        a: input function
        n_samples: number of weight samples
        key: JAX random key
    Returns:
        mean: (n_out,) empirical mean
        cov: (n_out, n_out) empirical covariance
    """
    w_samples = sample_weights(weight_uncertainty, n_samples, key)

    def eval_sample(w):
        out = model_fn_flat(model, w, a)
        return out.reshape(-1)

    predictions = jax.vmap(eval_sample)(w_samples)  # (n_samples, n_out)

    mean = jnp.mean(predictions, axis=0)
    centered = predictions - mean[None, :]
    cov = (centered.T @ centered) / (n_samples - 1)
    return mean, cov


def sample_pushforward_samples(
    model: nnx.Module,
    weight_uncertainty: Union[IsotropicGaussian, LowRankLaplace],
    a: jax.Array,
    n_samples: int,
    key: jax.Array,
) -> jax.Array:
    """
    Draw functional samples via sample-based pushforward.

    Args:
        model: trained FNO
        weight_uncertainty: Gaussian weight-space belief
        a: input function
        n_samples: number of samples
        key: JAX random key
    Returns:
        samples: (n_samples, *out_shape)
    """
    w_samples = sample_weights(weight_uncertainty, n_samples, key)

    def eval_sample(w):
        return model_fn_flat(model, w, a)

    return jax.vmap(eval_sample)(w_samples)
