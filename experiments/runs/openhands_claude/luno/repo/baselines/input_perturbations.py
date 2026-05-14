"""
Input perturbation baseline for uncertainty quantification.

Following Pathak et al. (2022) / FourCastNet:
  - Generate an ensemble of predictions by forwarding perturbed versions of the input
  - Perturbations: ε_{x,t} ~ N(0, σ²) added pointwise to the input function
  - σ is calibrated on the validation set to minimize marginal NLL

The empirical mean and std over the perturbed predictions give the predictive distribution.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx


def input_perturbation_mean_std(
    model: nnx.Module,
    a: jax.Array,
    sigma: float,
    n_samples: int,
    key: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Compute predictive mean and std via input perturbations.

    Args:
        model: trained FNO
        a: input function, shape (1, n_x, d_in) or (1, n_x, n_y, d_in)
        sigma: perturbation standard deviation
        n_samples: number of perturbed inputs
        key: JAX random key
    Returns:
        mean: predictive mean, shape matching model output
        std: predictive std, shape matching model output
    """
    # Generate perturbed inputs: (n_samples, *a.shape[1:])
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, (n_samples,) + a.shape[1:]) * sigma
    a_perturbed = a + eps  # broadcast over batch dim

    # Forward pass for each perturbed input
    def forward_single(a_i):
        return model(a_i[None])  # add batch dim

    predictions = jax.vmap(forward_single)(a_perturbed)  # (n_samples, 1, *out_shape)
    predictions = predictions[:, 0]  # (n_samples, *out_shape)

    mean = jnp.mean(predictions, axis=0)
    std = jnp.std(predictions, axis=0)
    return mean, std


def input_perturbation_samples(
    model: nnx.Module,
    a: jax.Array,
    sigma: float,
    n_samples: int,
    key: jax.Array,
) -> jax.Array:
    """
    Draw samples from the input perturbation predictive distribution.

    Args:
        model: trained FNO
        a: input function
        sigma: perturbation standard deviation
        n_samples: number of samples
        key: JAX random key
    Returns:
        samples: (n_samples, *out_shape)
    """
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, (n_samples,) + a.shape[1:]) * sigma
    a_perturbed = a + eps

    def forward_single(a_i):
        return model(a_i[None])[:, 0]  # remove batch dim

    return jax.vmap(forward_single)(a_perturbed)
