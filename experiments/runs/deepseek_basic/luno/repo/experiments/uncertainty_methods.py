"""Implementation of all uncertainty quantification methods compared in LUNO.

Methods (Section 5, Appendix D.3):
1. Input Perturbations
2. Deep Ensembles
3. Isotropic Gaussian (*-Iso)
4. Laplace Approximation (*-LA)
5. Sample-* variants
6. LUNO-* variants
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Callable, Any, Dict
from dataclasses import dataclass
from functools import partial


@dataclass
class UQResults:
    """Container for uncertainty quantification results."""
    mean: jnp.ndarray        # Predictive mean
    variance: jnp.ndarray    # Predictive marginal variance
    covariance: jnp.ndarray  # Full predictive covariance (optional)
    samples: jnp.ndarray     # Samples from predictive (optional)


def input_perturbations(
    model_fn: Callable,
    params: Any,
    x_input: jnp.ndarray,
    sigma: float,
    n_perturbations: int = 200,
    rng_key: jax.random.PRNGKey = None,
) -> UQResults:
    """Input Perturbations method (Section D.3.1).
    
    Following Pathak et al. (2022): generate predictions by forwarding
    batch of pointwise perturbed versions of a single input.
    
    Perturbations: ε_{x,t} ~ N(0, σ²) for each input value.
    σ is calibrated to achieve accurate marginal uncertainty.
    
    Args:
        model_fn: Model function
        params: Model parameters
        x_input: Input to perturb
        sigma: Perturbation standard deviation
        n_perturbations: Number of perturbed forward passes
        rng_key: Random key
    
    Returns:
        UQResults with empirical mean and variance
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    keys = jax.random.split(rng_key, n_perturbations)
    
    predictions = []
    for k in range(n_perturbations):
        noise = sigma * jax.random.normal(keys[k], x_input.shape)
        x_perturbed = x_input + noise
        pred = model_fn(params, x_perturbed)
        predictions.append(pred)
    
    preds = jnp.stack(predictions)
    mean = jnp.mean(preds, axis=0)
    variance = jnp.var(preds, axis=0)
    
    return UQResults(
        mean=mean,
        variance=variance,
        covariance=None,
        samples=preds,
    )


def deep_ensemble(
    ensemble_models: list,
    ensemble_params: list,
    x_input: jnp.ndarray,
) -> UQResults:
    """Deep Ensemble method (Section D.3.2).
    
    10 independently trained FNOs with different random seeds.
    
    Args:
        ensemble_models: List of 10 model functions
        ensemble_params: List of 10 parameter sets
        x_input: Input to evaluate
    
    Returns:
        UQResults with ensemble mean and variance
    """
    predictions = []
    for model_fn, params in zip(ensemble_models, ensemble_params):
        pred = model_fn(params, x_input)
        predictions.append(pred)
    
    preds = jnp.stack(predictions)
    mean = jnp.mean(preds, axis=0)
    variance = jnp.var(preds, axis=0)
    
    # Covariance is rank-deficient (rank ≤ 9 for 10 members)
    preds_flat = preds.reshape(preds.shape[0], -1)
    centered = preds_flat - jnp.mean(preds_flat, axis=0, keepdims=True)
    cov = (centered.T @ centered) / (preds.shape[0] - 1)
    
    return UQResults(
        mean=mean,
        variance=variance,
        covariance=cov,
        samples=preds,
    )


def sample_based_iso(
    model_fn: Callable,
    params: Any,
    x_input: jnp.ndarray,
    sigma_squared: float,
    n_samples: int = 200,
    rng_key: jax.random.PRNGKey = None,
) -> UQResults:
    """Sample-Iso method: Sample from isotropic Gaussian weight belief.
    
    w ~ N(w*, σ²I), push through nonlinear model.
    
    Args:
        model_fn: Model function
        params: MAP weights w*
        x_input: Input
        sigma_squared: Isotropic variance
        n_samples: Number of weight samples (200 in paper)
        rng_key: Random key
    
    Returns:
        UQResults
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    leaves, tree_def = jax.tree_util.tree_flatten(params)
    flat_params = jnp.concatenate([l.ravel() for l in leaves])
    n_params = flat_params.shape[0]
    
    shapes = [l.shape for l in leaves]
    sizes = [int(jnp.prod(jnp.array(s))) for s in shapes]
    
    keys = jax.random.split(rng_key, n_samples)
    
    predictions = []
    for k in range(n_samples):
        # Sample weights
        w_sample = flat_params + jnp.sqrt(sigma_squared) * jax.random.normal(keys[k], (n_params,))
        
        # Unflatten
        splits = jnp.split(w_sample, jnp.cumsum(jnp.array(sizes))[:-1])
        w_tree = jax.tree_util.tree_unflatten(
            tree_def,
            [s.reshape(shape) for s, shape in zip(splits, shapes)]
        )
        
        pred = model_fn(w_tree, x_input)
        predictions.append(pred)
    
    preds = jnp.stack(predictions)
    mean = jnp.mean(preds, axis=0)
    variance = jnp.var(preds, axis=0)
    
    return UQResults(
        mean=mean,
        variance=variance,
        covariance=None,
        samples=preds,
    )


def sample_based_la(
    model_fn: Callable,
    params: Any,
    x_input: jnp.ndarray,
    laplace_belief,
    n_samples: int = 200,
    rng_key: jax.random.PRNGKey = None,
) -> UQResults:
    """Sample-LA method: Sample from Laplace-approximated weight belief.
    
    w ~ N(w*, Σ_LA), push through nonlinear model.
    
    Args:
        model_fn: Model function
        params: MAP weights w*
        x_input: Input
        laplace_belief: LowRankLaplace object
        n_samples: Number of weight samples
        rng_key: Random key
    
    Returns:
        UQResults
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    leaves, tree_def = jax.tree_util.tree_flatten(params)
    shapes = [l.shape for l in leaves]
    sizes = [int(jnp.prod(jnp.array(s))) for s in shapes]
    
    weight_samples = laplace_belief.sample(rng_key, n_samples)
    
    predictions = []
    for k in range(n_samples):
        w_sample = weight_samples[k]
        splits = jnp.split(w_sample, jnp.cumsum(jnp.array(sizes))[:-1])
        w_tree = jax.tree_util.tree_unflatten(
            tree_def,
            [s.reshape(shape) for s, shape in zip(splits, shapes)]
        )
        
        pred = model_fn(w_tree, x_input)
        predictions.append(pred)
    
    preds = jnp.stack(predictions)
    mean = jnp.mean(preds, axis=0)
    variance = jnp.var(preds, axis=0)
    
    return UQResults(
        mean=mean,
        variance=variance,
        covariance=None,
        samples=preds,
    )


def luno_iso(
    model_fn: Callable,
    params: Any,
    x_input: jnp.ndarray,
    sigma_squared: float,
) -> UQResults:
    """LUNO-Iso: Linearized pushforward with isotropic Gaussian belief.
    
    f_lin(x, w) = f(x, w*) + J(x)(w - w*)
    ⇒ f ~ N(f(x, w*), σ² J(x) J(x)^T)
    
    Args:
        model_fn: Model function
        params: MAP weights w*
        x_input: Input
        sigma_squared: Isotropic variance
    
    Returns:
        UQResults with linearized mean and covariance
    """
    # Mean prediction
    f_star = model_fn(params, x_input)
    
    # Compute Jacobian
    f_fn = lambda p: model_fn(p, x_input)
    jac = jax.jacrev(f_fn)(params)
    
    # Flatten Jacobian
    leaves, _ = jax.tree_util.tree_flatten(jac)
    J = jnp.concatenate([l.reshape(l.shape[0], -1) for l in leaves], axis=1)
    
    # Covariance: σ² J J^T
    cov = sigma_squared * (J @ J.T)
    variance = jnp.diag(cov)
    
    return UQResults(
        mean=f_star,
        variance=variance,
        covariance=cov,
        samples=None,
    )


def luno_la(
    model_fn: Callable,
    params: Any,
    x_input: jnp.ndarray,
    laplace_belief,
) -> UQResults:
    """LUNO-LA: Linearized pushforward with Laplace-approximated belief.
    
    f_lin(x, w) = f(x, w*) + J(x)(w - w*)
    ⇒ f ~ N(f(x, w*), J(x) Σ_LA J(x)^T)
    
    Args:
        model_fn: Model function
        params: MAP weights w*
        x_input: Input
        laplace_belief: LowRankLaplace object
    
    Returns:
        UQResults with linearized mean and covariance
    """
    # Mean prediction
    f_star = model_fn(params, x_input)
    
    # Compute Jacobian
    f_fn = lambda p: model_fn(p, x_input)
    jac = jax.jacrev(f_fn)(params)
    
    # Flatten Jacobian
    leaves, _ = jax.tree_util.tree_flatten(jac)
    J = jnp.concatenate([l.reshape(l.shape[0], -1) for l in leaves], axis=1)
    
    # Covariance: J Σ_LA J^T
    Sigma = laplace_belief.get_covariance_matrix()
    cov = J @ Sigma @ J.T
    variance = jnp.diag(cov)
    
    return UQResults(
        mean=f_star,
        variance=variance,
        covariance=cov,
        samples=None,
    )
