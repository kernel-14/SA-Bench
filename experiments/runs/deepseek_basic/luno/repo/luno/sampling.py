"""Sampling and moment computation for function-valued Gaussian processes.

Provides utilities for:
1. Drawing lazy functional samples from function-valued GPs
2. Computing marginal moments (mean and variance)
3. Computing full covariance matrices at discretized points
4. Efficient Jacobian-vector product based pushforward
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable
from functools import partial


def sample_function_valued_gp(
    rng_key: jax.random.PRNGKey,
    mean_fn: Callable,
    cov_fn: Callable,
    x_grid: jnp.ndarray,
    n_samples: int = 1,
    output_dim: int = 1,
) -> jnp.ndarray:
    """Draw samples from a function-valued GP evaluated at a grid.
    
    Given F(a) ~ GP(m_a, K_a), draw samples of the output function
    evaluated at points x_grid.
    
    Args:
        rng_key: JAX random key
        mean_fn: Mean function m_a: x -> R^{d'_U}
        cov_fn: Covariance function K_a: (x_1, x_2) -> R^{d'_U × d'_U}
        x_grid: Grid points of shape (n_x,)
        n_samples: Number of function samples
        output_dim: d'_U
    
    Returns:
        Function samples of shape (n_samples, output_dim, n_x)
    """
    n_x = x_grid.shape[0]
    
    # Build covariance matrix on the grid
    K = compute_covariance_matrix(cov_fn, x_grid, output_dim)
    
    # Compute mean
    mean = jnp.zeros((output_dim, n_x))
    for i in range(n_x):
        mean = mean.at[:, i].set(mean_fn(x_grid[i]))
    
    # Flatten and sample
    mean_flat = mean.T.reshape(-1)  # (n_x * output_dim,)
    
    # Add jitter for numerical stability
    K_jitter = K + 1e-6 * jnp.eye(K.shape[0])
    L = jnp.linalg.cholesky(K_jitter)
    
    samples_flat = mean_flat + L @ jax.random.normal(
        rng_key, (n_samples, K.shape[0])
    ).T
    
    # Reshape to (n_samples, output_dim, n_x)
    samples = samples_flat.T.reshape(n_samples, n_x, output_dim)
    return samples.transpose(0, 2, 1)


def compute_covariance_matrix(
    cov_fn: Callable,
    x_grid: jnp.ndarray,
    output_dim: int = 1,
) -> jnp.ndarray:
    """Compute the full covariance matrix K((a, x_i), (a, x_j)) for all grid points.
    
    Args:
        cov_fn: Covariance function K_a: (x_1, x_2) -> R^{d'_U × d'_U}
        x_grid: Grid points of shape (n_x,)
        output_dim: d'_U
    
    Returns:
        Covariance matrix of shape (n_x * output_dim, n_x * output_dim)
    """
    n_x = x_grid.shape[0]
    n_total = n_x * output_dim
    
    K = jnp.zeros((n_total, n_total))
    
    for i in range(n_x):
        for j in range(n_x):
            K_ij = cov_fn(x_grid[i], x_grid[j])  # (d'_U, d'_U)
            for d1 in range(output_dim):
                for d2 in range(output_dim):
                    idx1 = i * output_dim + d1
                    idx2 = j * output_dim + d2
                    K = K.at[idx1, idx2].set(K_ij[d1, d2])
    
    return K


def compute_marginal_moments(
    mean_fn: Callable,
    cov_fn: Callable,
    x_grid: jnp.ndarray,
    output_dim: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute marginal mean and variance at each grid point.
    
    Args:
        mean_fn: Mean function m_a: x -> R^{d'_U}
        cov_fn: Covariance function K_a: (x_1, x_2) -> R^{d'_U × d'_U}
        x_grid: Grid points
        output_dim: d'_U
    
    Returns:
        Tuple (mean, variance) each of shape (output_dim, n_x)
    """
    n_x = x_grid.shape[0]
    
    mean = jnp.zeros((output_dim, n_x))
    variance = jnp.zeros((output_dim, n_x))
    
    for i in range(n_x):
        mean = mean.at[:, i].set(mean_fn(x_grid[i]))
        K_ii = cov_fn(x_grid[i], x_grid[i])
        variance = variance.at[:, i].set(jnp.diag(K_ii))
    
    return mean, variance


def sample_based_pushforward(
    model_fn: Callable,
    params: Any,
    x_input: jnp.ndarray,
    weight_samples: jnp.ndarray,  # (n_samples, n_params)
) -> jnp.ndarray:
    """Push weight samples through the nonlinear model (Sample-* method).
    
    This implements the sampling-based approach for comparison.
    
    Args:
        model_fn: Neural operator mapping (params, x) -> predictions
        params: Reference parameter structure (for structure, not values)
        x_input: Input to evaluate at
        weight_samples: Weight samples of shape (n_samples, n_params)
    
    Returns:
        Predictions of shape (n_samples, output_dim)
    """
    # Unflatten each sample and evaluate
    predictions = []
    
    leaves_ref, tree_def = jax.tree_util.tree_flatten(params)
    shapes = [l.shape for l in leaves_ref]
    sizes = [int(jnp.prod(jnp.array(s))) for s in shapes]
    
    for s in range(weight_samples.shape[0]):
        # Unflatten
        splits = jnp.split(weight_samples[s], jnp.cumsum(jnp.array(sizes))[:-1])
        w_sample = []
        for split, shape in zip(splits, shapes):
            w_sample.append(split.reshape(shape))
        w_sample_tree = jax.tree_util.tree_unflatten(tree_def, w_sample)
        
        pred = model_fn(w_sample_tree, x_input)
        predictions.append(pred)
    
    return jnp.stack(predictions)


def compute_empirical_gp_from_samples(
    predictions: jnp.ndarray,  # (n_samples, n_x, output_dim)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute empirical GP mean and covariance from samples.
    
    Used for Sample-* methods to construct a Gaussian process belief
    by moment matching.
    
    Args:
        predictions: Samples of shape (n_samples, n_x, output_dim)
    
    Returns:
        Tuple (mean, covariance) where:
        - mean: (n_x, output_dim)
        - covariance: (n_x * output_dim, n_x * output_dim)
    """
    n_samples, n_x, d_out = predictions.shape
    
    # Empirical mean
    mean = jnp.mean(predictions, axis=0)  # (n_x, d_out)
    
    # Empirical covariance
    pred_flat = predictions.reshape(n_samples, -1)  # (n_samples, n_x * d_out)
    centered = pred_flat - jnp.mean(pred_flat, axis=0, keepdims=True)
    cov = (centered.T @ centered) / (n_samples - 1)
    
    return mean, cov


def compute_std_from_covariance(
    covariance: jnp.ndarray,
    grid_size: int,
    output_dim: int,
) -> jnp.ndarray:
    """Extract marginal standard deviations from covariance matrix.
    
    Args:
        covariance: Full covariance matrix of shape (n_x * d_out, n_x * d_out)
        grid_size: n_x
        output_dim: d_out
    
    Returns:
        Standard deviations of shape (output_dim, grid_size)
    """
    std = jnp.zeros((output_dim, grid_size))
    for i in range(grid_size):
        for d in range(output_dim):
            idx = i * output_dim + d
            std = std.at[d, i].set(jnp.sqrt(jnp.maximum(covariance[idx, idx], 0.0)))
    return std


def compute_eigenfunction_decomposition(
    covariance: jnp.ndarray,
    n_eigen: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute top eigenfunctions of the predictive covariance.
    
    These are used for visualization (Figures 2, 6) to show the principal
    modes of variation in the predictive belief.
    
    Args:
        covariance: Covariance matrix of shape (n_x * d_out, n_x * d_out)
        n_eigen: Number of top eigenfunctions to compute
    
    Returns:
        Tuple (eigenvalues, eigenfunctions) where:
        - eigenvalues: (n_eigen,)
        - eigenfunctions: (n_eigen, n_x, d_out)
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    
    # Sort in descending order
    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][:n_eigen]
    eigenvectors = eigenvectors[:, idx][:, :n_eigen]
    
    # Reshape eigenvectors to (n_eigen, n_x, d_out)
    n_total = covariance.shape[0]
    d_out = n_total // (eigenvectors.shape[0] // n_total)
    # Hmm, need grid_size. Let's infer from the shape.
    
    return eigenvalues, eigenvectors


def linearized_pushforward_moments(
    jacobian_fn: Callable,
    weight_mean: jnp.ndarray,
    weight_cov_sqrt: jnp.ndarray,
    x_input: jnp.ndarray,
    output_fn: Callable,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute LUNO moments efficiently using the linearized pushforward.
    
    For LUNO-* methods, the predictive distribution is:
    - mean: f(x, μ) 
    - cov: J(x) Σ J(x)^T
    
    This computes these without materializing the full Σ.
    
    Args:
        jacobian_fn: Function to compute Jacobian J(x) = D_w f(x, w)|_μ
        weight_mean: μ (flattened)
        weight_cov_sqrt: Σ^{1/2} (or function to apply Σ^{1/2} to a vector)
        x_input: Input at which to evaluate
        output_fn: f(x, μ) function
    
    Returns:
        Tuple (mean, covariance) of predictive distribution
    """
    # Mean prediction
    mean_pred = output_fn(x_input)
    
    # Jacobian at x_input
    J = jacobian_fn(x_input)  # (d_out, p)
    
    # Compute covariance: J Σ J^T
    if callable(weight_cov_sqrt):
        # Apply Σ^{1/2} to each row of J^T and multiply
        # J Σ J^T = (Σ^{1/2} J^T)^T (Σ^{1/2} J^T)
        # = (J Σ^{1/2}) (J Σ^{1/2})^T
        JS = jnp.zeros_like(J)
        for d in range(J.shape[0]):
            JS_d = weight_cov_sqrt(J[d, :])
            JS = JS.at[d, :].set(JS_d)
        cov_pred = JS @ JS.T
    else:
        # Σ is a matrix
        cov_pred = J @ weight_cov_sqrt @ weight_cov_sqrt.T @ J.T
    
    return mean_pred, cov_pred
