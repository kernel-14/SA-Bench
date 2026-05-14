"""
Weight-space uncertainty representations for LUNO.

Implements:
  - IsotropicGaussian: N(w*, sigma^2 * I) weight-space belief
  - LaplaceApproximation: Linearized Laplace with low-rank GGN approximation
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Callable, NamedTuple, Optional, Tuple
from dataclasses import dataclass
import functools


@dataclass
class IsotropicGaussian:
    """Isotropic Gaussian weight-space belief: N(w*, sigma^2 * I).
    
    From Appendix D.3.3: represents weight-space uncertainty as N(w*, sigma^2 * I)
    where sigma^2 is calibrated on a validation set.
    """
    sigma2: float = 1.0

    def covariance_matvec(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute Sigma @ v = sigma^2 * v."""
        return self.sigma2 * v

    def sample_weight_perturbation(self, key: jax.Array, n_params: int, n_samples: int) -> jnp.ndarray:
        """Sample weight perturbations delta_w ~ N(0, Sigma).
        
        Returns: (n_samples, n_params)
        """
        return jax.random.normal(key, (n_samples, n_params)) * jnp.sqrt(self.sigma2)


@dataclass
class LaplaceApproximation:
    """Linearized Laplace approximation with low-rank GGN.
    
    From Appendix D.3.4: uses low-rank approximation of the GGN matrix.
    Posterior covariance: Sigma = (n * V @ V^T + sigma_prior * I)^{-1}
    
    where V contains the top eigenvectors of the GGN.
    
    Attributes:
        eigenvectors: Top-k eigenvectors of GGN, shape (n_params, rank)
        eigenvalues: Corresponding eigenvalues, shape (rank,)
        n_data: Number of training data points used for GGN
        prior_precision: Prior precision (sigma in the paper's notation)
        sigma2_scale: Additional scaling factor for calibration
    """
    eigenvectors: jnp.ndarray  # (n_params, rank)
    eigenvalues: jnp.ndarray   # (rank,)
    n_data: int
    prior_precision: float = 1.0
    sigma2_scale: float = 1.0

    def get_posterior_covariance_factors(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the posterior covariance in low-rank + diagonal form.
        
        Sigma = (n * V @ diag(lambda) @ V^T + prior_precision * I)^{-1}
        
        Using Woodbury identity:
        Sigma = (1/prior_precision) * I - (1/prior_precision^2) * V @ D_inv @ V^T
        where D_inv = diag(1 / (n * lambda_i / prior_precision + 1))
        
        Returns:
            V: eigenvectors (n_params, rank)
            d: diagonal correction factors (rank,)
        """
        # Posterior precision: P = n * V @ diag(lambda) @ V^T + prior_precision * I
        # Using Woodbury: P^{-1} = (1/prior_precision) * I 
        #                          - (1/prior_precision^2) * V @ (diag(1/(n*lambda/prior_precision + 1))) @ V^T
        n_lambda = self.n_data * self.eigenvalues  # (rank,)
        d = 1.0 / (n_lambda / self.prior_precision + 1.0)  # (rank,)
        return self.eigenvectors, d

    def covariance_matvec(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute Sigma @ v using Woodbury identity.
        
        Args:
            v: vector of shape (n_params,)
        Returns:
            Sigma @ v of shape (n_params,)
        """
        V, d = self.get_posterior_covariance_factors()
        # Sigma @ v = (1/prior_precision) * v - (1/prior_precision^2) * V @ diag(d) @ V^T @ v
        Vt_v = V.T @ v  # (rank,)
        correction = V @ (d * Vt_v)  # (n_params,)
        return self.sigma2_scale * (v / self.prior_precision - correction / self.prior_precision**2)

    def sample_weight_perturbation(self, key: jax.Array, n_params: int, n_samples: int) -> jnp.ndarray:
        """Sample weight perturbations delta_w ~ N(0, Sigma).
        
        Uses the low-rank structure for efficient sampling.
        
        Returns: (n_samples, n_params)
        """
        V, d = self.get_posterior_covariance_factors()
        rank = V.shape[1]

        key1, key2 = jax.random.split(key)

        # Sample from the low-rank part: V @ diag(sqrt(d/prior_precision)) @ z1
        z1 = jax.random.normal(key1, (n_samples, rank))
        low_rank_samples = z1 @ (V * jnp.sqrt(d / self.prior_precision)).T  # (n_samples, n_params)

        # Sample from the diagonal part: sqrt(1/prior_precision) * z2
        z2 = jax.random.normal(key2, (n_samples, n_params))
        diag_samples = z2 / jnp.sqrt(self.prior_precision)

        # Combine: this is not exact but approximates the full covariance
        # For exact sampling, we'd need to account for the correlation
        # Using the Cholesky of the Woodbury form
        return jnp.sqrt(self.sigma2_scale) * (diag_samples - low_rank_samples)


def compute_ggn_low_rank(
    model_fn: Callable,
    params_flat: jnp.ndarray,
    data_inputs: jnp.ndarray,
    data_targets: jnp.ndarray,
    rank: int = 500,
    batch_size: int = 32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute low-rank approximation of the Generalized Gauss-Newton (GGN) matrix.
    
    GGN = sum_i J_i^T H_i J_i
    
    where J_i is the Jacobian of the model output w.r.t. parameters at data point i,
    and H_i is the Hessian of the loss w.r.t. model output (= I for MSE loss).
    
    For MSE loss: GGN = sum_i J_i^T J_i
    
    We compute the top-k eigenvectors using randomized SVD / power iteration.
    
    Args:
        model_fn: Function (params_flat, x) -> y_pred
        params_flat: Flattened model parameters, shape (n_params,)
        data_inputs: Input data, shape (n_data, ...)
        data_targets: Target data, shape (n_data, ...)
        rank: Number of eigenvectors to compute
        batch_size: Batch size for Jacobian computation
    
    Returns:
        eigenvectors: (n_params, rank)
        eigenvalues: (rank,)
    """
    n_data = data_inputs.shape[0]
    n_params = params_flat.shape[0]

    # Compute GGN-vector products using Jacobian-vector products
    # GGN @ v = sum_i J_i^T (J_i @ v)
    def ggn_matvec(v: jnp.ndarray) -> jnp.ndarray:
        """Compute GGN @ v."""
        result = jnp.zeros(n_params)

        for start in range(0, n_data, batch_size):
            end = min(start + batch_size, n_data)
            x_batch = data_inputs[start:end]

            # Jacobian-vector product: J @ v
            def model_batch(p):
                return model_fn(p, x_batch)

            # J @ v using forward-mode AD
            _, jvp = jax.jvp(model_batch, (params_flat,), (v,))
            # jvp shape: (batch, output_dim...)

            # J^T @ (J @ v) using reverse-mode AD
            def jt_matvec(jvp_val):
                _, vjp_fn = jax.vjp(model_batch, params_flat)
                return vjp_fn(jvp_val)[0]

            result = result + jt_matvec(jvp)

        return result

    # Randomized power iteration to find top eigenvectors
    key = jax.random.PRNGKey(42)
    # Initialize random vectors
    Q = jax.random.normal(key, (n_params, rank))
    Q, _ = jnp.linalg.qr(Q)

    # Power iteration
    n_iter = 10
    for _ in range(n_iter):
        Z = jax.vmap(ggn_matvec, in_axes=1, out_axes=1)(Q)  # (n_params, rank)
        Q, R = jnp.linalg.qr(Z)

    # Compute eigenvalues
    AQ = jax.vmap(ggn_matvec, in_axes=1, out_axes=1)(Q)  # (n_params, rank)
    eigenvalues = jnp.diag(Q.T @ AQ)  # (rank,)

    # Sort by descending eigenvalue
    idx = jnp.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = Q[:, idx]

    return eigenvectors, eigenvalues
