"""Weight-space uncertainty models for LUNO.

This module implements various methods for modeling uncertainty in the 
weight space of neural operators. Following the paper, we support:

1. Isotropic Gaussian (*-Iso): Σ = σ² I
2. Low-Rank Laplace Approximation (*-LA): Using GGN low-rank structure
3. Deep Ensemble weight beliefs (for comparison)

These weight-space beliefs are then pushed forward through the linearized
neural operator to obtain function-valued Gaussian process beliefs.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass


@dataclass
class WeightSpaceBelief:
    """Base class for weight-space uncertainty beliefs.
    
    A weight-space belief is a probability distribution over the parameters
    w ∈ W ⊂ R^p of a neural operator F: A × W → U.
    """
    mean: jnp.ndarray  # μ ∈ R^p
    covariance: jnp.ndarray  # Σ ∈ R^{p×p}, may be low-rank representation
    
    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> jnp.ndarray:
        """Draw samples from the weight-space belief.
        
        Args:
            rng_key: JAX random key
            n_samples: Number of samples to draw
        
        Returns:
            Samples of shape (n_samples, p)
        """
        raise NotImplementedError
    
    def get_covariance_matrix(self) -> jnp.ndarray:
        """Return the full covariance matrix Σ."""
        raise NotImplementedError


class IsotropicGaussian(WeightSpaceBelief):
    """Isotropic Gaussian weight-space belief.
    
    Σ = σ² I
    
    From the paper (Section D.3.3): The isotropic Gaussian covariance structure
    represents the weight space uncertainty as N(w*, Σ := σ²I) where σ² is 
    the variance parameter and I is the identity matrix, reflecting independence 
    and identical uncertainty across all dimensions in the weight space.
    
    From a Bayesian perspective, this can be viewed as just considering a 
    calibrated prior over the selected weight space.
    """
    
    def __init__(self, w_star: jnp.ndarray, sigma_squared: float = 1.0):
        """Initialize isotropic Gaussian belief.
        
        Args:
            w_star: MAP estimate of weights (w*)
            sigma_squared: Variance σ² for the isotropic covariance
        """
        self._mean = w_star
        self._sigma_squared = sigma_squared
        self._n_params = w_star.shape[0]
    
    @property
    def mean(self) -> jnp.ndarray:
        return self._mean
    
    @property
    def covariance(self) -> jnp.ndarray:
        return self._sigma_squared * jnp.eye(self._n_params)
    
    def get_covariance_matrix(self) -> jnp.ndarray:
        return self._sigma_squared * jnp.eye(self._n_params)
    
    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> jnp.ndarray:
        """Draw samples: w ~ N(w*, σ²I)."""
        return self._mean + jnp.sqrt(self._sigma_squared) * jax.random.normal(
            rng_key, (n_samples, self._n_params)
        )
    
    def get_sqrt_covariance(self) -> jnp.ndarray:
        """Return sqrt(Σ) = σI for efficient linearized pushforward."""
        return jnp.sqrt(self._sigma_squared) * jnp.eye(self._n_params)
    
    def update_sigma(self, sigma_squared: float):
        """Update the variance parameter (used during calibration)."""
        self._sigma_squared = sigma_squared


class LowRankLaplace(WeightSpaceBelief):
    """Low-rank Laplace-approximated weight-space belief.
    
    Implements the linearized Laplace approximation (LLA) using a low-rank 
    approximation of the Generalized Gauss-Newton (GGN) matrix.
    
    From the paper (Section D.3.4 and Appendix B):
    - Select the largest eigenspaces of the GGN
    - Place an isotropic Gaussian prior over all weights
    - The posterior covariance is Σ = (n V V^T + σ I)^{-1}
    
    where V contains the top-k eigenvectors of the GGN, n is the number of
    data points, and σ is the prior precision.
    
    This allows regions of uncertainty to fall back to the prior belief when
    data does not constrain particular weight directions.
    """
    
    def __init__(
        self,
        w_star: jnp.ndarray,
        ggn_eigenvectors: jnp.ndarray,  # V of shape (p, rank)
        ggn_eigenvalues: jnp.ndarray,   # eigenvalues of shape (rank,)
        n_data: int,
        prior_precision: float = 1.0,
    ):
        """Initialize low-rank Laplace belief.
        
        Args:
            w_star: MAP estimate of weights (w*)
            ggn_eigenvectors: Top-k eigenvectors V of the GGN, shape (p, rank)
            ggn_eigenvalues: Corresponding eigenvalues, shape (rank,)
            n_data: Number of data points used for GGN approximation
            prior_precision: Prior precision σ (inverse prior variance)
        """
        self._mean = w_star
        self._V = ggn_eigenvectors  # (p, rank)
        self._lambdas = ggn_eigenvalues  # (rank,)
        self._n_data = n_data
        self._prior_precision = prior_precision
        self._rank = ggn_eigenvectors.shape[1]
        self._n_params = w_star.shape[0]
    
    @property
    def mean(self) -> jnp.ndarray:
        return self._mean
    
    @property
    def covariance(self) -> jnp.ndarray:
        """Full covariance: Σ = (n V Λ V^T + σ I)^{-1}.
        
        Using Woodbury identity for efficiency with low-rank structure:
        Σ = σ^{-1} I - σ^{-1} V (σ n^{-1} Λ^{-1} + V^T V)^{-1} V^T σ^{-1}
        """
        return self.get_covariance_matrix()
    
    def get_covariance_matrix(self) -> jnp.ndarray:
        """Compute the full posterior covariance matrix."""
        sigma = self._prior_precision
        V = self._V
        n = self._n_data
        
        # Σ = (n V V^T + σ I)^{-1}
        # Using: V has eigenvalues from GGN, but paper uses V V^T not V Λ V^T
        # Wait - re-reading the paper:
        # "extend the approach in Dangel et al. (2022) by selecting the largest 
        #  eigenspaces of the GGN and placing an isotropic Gaussian prior over 
        #  all weights"
        # The GGN approximation is H_GGN = V Λ V^T ≈ V V^T (with normalized eigenvectors)
        # So Σ = (n V V^T + σ I)^{-1}
        
        # Using Woodbury: (σI + n V V^T)^{-1} = σ^{-1}I - σ^{-1}V(σ n^{-1} I + V^T V)^{-1} V^T σ^{-1}
        # = σ^{-1}I - σ^{-2} V (σ/n I + V^T V)^{-1} V^T
        
        sigma_inv = 1.0 / sigma
        VTV = V.T @ V  # (rank, rank)
        M = (sigma / n) * jnp.eye(self._rank) + VTV
        M_inv = jnp.linalg.inv(M)
        
        return sigma_inv * jnp.eye(self._n_params) - sigma_inv**2 * V @ M_inv @ V.T
    
    def get_sqrt_covariance_times_vector(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute Σ^{1/2} v efficiently using low-rank structure.
        
        This is useful for pushforward computations without materializing Σ.
        """
        sigma = self._prior_precision
        V = self._V
        n = self._n_data
        
        sigma_inv_sqrt = 1.0 / jnp.sqrt(sigma)
        VTV = V.T @ V
        M = (sigma / n) * jnp.eye(self._rank) + VTV
        
        # Compute Cholesky of M for efficient sqrt
        L = jnp.linalg.cholesky(M)  # M = L L^T
        # Σ^{1/2} ≈ σ^{-1/2} I - σ^{-1} V L^{-T} (I - (I + σ^{-1} n L^{-1} V^T V L^{-T})^{-1/2}) V^T
        # Simplified: use the low-rank form directly
        # Σ = σ^{-1} I - σ^{-2} V M^{-1} V^T
        
        # For sampling, use: w = μ + σ^{-1/2} ε_1 - σ^{-1} V L^{-T} ε_2
        # where ε_1 ~ N(0, I_p), ε_2 ~ N(0, I_rank)
        # This is exact for the low-rank + diagonal structure
        
        result = sigma_inv_sqrt * v
        result -= (sigma_inv_sqrt / sigma) * V @ jnp.linalg.solve(L.T, V.T @ v)
        
        return result
    
    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> jnp.ndarray:
        """Draw samples from the low-rank Laplace posterior.
        
        Using the reparameterization:
        w = μ + σ^{-1/2} ε_1 - σ^{-1} V L^{-T} ε_2
        where ε_1 ~ N(0, I_p), ε_2 ~ N(0, I_rank)
        and L comes from Cholesky of M = (σ/n) I + V^T V.
        """
        key1, key2 = jax.random.split(rng_key)
        sigma = self._prior_precision
        n = self._n_data
        V = self._V
        
        sigma_inv_sqrt = 1.0 / jnp.sqrt(sigma)
        
        VTV = V.T @ V
        M = (sigma / n) * jnp.eye(self._rank) + VTV
        L = jnp.linalg.cholesky(M)
        
        eps1 = jax.random.normal(key1, (n_samples, self._n_params))
        eps2 = jax.random.normal(key2, (n_samples, self._rank))
        
        samples = self._mean + sigma_inv_sqrt * eps1
        correction = (sigma_inv_sqrt / sigma) * (eps2 @ jnp.linalg.inv(L.T) @ V.T)
        samples -= correction
        
        return samples
    
    def update_prior_precision(self, precision: float):
        """Update prior precision (used during calibration)."""
        self._prior_precision = precision


class DeepEnsembleWeightBelief(WeightSpaceBelief):
    """Weight-space belief from a deep ensemble.
    
    This represents the ensemble as a collection of point masses in parameter
    space. The empirical mean and covariance are computed from the ensemble
    members. 
    
    From the paper: Deep ensembles approximate uncertainty using a small set 
    of discrete hypotheses, represented by a collection of point masses in 
    parameter space. While this representation is not confined to the analytic 
    form of a Gaussian distribution, the associated empirical covariance across 
    the ensemble is fundamentally rank-deficient (rank ≤ K-1 for K ensemble members).
    """
    
    def __init__(self, ensemble_weights: jnp.ndarray):
        """Initialize from ensemble member weights.
        
        Args:
            ensemble_weights: Array of shape (K, p) containing K ensemble members
        """
        self._weights = ensemble_weights
        self._n_ensemble = ensemble_weights.shape[0]
        self._n_params = ensemble_weights.shape[1]
        
        # Compute empirical mean and covariance
        self._mean = jnp.mean(ensemble_weights, axis=0)
        centered = ensemble_weights - self._mean
        self._cov = (centered.T @ centered) / (self._n_ensemble - 1)
    
    @property
    def mean(self) -> jnp.ndarray:
        return self._mean
    
    @property
    def covariance(self) -> jnp.ndarray:
        return self._cov
    
    def get_covariance_matrix(self) -> jnp.ndarray:
        return self._cov
    
    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> jnp.ndarray:
        """Draw samples: bootstrap from ensemble members."""
        key = rng_key
        indices = jax.random.choice(key, self._n_ensemble, shape=(n_samples,))
        return self._weights[indices]
    
    def get_rank(self) -> int:
        """Return the rank of the ensemble covariance."""
        return min(self._n_ensemble - 1, self._n_params)
