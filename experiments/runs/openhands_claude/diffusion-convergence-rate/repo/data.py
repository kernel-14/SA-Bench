"""
Data distributions for experiments.

Implements the target distributions used in the paper:
  - Gaussian (Appendix A, numerical experiments)
  - Gaussian mixture model (Example 2)
  - General data distribution interface
"""

import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod

from score_functions import (
    ScoreFunction,
    GaussianScoreFunction,
    GMMScoreFunction,
    make_gaussian_score,
    make_gmm_score,
)


class DataDistribution(ABC):
    """Abstract base class for target data distributions."""

    @abstractmethod
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample n points from the distribution."""

    @abstractmethod
    def get_score_function(self) -> ScoreFunction:
        """Return the exact score function for this distribution."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Data dimension d."""

    def second_moment(self) -> float:
        """E[||X_0||^2] (Assumption 1)."""
        raise NotImplementedError


class GaussianDistribution(DataDistribution):
    """
    Diagonal Gaussian distribution for numerical experiments (Appendix A).

    X_0 ~ N(0, diag(sigma_1^2, ..., sigma_d^2))

    First k components have sigma_i^2 ~ Unif[0, sigma_max],
    remaining d-k components have sigma_i^2 = 0 (degenerate).
    """

    def __init__(
        self,
        d: int,
        k: int,
        sigma_max: float = 10.0,
        seed: int = 42,
    ):
        """
        Args:
            d: data dimension
            k: number of non-zero variance components
            sigma_max: maximum variance value
            seed: random seed for generating variances
        """
        self.d = d
        self.k = k
        self.sigma_max = sigma_max

        rng = np.random.default_rng(seed)
        self.variances = np.zeros(d)
        self.variances[:k] = rng.uniform(0.0, sigma_max, size=k)
        self.sigmas = np.sqrt(self.variances)
        self._score_fn = GaussianScoreFunction(self.sigmas)

    @property
    def dimension(self) -> int:
        return self.d

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample n points from N(0, diag(sigma^2))."""
        if rng is None:
            rng = np.random.default_rng()
        z = rng.standard_normal((n, self.d))
        return z * self.sigmas[None, :]

    def get_score_function(self) -> GaussianScoreFunction:
        return self._score_fn

    def second_moment(self) -> float:
        """E[||X_0||^2] = sum_i sigma_i^2."""
        return float(np.sum(self.variances))

    def marginal_distribution(self, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Marginal distribution of X_tau = sqrt(1-tau) X_0 + sqrt(tau) Z.

        Returns:
            mu: shape (d,), mean (always zero)
            var: shape (d,), diagonal variances
        """
        mu = np.zeros(self.d)
        var = self._score_fn._sigma_t_sq(tau)
        return mu, var

    def kl_from_gaussian(self, mu_p: np.ndarray, var_p: np.ndarray, tau: float) -> float:
        """
        KL(q_K || p_{Y_K}) where q_K = N(0, Sigma_{tau}).

        Args:
            mu_p: estimated mean of p_{Y_K}
            var_p: estimated diagonal variance of p_{Y_K}
            tau: noise level tau_{K,0}

        Returns:
            KL divergence
        """
        from convergence_metrics import kl_divergence_gaussians_diagonal
        mu_q, var_q = self.marginal_distribution(tau)
        return kl_divergence_gaussians_diagonal(mu_q, var_q, mu_p, var_p)


class GMMDistribution(DataDistribution):
    """
    Gaussian mixture model distribution (Example 2).

    X_0 ~ sum_{h=1}^H gamma_h N(mu_h, sigma^2 I_d)
    """

    def __init__(
        self,
        d: int,
        H: int,
        sigma: float = 1.0,
        mu_scale: float = 1.0,
        weights: Optional[np.ndarray] = None,
        means: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        """
        Args:
            d: data dimension
            H: number of components
            sigma: component standard deviation
            mu_scale: scale of component means
            weights: shape (H,), mixture weights (uniform if None)
            means: shape (H, d), component means (random if None)
            seed: random seed
        """
        self.d = d
        self.H = H
        self.sigma = sigma

        rng = np.random.default_rng(seed)

        if weights is None:
            self.weights = np.ones(H) / H
        else:
            self.weights = np.asarray(weights)
            assert np.abs(self.weights.sum() - 1.0) < 1e-10

        if means is None:
            self.means = rng.standard_normal((H, d)) * mu_scale
        else:
            self.means = np.asarray(means)

        self._score_fn = GMMScoreFunction(self.weights, self.means, sigma)

    @property
    def dimension(self) -> int:
        return self.d

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample n points from the GMM."""
        if rng is None:
            rng = np.random.default_rng()

        # Sample component indices
        components = rng.choice(self.H, size=n, p=self.weights)
        samples = np.zeros((n, self.d))

        for h in range(self.H):
            mask = components == h
            n_h = mask.sum()
            if n_h > 0:
                samples[mask] = self.means[h] + self.sigma * rng.standard_normal((n_h, self.d))

        return samples

    def get_score_function(self) -> GMMScoreFunction:
        return self._score_fn

    def second_moment(self) -> float:
        """E[||X_0||^2] = sum_h gamma_h (||mu_h||^2 + d*sigma^2)."""
        return float(sum(
            self.weights[h] * (np.sum(self.means[h]**2) + self.d * self.sigma**2)
            for h in range(self.H)
        ))

    def non_uniform_lipschitz(self, T: int) -> float:
        """
        Non-uniform Lipschitz constant L from Example 2.
        L <= C1 * log(H * (T + d))
        """
        return np.log(self.H * (T + self.d))

    def uniform_lipschitz_lower_bound(self, tau: float) -> float:
        """
        Lower bound on uniform Lipschitz constant (from Example 2, lower bound).

        For X_0 ~ 1/2 N(mu, sigma^2 I) + 1/2 N(-mu, sigma^2 I):
        (1 - alpha_bar_t) ||nabla s_t*(x)||_op >= (1-alpha_bar_t) ||mu||^2 / (4(1-alpha_bar_t + sigma^2)^2)

        This can be large when sigma^2 is small.
        """
        if self.H == 2:
            mu = self.means[0]
            mu_norm_sq = np.sum(mu**2)
            sigma_t_sq = (1.0 - tau) * self.sigma**2 + tau
            return tau * mu_norm_sq / (4.0 * sigma_t_sq**2)
        return float("inf")


class TwoComponentGMM(GMMDistribution):
    """
    Two-component GMM: X_0 ~ 1/2 N(mu, sigma^2 I) + 1/2 N(-mu, sigma^2 I).

    Used in Example 2 to demonstrate the gap between uniform and non-uniform
    Lipschitz constants.
    """

    def __init__(self, d: int, mu_norm: float = 1.0, sigma: float = 0.1, seed: int = 42):
        """
        Args:
            d: data dimension
            mu_norm: ||mu||_2
            sigma: component standard deviation
            seed: random seed
        """
        rng = np.random.default_rng(seed)
        mu = np.zeros(d)
        mu[0] = mu_norm  # mu = (mu_norm, 0, ..., 0)

        super().__init__(
            d=d,
            H=2,
            sigma=sigma,
            weights=np.array([0.5, 0.5]),
            means=np.array([mu, -mu]),
            seed=seed,
        )
        self.mu = mu
        self.mu_norm = mu_norm


def make_gaussian_experiment_data(
    d: int,
    k: int,
    sigma_max: float = 10.0,
    seed: int = 42,
) -> GaussianDistribution:
    """
    Create Gaussian distribution for numerical experiments (Appendix A).

    Args:
        d: data dimension
        k: number of non-zero variance components
        sigma_max: maximum variance
        seed: random seed

    Returns:
        GaussianDistribution
    """
    return GaussianDistribution(d=d, k=k, sigma_max=sigma_max, seed=seed)


def make_gmm_experiment_data(
    d: int,
    H: int,
    sigma: float = 1.0,
    mu_scale: float = 1.0,
    seed: int = 42,
) -> GMMDistribution:
    """
    Create GMM distribution for experiments.

    Args:
        d: data dimension
        H: number of components
        sigma: component standard deviation
        mu_scale: scale of component means
        seed: random seed

    Returns:
        GMMDistribution
    """
    return GMMDistribution(d=d, H=H, sigma=sigma, mu_scale=mu_scale, seed=seed)
