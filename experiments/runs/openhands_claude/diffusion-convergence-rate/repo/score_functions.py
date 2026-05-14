"""
Exact score functions for target distributions used in the paper.

Implements:
  - Gaussian score (Example 1, Appendix C.1)
  - Gaussian mixture model (GMM) score (Example 2, Appendix C.2)
  - Abstract base class for score function interface

Score function definition (Definition 1):
  s_t*(x) = nabla log p_{X_t}(x)
           = -1/(1 - alpha_bar_t) * E_{x0 | x_t}[x_t - sqrt(alpha_bar_t) x_0]

Continuous-time version (tau = 1 - alpha_bar_t):
  s_tau*(x) = -1/tau * E_{x0 | x_tau}[x_tau - sqrt(1-tau) x_0]
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class ScoreFunction(ABC):
    """Abstract base class for score functions."""

    @abstractmethod
    def score(self, x: np.ndarray, tau: float) -> np.ndarray:
        """
        Compute score s_tau*(x) = nabla log p_{X_tau}(x).

        Args:
            x: shape (d,) or (batch, d)
            tau: noise level in [0, 1], tau = 1 - alpha_bar_t

        Returns:
            score: same shape as x
        """

    def score_batch(self, x: np.ndarray, tau: float) -> np.ndarray:
        """Batch score computation. Default: vectorize over first axis."""
        if x.ndim == 1:
            return self.score(x, tau)
        return np.stack([self.score(x[i], tau) for i in range(x.shape[0])])

    def jacobian(self, x: np.ndarray, tau: float) -> np.ndarray:
        """
        Compute Jacobian J_tau(x) = d s_tau*(x) / dx.

        Args:
            x: shape (d,)
            tau: noise level

        Returns:
            J: shape (d, d)
        """
        d = x.shape[0]
        eps = 1e-5
        J = np.zeros((d, d))
        for j in range(d):
            e = np.zeros(d)
            e[j] = eps
            J[:, j] = (self.score(x + e, tau) - self.score(x - e, tau)) / (2 * eps)
        return J


class GaussianScoreFunction(ScoreFunction):
    """
    Exact score for Gaussian target distribution (Example 1, Appendix C.1).

    X_0^(i) ~ N(0, sigma_i^2), i = 1, ..., d

    Forward process: X_t = sqrt(alpha_bar_t) X_0 + sqrt(1 - alpha_bar_t) W
    Marginal: X_t ~ N(0, Sigma_t) where (Sigma_t)_{ii} = alpha_bar_t * sigma_i^2 + (1 - alpha_bar_t)

    Score: s_t*(x) = -Sigma_t^{-1} x  (Eq. C.1)

    In continuous time (tau = 1 - alpha_bar_t):
      (Sigma_tau)_{ii} = (1 - tau) * sigma_i^2 + tau
      s_tau*(x) = -Sigma_tau^{-1} x
    """

    def __init__(self, sigmas: np.ndarray):
        """
        Args:
            sigmas: shape (d,), standard deviations of each component
        """
        self.sigmas = np.asarray(sigmas, dtype=np.float64)
        self.d = len(sigmas)

    def _sigma_t_sq(self, tau: float) -> np.ndarray:
        """
        Diagonal entries of Sigma_tau.
        (Sigma_tau)_{ii} = (1 - tau) * sigma_i^2 + tau
        """
        return (1.0 - tau) * self.sigmas**2 + tau

    def score(self, x: np.ndarray, tau: float) -> np.ndarray:
        """s_tau*(x) = -Sigma_tau^{-1} x"""
        sigma_t_sq = self._sigma_t_sq(tau)
        return -x / sigma_t_sq

    def score_batch(self, x: np.ndarray, tau: float) -> np.ndarray:
        sigma_t_sq = self._sigma_t_sq(tau)
        return -x / sigma_t_sq

    def jacobian(self, x: np.ndarray, tau: float) -> np.ndarray:
        """J_tau(x) = -Sigma_tau^{-1} (constant in x for Gaussian)."""
        sigma_t_sq = self._sigma_t_sq(tau)
        return -np.diag(1.0 / sigma_t_sq)

    def covariance_matrix(self, tau: float) -> np.ndarray:
        """Sigma_tau: marginal covariance at noise level tau."""
        return np.diag(self._sigma_t_sq(tau))

    def mean_vector(self, tau: float) -> np.ndarray:
        """Mean of X_tau (always zero for zero-mean Gaussian)."""
        return np.zeros(self.d)

    def lipschitz_constant_uniform(self, tau: float) -> float:
        """
        Uniform Lipschitz constant of s_tau* (Remark 1).
        ||s_tau*(x) - s_tau*(y)||_2 <= L_unif * ||x - y||_2
        L_unif = ||Sigma_tau^{-1}||_op = 1 / min_i (Sigma_tau)_{ii}
        """
        sigma_t_sq = self._sigma_t_sq(tau)
        return 1.0 / np.min(sigma_t_sq)

    def lipschitz_constant_normalized(self, tau: float) -> float:
        """
        Uniform Lipschitz constant of (1 - alpha_bar_t) s_t* = tau * s_tau*.
        From Example 1: tau * ||s_tau*(x) - s_tau*(y)||_2 <= ||x - y||_2
        So L_normalized = 1.
        """
        return 1.0


class GMMScoreFunction(ScoreFunction):
    """
    Exact score for Gaussian mixture model (Example 2, Appendix C.2).

    X_0 ~ sum_{h=1}^H gamma_h N(mu_h, sigma^2 I_d)

    Forward process marginal:
      p_{X_tau}(x) = sum_h gamma_h N(x; sqrt(1-tau) mu_h, sigma_tau^2 I_d)
      sigma_tau^2 = (1-tau) sigma^2 + tau

    Score (Eq. C.2):
      s_tau*(x) = -x/sigma_tau^2 + sqrt(1-tau)/sigma_tau^2 * sum_h pi_h(x) mu_h

    where pi_h(x) = gamma_h exp(-||x - sqrt(1-tau) mu_h||^2 / (2 sigma_tau^2))
                    / sum_i gamma_i exp(-||x - sqrt(1-tau) mu_i||^2 / (2 sigma_tau^2))
    """

    def __init__(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        sigma: float,
    ):
        """
        Args:
            weights: shape (H,), mixture weights gamma_h >= 0, sum = 1
            means: shape (H, d), component means mu_h
            sigma: scalar, component standard deviation
        """
        self.weights = np.asarray(weights, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.sigma = float(sigma)
        self.H, self.d = means.shape
        assert np.abs(self.weights.sum() - 1.0) < 1e-10, "Weights must sum to 1"

    def _sigma_tau_sq(self, tau: float) -> float:
        """sigma_tau^2 = (1-tau) sigma^2 + tau"""
        return (1.0 - tau) * self.sigma**2 + tau

    def _mixing_weights(self, x: np.ndarray, tau: float) -> np.ndarray:
        """
        Compute posterior mixing weights pi_h(x).

        pi_h(x) = gamma_h * phi(x; sqrt(1-tau) mu_h, sigma_tau^2 I)
                  / sum_i gamma_i * phi(x; sqrt(1-tau) mu_i, sigma_tau^2 I)
        """
        sigma_tau_sq = self._sigma_tau_sq(tau)
        sqrt_one_minus_tau = np.sqrt(1.0 - tau)
        scaled_means = sqrt_one_minus_tau * self.means  # (H, d)

        # Log unnormalized weights for numerical stability
        diff = x[None, :] - scaled_means  # (H, d)
        log_unnorm = -0.5 * np.sum(diff**2, axis=1) / sigma_tau_sq  # (H,)
        log_unnorm += np.log(self.weights + 1e-300)

        # Softmax for numerical stability
        log_unnorm -= log_unnorm.max()
        unnorm = np.exp(log_unnorm)
        return unnorm / unnorm.sum()

    def score(self, x: np.ndarray, tau: float) -> np.ndarray:
        """
        s_tau*(x) = -x/sigma_tau^2 + sqrt(1-tau)/sigma_tau^2 * sum_h pi_h(x) mu_h
        """
        sigma_tau_sq = self._sigma_tau_sq(tau)
        pi = self._mixing_weights(x, tau)  # (H,)
        weighted_mean = pi @ self.means    # (d,)
        return -x / sigma_tau_sq + np.sqrt(1.0 - tau) / sigma_tau_sq * weighted_mean

    def score_batch(self, x: np.ndarray, tau: float) -> np.ndarray:
        """Vectorized score computation for batch of points."""
        if x.ndim == 1:
            return self.score(x, tau)

        sigma_tau_sq = self._sigma_tau_sq(tau)
        sqrt_one_minus_tau = np.sqrt(1.0 - tau)
        scaled_means = sqrt_one_minus_tau * self.means  # (H, d)

        # x: (batch, d), scaled_means: (H, d)
        diff = x[:, None, :] - scaled_means[None, :, :]  # (batch, H, d)
        log_unnorm = -0.5 * np.sum(diff**2, axis=2) / sigma_tau_sq  # (batch, H)
        log_unnorm += np.log(self.weights[None, :] + 1e-300)
        log_unnorm -= log_unnorm.max(axis=1, keepdims=True)
        unnorm = np.exp(log_unnorm)
        pi = unnorm / unnorm.sum(axis=1, keepdims=True)  # (batch, H)

        weighted_mean = pi @ self.means  # (batch, d)
        return -x / sigma_tau_sq + sqrt_one_minus_tau / sigma_tau_sq * weighted_mean

    def non_uniform_lipschitz_constant(self, tau: float, T: int, C1: float = 1.0) -> float:
        """
        Non-uniform Lipschitz constant L from Definition 2 (Example 2).

        For GMM: L <= C1 * log(H * (T + d))  (Eq. in Example 2)
        This scales only logarithmically with H, T, d.
        """
        return C1 * np.log(self.H * (T + self.d))

    def jacobian_score(self, x: np.ndarray, tau: float) -> np.ndarray:
        """
        Jacobian of score function J_tau(x) = d s_tau*(x) / dx.

        From Appendix C.2.2:
        J_tau(x) = 1/sigma_tau^2 * [-I_d + alpha_bar_t/sigma_tau^2 * (sum_h gamma_h mu_h mu_h^T - mu_bar mu_bar^T)]

        where mu_bar = sum_h gamma_h mu_h, alpha_bar_t = 1 - tau.
        """
        sigma_tau_sq = self._sigma_tau_sq(tau)
        alpha_bar = 1.0 - tau

        mu_bar = self.weights @ self.means  # (d,)
        second_moment = sum(
            self.weights[h] * np.outer(self.means[h], self.means[h])
            for h in range(self.H)
        )  # (d, d)

        J = (1.0 / sigma_tau_sq) * (
            -np.eye(self.d)
            + (alpha_bar / sigma_tau_sq) * (second_moment - np.outer(mu_bar, mu_bar))
        )
        return J


class LearnedScoreFunction(ScoreFunction):
    """
    Wrapper for a learned score function (neural network approximation).

    Used in Assumption 2: access to estimate s_t(.) of s_t*(.).
    """

    def __init__(self, true_score: ScoreFunction, noise_level: float = 0.0):
        """
        Args:
            true_score: the true score function
            noise_level: epsilon_score, standard deviation of additive noise
        """
        self.true_score = true_score
        self.noise_level = noise_level
        self.rng = np.random.default_rng(0)

    def score(self, x: np.ndarray, tau: float) -> np.ndarray:
        s_true = self.true_score.score(x, tau)
        if self.noise_level > 0:
            noise = self.rng.standard_normal(x.shape) * self.noise_level
            return s_true + noise
        return s_true

    def score_batch(self, x: np.ndarray, tau: float) -> np.ndarray:
        s_true = self.true_score.score_batch(x, tau)
        if self.noise_level > 0:
            noise = self.rng.standard_normal(x.shape) * self.noise_level
            return s_true + noise
        return s_true


def make_gaussian_score(d: int, k: int, sigma_max: float = 10.0, seed: int = 42) -> GaussianScoreFunction:
    """
    Create Gaussian score function for numerical experiments (Appendix A).

    Target: d-dimensional Gaussian, zero mean, diagonal covariance.
    First k diagonal entries ~ Unif[0, sigma_max], rest = 0.

    Args:
        d: data dimension
        k: number of non-zero variance components
        sigma_max: maximum variance value
        seed: random seed

    Returns:
        GaussianScoreFunction with appropriate sigmas
    """
    rng = np.random.default_rng(seed)
    sigmas = np.zeros(d)
    sigmas[:k] = np.sqrt(rng.uniform(0.0, sigma_max, size=k))
    return GaussianScoreFunction(sigmas)


def make_gmm_score(
    d: int,
    H: int,
    sigma: float = 1.0,
    mu_scale: float = 1.0,
    seed: int = 42,
) -> GMMScoreFunction:
    """
    Create GMM score function.

    Args:
        d: data dimension
        H: number of components
        sigma: component standard deviation
        mu_scale: scale of component means
        seed: random seed

    Returns:
        GMMScoreFunction
    """
    rng = np.random.default_rng(seed)
    weights = np.ones(H) / H
    means = rng.standard_normal((H, d)) * mu_scale
    return GMMScoreFunction(weights, means, sigma)
