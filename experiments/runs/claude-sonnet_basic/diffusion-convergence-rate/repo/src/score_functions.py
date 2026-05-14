"""
Score functions for various target distributions.

Implements exact score functions for:
1. Gaussian distributions (Example 1 in the paper)
2. Gaussian Mixture Models (Example 2 in the paper)

These are used in the numerical experiments (Appendix A) to validate
the theoretical convergence rate.
"""

import numpy as np
from typing import Optional


class GaussianScoreFunction:
    """
    Exact score function for a d-dimensional Gaussian target distribution.

    For target X_0 ~ N(0, Sigma) where Sigma = diag(sigma_1^2, ..., sigma_d^2),
    the forward process gives:
        X_t = sqrt(alpha_bar_t) * X_0 + sqrt(1 - alpha_bar_t) * W

    The marginal distribution is:
        X_t ~ N(0, Sigma_t)
    where Sigma_t = diag(alpha_bar_t * sigma_i^2 + (1 - alpha_bar_t))

    The score function is:
        s_t^*(x) = -Sigma_t^{-1} x

    In continuous time with tau = 1 - alpha_bar_t:
        X_tau ~ N(0, Sigma_tau)
    where (Sigma_tau)_{ii} = (1-tau) * sigma_i^2 + tau

    The score function is:
        s_tau^*(x) = -Sigma_tau^{-1} x
    """

    def __init__(self, sigma_sq: np.ndarray):
        """
        Args:
            sigma_sq: Diagonal variances of the target distribution, shape [d].
                      These are sigma_i^2 for i=1,...,d.
        """
        self.sigma_sq = sigma_sq
        self.d = len(sigma_sq)

    def __call__(self, tau: float, x: np.ndarray) -> np.ndarray:
        """
        Compute the score function s_tau^*(x) = -Sigma_tau^{-1} x.

        Args:
            tau: Continuous time in [0, 1].
            x: Current point, shape [d].

        Returns:
            Score vector, shape [d].
        """
        # Sigma_tau = diag((1-tau)*sigma_i^2 + tau)
        sigma_tau_sq = (1.0 - tau) * self.sigma_sq + tau
        return -x / sigma_tau_sq

    def get_marginal_covariance(self, tau: float) -> np.ndarray:
        """
        Get the covariance matrix of X_tau.

        Args:
            tau: Continuous time in [0, 1].

        Returns:
            Diagonal covariance, shape [d].
        """
        return (1.0 - tau) * self.sigma_sq + tau

    def kl_divergence_from_gaussian(self, mu: np.ndarray, cov_diag: np.ndarray,
                                     tau: float) -> float:
        """
        Compute KL(N(mu, diag(cov_diag)) || N(0, Sigma_tau)).

        This is used to compute the KL divergence between the sampler output
        and the target distribution at time tau.

        KL(p || q) = 0.5 * [tr(Sigma_q^{-1} Sigma_p) + mu^T Sigma_q^{-1} mu
                            - d + log(det(Sigma_q)/det(Sigma_p))]

        Args:
            mu: Mean of the first distribution, shape [d].
            cov_diag: Diagonal covariance of the first distribution, shape [d].
            tau: Time parameter for the target distribution.

        Returns:
            KL divergence value.
        """
        sigma_tau_sq = self.get_marginal_covariance(tau)

        # tr(Sigma_q^{-1} Sigma_p)
        trace_term = np.sum(cov_diag / sigma_tau_sq)

        # mu^T Sigma_q^{-1} mu
        quad_term = np.sum(mu ** 2 / sigma_tau_sq)

        # log(det(Sigma_q)/det(Sigma_p))
        log_det_term = np.sum(np.log(sigma_tau_sq) - np.log(cov_diag))

        return 0.5 * (trace_term + quad_term - self.d + log_det_term)


class GMMScoreFunction:
    """
    Exact score function for a Gaussian Mixture Model (GMM) target distribution.

    For target X_0 ~ sum_h gamma_h * N(mu_h, sigma^2 * I_d),
    the forward process gives:
        X_t | X_0 = x_0 ~ N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I_d)

    The marginal distribution is:
        X_t ~ sum_h gamma_h * N(sqrt(alpha_bar_t) * mu_h, sigma_t^2 * I_d)
    where sigma_t^2 = alpha_bar_t * sigma^2 + (1 - alpha_bar_t)

    The score function is (from Appendix C.2):
        s_t^*(x) = -x/sigma_t^2 + sqrt(alpha_bar_t)/sigma_t^2 * sum_h pi_h(x) * mu_h
    where pi_h(x) are the posterior weights.
    """

    def __init__(self, means: np.ndarray, weights: np.ndarray, sigma: float = 1.0):
        """
        Args:
            means: Component means, shape [H, d].
            weights: Component weights (must sum to 1), shape [H].
            sigma: Common standard deviation of each component.
        """
        self.means = means
        self.weights = weights
        self.sigma = sigma
        self.H, self.d = means.shape

    def _get_sigma_t_sq(self, tau: float) -> float:
        """
        Compute sigma_t^2 = (1-tau) * sigma^2 + tau.
        """
        return (1.0 - tau) * self.sigma ** 2 + tau

    def _get_posterior_weights(self, x: np.ndarray, tau: float) -> np.ndarray:
        """
        Compute posterior weights pi_h(x) for each component.

        pi_h(x) = gamma_h * exp(-||x - sqrt(1-tau)*mu_h||^2 / (2*sigma_t^2))
                  / sum_i gamma_i * exp(-||x - sqrt(1-tau)*mu_i||^2 / (2*sigma_t^2))

        Args:
            x: Current point, shape [d].
            tau: Continuous time in [0, 1].

        Returns:
            Posterior weights, shape [H].
        """
        sigma_t_sq = self._get_sigma_t_sq(tau)
        sqrt_1_minus_tau = np.sqrt(max(1.0 - tau, 1e-10))

        # Compute log-weights for numerical stability
        log_weights = np.zeros(self.H)
        for h in range(self.H):
            diff = x - sqrt_1_minus_tau * self.means[h]
            log_weights[h] = np.log(max(self.weights[h], 1e-300)) - np.sum(diff ** 2) / (2.0 * sigma_t_sq)

        # Softmax for numerical stability
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        return weights

    def __call__(self, tau: float, x: np.ndarray) -> np.ndarray:
        """
        Compute the score function s_tau^*(x).

        From Appendix C.2:
            s_t^*(x) = -x/sigma_t^2 + sqrt(alpha_bar_t)/sigma_t^2 * sum_h pi_h(x) * mu_h

        Args:
            tau: Continuous time in [0, 1].
            x: Current point, shape [d].

        Returns:
            Score vector, shape [d].
        """
        sigma_t_sq = self._get_sigma_t_sq(tau)
        sqrt_1_minus_tau = np.sqrt(max(1.0 - tau, 1e-10))

        pi = self._get_posterior_weights(x, tau)
        weighted_mean = np.sum(pi[:, np.newaxis] * self.means, axis=0)

        return -x / sigma_t_sq + sqrt_1_minus_tau / sigma_t_sq * weighted_mean
