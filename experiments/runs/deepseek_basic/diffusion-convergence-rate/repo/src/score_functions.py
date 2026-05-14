"""Score functions for various target distributions.

Implements exact score functions for:
  1. Gaussian distribution (Example 1, Appendix C.1)
  2. Gaussian Mixture Model (Example 2, Appendix C.2)

Also provides the non-uniform Lipschitz analysis from Definition 2.
"""

import numpy as np
from scipy.special import logsumexp


class GaussianScore:
    """Score function for Gaussian target distribution.

    X_0 ~ N(0, Sigma) where Sigma = diag(sigma^2_i).
    Used in Example 1 and numerical experiments (Appendix A).
    """

    def __init__(self, sigma2_diag):
        """Initialize with diagonal covariance entries.

        Args:
            sigma2_diag: array of shape (d,) with variance for each dimension.
        """
        self.sigma2 = np.asarray(sigma2_diag)
        self.d = len(self.sigma2)

    def score(self, x, bar_alpha):
        """Compute s_t^*(x) = -Sigma_t^{-1} x.

        Where Sigma_t = bar_alpha * Sigma_0 + (1 - bar_alpha) * I_d
        """
        sigma_t_diag = bar_alpha * self.sigma2 + (1 - bar_alpha)
        return -x / sigma_t_diag

    def score_unscaled(self, x, bar_alpha):
        """Compute (1 - bar_alpha) * s_t^*(x)."""
        sigma_t_diag = bar_alpha * self.sigma2 + (1 - bar_alpha)
        return -(1 - bar_alpha) * x / sigma_t_diag

    def true_lipschitz_uniform(self, bar_alpha):
        """Uniform Lipschitz constant of unscaled score.

        ||(1-bar_alpha)(s_t^*(x) - s_t^*(y))|| <= L_u ||x-y||
        L_u = (1-bar_alpha) / (bar_alpha * min(sigma2) + 1-bar_alpha)
        """
        min_s2 = np.min(self.sigma2)
        return (1 - bar_alpha) / (bar_alpha * min_s2 + 1 - bar_alpha)

    def sample_target(self, n=1, rng=None):
        """Sample from target N(0, Sigma)."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(0, np.sqrt(self.sigma2), size=(n, self.d))


class GMMScore:
    """Score function for Gaussian Mixture Model target distribution.

    X_0 ~ sum_{h=1}^H gamma_h * N(mu_h, sigma^2 I_d)

    As analyzed in Example 2 and Appendix C.2.
    """

    def __init__(self, means, weights, sigma2=0.0):
        """Initialize GMM score function.

        Args:
            means: array of shape (H, d) - component means
            weights: array of shape (H,) - mixing weights (sum to 1)
            sigma2: scalar variance for each component
        """
        self.means = np.asarray(means)
        self.weights = np.asarray(weights)
        self.sigma2 = sigma2
        self.H, self.d = self.means.shape

        # Precompute component covariances
        self.gamma = self.weights  # gamma_h

    def _sigma_t_sq(self, bar_alpha):
        """sigma_t^2 = bar_alpha * sigma^2 + 1 - bar_alpha"""
        return bar_alpha * self.sigma2 + 1 - bar_alpha

    def _log_responsibilities(self, x, bar_alpha):
        """Compute log of pi_h(x) up to normalization.

        pi_h(x) ~ gamma_h * exp(-||x - sqrt(bar_alpha)*mu_h||^2 / (2*sigma_t^2))
        """
        sigma_t_sq = self._sigma_t_sq(bar_alpha)
        sqrt_alpha = np.sqrt(bar_alpha)
        means_scaled = sqrt_alpha * self.means

        # ||x - sqrt(bar_alpha)*mu_h||^2 for all h
        diff = x[None, :] - means_scaled  # (H, d)
        sq_norms = np.sum(diff ** 2, axis=1)  # (H,)

        log_weights = np.log(self.weights + 1e-300)
        log_probs = log_weights - sq_norms / (2 * sigma_t_sq)

        return log_probs

    def _responsibilities(self, x, bar_alpha):
        """Compute pi_h(x): posterior probability of component h given x."""
        log_probs = self._log_responsibilities(x, bar_alpha)
        log_total = logsumexp(log_probs)
        return np.exp(log_probs - log_total)

    def score(self, x, bar_alpha):
        """Compute s_t^*(x) for GMM.

        s_t^*(x) = -x / sigma_t^2 + (sqrt(bar_alpha)/sigma_t^2) * sum_h pi_h(x) * mu_h

        As derived in Appendix C.2, equation after (C.2).
        """
        sigma_t_sq = self._sigma_t_sq(bar_alpha)
        pi = self._responsibilities(x, bar_alpha)
        weighted_mean = pi @ self.means  # sum_h pi_h(x) * mu_h
        sqrt_alpha = np.sqrt(bar_alpha)
        return -x / sigma_t_sq + (sqrt_alpha / sigma_t_sq) * weighted_mean

    def score_unscaled(self, x, bar_alpha):
        """Compute (1 - bar_alpha) * s_t^*(x)."""
        return (1 - bar_alpha) * self.score(x, bar_alpha)

    def jacobian(self, x, bar_alpha):
        """Compute the Jacobian matrix J_t(x) = d s_t^*(x) / dx.

        J_t(x) = (1/sigma_t^2) * [-I_d + (bar_alpha/sigma_t^2) * (sum_h pi_h mu_h mu_h^T - \bar{mu} \bar{mu}^T)]

        As derived in Appendix C.2.2.
        """
        sigma_t_sq = self._sigma_t_sq(bar_alpha)
        pi = self._responsibilities(x, bar_alpha)

        # Weighted mean
        weighted_mean = pi @ self.means  # sum_h pi_h(x) * mu_h

        # Covariance of component means under pi
        # sum_h pi_h mu_h mu_h^T - weighted_mean * weighted_mean^T
        outer_sum = np.zeros((self.d, self.d))
        for h in range(self.H):
            outer_sum += pi[h] * np.outer(self.means[h], self.means[h])

        bar_mu_outer = np.outer(weighted_mean, weighted_mean)

        J = (-np.eye(self.d) + (bar_alpha / sigma_t_sq) * (outer_sum - bar_mu_outer)) / sigma_t_sq
        return J

    def sample_target(self, n=1, rng=None):
        """Sample from target GMM."""
        if rng is None:
            rng = np.random.default_rng()
        components = rng.choice(self.H, size=n, p=self.weights)
        samples = rng.normal(0, np.sqrt(self.sigma2), size=(n, self.d))
        samples += self.means[components]
        return samples

    def non_uniform_lipschitz_upper_bound(self, bar_alpha, T, d_effective=None):
        """Compute the theoretical upper bound for L from Example 2.

        The paper shows that for GMM:
            L <= O(log(H*(T+d)))

        Args:
            bar_alpha: current bar_alpha_t
            T: total iterations
            d_effective: effective dimension (default: self.d)

        Returns:
            L_bound: logarithmic bound on non-uniform Lipschitz constant
        """
        if d_effective is None:
            d_effective = self.d
        # C_1 * log(H * (T + d)) from Equation in Example 2
        # C_1 is a universal constant (can set to ~1 for illustration)
        C_1 = 1.0
        return C_1 * np.log(self.H * (T + d_effective))

    def uniform_lipschitz_lower_bound(self, bar_alpha):
        """Compute the theoretical lower bound for uniform Lipschitz constant.

        From Example 2, for a two-component symmetric GMM:
            ||(1-bar_alpha) * grad s_t^*(x)||_op >= (1-bar_alpha)*||mu||^2 / (4*(1-bar_alpha+sigma^2)^2)

        when bar_alpha > 1/2.
        """
        if bar_alpha <= 0.5:
            return 0.0

        sigma_t_sq = self._sigma_t_sq(bar_alpha)
        mu_norm_sq = np.mean(np.sum(self.means ** 2, axis=1))
        return (1 - bar_alpha) * mu_norm_sq / (4 * sigma_t_sq ** 2)


def compute_covariance_matrix_tau(Sigma_0, tau):
    """Compute Sigma_tau = Cov[Z | sqrt(1-tau)*X_0 + sqrt(tau)*Z = x].

    For Gaussian target, this has a closed form.
    Sigma_tau = tau * (tau*I + (1-tau)*Sigma_0)^{-1} * Sigma_0

    This is used in Lemma 3, 9, and 10.
    """
    d = Sigma_0.shape[0]
    tau_I = tau * np.eye(d)
    inv_term = np.linalg.inv(tau_I + (1 - tau) * Sigma_0)
    return tau * inv_term @ Sigma_0


def compute_tr_sigma_tau_sq(Sigma_0, tau):
    """Compute E[Tr(Sigma_tau^2(x_tau))] for Gaussian target.

    This is a key quantity in the discretization error analysis (Lemma 3, Eq 35).
    """
    Sigma_tau = compute_covariance_matrix_tau(Sigma_0, tau)
    return np.trace(Sigma_tau @ Sigma_tau)
