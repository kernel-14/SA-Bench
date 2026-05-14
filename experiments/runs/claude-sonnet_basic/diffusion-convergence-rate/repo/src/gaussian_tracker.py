"""
Gaussian distribution tracker for the randomized midpoint sampler.

For a Gaussian target distribution, the sampler output Y_{k,n} remains
Gaussian at each step. This module tracks the mean and covariance of
Y_{k,n} analytically, enabling exact computation of KL divergence.

This is used in the numerical experiments (Appendix A) to validate
the theoretical convergence rate O(log^4(T) / T^3) in KL divergence.
"""

import numpy as np
from typing import Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from sampler import compute_alpha_hat_schedule


class GaussianSamplerTracker:
    """
    Tracks the Gaussian distribution of Y_{k,n} analytically.

    For a Gaussian target X_0 ~ N(0, Sigma) with Sigma = diag(sigma_sq),
    the forward process gives X_tau ~ N(0, Sigma_tau) where
    (Sigma_tau)_{ii} = (1-tau) * sigma_sq_i + tau.

    The sampler output Y_{k,n} is also Gaussian (since the score function
    is linear in x for Gaussian targets). We track its mean and covariance.

    The probability flow ODE for Gaussian targets:
        d(x_tau / sqrt(1-tau)) = -s_tau^*(x_tau) / (2*(1-tau)^{3/2}) d_tau
                                = x_tau / (2*(1-tau)^{3/2} * Sigma_tau) d_tau

    For the normalized variable u_tau = x_tau / sqrt(1-tau):
        du_tau = -s_tau^*(sqrt(1-tau) * u_tau) / (2*(1-tau)^{3/2}) d_tau
               = u_tau / (2*(1-tau) * Sigma_tau) d_tau

    This is a linear ODE, so the solution is Gaussian.
    """

    def __init__(self, sigma_sq: np.ndarray, T: int, K: int,
                 c0: float = 5.0, c1: float = 10.0,
                 rng: Optional[np.random.Generator] = None):
        """
        Args:
            sigma_sq: Diagonal variances of the target distribution, shape [d].
            T: Total number of iterations.
            K: Number of rounds.
            c0, c1: Schedule constants.
            rng: Random number generator.
        """
        self.sigma_sq = sigma_sq
        self.d = len(sigma_sq)
        self.T = T
        self.K = K
        self.N = 2 * T // K
        self.c0 = c0
        self.c1 = c1
        self.rng = rng if rng is not None else np.random.default_rng()

        # Precompute the deterministic schedule
        self.hat_alpha = compute_alpha_hat_schedule(T, c0, c1)

    def _get_hat_alpha(self, idx: int) -> float:
        if idx < 0:
            return 1.0
        if idx >= len(self.hat_alpha):
            return 0.0
        return float(self.hat_alpha[idx])

    def _get_hat_tau(self, k: int, n: int) -> float:
        idx = self.T - k * self.N // 2 - n
        return 1.0 - self._get_hat_alpha(idx)

    def _sample_tau(self, k: int, n: int) -> float:
        tau_lo = self._get_hat_tau(k, n)
        tau_hi = self._get_hat_tau(k, n - 1)
        if tau_hi <= tau_lo:
            return tau_lo
        return float(self.rng.uniform(tau_lo, tau_hi))

    def _sigma_tau(self, tau: float) -> np.ndarray:
        """Marginal variance of X_tau: (1-tau)*sigma_sq + tau."""
        return (1.0 - tau) * self.sigma_sq + tau

    def _ode_step_gaussian(self, mu: np.ndarray, cov_diag: np.ndarray,
                            tau_start: float, tau_end: float,
                            tau_score: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate a Gaussian distribution through one ODE step.

        The ODE step uses the score at tau_score to advance from tau_start to tau_end.
        For Gaussian targets, the score is linear: s_tau(x) = -x / Sigma_tau.

        The update in normalized coordinates u = x / sqrt(1-tau):
            u_{tau_end} = u_{tau_start} + s_{tau_score}(x_{tau_score}) / (2*(1-tau_score)^{3/2})
                          * (tau_score - tau_end)  [or appropriate interval]

        For Gaussian distributions, this is a linear transformation.

        Args:
            mu: Mean of current distribution, shape [d].
            cov_diag: Diagonal covariance of current distribution, shape [d].
            tau_start: Starting time.
            tau_end: Ending time.
            tau_score: Time at which score is evaluated.

        Returns:
            New mean and diagonal covariance.
        """
        # Score at tau_score: s_{tau_score}(x) = -x / Sigma_{tau_score}
        # For x ~ N(mu_x, Sigma_x), s_{tau_score}(x) ~ N(-mu_x/Sigma_{tau_score}, Sigma_x/Sigma_{tau_score}^2)

        sigma_score = self._sigma_tau(tau_score)
        sqrt_1_minus_tau_score = np.sqrt(max(1.0 - tau_score, 1e-10))

        # The ODE update in normalized coordinates:
        # u_{end} = u_{start} + s(x_{score}) / (2*(1-tau_score)^{3/2}) * delta_tau
        # where x_{score} = sqrt(1-tau_score) * u_{score}
        # and s(x) = -x / sigma_score

        # For the sampler, the score is evaluated at the current y_{k,n-1}
        # which has distribution N(mu, cov_diag) in x-space
        # In normalized space: u = x / sqrt(1-tau_start)
        # u ~ N(mu/sqrt(1-tau_start), cov_diag/(1-tau_start))

        # The update coefficient
        delta_tau = tau_start - tau_end  # positive since tau decreases
        coeff = delta_tau / (2.0 * sqrt_1_minus_tau_score ** 3)

        # Score contribution: s(x) = -x / sigma_score
        # In normalized space: s(sqrt(1-tau_score)*u) = -sqrt(1-tau_score)*u / sigma_score
        # Contribution to u: coeff * s(x) = -coeff * x / sigma_score
        #                                  = -coeff * sqrt(1-tau_score) * u / sigma_score

        # But x here is the score point (at tau_score), not tau_start
        # For the sampler, we use x at tau_start (y_{k,n-1}) to evaluate score
        # This is an approximation; the exact ODE would use x at tau_score

        # Linear coefficient for the update
        # u_end = u_start + coeff * s(x_start)
        #       = u_start - coeff * x_start / sigma_score
        #       = u_start - coeff * sqrt(1-tau_start) * u_start / sigma_score
        #       = u_start * (1 - coeff * sqrt(1-tau_start) / sigma_score)

        # In x-space:
        # x_end = sqrt(1-tau_end) * u_end
        #       = sqrt(1-tau_end) * u_start * (1 - coeff * sqrt(1-tau_start) / sigma_score)
        #       = x_start * sqrt(1-tau_end)/sqrt(1-tau_start) * (1 - coeff * sqrt(1-tau_start) / sigma_score)

        sqrt_ratio = np.sqrt(max(1.0 - tau_end, 1e-10) / max(1.0 - tau_start, 1e-10))
        linear_coeff = sqrt_ratio * (1.0 - coeff * np.sqrt(max(1.0 - tau_start, 1e-10)) / sigma_score)

        new_mu = linear_coeff * mu
        new_cov_diag = linear_coeff ** 2 * cov_diag

        return new_mu, new_cov_diag

    def compute_kl_divergence(self, T_val: int, n_trials: int = 100) -> float:
        """
        Compute the KL divergence between Y_K and q_K = X_{tau_{K,0}} analytically.

        For Gaussian targets, Y_K is Gaussian and q_K = X_{tau_{K,0}} is Gaussian.
        We track the distribution of Y_K through the sampler and compute KL divergence.

        Since the sampler uses randomized tau values, we average over multiple trials.

        Args:
            T_val: Number of iterations T.
            n_trials: Number of Monte Carlo trials for averaging over tau randomness.

        Returns:
            Average KL divergence.
        """
        kl_values = []

        for _ in range(n_trials):
            # Initialize Y_0 ~ N(0, I_d)
            mu = np.zeros(self.d)
            cov_diag = np.ones(self.d)

            for k in range(self.K):
                # Sample tau values for this round
                taus = [self._sample_tau(k, n) for n in range(self.N + 1)]

                tau_k0 = taus[0]
                hat_tau_k0 = self._get_hat_tau(k, 0)

                # Normalize: u = x / sqrt(1-tau_k0)
                sqrt_1_minus_tau_k0 = np.sqrt(max(1.0 - tau_k0, 1e-10))
                mu_u = mu / sqrt_1_minus_tau_k0
                cov_u = cov_diag / (1.0 - tau_k0)

                # First step: add score contribution at tau_k0
                # s(x) = -x / sigma_{tau_k0}
                sigma_k0 = self._sigma_tau(tau_k0)
                coeff_0 = (tau_k0 - hat_tau_k0) / (2.0 * sqrt_1_minus_tau_k0 ** 3)
                # u_new = u + coeff_0 * s(x) = u - coeff_0 * x / sigma_k0
                #       = u - coeff_0 * sqrt(1-tau_k0) * u / sigma_k0
                #       = u * (1 - coeff_0 * sqrt(1-tau_k0) / sigma_k0)
                factor_0 = 1.0 - coeff_0 * sqrt_1_minus_tau_k0 / sigma_k0
                mu_u = factor_0 * mu_u
                cov_u = factor_0 ** 2 * cov_u

                # Iterate through N steps
                for n in range(1, self.N + 1):
                    tau_kn = taus[n]
                    hat_tau_kn_minus1 = self._get_hat_tau(k, n - 1)
                    hat_tau_kn = self._get_hat_tau(k, n)
                    tau_kn_minus1 = taus[n - 1]

                    # Current x distribution: x = sqrt(1-tau_kn_minus1) * u
                    sqrt_1_minus_tau_kn_minus1 = np.sqrt(max(1.0 - tau_kn_minus1, 1e-10))
                    mu_x = sqrt_1_minus_tau_kn_minus1 * mu_u
                    cov_x = (1.0 - tau_kn_minus1) * cov_u

                    # Score at tau_kn_minus1: s(x) = -x / sigma_{tau_kn_minus1}
                    sigma_kn_minus1 = self._sigma_tau(tau_kn_minus1)
                    coeff_last = (hat_tau_kn_minus1 - tau_kn) / (2.0 * sqrt_1_minus_tau_kn_minus1 ** 3)

                    # u_new = u + coeff_last * s(x) = u - coeff_last * x / sigma
                    #       = u - coeff_last * sqrt(1-tau_kn_minus1) * u / sigma
                    factor_last = 1.0 - coeff_last * sqrt_1_minus_tau_kn_minus1 / sigma_kn_minus1
                    mu_u_n = factor_last * mu_u
                    cov_u_n = factor_last ** 2 * cov_u

                    # Convert to x-space at tau_kn
                    sqrt_1_minus_tau_kn = np.sqrt(max(1.0 - tau_kn, 1e-10))
                    mu_x_n = sqrt_1_minus_tau_kn * mu_u_n
                    cov_x_n = (1.0 - tau_kn) * cov_u_n

                    # Update u for next step (add middle term)
                    if n < self.N:
                        sigma_kn = self._sigma_tau(tau_kn)
                        coeff_mid = (hat_tau_kn_minus1 - hat_tau_kn) / (2.0 * sqrt_1_minus_tau_kn ** 3)
                        factor_mid = 1.0 - coeff_mid * sqrt_1_minus_tau_kn / sigma_kn
                        mu_u = factor_mid * mu_u_n
                        cov_u = factor_mid ** 2 * cov_u_n
                    else:
                        mu_u = mu_u_n
                        cov_u = cov_u_n

                # After N steps, we have Y_{k,N} ~ N(mu_x_n, diag(cov_x_n))
                mu = mu_x_n
                cov_diag = cov_x_n

                # Noise injection: Y_{k+1} = scale_y * Y_{k,N} + scale_z * Z_k
                tau_k1_0 = self._get_hat_tau(k + 1, 0)
                tau_kN = self._get_hat_tau(k, self.N)

                denom = max(1.0 - tau_kN, 1e-10)
                if tau_k1_0 > tau_kN:
                    scale_y = np.sqrt((1.0 - tau_k1_0) / denom)
                    scale_z_sq = (tau_k1_0 - tau_kN) / denom
                    mu = scale_y * mu
                    cov_diag = scale_y ** 2 * cov_diag + scale_z_sq

            # Compute KL divergence between Y_K ~ N(mu, diag(cov_diag))
            # and q_K = X_{tau_{K,0}} ~ N(0, Sigma_{tau_{K,0}})
            tau_K0 = self._get_hat_tau(self.K, 0)
            sigma_K0 = self._sigma_tau(tau_K0)

            # KL(N(mu, diag(cov)) || N(0, diag(sigma_K0)))
            # = 0.5 * [sum(cov/sigma_K0) + sum(mu^2/sigma_K0) - d + sum(log(sigma_K0/cov))]
            kl = 0.5 * (np.sum(cov_diag / sigma_K0) + np.sum(mu ** 2 / sigma_K0)
                        - self.d + np.sum(np.log(sigma_K0) - np.log(np.maximum(cov_diag, 1e-300))))
            kl_values.append(max(kl, 0.0))

        return float(np.mean(kl_values))
