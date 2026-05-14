## sampler.py

"""
Exact covariance propagation for the randomized midpoint diffusion sampler
applied to a Gaussian target distribution.

Because the score function is linear and the initial noise is Gaussian,
the whole sampling process is Gaussian with zero mean.  Hence it suffices to
track the marginal variances (diagonal covariance).  The module provides a
`Sampler` class that, given the data covariance `sigma_diag` and a `Schedule`,
returns the variance vector of the final sample after K rounds.
"""

import numpy as np
from typing import Tuple
from schedule import Schedule   # assumed to be in the same package


class Sampler:
    """
    Exact analytic sampler for linear score functions (Gaussian target).
    """

    def __init__(self, sigma_diag: np.ndarray, schedule: Schedule):
        """
        Initialize the sampler.

        Args:
            sigma_diag: 1D numpy array of length d containing the marginal
                variances of the target Gaussian distribution.
            schedule: A `Schedule` instance that provides the time discretisation
                (tau and hat_tau) for each round.
        """
        if sigma_diag.ndim != 1:
            raise ValueError("sigma_diag must be a 1D array")
        if np.any(sigma_diag < 0):
            raise ValueError("sigma_diag must contain non‑negative entries")

        self.sigma_diag = sigma_diag.astype(np.float64)
        self.d = len(sigma_diag)
        self.schedule = schedule

    def _score_matrix(self, tau: float) -> np.ndarray:
        """
        Return the vector of diagonal entries of the inverse covariance matrix
        that defines the score function at noise level tau.

        Specifically, if the forward marginal covariance is
            (1 - tau) * diag(sigma_diag) + tau * I_d,
        then the score matrix is -((1-tau)*Sigma + tau*I)^{-1}, which is diagonal.
        This method returns the positive vector
            d_tau[j] = 1 / ((1 - tau) * sigma_diag[j] + tau).

        Args:
            tau: noise level in (0, 1).

        Returns:
            1D array of length d containing the diagonal entries.
        """
        return 1.0 / ((1.0 - tau) * self.sigma_diag + tau)

    def propagate_covariance(self) -> np.ndarray:
        """
        Iterate over all K rounds and return the variance vector of the final
        sample Y_K (zero mean).

        Returns:
            1D numpy array of length d containing the marginal variances.
        """
        v = np.ones(self.d, dtype=np.float64)  # Y_0 ~ N(0, I_d)

        K = self.schedule.K
        for k in range(K):
            tau_k, hat_tau_k = self.schedule.get_round_schedule(k)
            # tau_{k+1,0} is needed for the noise injection
            if k + 1 < K:
                tau_next_0 = self.schedule.tau_all[k + 1][0]
            else:
                # For the last round we use tau_{K,0} (also stored in schedule)
                tau_next_0 = self.schedule.tau_all[K][0]

            v = self._run_round(v, tau_k, hat_tau_k, tau_next_0)

        return v

    def _run_round(
        self,
        v0: np.ndarray,
        tau: np.ndarray,
        hat_tau: np.ndarray,
        tau_next_0: float,
    ) -> np.ndarray:
        """
        Propagate the variance vector through a single round consisting of N
        deterministic steps (Eq. (16) of the paper) followed by Gaussian noise
        injection.

        Args:
            v0: variance vector of Y_{k,0} (length d).
            tau: 1D array of shape (N+1,) containing tau_{k,n} for n=0..N.
            hat_tau: 1D array of shape (N+1,) containing hat_tau_{k,n}
                     for n=-1..N-1, stored at index j = n+1.
            tau_next_0: scalar tau_{k+1,0}, used for the noise injection step.

        Returns:
            Variance vector of Y_{k+1,0} (length d).
        """
        N = len(tau) - 1  # number of substeps in the round
        d = self.d

        # Precompute the score vectors d_i for i = 0 ... N-1
        # (only needed for i that appear in the update; we can compute lazily)
        d_vectors = np.empty((N, d), dtype=np.float64)
        for i in range(N):
            d_vectors[i] = self._score_matrix(tau[i])

        # L_n will hold the linear coefficient vector such that
        # Y_{k,n} = diag(L_n) * Y_{k,0}.
        L = [np.ones(d, dtype=np.float64)]  # L_0

        for n in range(1, N + 1):
            tau_n = tau[n]
            # coefficient for i = 0
            term0 = np.sqrt((1.0 - tau_n) / (1.0 - tau[0]))
            coef0 = -np.sqrt(1.0 - tau_n) * (tau[0] - hat_tau[1]) / (2.0 * (1.0 - tau[0]) ** 1.5)
            Ln = term0 + coef0 * d_vectors[0]

            # contributions for i = 1 .. n-2 (if any)
            for i in range(1, n - 1):
                # interval: hat_tau_{i-1} - hat_tau_i
                delta_hat = hat_tau[i] - hat_tau[i + 1]
                coef = -np.sqrt(1.0 - tau_n) * delta_hat / (2.0 * (1.0 - tau[i]) ** 1.5)
                Ln += coef * d_vectors[i] * L[i]

            # contribution for i = n-1 (the last substep)
            if n >= 2:
                delta_last = hat_tau[n - 1] - tau_n   # = hat_tau_{n-1} - tau_n
                coef_last = -np.sqrt(1.0 - tau_n) * delta_last / (2.0 * (1.0 - tau[n - 1]) ** 1.5)
                Ln += coef_last * d_vectors[n - 1] * L[n - 1]
            else:  # n == 1, the "last" is also the i=0 term, already included.
                # Actually for n=1, there are no i in 1..n-2, and the last term is the same as i=0?
                # The formula includes only i=0. So do nothing extra.
                pass

            L.append(Ln)

        # variance after N steps (before noise injection)
        vN = (L[N] ** 2) * v0

        # noise injection step (Eq. (15) of the paper)
        tauN = tau[N]
        scale = (1.0 - tau_next_0) / (1.0 - tauN)
        noise_scale = (tau_next_0 - tauN) / (1.0 - tauN)
        v_new = scale * vN + noise_scale * np.ones(d, dtype=np.float64)

        return v_new

