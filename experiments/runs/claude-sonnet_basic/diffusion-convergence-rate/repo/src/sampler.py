"""
Randomized midpoint sampler for score-based diffusion models.

Implements the sampling algorithm from:
"Instance-dependent Convergence Theory for Diffusion Models"
by Yuchen Jiao and Gen Li (2025).

The sampler is based on the probability flow ODE discretization with
a randomized learning rate schedule, as described in Section 2.2.
"""

import numpy as np
from typing import Callable, Optional, Tuple


def compute_alpha_hat_schedule(T: int, c0: float = 5.0, c1: float = 10.0) -> np.ndarray:
    """
    Compute the deterministic schedule hat_alpha_t as defined in equation (9).

    The schedule is defined recursively:
        hat_alpha_{T+1} = 1 / T^{c0}
        hat_alpha_{t-1} = hat_alpha_t + c1 * hat_alpha_t * (1 - hat_alpha_t) * log(T) / T

    Args:
        T: Total number of iterations.
        c0: Constant controlling the initial value (sufficiently large).
        c1: Constant controlling the step size (c1/c0 sufficiently large).

    Returns:
        Array of hat_alpha values indexed from 0 to T+1.
    """
    log_T = np.log(T)
    n_steps = T + 2
    hat_alpha = np.zeros(n_steps + 1)
    hat_alpha[T + 1] = 1.0 / (T ** c0)

    for t in range(T + 1, 0, -1):
        a = hat_alpha[t]
        hat_alpha[t - 1] = a + c1 * a * (1.0 - a) * log_T / T
        hat_alpha[t - 1] = min(hat_alpha[t - 1], 1.0)

    return hat_alpha


class RandomizedMidpointSampler:
    """
    Implements the randomized midpoint sampler from Section 2.2.

    The sampler operates over K rounds, each consisting of N steps.
    In each round, it discretizes the probability flow ODE using a
    randomized learning rate schedule.

    Algorithm:
    1. Initialize Y_0 ~ N(0, I_d)
    2. For k = 0, ..., K-1:
       a. Compute Y_{k,n} for n=1,...,N using the ODE discretization (eq. 10)
       b. Inject noise to get Y_{k+1} (eq. 11)
    3. Return Y_K
    """

    def __init__(self,
                 score_fn: Callable,
                 d: int,
                 T: int,
                 K: int,
                 c0: float = 5.0,
                 c1: float = 10.0,
                 rng: Optional[np.random.Generator] = None):
        """
        Args:
            score_fn: Score function estimate. Signature: score_fn(tau, x) -> np.ndarray
                      where tau is the continuous time in [0,1] and x is the current point.
            d: Data dimension.
            T: Total number of iterations (T = KN/2).
            K: Number of rounds.
            c0: Schedule constant (sufficiently large, default 5).
            c1: Schedule constant (c1/c0 sufficiently large, default 10).
            rng: Random number generator.
        """
        self.score_fn = score_fn
        self.d = d
        self.T = T
        self.K = K
        self.N = 2 * T // K  # Steps per round
        self.c0 = c0
        self.c1 = c1
        self.rng = rng if rng is not None else np.random.default_rng()

        # Precompute the deterministic schedule
        self.hat_alpha = compute_alpha_hat_schedule(T, c0, c1)

    def _get_hat_alpha(self, idx: int) -> float:
        """Get hat_alpha at index idx, clamped to [0, 1]."""
        if idx < 0:
            return 1.0
        if idx >= len(self.hat_alpha):
            return 0.0
        return float(self.hat_alpha[idx])

    def _get_hat_tau(self, k: int, n: int) -> float:
        """
        Compute hat_tau_{k,n} = 1 - hat_alpha_{T - kN/2 - n}.
        """
        idx = self.T - k * self.N // 2 - n
        return 1.0 - self._get_hat_alpha(idx)

    def _sample_tau(self, k: int, n: int) -> float:
        """
        Sample tau_{k,n} ~ Unif(hat_tau_{k,n}, hat_tau_{k,n-1}).
        """
        tau_lo = self._get_hat_tau(k, n)
        tau_hi = self._get_hat_tau(k, n - 1)
        if tau_hi <= tau_lo:
            return tau_lo
        return float(self.rng.uniform(tau_lo, tau_hi))

    def sample_one_round(self, y_k: np.ndarray, k: int) -> Tuple[np.ndarray, list]:
        """
        Execute one round of the sampler.

        Implements equation (10) from the paper. The update rule is:
            Y_{k,n} / sqrt(1 - tau_{k,n}) = Y_{k,0} / sqrt(1 - tau_{k,0})
                + s(Y_{k,0}) / (2*(1-tau_{k,0})^{3/2}) * (tau_{k,0} - hat_tau_{k,0})
                + sum_{i=1}^{n-1} s(Y_{k,i}) / (2*(1-tau_{k,i})^{3/2}) * (hat_tau_{k,i-1} - hat_tau_{k,i})
                + s(Y_{k,n-1}) / (2*(1-tau_{k,n-1})^{3/2}) * (hat_tau_{k,n-1} - tau_{k,n})

        Args:
            y_k: Current sample Y_k (shape: [d]).
            k: Current round index.

        Returns:
            y_kN: Final sample Y_{k,N} after N steps.
            taus: List of sampled tau values.
        """
        N = self.N

        # Sample all tau values for this round
        taus = [self._sample_tau(k, n) for n in range(N + 1)]

        tau_k0 = taus[0]
        hat_tau_k0 = self._get_hat_tau(k, 0)

        # Initialize: y_{k,0} = y_k
        y_kn = y_k.copy()

        # Compute score at initial point
        s_k0 = self.score_fn(tau_k0, y_kn)

        # Accumulate the ODE integral in normalized coordinates
        # integral = y_{k,n} / sqrt(1 - tau_{k,n})
        sqrt_1_minus_tau_k0 = np.sqrt(max(1.0 - tau_k0, 1e-10))
        integral = y_kn / sqrt_1_minus_tau_k0

        # Add first term: s(Y_{k,0}) / (2*(1-tau_{k,0})^{3/2}) * (tau_{k,0} - hat_tau_{k,0})
        coeff_0 = (tau_k0 - hat_tau_k0) / (2.0 * sqrt_1_minus_tau_k0 ** 3)
        integral = integral + s_k0 * coeff_0

        # Store scores for accumulation
        s_values = [s_k0.copy()]

        for n in range(1, N + 1):
            tau_kn = taus[n]
            hat_tau_kn_minus1 = self._get_hat_tau(k, n - 1)
            hat_tau_kn = self._get_hat_tau(k, n)
            tau_kn_minus1 = taus[n - 1]

            # Add last term: s(Y_{k,n-1}) / (2*(1-tau_{k,n-1})^{3/2}) * (hat_tau_{k,n-1} - tau_{k,n})
            sqrt_1_minus_tau_kn_minus1 = np.sqrt(max(1.0 - tau_kn_minus1, 1e-10))
            coeff_last = (hat_tau_kn_minus1 - tau_kn) / (2.0 * sqrt_1_minus_tau_kn_minus1 ** 3)
            integral_n = integral + s_values[-1] * coeff_last

            # Recover y_{k,n} = integral_n * sqrt(1 - tau_{k,n})
            sqrt_1_minus_tau_kn = np.sqrt(max(1.0 - tau_kn, 1e-10))
            y_kn_new = integral_n * sqrt_1_minus_tau_kn

            # Compute score at new point
            s_kn = self.score_fn(tau_kn, y_kn_new)

            # Update integral for next step: add middle term
            if n < N:
                coeff_mid = (hat_tau_kn_minus1 - hat_tau_kn) / (2.0 * sqrt_1_minus_tau_kn ** 3)
                integral = integral_n + s_kn * coeff_mid
            else:
                integral = integral_n

            s_values.append(s_kn.copy())
            y_kn = y_kn_new

        return y_kn, taus

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Generate samples using the randomized midpoint sampler.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Samples of shape [n_samples, d].
        """
        samples = []
        for _ in range(n_samples):
            # Step 1: Initialize Y_0 ~ N(0, I_d)
            y = self.rng.standard_normal(self.d)

            # Step 2: Iterate over K rounds
            for k in range(self.K):
                # Run one round of ODE discretization
                y_kN, taus = self.sample_one_round(y, k)

                # Step 3: Noise injection (equation 11)
                # Y_{k+1} = sqrt((1-tau_{k+1,0})/(1-tau_{k,N})) * Y_{k,N}
                #           + sqrt((tau_{k+1,0} - tau_{k,N})/(1-tau_{k,N})) * Z_k
                tau_k1_0 = self._get_hat_tau(k + 1, 0)
                tau_kN = self._get_hat_tau(k, self.N)

                denom = max(1.0 - tau_kN, 1e-10)
                if tau_k1_0 > tau_kN:
                    scale_y = np.sqrt((1.0 - tau_k1_0) / denom)
                    scale_z = np.sqrt((tau_k1_0 - tau_kN) / denom)
                    z_k = self.rng.standard_normal(self.d)
                    y = scale_y * y_kN + scale_z * z_k
                else:
                    y = y_kN

            samples.append(y)

        return np.array(samples)
