"""
Forward process for score-based diffusion models.

Implements the discrete and continuous forward processes from Section 2.1:

Discrete (Eq. 3):
  X_t = sqrt(alpha_t) X_{t-1} + sqrt(1 - alpha_t) W_t

Continuous (Eq. 4):
  dX_tau = -1/(2(1-tau)) X_tau dtau + 1/sqrt(1-tau) dB_tau

Marginal (Eq. 6):
  X_tau = sqrt(1-tau) X_0 + sqrt(tau) Z,  Z ~ N(0, I_d)
"""

import numpy as np
from typing import Tuple, Optional
from score_functions import ScoreFunction


class ForwardProcess:
    """
    Forward diffusion process.

    Parameterization: tau = 1 - alpha_bar_t in [0, 1].
    At tau=0: X_tau = X_0 (clean data).
    At tau=1: X_tau ~ N(0, I_d) (pure noise).
    """

    def __init__(self, score_fn: ScoreFunction):
        self.score_fn = score_fn

    def sample_marginal(
        self,
        x0: np.ndarray,
        tau: float,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample X_tau | X_0 = x0 using the marginal formula (Eq. 6 in continuous time):
          X_tau = sqrt(1 - tau) X_0 + sqrt(tau) Z,  Z ~ N(0, I_d)

        Args:
            x0: shape (d,) or (batch, d), clean data sample
            tau: noise level in [0, 1]
            rng: random number generator

        Returns:
            x_tau: same shape as x0
        """
        if rng is None:
            rng = np.random.default_rng()
        z = rng.standard_normal(x0.shape)
        return np.sqrt(1.0 - tau) * x0 + np.sqrt(tau) * z

    def sample_marginal_batch(
        self,
        x0_samples: np.ndarray,
        tau: float,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample X_tau for a batch of X_0 samples.

        Args:
            x0_samples: shape (n, d)
            tau: noise level
            rng: random number generator

        Returns:
            x_tau: shape (n, d)
        """
        if rng is None:
            rng = np.random.default_rng()
        z = rng.standard_normal(x0_samples.shape)
        return np.sqrt(1.0 - tau) * x0_samples + np.sqrt(tau) * z

    def score_at_tau(self, x: np.ndarray, tau: float) -> np.ndarray:
        """
        Evaluate score function s_tau*(x) = nabla log p_{X_tau}(x).

        Args:
            x: shape (d,) or (batch, d)
            tau: noise level

        Returns:
            score: same shape as x
        """
        if x.ndim == 1:
            return self.score_fn.score(x, tau)
        return self.score_fn.score_batch(x, tau)

    def probability_flow_ode_rhs(self, x: np.ndarray, tau: float) -> np.ndarray:
        """
        Right-hand side of the probability flow ODE (Eq. 5):
          dY_tau/dtau = -1/(2(1-tau)) * (Y_tau + nabla log p_{X_tau}(Y))

        Note: this is the ODE for the reverse process (decreasing tau).
        The sign convention here gives the drift for increasing tau.

        Args:
            x: shape (d,), current state
            tau: current noise level

        Returns:
            drift: shape (d,)
        """
        s = self.score_fn.score(x, tau)
        return -1.0 / (2.0 * (1.0 - tau)) * (x + s)

    def ode_flow_map(
        self,
        x_tau1: np.ndarray,
        tau1: float,
        tau2: float,
        n_steps: int = 100,
    ) -> np.ndarray:
        """
        Integrate the probability flow ODE from tau1 to tau2 using Euler method.
        Used to compute the exact ODE map Phi_{tau1 -> tau2} (Eq. 17).

        Args:
            x_tau1: shape (d,), starting point at tau1
            tau1: starting noise level
            tau2: ending noise level (tau2 < tau1 for reverse process)
            n_steps: number of Euler steps

        Returns:
            x_tau2: shape (d,), endpoint at tau2
        """
        x = x_tau1.copy()
        tau = tau1
        dtau = (tau2 - tau1) / n_steps

        for _ in range(n_steps):
            s = self.score_fn.score(x, tau)
            # ODE: d(x/sqrt(1-tau))/dtau = -s_tau*(x) / (2(1-tau)^{3/2})
            # Equivalently: dx/dtau = -x/(2(1-tau)) - s_tau*(x)/(2(1-tau))
            drift = -1.0 / (2.0 * (1.0 - tau)) * (x + s)
            x = x + dtau * drift
            tau = tau + dtau

        return x

    def ode_flow_map_normalized(
        self,
        x_tau1: np.ndarray,
        tau1: float,
        tau2: float,
        n_steps: int = 100,
    ) -> np.ndarray:
        """
        Integrate the normalized ODE: d(x/sqrt(1-tau))/dtau = -s_tau*(x)/(2(1-tau)^{3/2})

        This is the form used in the sampler update (Eq. 10, 16).

        Args:
            x_tau1: shape (d,), starting point at tau1
            tau1: starting noise level
            tau2: ending noise level
            n_steps: number of Euler steps

        Returns:
            x_tau2: shape (d,), endpoint at tau2
        """
        # Work in normalized coordinates: y = x / sqrt(1 - tau)
        y = x_tau1 / np.sqrt(1.0 - tau1)
        tau = tau1
        dtau = (tau2 - tau1) / n_steps

        for _ in range(n_steps):
            x = y * np.sqrt(1.0 - tau)
            s = self.score_fn.score(x, tau)
            # dy/dtau = -s_tau*(x) / (2(1-tau)^{3/2})
            dy = -s / (2.0 * (1.0 - tau)**1.5)
            y = y + dtau * dy
            tau = tau + dtau

        return y * np.sqrt(1.0 - tau)


class LearningRateSchedule:
    """
    Randomized learning rate schedule from Section 2.2 (Eq. 8, 9).

    Defines the discretization points alpha_hat_t and the randomized
    tau_{k,n} ~ Unif(tau_hat_{k,n}, tau_hat_{k,n-1}).
    """

    def __init__(self, T: int, K: int, c0: float = 5.0, c1: float = 50.0):
        """
        Args:
            T: total number of score evaluations
            K: number of rounds
            c0: exponent for initial alpha_hat (alpha_hat_{T+1} = 1/T^{c0})
            c1: step-size coefficient (c1/c0 must be sufficiently large)
        """
        self.T = T
        self.K = K
        self.N = 2 * T // K  # steps per round
        self.c0 = c0
        self.c1 = c1
        self._build_schedule()

    def _build_schedule(self):
        """
        Build the deterministic schedule alpha_hat_t (Eq. 8).

        alpha_hat_{T+1} = 1/T^{c0}
        alpha_hat_{t-1} = alpha_hat_t + c1 * alpha_hat_t * (1 - alpha_hat_t) * log(T) / T
        for t = -N/2 + 1, ..., T+1
        """
        T, N, c0, c1 = self.T, self.N, self.c0, self.c1
        log_T = np.log(T)

        # Total number of steps needed: from T+1 down to T - K*N/2 - N
        # Index range: t from T+1 down to T - K*N/2 - N + 1
        # We store alpha_hat indexed by t
        n_total = T + 1 + N // 2 + 2  # extra buffer

        alpha_hat = np.zeros(n_total + 1)
        # alpha_hat[T+1] = 1/T^{c0}
        alpha_hat[T + 1] = 1.0 / (T**c0)

        # Recurse: alpha_hat[t-1] = alpha_hat[t] + c1 * alpha_hat[t] * (1 - alpha_hat[t]) * log(T) / T
        for t in range(T + 1, 0, -1):
            if t - 1 < 0:
                break
            a = alpha_hat[t]
            alpha_hat[t - 1] = a + c1 * a * (1.0 - a) * log_T / T
            alpha_hat[t - 1] = min(alpha_hat[t - 1], 1.0)

        self.alpha_hat = alpha_hat

    def tau_hat(self, k: int, n: int) -> float:
        """
        Compute tau_hat_{k,n} = 1 - alpha_hat_{T - kN/2 - n} (Eq. 9).

        Args:
            k: round index (0 <= k < K)
            n: step index within round (-1 <= n <= N)

        Returns:
            tau_hat_{k,n} in [0, 1]
        """
        idx = self.T - k * self.N // 2 - n
        idx = max(0, min(idx, len(self.alpha_hat) - 1))
        return 1.0 - self.alpha_hat[idx]

    def sample_tau(
        self,
        k: int,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Sample tau_{k,n} ~ Unif(tau_hat_{k,n}, tau_hat_{k,n-1}) (Eq. 9).

        Args:
            k: round index
            n: step index
            rng: random number generator

        Returns:
            tau_{k,n}: sampled noise level
        """
        if rng is None:
            rng = np.random.default_rng()
        lo = self.tau_hat(k, n)
        hi = self.tau_hat(k, n - 1)
        return rng.uniform(lo, hi)

    def get_round_taus(
        self,
        k: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample all tau_{k,n} for round k.

        Returns:
            tau_hat_arr: shape (N+2,), tau_hat_{k,n} for n = -1, 0, ..., N
                         tau_hat_arr[j] = tau_hat_{k, j-1}
            tau_arr: shape (N+1,), sampled tau_{k,n} for n = 0, 1, ..., N
        """
        if rng is None:
            rng = np.random.default_rng()

        N = self.N
        # tau_hat_arr[j] = tau_hat_{k, j-1} for j = 0, 1, ..., N+1
        # j=0: tau_hat_{k,-1}, j=1: tau_hat_{k,0}, ..., j=N+1: tau_hat_{k,N}
        tau_hat_arr = np.array([self.tau_hat(k, n) for n in range(-1, N + 1)])
        # tau_arr[n] = tau_{k,n} ~ Unif(tau_hat_{k,n}, tau_hat_{k,n-1}) for n = 0, ..., N
        tau_arr = np.array([self.sample_tau(k, n, rng) for n in range(0, N + 1)])
        return tau_hat_arr, tau_arr

    def alpha_bar(self, tau: float) -> float:
        """alpha_bar_t = 1 - tau."""
        return 1.0 - tau

    def noise_injection_variance(self, k: int, tau_kN: float) -> float:
        """
        Variance for noise injection step (Eq. 11):
          sigma_k^2 = (tau_{k+1,0} - tau_{k,N}) / (1 - tau_{k,N})

        Args:
            k: current round index
            tau_kN: tau_{k,N} (end of round k)

        Returns:
            sigma_k^2
        """
        tau_next_0 = self.tau_hat(k + 1, 0)
        return (tau_next_0 - tau_kN) / (1.0 - tau_kN)
