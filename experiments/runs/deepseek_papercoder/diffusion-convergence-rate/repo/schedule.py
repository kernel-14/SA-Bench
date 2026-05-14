## schedule.py

"""
Implements the randomised midpoint schedule as described in Section 2.2 of the paper.
The schedule consists of:
1. A deterministic grid `hat_alpha` based on Equation (8).
2. A random uniform sampling of `alpha_bar` between consecutive `hat_alpha` values (Equation (9)).
3. Conversion to continuous‑time variables `hat_tau` and `tau` for each round `k` and substep `n`.

All algorithmic constants (c0, c1, K, T) are passed to the constructor; they may be taken from config.yaml by the caller.
"""

import numpy as np
from typing import Tuple, List

class Schedule:
    """
    Generates and stores the randomised midpoint schedule for the diffusion sampler.
    """

    def __init__(self, T: int, K: int, c0: float, c1: float, seed: int):
        """
        Initialize the schedule.

        Args:
            T: Total number of discretisation steps (must be such that 2*T/K is an integer).
            K: Number of rounds (default 10).
            c0: Exponent for initial alpha window (must satisfy c1 > 3*c0).
            c1: Factor controlling the per-step alpha change.
            seed: Seed for the random number generator (sampling alpha_bar).
        """
        if not isinstance(T, int) or T <= 0:
            raise ValueError("T must be a positive integer")
        if not isinstance(K, int) or K <= 0:
            raise ValueError("K must be a positive integer")
        if not (isinstance(c0, (int, float)) and c0 > 0):
            raise ValueError("c0 must be positive")
        if not (isinstance(c1, (int, float)) and c1 > 0):
            raise ValueError("c1 must be positive")
        if c1 <= 3 * c0:
            raise ValueError("c1 must be greater than 3*c0")
        if (2 * T) % K != 0:
            raise ValueError(f"2*T must be divisible by K, but 2*{T} is not divisible by {K}")

        self.T = T
        self.K = K
        self.c0 = c0
        self.c1 = c1
        self.N = (2 * T) // K          # number of steps per round
        self.start_idx = -(self.N // 2)  # because N is even (T multiple of 5, K=10 ensures even)
        self.offset = -self.start_idx    # to map t to array index: array_idx = t + offset

        # Set up random number generator
        self.rng = np.random.default_rng(seed)

        # Array length for hat_alpha and alpha_bar:
        # Indices range from start_idx to T+1 inclusive.
        # alpha_bar only needs from start_idx+1 to T+1, but we allocate same length for simplicity.
        self.length = T + 1 - self.start_idx + 1
        self.hat_alpha = np.empty(self.length, dtype=np.float64)
        self.alpha_bar = np.empty(self.length, dtype=np.float64)
        self.alpha_bar.fill(np.nan)   # only indices start_idx+1..T+1 will be filled

        # Generate schedule parts
        self._generate_hat_alpha()
        self._sample_alpha_bar()
        self._compute_taus()

    def _generate_hat_alpha(self) -> None:
        """
        Compute the deterministic hat_alpha grid per Equation (8).
        The recurrence is: hat_alpha_{t-1} = hat_alpha_t + c1 * hat_alpha_t * (1 - hat_alpha_t) * log(T) / T
        We compute backwards starting from t = T+1.
        """
        # The last index is T+1, which maps to position T+1 + offset.
        idx_end = T_plus_1 = self.T + 1 + self.offset
        self.hat_alpha[idx_end] = 1.0 / (self.T ** self.c0)

        log_T = np.log(self.T)
        factor = (self.c1 * log_T) / self.T

        for t in range(self.T + 1, self.start_idx + 1, -1):  # t decreases from T+1 to start_idx+1
            idx_cur = t + self.offset          # hat_alpha[t]
            idx_prev = t - 1 + self.offset     # hat_alpha[t-1]
            cur_val = self.hat_alpha[idx_cur]
            self.hat_alpha[idx_prev] = cur_val + factor * cur_val * (1.0 - cur_val)

    def _sample_alpha_bar(self) -> None:
        """
        Sample alpha_bar uniformly between consecutive hat_alpha values:
            alpha_bar_t ~ Uniform(hat_alpha_{t-1}, hat_alpha_t)
        for t = start_idx+1 to T+1.
        """
        for t in range(self.start_idx + 1, self.T + 2):  # t from start_idx+1 up to T+1
            idx_prev = t - 1 + self.offset
            idx_cur = t + self.offset
            lo, hi = sorted((self.hat_alpha[idx_prev], self.hat_alpha[idx_cur]))
            self.alpha_bar[idx_cur] = self.rng.uniform(lo, hi)

    def _compute_taus(self) -> None:
        """
        Convert hat_alpha and alpha_bar to continuous-time tau and hat_tau.
        For each round k (0 to K-1) and substep n:
            hat_tau_{k,n} = 1 - hat_alpha[T - (k*N//2) - n]
            tau_{k,n}      = 1 -  alpha_bar[T - (k*N//2) - n + 1]
        n ranges from -1 to N-1 for hat_tau, and 0 to N for tau.
        We store these as lists of arrays.
        """
        # We need tau and hat_tau up to k = K (to include tau_{K,0}). For k=K, only tau_{K,0} is needed.
        self.tau_all: List[np.ndarray] = []
        self.hat_tau_all: List[np.ndarray] = []

        half_N = self.N // 2

        for k in range(self.K + 1):   # k = 0 .. K
            # Pre-allocate arrays: hat_tau has indices n = -1,0,...,N-1 => N+1 elements (mapping j = n+1)
            # tau has indices n = 0,1,...,N => N+1 elements
            tau_k = np.empty(self.N + 1, dtype=np.float64)
            hat_tau_k = np.empty(self.N + 1, dtype=np.float64)

            # Compute hat_tau: n goes from -1 to N-1
            for n in range(-1, self.N):   # n = -1,0,...,N-1
                idx = self.T - (k * half_N) - n
                hat_tau_k[n + 1] = 1.0 - self.hat_alpha[idx + self.offset]

            # Compute tau: n goes from 0 to N
            for n in range(0, self.N + 1):  # n = 0,...,N
                idx = self.T - (k * half_N) - n + 1
                tau_k[n] = 1.0 - self.alpha_bar[idx + self.offset]

            self.hat_tau_all.append(hat_tau_k)
            self.tau_all.append(tau_k)

    def get_round_schedule(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the time arrays for round k.

        Args:
            k: round index (0 <= k < K)

        Returns:
            tau_k: array of shape (N+1,) containing tau_{k,n} for n=0..N.
            hat_tau_k: array of shape (N+1,) containing hat_tau_{k,n} for n=-1..N-1
                       (index j = n+1, so hat_tau_k[0] = hat_tau_{k,-1}, hat_tau_k[1] = hat_tau_{k,0}, etc.)
        """
        if not (0 <= k < self.K):
            raise IndexError(f"Round index k must be between 0 and {self.K-1}")
        return self.tau_all[k], self.hat_tau_all[k]
