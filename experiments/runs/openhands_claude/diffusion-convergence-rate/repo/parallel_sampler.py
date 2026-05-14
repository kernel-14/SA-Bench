"""
Parallel randomized midpoint sampler (Section 3.3, Appendix E).

Implements the parallel version of the sampler where N processors
run M iterations per round, achieving the same accuracy with
O(min{L, d} * log^2(Ld/epsilon)) parallel rounds.

Theorem 2: To achieve TV(q_K, p_{Y_K}) <= epsilon, it suffices to choose:
  N >= (min{d^{2/3} L^{-2/3}, d^{1/3}} + 1) * log^{5/3}(T) / epsilon^{2/3}
  MK >= min{d*log(T), L} * log^2(T)
  epsilon_score^2 <= epsilon^2 / log(T)
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from score_functions import ScoreFunction
from forward_process import LearningRateSchedule
from sampler import SamplerResult


@dataclass
class ParallelSamplerResult:
    """Result of running the parallel sampler."""
    samples: np.ndarray       # final samples Y_K, shape (d,) or (n_samples, d)
    score_evals: int          # total score evaluations
    parallel_rounds: int      # total parallel rounds MK
    n_processors: int         # number of parallel processors N
    tau_K0: float             # tau_{K,0}


class ParallelRandomizedMidpointSampler:
    """
    Parallel implementation of the randomized midpoint sampler (Appendix E.1).

    In each round k, N processors update Y_{m,k,n} simultaneously for M iterations.

    Update rule (Eq. E.2):
      Y_{m,k,n}/sqrt(1-tau_{k,n}) = Y_k/sqrt(1-tau_{k,0})
        + s_{tau_{k,0}}(Y_k) / (2(1-tau_{k,0})^{3/2}) * (tau_{k,0} - tau_hat_{k,0})
        + sum_{i=1}^{n-1} s_{tau_{k,i}}(Y_{m-1,k,i}) / (2(1-tau_{k,i})^{3/2}) * (tau_hat_{k,i-1} - tau_hat_{k,i})
        + s_{tau_{k,n-1}}(Y_{m-1,k,n-1}) / (2(1-tau_{k,n-1})^{3/2}) * (tau_hat_{k,n-1} - tau_{k,n})

    Convergence: after M >= log(T) iterations, the error is O(1/poly(T)).
    """

    def __init__(
        self,
        score_fn: ScoreFunction,
        N_parallel: int,
        M: int,
        K: int,
        c0: float = 5.0,
        c1: float = 50.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            score_fn: score function estimate
            N_parallel: number of parallel processors
            M: number of parallel iterations per round (M << N_parallel)
            K: number of rounds
            c0: schedule parameter
            c1: schedule parameter
            seed: random seed
        """
        self.score_fn = score_fn
        self.N = N_parallel
        self.M = M
        self.K = K
        self.T = K * N_parallel // 2  # total iterations T = KN/2
        self.schedule = LearningRateSchedule(self.T, K, c0, c1)
        self.rng = np.random.default_rng(seed)

    def _parallel_round(
        self,
        Y_k0: np.ndarray,
        k: int,
        tau_hat_arr: np.ndarray,
        tau_arr: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Execute one parallel round with M iterations of N processors.

        Each iteration m uses Y_{m-1,k,i} from the previous iteration.
        After M iterations, the error is O((N*min{d,L}*log(T)/T)^M) = O(1/poly(T)).

        Args:
            Y_k0: shape (d,), Y_{k,0}
            k: round index
            tau_hat_arr: shape (N+2,), tau_hat_{k,n} for n=-1,...,N
            tau_arr: shape (N+1,), sampled tau_{k,n} for n=0,...,N

        Returns:
            Y_M_kN: shape (d,), Y_{M,k,N} after M iterations
            score_evals: number of score evaluations
        """
        N, M = self.N, self.M
        d = Y_k0.shape[0]
        score_evals = 0

        tau_k0 = tau_arr[0]

        # Evaluate score at initial point (shared across all processors)
        s0 = self.score_fn.score(Y_k0, tau_k0)
        score_evals += 1

        # Initialize: Y_{0,k,n} = Y_k for all n (Eq. E.1)
        Y_m = np.tile(Y_k0, (N + 1, 1))  # shape (N+1, d)

        # M parallel iterations
        for m in range(1, M + 1):
            Y_m_new = np.zeros((N + 1, d))
            Y_m_new[0] = Y_k0

            # Precompute scores for all n from previous iteration
            scores_prev = np.zeros((N + 1, d))
            scores_prev[0] = s0
            for n in range(1, N + 1):
                scores_prev[n] = self.score_fn.score(Y_m[n], tau_arr[n])
                score_evals += 1

            # Step n=1
            tau_k1 = tau_arr[1]
            y_norm = Y_k0 / np.sqrt(1.0 - tau_k0)
            y_norm = y_norm + s0 / (2.0 * (1.0 - tau_k0)**1.5) * (tau_k0 - tau_k1)
            Y_m_new[1] = y_norm * np.sqrt(1.0 - tau_k1)

            # Steps n=2,...,N using incremental update with previous iteration's scores
            for n in range(2, N + 1):
                tau_kn = tau_arr[n]
                tau_hat_kn_minus2 = tau_hat_arr[n - 1]  # tau_hat_{k,n-2}
                tau_prev_prev = tau_arr[n - 2]
                tau_prev = tau_arr[n - 1]

                s_nm2 = scores_prev[n - 2]
                s_nm1 = scores_prev[n - 1]

                correction = s_nm2 / (2.0 * (1.0 - tau_prev_prev)**1.5) * (tau_prev - tau_hat_kn_minus2)
                new_last = s_nm1 / (2.0 * (1.0 - tau_prev)**1.5) * (tau_hat_kn_minus2 - tau_kn)

                y_norm = y_norm + correction + new_last
                Y_m_new[n] = y_norm * np.sqrt(1.0 - tau_kn)

            Y_m = Y_m_new

        return Y_m[N], score_evals

    def _parallel_round_efficient(
        self,
        Y_k0: np.ndarray,
        k: int,
        tau_hat_arr: np.ndarray,
        tau_arr: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Efficient parallel round: all N processors update simultaneously.

        In practice, processors share score evaluations from previous iteration.
        This is the intended parallel implementation where each processor n
        uses outputs from all processors i < n from the previous iteration.

        Args:
            Y_k0: shape (d,), Y_{k,0}
            k: round index
            tau_hat_arr: shape (N+2,), tau_hat values
            tau_arr: shape (N+1,), sampled tau values

        Returns:
            Y_M_kN: shape (d,), final output
            score_evals: number of score evaluations (sequential depth)
        """
        N, M = self.N, self.M
        d = Y_k0.shape[0]
        score_evals = 0

        tau_k0 = tau_arr[0]
        tau_hat_k0 = tau_hat_arr[1]

        # Evaluate score at initial point
        s0 = self.score_fn.score(Y_k0, tau_k0)
        score_evals += 1

        base_norm = Y_k0 / np.sqrt(1.0 - tau_k0)
        base_norm = base_norm + s0 / (2.0 * (1.0 - tau_k0)**1.5) * (tau_k0 - tau_hat_k0)

        # Initialize all processors with Y_k
        Y_prev_iter = np.tile(Y_k0, (N + 1, 1))  # shape (N+1, d)

        for m in range(M):
            Y_curr_iter = np.zeros((N + 1, d))
            Y_curr_iter[0] = Y_k0

            # Precompute scores for all processors from previous iteration
            scores_prev = np.zeros((N + 1, d))
            for n in range(1, N + 1):
                tau_n = tau_arr[n - 1] if n > 0 else tau_k0
                scores_prev[n] = self.score_fn.score(Y_prev_iter[n], tau_arr[n - 1] if n > 0 else tau_k0)
                score_evals += 1

            # All N processors update in parallel
            for n in range(1, N + 1):
                tau_kn = tau_arr[n]
                tau_hat_kn_minus1 = tau_hat_arr[n]

                running_sum = np.zeros(d)
                for i in range(1, n):
                    tau_hat_i_minus1 = tau_hat_arr[i]
                    tau_hat_i = tau_hat_arr[i + 1]
                    s_i = self.score_fn.score(Y_prev_iter[i], tau_arr[i])
                    running_sum += s_i / (2.0 * (1.0 - tau_arr[i])**1.5) * (tau_hat_i_minus1 - tau_hat_i)

                tau_prev = tau_arr[n - 1]
                tau_hat_prev = tau_hat_arr[n]
                s_prev = self.score_fn.score(Y_prev_iter[n - 1], tau_prev)
                last_term = s_prev / (2.0 * (1.0 - tau_prev)**1.5) * (tau_hat_prev - tau_kn)

                y_kn_norm = base_norm + running_sum + last_term
                Y_curr_iter[n] = y_kn_norm * np.sqrt(1.0 - tau_kn)

            Y_prev_iter = Y_curr_iter

        return Y_prev_iter[N], score_evals

    def _noise_injection(
        self,
        Y_kN: np.ndarray,
        k: int,
        tau_kN: float,
    ) -> Tuple[np.ndarray, float]:
        """Noise injection step (same as sequential sampler, Eq. E.3)."""
        tau_k1_0 = self.schedule.tau_hat(k + 1, 0)
        scale_signal = np.sqrt((1.0 - tau_k1_0) / (1.0 - tau_kN))
        scale_noise = np.sqrt((tau_k1_0 - tau_kN) / (1.0 - tau_kN))
        Z_k = self.rng.standard_normal(Y_kN.shape)
        Y_k1 = scale_signal * Y_kN + scale_noise * Z_k
        return Y_k1, tau_k1_0

    def sample(self, d: int, n_samples: int = 1) -> ParallelSamplerResult:
        """
        Run the parallel sampling procedure.

        Args:
            d: data dimension
            n_samples: number of independent samples

        Returns:
            ParallelSamplerResult
        """
        all_samples = []
        total_score_evals = 0
        tau_K0 = self.schedule.tau_hat(self.K, 0)

        for _ in range(n_samples):
            Y_k = self.rng.standard_normal(d)
            score_evals = 0

            for k in range(self.K):
                tau_hat_arr, tau_arr = self.schedule.get_round_taus(k, self.rng)
                Y_kN, evals = self._parallel_round(Y_k, k, tau_hat_arr, tau_arr)
                score_evals += evals

                tau_kN = tau_arr[self.N]
                if k < self.K - 1:
                    Y_k, _ = self._noise_injection(Y_kN, k, tau_kN)
                else:
                    Y_k = Y_kN

            all_samples.append(Y_k)
            total_score_evals = score_evals

        samples = np.stack(all_samples) if n_samples > 1 else all_samples[0]

        return ParallelSamplerResult(
            samples=samples,
            score_evals=total_score_evals,
            parallel_rounds=self.M * self.K,
            n_processors=self.N,
            tau_K0=tau_K0,
        )


def compute_parallel_requirements(
    d: int,
    L: float,
    epsilon: float,
    T: Optional[int] = None,
) -> Dict:
    """
    Compute parallel sampler requirements from Theorem 2.

    N >= (min{d^{2/3} L^{-2/3}, d^{1/3}} + 1) * log^{5/3}(T) / epsilon^{2/3}
    MK >= min{d*log(T), L} * log^2(T)
    epsilon_score^2 <= epsilon^2 / log(T)

    Args:
        d: data dimension
        L: non-uniform Lipschitz constant
        epsilon: target TV distance
        T: total iterations (if None, computed from Theorem 1)

    Returns:
        dict with N, MK, epsilon_score requirements
    """
    if T is None:
        from sampler import compute_required_T
        T = compute_required_T(d, L, epsilon)

    log_T = max(np.log(T), 1.0)

    # Number of parallel processors
    if L == float("inf"):
        N_factor = d**(1/3)
    else:
        N_factor = min(d**(2/3) * L**(-2/3), d**(1/3))
    N = int(np.ceil((N_factor + 1) * log_T**(5/3) / epsilon**(2/3)))

    # Number of parallel rounds
    if L == float("inf"):
        MK = int(np.ceil(d * log_T * log_T**2))
    else:
        MK = int(np.ceil(min(d * log_T, L) * log_T**2))

    # Score estimation error requirement
    eps_score_sq = epsilon**2 / log_T

    return {
        "N": N,
        "MK": MK,
        "epsilon_score_sq": eps_score_sq,
        "T": T,
    }
