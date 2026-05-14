"""
Randomized midpoint sampler for score-based diffusion models.

Implements the sampling algorithm from Section 2.2 (Algorithm 1):
  - Randomized learning rate schedule (Eq. 8, 9)
  - Iterative ODE discretization update (Eq. 10)
  - Noise injection between rounds (Eq. 11)

This is the core algorithm whose convergence is analyzed in Theorem 1.

Convergence guarantee (Theorem 1):
  TV(q_K, p_{Y_K}) <= C * min{d^{3/2}, d*L^{1/2}, d^{1/2}*L^{3/2}} * log^4(T) / T^{3/2}
                     + C * epsilon_score * log^{1/2}(T)

Iteration complexity to achieve TV <= epsilon:
  T >= min{d, d^{2/3}*L^{1/3}, d^{1/3}*L} * epsilon^{-2/3} * log^{8/3}(T)
"""

import numpy as np
from typing import Optional, Callable, List, Tuple, Dict
from dataclasses import dataclass

from score_functions import ScoreFunction
from forward_process import LearningRateSchedule


@dataclass
class SamplerState:
    """State of the sampler at a given round."""
    k: int                    # current round index
    Y_k: np.ndarray           # current sample Y_k, shape (d,)
    tau_k0: float             # tau_{k,0}
    score_evals: int = 0      # total score evaluations so far


@dataclass
class SamplerResult:
    """Result of running the sampler."""
    samples: np.ndarray       # final samples Y_K, shape (n_samples, d) or (d,)
    score_evals: int          # total score evaluations
    tau_K0: float             # tau_{K,0} (starting point of forward process)
    intermediate_states: Optional[List[np.ndarray]] = None


class RandomizedMidpointSampler:
    """
    Randomized midpoint sampler based on probability flow ODE discretization.

    Algorithm (Section 2.2):
    1. Initialize Y_0 ~ N(0, I_d)
    2. For k = 0, ..., K-1:
       a. Compute Y_{k,n} for n = 1, ..., N using ODE discretization (Eq. 10)
       b. Inject noise: Y_{k+1} = sqrt((1-tau_{k+1,0})/(1-tau_{k,N})) Y_{k,N}
                                  + sqrt((tau_{k+1,0}-tau_{k,N})/(1-tau_{k,N})) Z_k
    3. Return Y_K

    The key feature is the randomized tau_{k,n} ~ Unif(tau_hat_{k,n}, tau_hat_{k,n-1}),
    which enables the unbiased estimation of the ODE integral (Eq. 57 in proof).
    """

    def __init__(
        self,
        score_fn: ScoreFunction,
        T: int,
        K: int,
        c0: float = 5.0,
        c1: float = 50.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            score_fn: score function estimate s_t(.) (Assumption 2)
            T: total number of score evaluations
            K: number of rounds (K = c2 * min{d*log^2(T), L*log(T)})
            c0: schedule parameter (alpha_hat_{T+1} = 1/T^{c0})
            c1: schedule parameter (step-size coefficient)
            seed: random seed
        """
        self.score_fn = score_fn
        self.T = T
        self.K = K
        self.N = 2 * T // K  # steps per round
        self.schedule = LearningRateSchedule(T, K, c0, c1)
        self.rng = np.random.default_rng(seed)

    def _compute_round_from_scratch(
        self,
        Y_k0: np.ndarray,
        k: int,
        tau_hat_arr: np.ndarray,
        tau_arr: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute Y_{k,N} using the incremental update (Eq. 10).

        Indexing convention:
          tau_hat_arr[j] = tau_hat_{k, j-1}  for j = 0, 1, ..., N+1
            => tau_hat_arr[0] = tau_hat_{k,-1}, tau_hat_arr[1] = tau_hat_{k,0}, ...
          tau_arr[n] = tau_{k,n}  for n = 0, 1, ..., N

        The update rule (Eq. 10) in normalized coordinates y = Y/sqrt(1-tau):

          y_{k,n} = Y_{k,0}/sqrt(1-tau_{k,0})
                  + s0/(2(1-tau_{k,0})^{3/2}) * (tau_{k,0} - tau_hat_{k,0})
                  + sum_{i=1}^{n-1} s_i/(2(1-tau_{k,i})^{3/2}) * (tau_hat_{k,i-1} - tau_hat_{k,i})
                  + s_{n-1}/(2(1-tau_{k,n-1})^{3/2}) * (tau_hat_{k,n-1} - tau_{k,n})

        Incremental update from step n-1 to step n (n >= 2):
          y_{k,n} = y_{k,n-1}
                  + s_{n-2}/(2(1-tau_{k,n-2})^{3/2}) * (tau_{k,n-1} - tau_hat_{k,n-1})  [correction]
                  + s_{n-1}/(2(1-tau_{k,n-1})^{3/2}) * (tau_hat_{k,n-1} - tau_{k,n})    [new last term]

        At each step n, only one new score evaluation s_{n-1} = s(Y_{k,n-1}) is needed.
        Total score evaluations per round: N (one per step).
        """
        N = self.N
        score_evals = 0

        tau_k0 = tau_arr[0]
        tau_hat_k0 = tau_hat_arr[1]   # tau_hat_{k,0}

        # Evaluate score at initial point Y_{k,0}
        s_prev_prev = None  # s_{k,n-2} (not needed at n=1)
        s_prev = self.score_fn.score(Y_k0, tau_k0)  # s_{k,0}
        score_evals += 1

        # Step n=1: y_{k,1} = Y_{k,0}/sqrt(1-tau_{k,0})
        #                    + s0/(2(1-tau_{k,0})^{3/2}) * (tau_{k,0} - tau_{k,1})
        # (first term + last term with tau_hat_{k,0} - tau_{k,1} + tau_{k,0} - tau_hat_{k,0})
        tau_k1 = tau_arr[1]
        y_norm = Y_k0 / np.sqrt(1.0 - tau_k0)
        y_norm = y_norm + s_prev / (2.0 * (1.0 - tau_k0)**1.5) * (tau_k0 - tau_k1)

        Y_prev_prev = Y_k0          # Y_{k,n-2}
        tau_prev_prev = tau_k0      # tau_{k,n-2}
        Y_prev = y_norm * np.sqrt(1.0 - tau_k1)  # Y_{k,1}
        tau_prev = tau_k1           # tau_{k,n-1}

        if N == 1:
            return Y_prev, score_evals

        # Steps n=2,...,N: incremental update
        for n in range(2, N + 1):
            tau_kn = tau_arr[n]
            # tau_hat_{k,n-2} = tau_hat_arr[n-1]  (used in both correction and new last term)
            tau_hat_kn_minus2 = tau_hat_arr[n - 1]

            # Evaluate score at Y_{k,n-1}
            s_curr = self.score_fn.score(Y_prev, tau_prev)
            score_evals += 1

            # Incremental update (derived from Eq. 10):
            # y_{k,n} = y_{k,n-1}
            #   + s_{n-2}/(2(1-tau_{k,n-2})^{3/2}) * (tau_{k,n-1} - tau_hat_{k,n-2})  [correction]
            #   + s_{n-1}/(2(1-tau_{k,n-1})^{3/2}) * (tau_hat_{k,n-2} - tau_{k,n})    [new last term]
            correction = s_prev / (2.0 * (1.0 - tau_prev_prev)**1.5) * (tau_prev - tau_hat_kn_minus2)
            new_last = s_curr / (2.0 * (1.0 - tau_prev)**1.5) * (tau_hat_kn_minus2 - tau_kn)

            y_norm = y_norm + correction + new_last
            Y_kn = y_norm * np.sqrt(1.0 - tau_kn)

            # Shift state
            Y_prev_prev = Y_prev
            tau_prev_prev = tau_prev
            s_prev = s_curr
            Y_prev = Y_kn
            tau_prev = tau_kn

        return Y_prev, score_evals

    def _noise_injection(
        self,
        Y_kN: np.ndarray,
        k: int,
        tau_kN: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Noise injection step (Eq. 11):
          Y_{k+1} = sqrt((1-tau_{k+1,0})/(1-tau_{k,N})) * Y_{k,N}
                  + sqrt((tau_{k+1,0} - tau_{k,N})/(1-tau_{k,N})) * Z_k

        Args:
            Y_kN: shape (d,), Y_{k,N}
            k: current round index
            tau_kN: tau_{k,N}

        Returns:
            Y_k1: shape (d,), Y_{k+1}
            tau_k1_0: tau_{k+1,0}
        """
        tau_k1_0 = self.schedule.tau_hat(k + 1, 0)

        scale_signal = np.sqrt((1.0 - tau_k1_0) / (1.0 - tau_kN))
        scale_noise = np.sqrt((tau_k1_0 - tau_kN) / (1.0 - tau_kN))

        Z_k = self.rng.standard_normal(Y_kN.shape)
        Y_k1 = scale_signal * Y_kN + scale_noise * Z_k

        return Y_k1, tau_k1_0

    def sample(
        self,
        d: int,
        n_samples: int = 1,
        return_intermediate: bool = False,
    ) -> SamplerResult:
        """
        Run the full sampling procedure.

        Args:
            d: data dimension
            n_samples: number of independent samples to generate
            return_intermediate: whether to store intermediate Y_k values

        Returns:
            SamplerResult with final samples and metadata
        """
        all_samples = []
        total_score_evals = 0

        for _ in range(n_samples):
            result = self._sample_single(d, return_intermediate)
            all_samples.append(result.samples)
            total_score_evals = result.score_evals

        samples = np.stack(all_samples) if n_samples > 1 else all_samples[0]

        return SamplerResult(
            samples=samples,
            score_evals=total_score_evals,
            tau_K0=result.tau_K0,
        )

    def _sample_single(
        self,
        d: int,
        return_intermediate: bool = False,
    ) -> SamplerResult:
        """
        Generate a single sample Y_K.

        Args:
            d: data dimension
            return_intermediate: whether to store Y_k for each round

        Returns:
            SamplerResult
        """
        K, N = self.K, self.N
        total_score_evals = 0
        intermediate = [] if return_intermediate else None

        # Step 1: Initialize Y_0 ~ N(0, I_d)
        Y_k = self.rng.standard_normal(d)

        if return_intermediate:
            intermediate.append(Y_k.copy())

        tau_K0 = self.schedule.tau_hat(K, 0)

        # Step 2: Iterate over K rounds
        for k in range(K):
            # Sample randomized tau values for this round
            tau_hat_arr, tau_arr = self.schedule.get_round_taus(k, self.rng)

            # Execute round: compute Y_{k,N}
            Y_kN, score_evals = self._compute_round_from_scratch(
                Y_k, k, tau_hat_arr, tau_arr
            )
            total_score_evals += score_evals

            tau_kN = tau_arr[N]

            # Step 3: Noise injection (except after last round)
            if k < K - 1:
                Y_k, _ = self._noise_injection(Y_kN, k, tau_kN)
            else:
                Y_k = Y_kN

            if return_intermediate:
                intermediate.append(Y_k.copy())

        return SamplerResult(
            samples=Y_k,
            score_evals=total_score_evals,
            tau_K0=tau_K0,
            intermediate_states=intermediate,
        )

    def sample_trajectory(self, d: int) -> Dict:
        """
        Generate a single sample and return the full trajectory.

        Returns:
            dict with keys: 'Y_k' (list of Y_k for k=0,...,K),
                           'tau_k0' (list of tau_{k,0}),
                           'score_evals'
        """
        result = self._sample_single(d, return_intermediate=True)
        return {
            "Y_k": result.intermediate_states,
            "tau_K0": result.tau_K0,
            "score_evals": result.score_evals,
            "Y_final": result.samples,
        }


def compute_required_K(d: int, L: float, T: int) -> int:
    """
    Compute the required number of rounds K from Theorem 1.

    K = c2 * min{d * log^2(T), L * log(T)}

    Args:
        d: data dimension
        L: non-uniform Lipschitz constant (Definition 2)
        T: total iterations

    Returns:
        K: number of rounds
    """
    log_T = np.log(T)
    c2 = 1.0  # universal constant
    K = int(c2 * min(d * log_T**2, L * log_T))
    K = max(K, 2)  # at least 2 rounds
    K = min(K, T)  # at most T rounds
    return K


def compute_required_T(
    d: int,
    L: float,
    epsilon: float,
    log_factor: bool = True,
) -> int:
    """
    Compute the required number of iterations T to achieve TV <= epsilon (Eq. 14).

    T >= min{d, d^{2/3} L^{1/3}, d^{1/3} L} * epsilon^{-2/3} * log^{8/3}(T)

    Args:
        d: data dimension
        L: non-uniform Lipschitz constant
        epsilon: target TV distance
        log_factor: whether to include log factors

    Returns:
        T: required iterations (approximate, ignoring log factors in T)
    """
    complexity = min(d, d**(2/3) * L**(1/3), d**(1/3) * L)
    T_base = complexity * epsilon**(-2/3)

    if log_factor:
        # Approximate log factor: iterate to find T
        T = T_base
        for _ in range(10):
            log_T = max(np.log(T), 1.0)
            T = T_base * log_T**(8/3)
        return int(np.ceil(T))
    return int(np.ceil(T_base))
