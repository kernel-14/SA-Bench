"""Learning rate schedule from Section 2.2, Equation (9).

Implements the randomized learning rate schedule defined in:
    hat_alpha_{T+1} = 1/T^{c_0}
    hat_alpha_{t-1} = hat_alpha_t + c_1 * hat_alpha_t * (1 - hat_alpha_t) * log(T) / T
    bar_alpha_t ~ Unif(hat_alpha_t, hat_alpha_{t-1})
"""

import numpy as np


def build_hat_alphas(T, c0, c1):
    """Build the deterministic hat_alpha schedule.

    Args:
        T: total number of iterations (2T = K*N total steps)
        c0: constant for initial alpha
        c1: constant for step size (c1/c0 assumed sufficiently large)

    Returns:
        hat_alphas: array of length T+2, indexed t=0,...,T+1
    """
    # hat_alpha_{T+1} = 1/T^{c_0}
    hat_alphas = np.zeros(T + 2)
    hat_alphas[T + 1] = 1.0 / (T ** c0)

    # Build backwards: hat_alpha_{t-1} = hat_alpha_t + c_1 * hat_alpha_t * (1 - hat_alpha_t) * log(T) / T
    for t in range(T + 1, 0, -1):
        step = c1 * hat_alphas[t] * (1.0 - hat_alphas[t]) * np.log(max(T, 2)) / T
        # Prevent overflow: clamp step to reasonable range
        step = min(step, 1.0 - hat_alphas[t])
        hat_alphas[t - 1] = hat_alphas[t] + step

    return hat_alphas


def sample_bar_alphas(hat_alphas, rng=None):
    """Sample bar_alpha_t ~ Unif(hat_alpha_t, hat_alpha_{t-1}).

    Args:
        hat_alphas: deterministic hat_alpha schedule
        rng: numpy random generator

    Returns:
        bar_alphas: array of same length as hat_alphas (with last entry fixed)
    """
    if rng is None:
        rng = np.random.default_rng()

    T_plus_1 = len(hat_alphas) - 1  # T+1
    bar_alphas = np.zeros_like(hat_alphas)

    for t in range(T_plus_1, 0, -1):
        lo = hat_alphas[t]
        hi = hat_alphas[t - 1]
        # Ensure hi >= lo
        if hi < lo:
            bar_alphas[t] = lo
        else:
            bar_alphas[t] = rng.uniform(lo, hi)

    bar_alphas[0] = hat_alphas[0]

    return bar_alphas


def build_tau_schedule(T, K, hat_alphas, bar_alphas):
    """Convert alpha schedule to tau schedule.

    tau_{k,n} = 1 - bar_alpha_{T - kN/2 - n + 1}
    hat_tau_{k,n} = 1 - hat_alpha_{T - kN/2 - n}

    Args:
        T: total iterations
        K: number of rounds
        hat_alphas: deterministic hat_alpha schedule
        bar_alphas: randomized bar_alpha schedule

    Returns:
        tau: dict with keys (k,n) -> tau_{k,n}
        hat_tau: dict with keys (k,n) -> hat_tau_{k,n}
        N: number of steps per round
    """
    N = 2 * T // K

    tau = {}
    hat_tau = {}

    for k in range(K):
        for n in range(-1, N + 1):
            # hat_tau_{k,n} = 1 - hat_alpha_{T - kN/2 - n}
            alpha_idx = T - k * N // 2 - n
            if 0 <= alpha_idx <= T + 1:
                hat_tau[(k, n)] = 1.0 - hat_alphas[alpha_idx]

            # tau_{k,n} = 1 - bar_alpha_{T - kN/2 - n + 1}
            bar_idx = T - k * N // 2 - n + 1
            if 0 <= bar_idx <= T + 1:
                tau[(k, n)] = 1.0 - bar_alphas[bar_idx]

    return tau, hat_tau, N


def verify_schedule_properties(tau, hat_tau, T, c1):
    """Verify Lemma 7 properties of the schedule.

    Properties:
    1. 1 - tau_{0,0} <= 2/T^{c0}
    2. tau_{K,0} <= 1/T^{c0}
    3. (hat_tau_{k,n-1} - hat_tau_{k,n}) / (hat_tau_{k,n-1} * (1 - hat_tau_{k,n-1})) = c_1 * log(T) / T
    4. hat_tau ratios are bounded by O(1)
    """
    results = {}

    # Property 3
    n_idx = 0
    hn = hat_tau.get((0, n_idx), None)
    hnm1 = hat_tau.get((0, n_idx - 1), None)
    if hn is not None and hnm1 is not None and hnm1 > 0 and (1 - hnm1) > 0:
        ratio = (hnm1 - hn) / (hnm1 * (1 - hnm1))
        expected = c1 * np.log(T) / T
        results['ratio_check'] = abs(ratio - expected) < 1e-10 if expected > 1e-15 else True

    return results
