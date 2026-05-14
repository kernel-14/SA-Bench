"""Parallel sampler analysis from Section 3.3 and Theorem 2.

Extends the randomized midpoint sampler to a parallel implementation
where N parallel processors are used across MK rounds.

Theorem 2: Under the same assumptions as Theorem 1, to achieve
TV(q_K, p_{Y_K}) <= epsilon, it suffices to choose:
    N >= O((min{d^{2/3} L^{-2/3}, d^{1/3}} + 1) * log^{5/3}(T) / epsilon^{2/3})
    MK >= O(min{d log(T), L} * log^2(T))
    epsilon_score^2 <= O(epsilon^2 / log(T))
"""

import numpy as np


def parallel_sampler_bounds(d, L, epsilon, T=None):
    """Compute the parallel sampler bounds from Theorem 2.

    Args:
        d: data dimension
        L: non-uniform Lipschitz constant
        epsilon: target TV distance
        T: total iterations (optional, for log terms)

    Returns:
        dict with N_bound, MK_bound, epsilon_score_bound
    """
    if T is None:
        T = min(d, d ** (2/3) * L ** (1/3), d ** (1/3) * L) * epsilon ** (-2/3)

    log_T = np.log(T)

    # N bound: number of parallel processors
    N_term1 = d ** (2/3) * L ** (-2/3) if L > 0 else float('inf')
    N_term2 = d ** (1/3)
    N_factor = min(N_term1, N_term2) + 1
    N_bound = N_factor * (log_T ** (5/3)) / (epsilon ** (2/3))

    # MK bound: number of parallel rounds
    MK_bound = min(d * log_T, L) * (log_T ** 2)

    # Score estimation error bound
    eps_score_bound = epsilon ** 2 / log_T

    return {
        'N_bound': N_bound,
        'MK_bound': MK_bound,
        'epsilon_score_bound': eps_score_bound,
        'T': T,
    }


def compare_serial_vs_parallel(d, L, epsilon):
    """Compare serial (Theorem 1) and parallel (Theorem 2) requirements.

    Serial: T ~ min{d, d^{2/3} L^{1/3}, d^{1/3} L} * epsilon^{-2/3}
    Parallel: N ~ min{d^{2/3} L^{-2/3}, d^{1/3}} * epsilon^{-2/3}
              MK ~ min{d, L} (log factors)

    The parallel implementation needs fewer processors when L is large.
    """
    serial_T = min(d, d ** (2/3) * L ** (1/3), d ** (1/3) * L) * epsilon ** (-2/3)

    par = parallel_sampler_bounds(d, L, epsilon, T=serial_T)

    return {
        'serial_iterations': serial_T,
        'parallel_processors': par['N_bound'],
        'parallel_rounds': par['MK_bound'],
        'speedup': serial_T / (par['N_bound'] * par['MK_bound']) if par['MK_bound'] > 0 else 0,
    }
