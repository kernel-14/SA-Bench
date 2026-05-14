"""
Theoretical convergence bounds from the paper.

Implements the convergence rate formulas from:
"Instance-dependent Convergence Theory for Diffusion Models"
by Yuchen Jiao and Gen Li (2025).

Main result (Theorem 1):
    TV(q_K, p_{Y_K}) <= C * min{d^{3/2}, d*L^{1/2}, d^{1/2}*L^{3/2}} * log^4(T) / T^{3/2}
                       + C * eps_score * log^{1/2}(T)

Iteration complexity (to achieve TV <= eps):
    T >= min{d, d^{2/3}*L^{1/3}, d^{1/3}*L} * eps^{-2/3} * log^{8/3}(T)
"""

import numpy as np
from typing import Optional


def tv_bound_theorem1(T: int, L: float, d: int, eps_score: float = 0.0,
                       C: float = 1.0) -> float:
    """
    Compute the TV distance upper bound from Theorem 1.

    TV(q_K, p_{Y_K}) <= C * min{d^{3/2}, d*L^{1/2}, d^{1/2}*L^{3/2}} * log^4(T) / T^{3/2}
                       + C * eps_score * log^{1/2}(T)

    Args:
        T: Total number of iterations.
        L: Non-uniform Lipschitz constant (Definition 2).
        d: Data dimension.
        eps_score: Score estimation error (Assumption 2).
        C: Universal constant.

    Returns:
        TV distance upper bound.
    """
    log_T = np.log(max(T, 2))

    # Discretization error term
    disc_error = C * min(d**(3/2), d * L**(1/2), d**(1/2) * L**(3/2)) * log_T**4 / T**(3/2)

    # Score estimation error term
    score_error = C * eps_score * log_T**(1/2)

    return disc_error + score_error


def iteration_complexity(eps: float, L: float, d: int, log_factor: float = 1.0) -> float:
    """
    Compute the iteration complexity to achieve TV distance <= eps.

    T >= min{d, d^{2/3}*L^{1/3}, d^{1/3}*L} * eps^{-2/3} * log^{8/3}(T)

    This ignores the logarithmic factor in T (which requires solving a fixed-point equation).

    Args:
        eps: Target TV distance accuracy.
        L: Non-uniform Lipschitz constant.
        d: Data dimension.
        log_factor: Logarithmic factor (default 1.0, ignoring log terms).

    Returns:
        Iteration complexity (up to logarithmic factors).
    """
    return min(d, d**(2/3) * L**(1/3), d**(1/3) * L) * eps**(-2/3) * log_factor


def number_of_rounds(d: int, L: float, c2: float = 1.0) -> int:
    """
    Compute the number of rounds K = c2 * min{d * log^2(T), L * log(T)}.

    From Theorem 1: K = c2 * min{d * log^2(T), L * log(T)}.

    Args:
        d: Data dimension.
        L: Non-uniform Lipschitz constant.
        c2: Constant (default 1.0).

    Returns:
        Number of rounds K.
    """
    # This is a simplified version ignoring the log(T) factor
    return int(c2 * min(d, L))


def compare_with_prior_works(T: int, L: float, d: int) -> dict:
    """
    Compare TV distance bounds from various works.

    Args:
        T: Total number of iterations.
        L: Lipschitz constant.
        d: Data dimension.

    Returns:
        Dictionary mapping method name to TV distance bound.
    """
    log_T = np.log(max(T, 2))

    results = {}

    # Our result (Theorem 1)
    results["Ours (Theorem 1)"] = min(d**(3/2), d * L**(1/2), d**(1/2) * L**(3/2)) * log_T**4 / T**(3/2)

    # Li and Jiao (2024): O(d^{1/2} * L^{3/2} * log^4(T) / T^{3/2})
    results["Li & Jiao (2024)"] = d**(1/2) * L**(3/2) * log_T**4 / T**(3/2)

    # Li and Yan (2024a): O(d * log^2(T) / T)
    results["Li & Yan (2024a)"] = d * log_T**2 / T

    # Benton et al. (2023): O(d^{1/2} * log(T) / T^{1/2})
    results["Benton et al. (2023)"] = d**(1/2) * log_T / T**(1/2)

    # Li and Cai (2024): O(d^{5/8} * log^{1/4}(T) / T^{1/4})
    results["Li & Cai (2024)"] = d**(5/8) * log_T**(1/4) / T**(1/4)

    # Clip to [0, 1]
    for key in results:
        results[key] = min(results[key], 1.0)

    return results


def improvement_factor(L: float, d: float) -> float:
    """
    Compute the improvement factor of our result over Li and Jiao (2024).

    Our result improves Li & Jiao (2024) by a factor of:
        max{d^{-2/3}*L, d^{-1/3}*L^{2/3}, 1}

    This is significant when L >= sqrt(d).

    Args:
        L: Lipschitz constant.
        d: Data dimension.

    Returns:
        Improvement factor.
    """
    return max(d**(-2/3) * L, d**(-1/3) * L**(2/3), 1.0)


def parallel_sampler_complexity(eps: float, L: float, d: int) -> dict:
    """
    Compute the parallel sampler complexity from Theorem 2.

    From Theorem 2:
        N >= (min{d^{2/3}*L^{-2/3}, d^{1/3}} + 1) * log^{5/3}(T) / eps^{2/3}
        MK >= min{d*log(T), L} * log^2(T)
        eps_score^2 <= eps^2 / log(T)

    Args:
        eps: Target TV distance accuracy.
        L: Non-uniform Lipschitz constant.
        d: Data dimension.

    Returns:
        Dictionary with parallel complexity parameters.
    """
    # Parallel processors needed
    N_parallel = (min(d**(2/3) * L**(-2/3), d**(1/3)) + 1) * eps**(-2/3)

    # Parallel rounds needed
    MK_rounds = min(d, L)

    return {
        "N_parallel_processors": N_parallel,
        "MK_parallel_rounds": MK_rounds,
        "description": "Parallel sampler achieves eps-accuracy with O(min{L,d}*log^2(Ld/eps)) rounds"
    }
