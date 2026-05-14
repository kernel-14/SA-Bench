"""
Theoretical complexity bounds for diffusion model samplers.

Implements the iteration complexity formulas from the paper and prior works
for comparison (Figure 1, Figure 3, Appendix B).

Our result (Theorem 1):
  TV <= C * min{d^{3/2}, d*L^{1/2}, d^{1/2}*L^{3/2}} * log^4(T) / T^{3/2}

Iteration complexity to achieve TV <= epsilon:
  T >= min{d, d^{2/3}*L^{1/3}, d^{1/3}*L} * epsilon^{-2/3} * log^{8/3}(T)

Prior works:
  - Benton et al. (2023): T >= d * epsilon^{-2}
  - Li and Yan (2024a): T >= d * epsilon^{-1}
  - Li and Cai (2024): T >= d^{5/4} * epsilon^{-1/2}
  - Li and Jiao (2024): T >= d^{1/3} * L * epsilon^{-2/3}
  - Gupta et al. (2024): T >= d^{2/3} * L^{1/3} * epsilon^{-2/3} (under uniform Lipschitz)
"""

import numpy as np
from typing import Dict, Optional


def our_tv_bound(d: int, L: float, T: int, log_factor: bool = True) -> float:
    """
    TV distance bound from Theorem 1 (ignoring score estimation error).

    TV <= C * min{d^{3/2}, d*L^{1/2}, d^{1/2}*L^{3/2}} * log^4(T) / T^{3/2}

    Args:
        d: data dimension
        L: non-uniform Lipschitz constant (inf for no smoothness)
        T: total iterations
        log_factor: whether to include log^4(T) factor

    Returns:
        TV upper bound
    """
    if L == float("inf"):
        complexity = d**(3/2)
    else:
        complexity = min(d**(3/2), d * L**(1/2), d**(1/2) * L**(3/2))

    log_T = np.log(max(T, 2))
    log_fac = log_T**4 if log_factor else 1.0
    return complexity * log_fac / T**(3/2)


def our_iteration_complexity(
    d: int,
    L: float,
    epsilon: float,
    log_factor: bool = True,
) -> float:
    """
    Iteration complexity from Theorem 1 (Eq. 14).

    T >= min{d, d^{2/3}*L^{1/3}, d^{1/3}*L} * epsilon^{-2/3} * log^{8/3}(T)

    Args:
        d: data dimension
        L: non-uniform Lipschitz constant
        epsilon: target TV distance
        log_factor: whether to include log factors

    Returns:
        Required T (approximate)
    """
    if L == float("inf"):
        complexity = d
    else:
        complexity = min(d, d**(2/3) * L**(1/3), d**(1/3) * L)

    T_base = complexity * epsilon**(-2/3)

    if log_factor:
        T = T_base
        for _ in range(20):
            log_T = max(np.log(T), 1.0)
            T = T_base * log_T**(8/3)
        return T
    return T_base


def benton_2023_tv_bound(d: int, T: int) -> float:
    """
    TV bound from Benton et al. (2023): nearly d-linear convergence.

    TV <= C * d / sqrt(T)  (simplified from d*epsilon^{-2} complexity)

    Actually: T >= d * epsilon^{-2} implies TV <= C * sqrt(d/T)
    """
    return np.sqrt(d / T)


def benton_2023_iteration_complexity(d: int, epsilon: float) -> float:
    """
    Iteration complexity from Benton et al. (2023): T >= d * epsilon^{-2}.
    """
    return d * epsilon**(-2)


def li_yan_2024a_tv_bound(d: int, T: int) -> float:
    """
    TV bound from Li and Yan (2024a): O(d/T) convergence.

    TV <= C * d / T  (from T >= d * epsilon^{-1} complexity)
    """
    return d / T


def li_yan_2024a_iteration_complexity(d: int, epsilon: float) -> float:
    """
    Iteration complexity from Li and Yan (2024a): T >= d * epsilon^{-1}.
    """
    return d * epsilon**(-1)


def li_cai_2024_tv_bound(d: int, T: int) -> float:
    """
    TV bound from Li and Cai (2024): accelerated sampler.

    TV <= C * d^{5/4} / T  (from T >= d^{5/4} * epsilon^{-1/2} complexity)
    Actually: TV <= C * (d^{5/4}/T)^{1/2} ... let me re-derive.

    T >= d^{5/4} * epsilon^{-1/2} means epsilon <= (d^{5/4}/T)^2 = d^{5/2}/T^2
    Wait: epsilon^{1/2} <= d^{5/4}/T => epsilon <= d^{5/2}/T^2
    So TV <= C * d^{5/4} / T^{1/2}... let me check.

    From the paper: "O(d^{5/4} epsilon^{-1/2})" means T = O(d^{5/4} / epsilon^{1/2})
    So epsilon = O((d^{5/4}/T)^2) = O(d^{5/2}/T^2)... that seems too fast.

    Actually: T >= d^{5/4} * epsilon^{-1/2}
    => epsilon^{1/2} >= d^{5/4} / T
    => epsilon >= d^{5/2} / T^2

    So TV <= C * d^{5/4} / T (from T^{1/2} scaling... let me be careful)

    T = d^{5/4} * epsilon^{-1/2} => epsilon = (d^{5/4}/T)^2 = d^{5/2}/T^2
    TV = epsilon^{1/2} = d^{5/4}/T

    So TV <= C * d^{5/4} / T.
    """
    return d**(5/4) / T


def li_cai_2024_iteration_complexity(d: int, epsilon: float) -> float:
    """
    Iteration complexity from Li and Cai (2024): T >= d^{5/4} * epsilon^{-1/2}.
    """
    return d**(5/4) * epsilon**(-1/2)


def li_jiao_2024_tv_bound(d: int, L: float, T: int, log_factor: bool = True) -> float:
    """
    TV bound from Li and Jiao (2024): T >= d^{1/3} * L * epsilon^{-2/3}.

    TV <= C * (d^{1/3} * L)^{3/2} * log^4(T) / T^{3/2}
       = C * d^{1/2} * L^{3/2} * log^4(T) / T^{3/2}
    """
    log_T = np.log(max(T, 2))
    log_fac = log_T**4 if log_factor else 1.0
    return d**(1/2) * L**(3/2) * log_fac / T**(3/2)


def li_jiao_2024_iteration_complexity(d: int, L: float, epsilon: float, log_factor: bool = True) -> float:
    """
    Iteration complexity from Li and Jiao (2024): T >= d^{1/3} * L * epsilon^{-2/3}.
    """
    T_base = d**(1/3) * L * epsilon**(-2/3)
    if log_factor:
        T = T_base
        for _ in range(20):
            log_T = max(np.log(T), 1.0)
            T = T_base * log_T**(8/3)
        return T
    return T_base


def gupta_2024_tv_bound(d: int, L: float, T: int, log_factor: bool = True) -> float:
    """
    TV bound from Gupta et al. (2024): randomized midpoints under uniform Lipschitz.

    T >= d^{2/3} * L^{1/3} * epsilon^{-2/3}
    TV <= C * (d^{2/3} * L^{1/3})^{3/2} * log^4(T) / T^{3/2}
       = C * d * L^{1/2} * log^4(T) / T^{3/2}
    """
    log_T = np.log(max(T, 2))
    log_fac = log_T**4 if log_factor else 1.0
    return d * L**(1/2) * log_fac / T**(3/2)


def gupta_2024_iteration_complexity(d: int, L: float, epsilon: float, log_factor: bool = True) -> float:
    """
    Iteration complexity from Gupta et al. (2024): T >= d^{2/3} * L^{1/3} * epsilon^{-2/3}.
    """
    T_base = d**(2/3) * L**(1/3) * epsilon**(-2/3)
    if log_factor:
        T = T_base
        for _ in range(20):
            log_T = max(np.log(T), 1.0)
            T = T_base * log_T**(8/3)
        return T
    return T_base


def all_tv_bounds(d: int, L: float, T: int, log_factor: bool = False) -> Dict[str, float]:
    """
    Compute TV bounds from all methods for comparison.

    Args:
        d: data dimension
        L: Lipschitz constant
        T: total iterations
        log_factor: whether to include log factors

    Returns:
        dict mapping method name to TV bound
    """
    bounds = {
        "Ours (Theorem 1)": our_tv_bound(d, L, T, log_factor),
        "Benton et al. (2023)": benton_2023_tv_bound(d, T),
        "Li & Yan (2024a)": li_yan_2024a_tv_bound(d, T),
        "Li & Cai (2024)": li_cai_2024_tv_bound(d, T),
    }

    if L < float("inf"):
        bounds["Li & Jiao (2024)"] = li_jiao_2024_tv_bound(d, L, T, log_factor)
        bounds["Gupta et al. (2024)"] = gupta_2024_tv_bound(d, L, T, log_factor)

    return bounds


def all_iteration_complexities(
    d: int,
    L: float,
    epsilon: float,
    log_factor: bool = False,
) -> Dict[str, float]:
    """
    Compute iteration complexities from all methods for comparison.

    Args:
        d: data dimension
        L: Lipschitz constant
        epsilon: target TV distance
        log_factor: whether to include log factors

    Returns:
        dict mapping method name to iteration complexity
    """
    complexities = {
        "Ours (Theorem 1)": our_iteration_complexity(d, L, epsilon, log_factor),
        "Benton et al. (2023)": benton_2023_iteration_complexity(d, epsilon),
        "Li & Yan (2024a)": li_yan_2024a_iteration_complexity(d, epsilon),
        "Li & Cai (2024)": li_cai_2024_iteration_complexity(d, epsilon),
    }

    if L < float("inf"):
        complexities["Li & Jiao (2024)"] = li_jiao_2024_iteration_complexity(d, L, epsilon, log_factor)
        complexities["Gupta et al. (2024)"] = gupta_2024_iteration_complexity(d, L, epsilon, log_factor)

    return complexities


def improvement_factor_over_li_jiao(d: int, L: float) -> float:
    """
    Improvement factor of our result over Li and Jiao (2024).

    Our complexity: min{d, d^{2/3}L^{1/3}, d^{1/3}L} * epsilon^{-2/3}
    Li & Jiao:      d^{1/3} * L * epsilon^{-2/3}

    Improvement = d^{1/3} * L / min{d, d^{2/3}L^{1/3}, d^{1/3}L}
                = max{d^{-2/3}L, d^{-1/3}L^{2/3}, 1}

    From Section 1.1: improvement by factor max{d^{-2/3}L, d^{-1/3}L^{2/3}, 1}
    """
    return max(d**(-2/3) * L, d**(-1/3) * L**(2/3), 1.0)


def improvement_factor_over_li_yan(d: int, L: float, epsilon: float) -> float:
    """
    Improvement factor of our result over Li and Yan (2024a).

    Our complexity: min{d, d^{2/3}L^{1/3}, d^{1/3}L} * epsilon^{-2/3}
    Li & Yan:       d * epsilon^{-1}

    Improvement = d * epsilon^{-1} / (min{d, d^{2/3}L^{1/3}, d^{1/3}L} * epsilon^{-2/3})
                = d / min{d, d^{2/3}L^{1/3}, d^{1/3}L} * epsilon^{-1/3}
                = max{1, d^{1/3}L^{-1/3}, d^{2/3}L^{-1}} * epsilon^{-1/3}
    """
    if L == float("inf"):
        return epsilon**(-1/3)
    return max(1.0, d**(1/3) * L**(-1/3), d**(2/3) * L**(-1)) * epsilon**(-1/3)


def non_uniform_vs_uniform_lipschitz_gmm(H: int, T: int, d: int) -> Dict[str, float]:
    """
    Compare non-uniform and uniform Lipschitz constants for GMM (Example 2).

    Non-uniform L: O(log(H*(T+d)))  (scales logarithmically)
    Uniform L: can be extremely large when sigma^2 is small

    Args:
        H: number of GMM components
        T: number of iterations
        d: data dimension

    Returns:
        dict with L_nonuniform and description of L_uniform
    """
    L_nonuniform = np.log(H * (T + d))
    return {
        "L_nonuniform": L_nonuniform,
        "L_nonuniform_formula": f"log(H*(T+d)) = log({H}*({T}+{d})) = {L_nonuniform:.2f}",
        "L_uniform_description": "Can be O(||mu||^2 / sigma^4) which is O(d/sigma^4) >> L_nonuniform",
    }
