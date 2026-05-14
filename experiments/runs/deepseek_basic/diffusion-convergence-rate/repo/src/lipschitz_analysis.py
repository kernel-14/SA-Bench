"""Non-uniform Lipschitz analysis from Section 3.1 and Appendix C.

Implements:
  1. Definition 2: Non-uniform Lipschitz property
  2. Example 1: Gaussian distribution (Appendix C.1)
  3. Example 2: GMM (Appendix C.2) 
     - Upper bound: L <= O(log(H*(T+d))) with high probability
     - Lower bound: uniform Lipschitz can be extremely large
"""

import numpy as np
from scipy.special import logsumexp


def verify_gaussian_lipschitz(sigma2_diag, bar_alpha, num_trials=1000):
    """Verify the Lipschitz property for Gaussian target (Example 1).

    Properties from Example 1:
    - For all t > 0, all x, x': 
      (1-bar_alpha_t)||s_t^*(x) - s_t^*(x')||_2 <= ||x - x'||_2
    - For t >= 0, exists x, x' such that:
      ||s_t^*(x) - s_t^*(x')||_2 >= (1-bar_alpha_t)^{-1} ||x - x'||_2
      (when min(sigma2_i) = 0)

    Args:
        sigma2_diag: array of shape (d,) with variances
        bar_alpha: current bar_alpha_t
        num_trials: number of random trials

    Returns:
        dict with empirical verification results
    """
    d = len(sigma2_diag)
    rng = np.random.default_rng(42)

    # Score function
    def score(x):
        sigma_t_diag = bar_alpha * sigma2_diag + (1 - bar_alpha)
        return -x / sigma_t_diag

    # Maximum empirical ratio
    max_ratio = 0.0
    max_ratio_unscaled = 0.0

    for _ in range(num_trials):
        x = rng.normal(0, 10, size=(d,))
        xp = rng.normal(0, 10, size=(d,))

        s_diff = np.linalg.norm(score(x) - score(xp))
        s_diff_us = (1 - bar_alpha) * s_diff
        xy_diff = np.linalg.norm(x - xp)

        if xy_diff > 1e-10:
            max_ratio = max(max_ratio, s_diff / xy_diff)
            max_ratio_unscaled = max(max_ratio_unscaled, s_diff_us / xy_diff)

    # Theoretical bounds
    min_s2 = np.min(sigma2_diag)
    L_uniform = 1.0 / (bar_alpha * min_s2 + 1 - bar_alpha)
    L_unscaled_uniform = (1 - bar_alpha) / (bar_alpha * min_s2 + 1 - bar_alpha)

    return {
        'empirical_max_ratio': max_ratio,
        'empirical_max_ratio_unscaled': max_ratio_unscaled,
        'theoretical_L_uniform': L_uniform,
        'theoretical_L_unscaled_uniform': L_unscaled_uniform,
        'unscaled_bound_holds': max_ratio_unscaled <= 1.0 + 1e-8,
        'min_sigma2': min_s2,
    }


def verify_gmm_lipschitz_upper_bound(gmm_score, bar_alpha, T, num_x_samples=100, num_xp_per_x=10):
    """Verify the GMM Lipschitz upper bound (Example 2, Appendix C.2.1).

    The claim: With probability >= 1 - c/(T+d)^4,
        (1-bar_alpha_t)||s_t^*(x) - s_t^*(x')||_2 <= C_1 * log(H*(T+d)) * ||x-x'||_2
    for all ||x-x'||_2 <= C_2 * sqrt(d*(1-bar_alpha_t))

    Args:
        gmm_score: GMMScore instance
        bar_alpha: current bar_alpha_t
        T: total iterations
        num_x_samples: number of x samples for empirical check
        num_xp_per_x: number of x' per x

    Returns:
        dict with empirical verification results
    """
    rng = np.random.default_rng(42)
    H = gmm_score.H
    d = gmm_score.d

    # Theoretical L bound: C_1 * log(H * (T + d))
    L_theory = gmm_score.non_uniform_lipschitz_upper_bound(bar_alpha, T)

    # The neighborhood radius from Definition 2
    # ||x' - x||_2 <= C * sqrt(d * (1-bar_alpha) * log(T)) / L
    radius = np.sqrt(d * (1 - bar_alpha) * np.log(max(T, 2)))

    max_ratio = 0.0
    num_in_radius = 0

    for _ in range(num_x_samples):
        x = gmm_score.sample_target(n=1, rng=rng)[0]
        # Add noise to get x in forward process
        x = np.sqrt(bar_alpha) * x + np.sqrt(1 - bar_alpha) * rng.normal(0, 1, size=(d,))

        for _ in range(num_xp_per_x):
            direction = rng.normal(0, 1, size=(d,))
            direction /= np.linalg.norm(direction) + 1e-15
            dist = rng.uniform(0, radius * 0.5)  # stay well within radius
            xp = x + dist * direction

            s_x = gmm_score.score(x, bar_alpha)
            s_xp = gmm_score.score(xp, bar_alpha)

            s_diff = np.linalg.norm((1 - bar_alpha) * (s_x - s_xp))
            xy_diff = np.linalg.norm(x - xp)

            if xy_diff > 1e-10:
                ratio = s_diff / xy_diff
                max_ratio = max(max_ratio, ratio)
                num_in_radius += 1

    return {
        'theoretical_L_bound': L_theory,
        'empirical_max_ratio': max_ratio,
        'bound_holds': max_ratio <= L_theory + 1e-8 or num_in_radius == 0,
        'num_samples': num_in_radius,
        'neighborhood_radius': radius,
    }


def compute_gmm_uniform_lower_bound(gmm_score, bar_alpha):
    """Compute the lower bound for uniform Lipschitz constant of GMM.

    From Example 2 / Appendix C.2.2:
    For X_0 ~ (1/2)N(mu, sigma^2 I_d) + (1/2)N(-mu, sigma^2 I_d):
        ||(1-bar_alpha) * grad s_t^*(x)||_op >= (1-bar_alpha)*||mu||^2 / (4*(1-bar_alpha+sigma^2)^2)
    when bar_alpha > 1/2.

    This shows the uniform Lipschitz constant can be O(d) while the
    non-uniform one is O(log(H*(T+d))).
    """
    return gmm_score.uniform_lipschitz_lower_bound(bar_alpha)


def compare_lipschitz_constants(gmm_score, T, d, bar_alphas_to_check=None):
    """Compare uniform vs non-uniform Lipschitz constants for GMM.

    Demonstrates that the non-uniform Lipschitz constant scales 
    logarithmically with H and T+d, while the uniform one can be
    extremely large when component variance is small.

    Args:
        gmm_score: GMMScore instance
        T: total iterations
        d: dimension
        bar_alphas_to_check: list of bar_alpha values to check

    Returns:
        dict with comparison results
    """
    if bar_alphas_to_check is None:
        bar_alphas_to_check = [0.5, 0.7, 0.9, 0.95, 0.99]

    results = []
    for bar_alpha in bar_alphas_to_check:
        L_non_uniform = gmm_score.non_uniform_lipschitz_upper_bound(bar_alpha, T)
        L_uniform_lb = compute_gmm_uniform_lower_bound(gmm_score, bar_alpha)
        results.append({
            'bar_alpha': bar_alpha,
            'L_non_uniform': L_non_uniform,
            'L_uniform_lower_bound': L_uniform_lb,
            'ratio': L_uniform_lb / max(L_non_uniform, 1e-10),
        })

    return results


def theoretical_convergence_rate(T, d, L, epsilon=None):
    """Compute the theoretical convergence rate from Theorem 1.

    Iteration complexity: min{d, d^{2/3} L^{1/3}, d^{1/3} L} * epsilon^{-2/3}

    TV distance bound: C * min{d^{3/2}, d L^{1/2}, d^{1/2} L^{3/2}} * log^4(T) / T^{3/2}
                        + C * epsilon_score * log^{1/2}(T)

    Args:
        T: number of iterations
        d: dimension
        L: non-uniform Lipschitz constant
        epsilon: target TV distance (for complexity computation)

    Returns:
        dict with rate analysis
    """
    log_term = (np.log(T)) ** 4

    # TV bound (first term, ignoring constants)
    term_d = d ** 1.5
    term_dL = d * np.sqrt(L)
    term_dL2 = np.sqrt(d) * (L ** 1.5)
    tv_bound_factor = min(term_d, term_dL, term_dL2)
    tv_bound = tv_bound_factor * log_term / (T ** 1.5)

    result = {
        'T': T,
        'd': d,
        'L': L,
        'log_term': log_term,
        'term_d^{3/2}': term_d,
        'term_d L^{1/2}': term_dL,
        'term_d^{1/2} L^{3/2}': term_dL2,
        'min_factor': tv_bound_factor,
        'tv_bound': tv_bound,
    }

    if epsilon is not None:
        # Iteration complexity: solve for T
        complexity = min(d, d ** (2/3) * L ** (1/3), d ** (1/3) * L)
        complexity *= epsilon ** (-2/3)
        result['iteration_complexity'] = complexity
        result['epsilon'] = epsilon

    return result


def compare_with_prior_works(d, L, T, epsilon=0.1):
    """Compare Theorem 1 with prior convergence results.

    As discussed in Section 1.1 and Appendix B:
    - Benton et al. (2023): O(d * epsilon^{-2})
    - Li and Yan (2024a): O(d * epsilon^{-1})
    - Li and Cai (2024): O(d^{5/4} * epsilon^{-1/2})
    - Li and Jiao (2024): O(d^{1/3} * L * epsilon^{-2/3})
    - This work: O(min{d, d^{2/3} L^{1/3}, d^{1/3} L} * epsilon^{-2/3})
    """
    rates = {}

    # Benton et al. (2023)
    T_benton = d / (epsilon ** 2)
    rates['Benton2023_d_eps^{-2}'] = T_benton

    # Li and Yan (2024a)
    T_li_yan = d / epsilon
    rates['LiYan2024_d_eps^{-1}'] = T_li_yan

    # Li and Cai (2024)
    T_li_cai = (d ** 1.25) / np.sqrt(epsilon)
    rates['LiCai2024_d^{5/4}_eps^{-1/2}'] = T_li_cai

    # Li and Jiao (2024)
    T_li_jiao = (d ** (1/3)) * L * (epsilon ** (-2/3))
    rates['LiJiao2024_d^{1/3}L_eps^{-2/3}'] = T_li_jiao

    # This work
    T_ours = min(d, d ** (2/3) * L ** (1/3), d ** (1/3) * L) * (epsilon ** (-2/3))
    rates['ThisWork'] = T_ours

    # Improvement factor over Li and Yan
    improvement_ly = T_li_yan / max(T_ours, 1e-15)
    # Improvement factor over Li and Jiao
    improvement_lj = T_li_jiao / max(T_ours, 1e-15)

    rates['improvement_over_LiYan'] = improvement_ly
    rates['improvement_over_LiJiao'] = improvement_lj

    return rates
