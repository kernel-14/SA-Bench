"""
Numerical experiments reproducing the paper's results.

Implements:
  - Figure 2 (Appendix A): KL divergence vs T for Gaussian target
  - Figure 1: Iteration complexity comparison as function of L and epsilon
  - Figure 3 (Appendix B): TV distance comparison for fixed T

All experiments use exact score functions (no score estimation error).
"""

import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from data import GaussianDistribution, GMMDistribution, TwoComponentGMM
from score_functions import GaussianScoreFunction, GMMScoreFunction
from sampler import RandomizedMidpointSampler, compute_required_K
from convergence_metrics import (
    kl_divergence_gaussians_diagonal,
    tv_from_kl,
    theoretical_tv_bound,
)
from theory import (
    our_tv_bound,
    our_iteration_complexity,
    benton_2023_tv_bound,
    li_yan_2024a_tv_bound,
    li_cai_2024_tv_bound,
    li_jiao_2024_tv_bound,
    gupta_2024_tv_bound,
    all_tv_bounds,
    all_iteration_complexities,
)


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    T_values: np.ndarray
    kl_values: np.ndarray
    tv_values: np.ndarray
    theoretical_rate: np.ndarray
    d: int
    k: int
    K: int
    n_trials: int
    metadata: Dict = field(default_factory=dict)


def run_gaussian_convergence_experiment(
    d: int,
    k: int,
    T_values: List[int],
    K: int = 10,
    c0: float = 5.0,
    c1: float = 50.0,
    n_samples: int = 5000,
    n_trials: int = 5,
    seed: int = 0,
    sigma_max: float = 10.0,
) -> ExperimentResult:
    """
    Run convergence experiment for Gaussian target (Appendix A, Figure 2).

    For each T in T_values:
    1. Create sampler with K rounds, N = 2T/K steps per round
    2. Generate n_samples samples Y_K
    3. Compute KL(q_K || p_{Y_K}) using closed-form for Gaussians

    Args:
        d: data dimension
        k: number of non-zero variance components
        T_values: list of total iteration counts
        K: number of rounds
        c0: schedule parameter
        c1: schedule parameter
        n_samples: number of samples for KL estimation
        n_trials: number of independent trials
        seed: random seed
        sigma_max: maximum variance for Gaussian components

    Returns:
        ExperimentResult with KL and TV values for each T
    """
    data = GaussianDistribution(d=d, k=k, sigma_max=sigma_max, seed=seed)
    score_fn = data.get_score_function()

    kl_means = []
    kl_stds = []
    tv_means = []

    for T in T_values:
        N = 2 * T // K
        if N < 2:
            kl_means.append(float("nan"))
            kl_stds.append(float("nan"))
            tv_means.append(float("nan"))
            continue

        trial_kls = []

        for trial in range(n_trials):
            trial_seed = seed + trial * 1000

            sampler = RandomizedMidpointSampler(
                score_fn=score_fn,
                T=T,
                K=K,
                c0=c0,
                c1=c1,
                seed=trial_seed,
            )

            # Get tau_{K,0} for q_K
            tau_K0 = sampler.schedule.tau_hat(K, 0)

            # q_K = N(0, Sigma_{tau_{K,0}})
            mu_q = np.zeros(d)
            var_q = score_fn._sigma_t_sq(tau_K0)

            # Generate samples from p_{Y_K}
            result = sampler.sample(d, n_samples=n_samples)
            samples = result.samples  # shape (n_samples, d)

            # Estimate p_{Y_K} as Gaussian
            mu_p = np.mean(samples, axis=0)
            var_p = np.var(samples, axis=0)

            # KL(q_K || p_{Y_K})
            kl = kl_divergence_gaussians_diagonal(mu_q, var_q, mu_p, var_p)
            trial_kls.append(kl)

        kl_mean = np.mean(trial_kls)
        kl_std = np.std(trial_kls)
        kl_means.append(kl_mean)
        kl_stds.append(kl_std)
        tv_means.append(tv_from_kl(kl_mean))

        print(f"  T={T:5d}: KL = {kl_mean:.4e} ± {kl_std:.4e}, TV <= {tv_from_kl(kl_mean):.4e}")

    T_arr = np.array(T_values, dtype=float)
    kl_arr = np.array(kl_means)
    tv_arr = np.array(tv_means)

    # Theoretical rate: O(log^4(T) / T^3) for KL divergence
    # (TV^2 <= KL/2, and TV = O(log^4(T)/T^{3/2}) => KL = O(log^8(T)/T^3))
    # From Theorem 1: TV <= C * d^{3/2} * log^4(T) / T^{3/2}
    # => KL <= 2 * TV^2 <= C * d^3 * log^8(T) / T^3
    # Paper says: KL convergence rate O(log^4(T)/T^3) (Appendix A)
    theoretical_rate = np.log(np.maximum(T_arr, 2))**4 / T_arr**3

    # Normalize to match empirical scale
    valid = ~np.isnan(kl_arr)
    if valid.any():
        scale = np.nanmedian(kl_arr[valid] / theoretical_rate[valid])
        theoretical_rate = scale * theoretical_rate

    return ExperimentResult(
        T_values=T_arr,
        kl_values=kl_arr,
        tv_values=tv_arr,
        theoretical_rate=theoretical_rate,
        d=d,
        k=k,
        K=K,
        n_trials=n_trials,
        metadata={
            "kl_stds": np.array(kl_stds),
            "sigma_max": sigma_max,
            "c0": c0,
            "c1": c1,
        },
    )


def run_complexity_comparison_vs_L(
    d: int,
    epsilon: float,
    L_values: np.ndarray,
    log_factor: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute iteration complexity as function of L for all methods (Figure 1 left).

    Args:
        d: data dimension
        epsilon: target TV distance
        L_values: array of Lipschitz constant values
        log_factor: whether to include log factors

    Returns:
        dict mapping method name to array of complexities
    """
    results = {
        "Ours (Theorem 1)": [],
        "Benton et al. (2023)": [],
        "Li & Yan (2024a)": [],
        "Li & Cai (2024)": [],
        "Li & Jiao (2024)": [],
        "Gupta et al. (2024)": [],
    }

    for L in L_values:
        complexities = all_iteration_complexities(d, L, epsilon, log_factor)
        for method, T in complexities.items():
            if method in results:
                results[method].append(T)

    return {k: np.array(v) for k, v in results.items()}


def run_tv_comparison_vs_epsilon(
    d: int,
    L: float,
    T_values: np.ndarray,
    log_factor: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute TV distance as function of T for all methods (Figure 1 right, L=inf).

    Args:
        d: data dimension
        L: Lipschitz constant (inf for no smoothness)
        T_values: array of iteration counts
        log_factor: whether to include log factors

    Returns:
        dict mapping method name to array of TV bounds
    """
    results = {}

    for T in T_values:
        bounds = all_tv_bounds(d, L, int(T), log_factor)
        for method, tv in bounds.items():
            if method not in results:
                results[method] = []
            results[method].append(tv)

    return {k: np.array(v) for k, v in results.items()}


def run_tv_comparison_vs_L(
    d: int,
    T: int,
    L_values: np.ndarray,
    log_factor: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute TV distance as function of L for fixed T (Figure 3).

    Args:
        d: data dimension
        T: total iterations
        L_values: array of Lipschitz constant values
        log_factor: whether to include log factors

    Returns:
        dict mapping method name to array of TV bounds
    """
    results = {}

    for L in L_values:
        bounds = all_tv_bounds(d, L, T, log_factor)
        for method, tv in bounds.items():
            if method not in results:
                results[method] = []
            results[method].append(tv)

    return {k: np.array(v) for k, v in results.items()}


def run_lipschitz_comparison_gmm(
    d: int,
    H_values: List[int],
    T: int,
    sigma_values: List[float],
) -> Dict:
    """
    Compare non-uniform vs uniform Lipschitz constants for GMM (Example 2).

    Args:
        d: data dimension
        H_values: list of number of components
        T: number of iterations
        sigma_values: list of component standard deviations

    Returns:
        dict with comparison results
    """
    results = {}

    for H in H_values:
        for sigma in sigma_values:
            gmm = TwoComponentGMM(d=d, mu_norm=np.sqrt(d), sigma=sigma)
            score_fn = gmm.get_score_function()

            L_nonuniform = score_fn.non_uniform_lipschitz_constant(tau=0.5, T=T)

            # Uniform Lipschitz lower bound at tau=0.5
            L_uniform_lb = gmm.uniform_lipschitz_lower_bound(tau=0.5)

            key = f"H={H}, sigma={sigma}"
            results[key] = {
                "L_nonuniform": L_nonuniform,
                "L_uniform_lower_bound": L_uniform_lb,
                "ratio": L_uniform_lb / L_nonuniform if L_nonuniform > 0 else float("inf"),
            }

    return results


def run_score_estimation_robustness(
    d: int,
    k: int,
    T: int,
    K: int,
    epsilon_score_values: List[float],
    n_samples: int = 2000,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Test robustness to score estimation error (Assumption 2).

    For each epsilon_score, add Gaussian noise to the score function
    and measure the resulting TV distance.

    Args:
        d: data dimension
        k: number of non-zero variance components
        T: total iterations
        K: number of rounds
        epsilon_score_values: list of score estimation error levels
        n_samples: number of samples
        seed: random seed

    Returns:
        dict with TV values for each epsilon_score
    """
    from score_functions import LearnedScoreFunction

    data = GaussianDistribution(d=d, k=k, seed=seed)
    true_score_fn = data.get_score_function()
    tau_K0 = RandomizedMidpointSampler(true_score_fn, T, K, seed=seed).schedule.tau_hat(K, 0)

    mu_q = np.zeros(d)
    var_q = true_score_fn._sigma_t_sq(tau_K0)

    tv_values = []

    for eps_score in epsilon_score_values:
        noisy_score_fn = LearnedScoreFunction(true_score_fn, noise_level=eps_score)
        sampler = RandomizedMidpointSampler(noisy_score_fn, T, K, seed=seed)

        result = sampler.sample(d, n_samples=n_samples)
        samples = result.samples

        mu_p = np.mean(samples, axis=0)
        var_p = np.var(samples, axis=0)

        kl = kl_divergence_gaussians_diagonal(mu_q, var_q, mu_p, var_p)
        tv = tv_from_kl(kl)
        tv_values.append(tv)

        # Theoretical bound: C * epsilon_score * log^{1/2}(T)
        tv_theory = eps_score * np.log(T)**0.5

        print(f"  eps_score={eps_score:.4f}: TV={tv:.4e}, theory={tv_theory:.4e}")

    return {
        "epsilon_score_values": np.array(epsilon_score_values),
        "tv_values": np.array(tv_values),
    }


def run_all_figure2_experiments(
    T_values: Optional[List[int]] = None,
    n_samples: int = 3000,
    n_trials: int = 3,
    seed: int = 0,
) -> Dict[str, ExperimentResult]:
    """
    Run all experiments for Figure 2 (Appendix A).

    Three settings:
    (a) d=10, k=10
    (b) d=100, k=10
    (c) d=500, k=100

    Args:
        T_values: list of T values (default: paper values)
        n_samples: number of samples per T
        n_trials: number of independent trials
        seed: random seed

    Returns:
        dict mapping setting name to ExperimentResult
    """
    if T_values is None:
        T_values = [50, 100, 200, 500, 1000, 2000, 5000]

    settings = {
        "fig2a": {"d": 10, "k": 10, "K": 10},
        "fig2b": {"d": 100, "k": 10, "K": 10},
        "fig2c": {"d": 500, "k": 100, "K": 10},
    }

    results = {}

    for name, cfg in settings.items():
        print(f"\nRunning {name}: d={cfg['d']}, k={cfg['k']}, K={cfg['K']}")
        result = run_gaussian_convergence_experiment(
            d=cfg["d"],
            k=cfg["k"],
            T_values=T_values,
            K=cfg["K"],
            n_samples=n_samples,
            n_trials=n_trials,
            seed=seed,
        )
        results[name] = result

    return results
