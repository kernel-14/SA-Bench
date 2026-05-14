"""Demonstration of the core contributions.

Runs through all the main results of the paper:
  1. Schedule construction (Section 2.2)
  2. Gaussian score and Lipschitz (Example 1)
  3. GMM score and non-uniform Lipschitz (Example 2)
  4. Convergence rate computation (Theorem 1)
  5. Parallel sampler bounds (Theorem 2)
  6. Comparison with prior works (Section 1.1, Appendix B)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.schedule import build_hat_alphas, sample_bar_alphas
from src.score_functions import GaussianScore, GMMScore
from src.lipschitz_analysis import (
    verify_gaussian_lipschitz,
    verify_gmm_lipschitz_upper_bound,
    compare_lipschitz_constants,
    theoretical_convergence_rate,
    compare_with_prior_works,
)
from src.parallel_sampler import parallel_sampler_bounds, compare_serial_vs_parallel


def demo_schedule():
    """Demonstrate the learning rate schedule (Section 2.2, Eq 9)."""
    print("=" * 60)
    print("1. Learning Rate Schedule (Section 2.2)")
    print("=" * 60)

    T = 100
    c0 = 10.0
    c1 = 100.0

    hat_alphas = build_hat_alphas(T, c0, c1)
    bar_alphas = sample_bar_alphas(hat_alphas)

    print(f"  T = {T}, c0 = {c0}, c1 = {c1}")
    print(f"  hat_alpha_{T+1} = {hat_alphas[T+1]:.6e}  (should be 1/T^{c0} = {1.0/T**c0:.6e})")
    print(f"  hat_alpha_0 = {hat_alphas[0]:.6f}")
    print(f"  Number of steps: {len(hat_alphas)}")
    print(f"  First few hat_alphas: {hat_alphas[-5:]}")
    print(f"  First few bar_alphas: {bar_alphas[-5:]}")


def demo_gaussian_lipschitz():
    """Demonstrate Example 1: Gaussian Lipschitz property."""
    print("\n" + "=" * 60)
    print("2. Gaussian Example (Example 1, Appendix C.1)")
    print("=" * 60)

    d = 5
    sigma2 = np.array([10.0, 5.0, 1.0, 0.5, 0.0])  # last one is 0 (degenerate)
    bar_alpha = 0.5

    result = verify_gaussian_lipschitz(sigma2, bar_alpha)

    print(f"  Dimension d = {d}")
    print(f"  Variance entries: {sigma2}")
    print(f"  bar_alpha = {bar_alpha}")
    print(f"  min(sigma2_i) = {result['min_sigma2']}")
    print(f"  Empirical max ratio (unscaled): {result['empirical_max_ratio_unscaled']:.6f}")
    print(f"  Theoretical L (unscaled): {result['theoretical_L_unscaled_uniform']:.6f}")
    print(f"  Unscaled bound <= 1: {result['unscaled_bound_holds']}")

    # Also show the first property: (1-bar_alpha)||s_t(x)-s_t(x')|| <= ||x-x'||
    print(f"\n  Property (1-bar_alpha)||s_t*(x)-s_t*(x')|| <= ||x-x'|| holds: "
          f"{result['empirical_max_ratio_unscaled'] <= 1.0 + 1e-8}")

    # Second property: when min(sigma2)=0, ||s_t(x)-s_t(x')|| can be >= (1-bar_alpha)^{-1}||x-x'||
    if result['min_sigma2'] == 0:
        print(f"  When min(sigma2)=0: max ratio {result['empirical_max_ratio']:.6f} "
              f"vs theoretical (1-bar_alpha)^(-1) = {1/(1-bar_alpha):.6f}")


def demo_gmm_lipschitz():
    """Demonstrate Example 2: GMM non-uniform Lipschitz."""
    print("\n" + "=" * 60)
    print("3. GMM Example (Example 2, Appendix C.2)")
    print("=" * 60)

    d = 10
    H = 5
    rng = np.random.default_rng(42)

    # Create GMM with separated means and small sigma
    means = rng.normal(0, 5, size=(H, d))
    weights = np.ones(H) / H
    sigma2 = 0.1  # small sigma to make uniform Lipschitz large

    gmm = GMMScore(means, weights, sigma2)
    bar_alpha = 0.9
    T = 1000

    # Non-uniform Lipschitz upper bound
    L_nu = gmm.non_uniform_lipschitz_upper_bound(bar_alpha, T)
    print(f"  H = {H}, d = {d}, sigma^2 = {sigma2}")
    print(f"  bar_alpha = {bar_alpha}, T = {T}")
    print(f"  Non-uniform L bound (logarithmic): {L_nu:.4f}")

    # Uniform Lipschitz lower bound
    L_u_lb = gmm.uniform_lipschitz_lower_bound(bar_alpha)
    print(f"  Uniform L lower bound: {L_u_lb:.4f}")
    print(f"  Ratio (uniform/non-uniform): {L_u_lb/L_nu:.1f}x")
    print(f"  Note: uniform can be O(d) while non-uniform is O(log(H*(T+d)))")

    # Verify empirically
    verify = verify_gmm_lipschitz_upper_bound(gmm, bar_alpha, T, num_x_samples=50)
    print(f"\n  Empirical verification:")
    print(f"    Max ratio in neighborhood: {verify['empirical_max_ratio']:.4f}")
    print(f"    Theoretical bound: {verify['theoretical_L_bound']:.4f}")
    print(f"    Bound holds: {verify['bound_holds']}")
    print(f"    Samples in neighborhood: {verify['num_samples']}")


def demo_convergence_rate():
    """Demonstrate Theorem 1 convergence rate."""
    print("\n" + "=" * 60)
    print("4. Convergence Rate (Theorem 1)")
    print("=" * 60)

    configs = [
        {'d': 100, 'L': 10, 'label': 'L = 10 (small)'},
        {'d': 100, 'L': 100, 'label': 'L = 100 = d'},
        {'d': 100, 'L': 1000, 'label': 'L = 1000 > d'},
        {'d': 100, 'L': np.inf, 'label': 'L = inf'},
    ]

    for cfg in configs:
        d, L = cfg['d'], cfg['L']
        rate = theoretical_convergence_rate(10000, d, L, epsilon=0.1)

        print(f"\n  {cfg['label']}:")
        print(f"    d = {d}, L = {L}")
        print(f"    min{{d^(3/2), d L^(1/2), d^(1/2) L^(3/2)}} = {rate['min_factor']:.2f}")
        print(f"    TV bound: {rate['tv_bound']:.6e}")

        if 'iteration_complexity' in rate:
            print(f"    Iteration complexity (eps=0.1): {rate['iteration_complexity']:.1f}")


def demo_comparison():
    """Demonstrate comparison with prior works."""
    print("\n" + "=" * 60)
    print("5. Comparison with Prior Works (Section 1.1)")
    print("=" * 60)

    d = 100
    for L in [10, np.sqrt(d), d, d**2, np.inf]:
        label = f"L={L:.1f}" if np.isfinite(L) else "L=inf"
        rates = compare_with_prior_works(d, L, 1000, epsilon=0.1)
        print(f"\n  {label}:")
        for key, val in rates.items():
            if 'improvement' in key:
                print(f"    {key}: {val:.1f}x")


def demo_parallel():
    """Demonstrate Theorem 2 parallel sampler."""
    print("\n" + "=" * 60)
    print("6. Parallel Sampler Bounds (Theorem 2)")
    print("=" * 60)

    d = 100
    epsilon = 0.1
    for L in [10, 100, 1000, np.inf]:
        label = f"L={L:.1f}" if np.isfinite(L) else "L=inf"
        comp = compare_serial_vs_parallel(d, L, epsilon)
        print(f"\n  {label}:")
        print(f"    Serial iterations: {comp['serial_iterations']:.1f}")
        print(f"    Parallel processors (N): {comp['parallel_processors']:.1f}")
        print(f"    Parallel rounds (MK): {comp['parallel_rounds']:.1f}")


def main():
    """Run full demonstration."""
    print("=" * 60)
    print("Improved Convergence Rate for Diffusion Probabilistic Models")
    print("Jiao & Li (2025) - Reproduction Demo")
    print("=" * 60)

    demo_schedule()
    demo_gaussian_lipschitz()
    demo_gmm_lipschitz()
    demo_convergence_rate()
    demo_comparison()
    demo_parallel()

    print("\n" + "=" * 60)
    print("Demo complete. See README.md for more details.")
    print("=" * 60)


if __name__ == '__main__':
    main()
