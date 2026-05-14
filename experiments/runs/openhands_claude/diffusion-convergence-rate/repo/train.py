"""
Main training/experiment runner.

Reproduces all experiments from the paper:
  - Figure 2 (Appendix A): KL divergence convergence for Gaussian target
  - Figure 1: Iteration complexity comparison
  - Figure 3 (Appendix B): TV distance comparison
  - Lipschitz constant comparison for GMM (Example 2)

Usage:
  python train.py --experiment all
  python train.py --experiment figure2
  python train.py --experiment figure1
  python train.py --experiment figure3
  python train.py --experiment lipschitz
"""

import argparse
import os
import json
import numpy as np
from typing import Optional

from config import ExperimentConfig, EXPERIMENT_CONFIGS
from data import GaussianDistribution, GMMDistribution, TwoComponentGMM
from score_functions import GaussianScoreFunction, GMMScoreFunction
from sampler import RandomizedMidpointSampler, compute_required_K, compute_required_T
from parallel_sampler import ParallelRandomizedMidpointSampler, compute_parallel_requirements
from convergence_metrics import (
    kl_divergence_gaussians_diagonal,
    tv_from_kl,
    theoretical_tv_bound,
)
from theory import (
    our_tv_bound,
    our_iteration_complexity,
    all_tv_bounds,
    all_iteration_complexities,
    improvement_factor_over_li_jiao,
    improvement_factor_over_li_yan,
    non_uniform_vs_uniform_lipschitz_gmm,
)
from experiments import (
    run_gaussian_convergence_experiment,
    run_all_figure2_experiments,
    run_complexity_comparison_vs_L,
    run_tv_comparison_vs_epsilon,
    run_tv_comparison_vs_L,
    run_lipschitz_comparison_gmm,
    run_score_estimation_robustness,
)
from visualization import (
    plot_figure1_left,
    plot_figure1_right,
    plot_figure2,
    plot_figure3,
    plot_lipschitz_comparison,
    save_results_to_csv,
    HAS_MATPLOTLIB,
)


def run_figure2_experiment(
    output_dir: str,
    T_values: Optional[list] = None,
    n_samples: int = 3000,
    n_trials: int = 3,
    seed: int = 0,
):
    """
    Reproduce Figure 2 (Appendix A): KL divergence convergence.

    Verifies the theoretical rate O(log^4(T)/T^3) for KL divergence.
    """
    print("=" * 60)
    print("Running Figure 2 experiments (Appendix A)")
    print("=" * 60)

    if T_values is None:
        T_values = [50, 100, 200, 500, 1000, 2000, 5000]

    results = run_all_figure2_experiments(
        T_values=T_values,
        n_samples=n_samples,
        n_trials=n_trials,
        seed=seed,
    )

    # Save numerical results
    save_results_to_csv(results, os.path.join(output_dir, "figure2"))

    # Print summary
    print("\nSummary of Figure 2 results:")
    for name, result in results.items():
        print(f"\n{name} (d={result.d}, k={result.k}):")
        valid = ~np.isnan(result.kl_values)
        T_valid = result.T_values[valid]
        kl_valid = result.kl_values[valid]
        for T, kl in zip(T_valid, kl_valid):
            print(f"  T={int(T):5d}: KL = {kl:.4e}")

    # Generate plots
    if HAS_MATPLOTLIB:
        fig = plot_figure2(
            results,
            output_path=os.path.join(output_dir, "figure2", "figure2.pdf"),
        )
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)

    return results


def run_figure1_experiment(
    output_dir: str,
    d: int = 100,
    epsilon: float = 1.0,
):
    """
    Reproduce Figure 1: Iteration complexity comparison.

    Left: complexity vs L for epsilon = O(1)
    Right: TV distance vs T for L = infinity
    """
    print("=" * 60)
    print("Running Figure 1 experiments")
    print("=" * 60)

    os.makedirs(os.path.join(output_dir, "figure1"), exist_ok=True)

    # Figure 1 left: complexity vs L
    print(f"\nFigure 1 (left): d={d}, epsilon={epsilon}")
    L_values = np.logspace(-1, 4, 200)

    complexity_results = run_complexity_comparison_vs_L(
        d=d,
        epsilon=epsilon,
        L_values=L_values,
        log_factor=False,
    )

    # Save to CSV
    header = "L," + ",".join(complexity_results.keys())
    data = np.column_stack([L_values] + list(complexity_results.values()))
    np.savetxt(
        os.path.join(output_dir, "figure1", "complexity_vs_L.csv"),
        data,
        delimiter=",",
        header=header,
        comments="",
    )

    # Print key values
    print("\nIteration complexity at key L values:")
    for L in [1.0, np.sqrt(d), d, d**2]:
        complexities = all_iteration_complexities(d, L, epsilon, log_factor=False)
        print(f"\n  L = {L:.1f}:")
        for method, T in complexities.items():
            print(f"    {method}: T = {T:.2e}")

    # Figure 1 right: TV vs T for L=inf
    print(f"\nFigure 1 (right): d={d}, L=inf")
    T_values = np.logspace(1, 6, 100)

    tv_results = run_tv_comparison_vs_epsilon(
        d=d,
        L=float("inf"),
        T_values=T_values,
        log_factor=False,
    )

    header = "T," + ",".join(tv_results.keys())
    data = np.column_stack([T_values] + list(tv_results.values()))
    np.savetxt(
        os.path.join(output_dir, "figure1", "tv_vs_T.csv"),
        data,
        delimiter=",",
        header=header,
        comments="",
    )

    # Generate plots
    if HAS_MATPLOTLIB:
        fig_left = plot_figure1_left(
            d=d,
            epsilon=epsilon,
            output_path=os.path.join(output_dir, "figure1", "figure1_left.pdf"),
        )
        fig_right = plot_figure1_right(
            d=d,
            L=float("inf"),
            output_path=os.path.join(output_dir, "figure1", "figure1_right.pdf"),
        )
        import matplotlib.pyplot as plt
        if fig_left is not None:
            plt.close(fig_left)
        if fig_right is not None:
            plt.close(fig_right)

    return complexity_results, tv_results


def run_figure3_experiment(
    output_dir: str,
    d: int = 100,
):
    """
    Reproduce Figure 3 (Appendix B): TV distance vs L for fixed T.
    """
    print("=" * 60)
    print("Running Figure 3 experiments (Appendix B)")
    print("=" * 60)

    os.makedirs(os.path.join(output_dir, "figure3"), exist_ok=True)

    T_settings = {
        "T=O(d)": d,
        "T=O(d^1.5)": int(d**1.5),
        "T=O(d^2)": d**2,
    }

    L_values = np.logspace(-1, 4, 200)

    all_results = {}
    for T_label, T in T_settings.items():
        print(f"\n{T_label} (T={T}):")
        tv_results = run_tv_comparison_vs_L(
            d=d,
            T=T,
            L_values=L_values,
            log_factor=False,
        )
        all_results[T_label] = tv_results

        # Save to CSV
        header = "L," + ",".join(tv_results.keys())
        data = np.column_stack([L_values] + list(tv_results.values()))
        np.savetxt(
            os.path.join(output_dir, "figure3", f"tv_vs_L_{T_label.replace('=', '_').replace('^', '')}.csv"),
            data,
            delimiter=",",
            header=header,
            comments="",
        )

    # Generate plot
    if HAS_MATPLOTLIB:
        fig = plot_figure3(
            d=d,
            T_settings={k: v for k, v in T_settings.items()},
            output_path=os.path.join(output_dir, "figure3", "figure3.pdf"),
        )
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)

    return all_results


def run_lipschitz_experiment(
    output_dir: str,
    d: int = 100,
    T: int = 1000,
):
    """
    Reproduce Example 2: Lipschitz constant comparison for GMM.
    """
    print("=" * 60)
    print("Running Lipschitz comparison experiment (Example 2)")
    print("=" * 60)

    os.makedirs(os.path.join(output_dir, "lipschitz"), exist_ok=True)

    H_values = [2, 5, 10, 50, 100]
    sigma_values = [0.01, 0.1, 1.0]

    results = run_lipschitz_comparison_gmm(
        d=d,
        H_values=H_values,
        T=T,
        sigma_values=sigma_values,
    )

    print("\nLipschitz constant comparison:")
    for key, vals in results.items():
        print(f"\n  {key}:")
        print(f"    L_nonuniform = {vals['L_nonuniform']:.4f}")
        print(f"    L_uniform_lb = {vals['L_uniform_lower_bound']:.4f}")
        print(f"    Ratio = {vals['ratio']:.2f}")

    # Save results
    with open(os.path.join(output_dir, "lipschitz", "lipschitz_comparison.json"), "w") as f:
        json.dump(
            {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
            f,
            indent=2,
        )

    # Generate plot
    if HAS_MATPLOTLIB:
        fig = plot_lipschitz_comparison(
            d=d,
            H_values=H_values,
            sigma_values=sigma_values,
            T=T,
            output_path=os.path.join(output_dir, "lipschitz", "lipschitz_comparison.pdf"),
        )
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)

    return results


def run_theorem1_verification(
    output_dir: str,
    d: int = 50,
    K: int = 10,
    T_values: Optional[list] = None,
    n_samples: int = 5000,
    seed: int = 0,
):
    """
    Verify Theorem 1 numerically.

    Checks that TV(q_K, p_{Y_K}) <= C * min{d^{3/2}, dL^{1/2}, d^{1/2}L^{3/2}} * log^4(T) / T^{3/2}
    """
    print("=" * 60)
    print("Verifying Theorem 1 numerically")
    print("=" * 60)

    if T_values is None:
        T_values = [100, 200, 500, 1000, 2000]

    os.makedirs(os.path.join(output_dir, "theorem1"), exist_ok=True)

    data = GaussianDistribution(d=d, k=d, sigma_max=5.0, seed=seed)
    score_fn = data.get_score_function()

    # Estimate L (non-uniform Lipschitz constant)
    # For Gaussian: L = 1 (from Example 1)
    L = 1.0

    print(f"\nd={d}, K={K}, L={L}")
    print(f"Theoretical complexity: min{{d, d^{{2/3}}L^{{1/3}}, d^{{1/3}}L}} = {min(d, d**(2/3)*L**(1/3), d**(1/3)*L):.2f}")

    results = []
    for T in T_values:
        N = 2 * T // K
        if N < 2:
            continue

        sampler = RandomizedMidpointSampler(
            score_fn=score_fn,
            T=T,
            K=K,
            c0=5.0,
            c1=50.0,
            seed=seed,
        )

        tau_K0 = sampler.schedule.tau_hat(K, 0)
        mu_q = np.zeros(d)
        var_q = score_fn._sigma_t_sq(tau_K0)

        result = sampler.sample(d, n_samples=n_samples)
        samples = result.samples

        mu_p = np.mean(samples, axis=0)
        var_p = np.var(samples, axis=0)

        kl = kl_divergence_gaussians_diagonal(mu_q, var_q, mu_p, var_p)
        tv_empirical = tv_from_kl(kl)
        tv_theory = our_tv_bound(d, L, T, log_factor=True)

        results.append({
            "T": T,
            "KL": kl,
            "TV_empirical": tv_empirical,
            "TV_theory": tv_theory,
            "ratio": tv_empirical / tv_theory if tv_theory > 0 else float("inf"),
        })

        print(f"  T={T:5d}: TV_empirical={tv_empirical:.4e}, TV_theory={tv_theory:.4e}, ratio={tv_empirical/tv_theory:.3f}")

    # Save results
    with open(os.path.join(output_dir, "theorem1", "verification.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_parallel_sampler_experiment(
    output_dir: str,
    d: int = 50,
    epsilon: float = 0.1,
    seed: int = 0,
):
    """
    Verify Theorem 2: parallel sampler requirements.
    """
    print("=" * 60)
    print("Running parallel sampler experiment (Theorem 2)")
    print("=" * 60)

    os.makedirs(os.path.join(output_dir, "parallel"), exist_ok=True)

    L_values = [1.0, np.sqrt(d), d, float("inf")]

    print(f"\nd={d}, epsilon={epsilon}")
    print("\nParallel sampler requirements (Theorem 2):")

    requirements = {}
    for L in L_values:
        req = compute_parallel_requirements(d, L, epsilon)
        L_str = "inf" if L == float("inf") else f"{L:.1f}"
        requirements[L_str] = req
        print(f"\n  L={L_str}:")
        print(f"    N (processors) = {req['N']}")
        print(f"    MK (rounds) = {req['MK']}")
        print(f"    T = {req['T']}")
        print(f"    epsilon_score^2 <= {req['epsilon_score_sq']:.4e}")

    # Run actual parallel sampler for small example
    data = GaussianDistribution(d=d, k=d, sigma_max=5.0, seed=seed)
    score_fn = data.get_score_function()

    L = 1.0
    req = compute_parallel_requirements(d, L, epsilon)
    N_parallel = min(req["N"], 20)  # limit for speed
    M = max(int(np.log(req["T"])), 5)
    K = 10

    print(f"\nRunning parallel sampler: N={N_parallel}, M={M}, K={K}")

    parallel_sampler = ParallelRandomizedMidpointSampler(
        score_fn=score_fn,
        N_parallel=N_parallel,
        M=M,
        K=K,
        seed=seed,
    )

    result = parallel_sampler.sample(d, n_samples=100)
    print(f"  Generated {100} samples, score_evals={result.score_evals}")
    print(f"  Parallel rounds: {result.parallel_rounds}")

    with open(os.path.join(output_dir, "parallel", "requirements.json"), "w") as f:
        json.dump(
            {k: {kk: float(vv) for kk, vv in v.items()} for k, v in requirements.items()},
            f,
            indent=2,
        )

    return requirements


def print_theoretical_summary(d: int = 100):
    """Print a summary of theoretical results from the paper."""
    print("\n" + "=" * 70)
    print("THEORETICAL SUMMARY")
    print("=" * 70)

    print(f"\nData dimension: d = {d}")
    print("\nMain result (Theorem 1):")
    print("  TV(q_K, p_{Y_K}) <= C * min{d^{3/2}, d*L^{1/2}, d^{1/2}*L^{3/2}} * log^4(T) / T^{3/2}")
    print("                     + C * epsilon_score * log^{1/2}(T)")

    print("\nIteration complexity to achieve TV <= epsilon:")
    print("  T >= min{d, d^{2/3}*L^{1/3}, d^{1/3}*L} * epsilon^{-2/3} * log^{8/3}(T)")

    print("\nComparison with prior works (epsilon = 0.1):")
    epsilon = 0.1
    for L in [1.0, np.sqrt(d), d]:
        print(f"\n  L = {L:.2f}:")
        complexities = all_iteration_complexities(d, L, epsilon, log_factor=False)
        for method, T in sorted(complexities.items(), key=lambda x: x[1]):
            print(f"    {method:30s}: T = {T:.2e}")

    print("\nImprovement over Li & Jiao (2024):")
    for L in [np.sqrt(d), d, d**2]:
        factor = improvement_factor_over_li_jiao(d, L)
        print(f"  L = {L:.1f}: improvement factor = {factor:.2f}x")

    print("\nNon-uniform Lipschitz for GMM (Example 2):")
    for H in [2, 10, 100]:
        T = 1000
        info = non_uniform_vs_uniform_lipschitz_gmm(H, T, d)
        print(f"  H={H}: {info['L_nonuniform_formula']}")
        print(f"         {info['L_uniform_description']}")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce experiments from 'Instance-dependent Convergence Theory for Diffusion Models'"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "figure1", "figure2", "figure3", "lipschitz", "theorem1", "parallel", "summary"],
        help="Which experiment to run",
    )
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--d", type=int, default=100, help="Data dimension")
    parser.add_argument("--K", type=int, default=10, help="Number of rounds")
    parser.add_argument("--n_samples", type=int, default=3000, help="Number of samples")
    parser.add_argument("--n_trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--T_values",
        type=int,
        nargs="+",
        default=None,
        help="List of T values for convergence experiments",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.experiment in ("all", "summary"):
        print_theoretical_summary(d=args.d)

    if args.experiment in ("all", "figure2"):
        run_figure2_experiment(
            output_dir=args.output_dir,
            T_values=args.T_values,
            n_samples=args.n_samples,
            n_trials=args.n_trials,
            seed=args.seed,
        )

    if args.experiment in ("all", "figure1"):
        run_figure1_experiment(
            output_dir=args.output_dir,
            d=args.d,
        )

    if args.experiment in ("all", "figure3"):
        run_figure3_experiment(
            output_dir=args.output_dir,
            d=args.d,
        )

    if args.experiment in ("all", "lipschitz"):
        run_lipschitz_experiment(
            output_dir=args.output_dir,
            d=args.d,
        )

    if args.experiment in ("all", "theorem1"):
        run_theorem1_verification(
            output_dir=args.output_dir,
            d=min(args.d, 50),
            K=args.K,
            T_values=args.T_values,
            n_samples=args.n_samples,
            seed=args.seed,
        )

    if args.experiment in ("all", "parallel"):
        run_parallel_sampler_experiment(
            output_dir=args.output_dir,
            d=min(args.d, 50),
            seed=args.seed,
        )

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
