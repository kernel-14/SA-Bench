r"""Numerical experiment from Appendix A of:
"Instance-dependent Convergence Theory for Diffusion Models"
by Yuchen Jiao and Gen Li (2025).

Reproduces Figure 2: sampling error vs. number of iterations T
for different data dimensions d and numbers of non-zero components k.

The target distribution is a d-dimensional Gaussian with diagonal covariance.
First k diagonal entries ~ Uniform[0, 10]; remaining d-k = 0.

We use K = 10 rounds, exact score functions, and measure KL divergence
between the sampler output Y_K and the target distribution q_K.
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ExperimentConfig, DataConfig
from score_function import GaussianScoreFunction
from sampler import DiffusionSampler


def run_single_experiment(
    d: int,
    k: int,
    T: int,
    K: int,
    sigma_max: float,
    num_mc_samples: int,
    seed: int,
) -> Tuple[float, float]:
    """Run one sampler configuration and return KL divergence.

    Args:
        d: Data dimension.
        k: Number of non-zero variance components.
        T: Total iteration budget.
        K: Number of rounds.
        sigma_max: Maximum variance.
        num_mc_samples: Number of Monte Carlo samples for KL estimation.
        seed: Random seed.

    Returns:
        kl: KL divergence estimate.
        runtime: Wall-clock time in seconds.
    """
    torch.manual_seed(seed)
    score_fn = GaussianScoreFunction(d=d, k=k, sigma_max=sigma_max, seed=seed)

    c_0 = 15.0
    c_1 = 75.0

    sampler = DiffusionSampler(
        score_fn=score_fn,
        T=T,
        K=K,
        c_0=c_0,
        c_1=c_1,
    )

    t0 = time.time()
    kl = sampler.compute_kl_divergence(num_samples=num_mc_samples)
    runtime = time.time() - t0

    return kl, runtime


def compute_theoretical_rate(T: int, d: int) -> float:
    """Compute the theoretical rate O(log^4(T) / T^3) for KL divergence.

    This gives a reference slope for the convergence plot.
    """
    log_T = np.log(T)
    return (log_T ** 4) / (T ** 3)


def run_experiment_suite(
    config: ExperimentConfig,
    output_dir: str = "results",
) -> Dict:
    """Run the full suite of experiments matching Figure 2.

    Returns:
        results: Dictionary with all KL divergence values.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for d, k in zip(config.d_values, config.k_values):
        print(f"\n{'='*60}")
        print(f"Experiment: d={d}, k={k}")
        print(f"{'='*60}")

        kl_values = []
        t_values = sorted(config.T_values)
        runtimes = []

        for T in t_values:
            print(f"  T={T:5d} ... ", end="", flush=True)
            kl, rt = run_single_experiment(
                d=d,
                k=k,
                T=T,
                K=config.K,
                sigma_max=10.0,
                num_mc_samples=config.num_mc_samples,
                seed=42,
            )
            kl_values.append(kl)
            runtimes.append(rt)
            print(f"KL={kl:.6e}, time={rt:.1f}s")

        results[(d, k)] = {
            "T": t_values,
            "kl": kl_values,
            "runtime": runtimes,
        }

        plot_single_figure(
            d=d,
            k=k,
            t_values=t_values,
            kl_values=kl_values,
            output_dir=output_dir,
        )

    plot_combined_figure(results, output_dir)
    return results


def plot_single_figure(
    d: int,
    k: int,
    t_values: List[int],
    kl_values: List[float],
    output_dir: str,
):
    """Plot KL divergence vs T for one (d, k) configuration.

    Matches Figure 2 in the paper: blue line = empirical, black line = theory.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    t_arr = np.array(t_values, dtype=float)
    kl_arr = np.array(kl_values)

    # Empirical results (blue line).
    ax.loglog(t_arr, kl_arr, "b-o", linewidth=2, markersize=6, label="Empirical")

    # Theoretical rate O(log^4(T) / T^3) for KL divergence.
    theory = np.array([compute_theoretical_rate(int(t), d) for t in t_values])
    # Rescale theory to match empirical at the largest T.
    scale = kl_arr[-1] / theory[-1]
    ax.loglog(
        t_arr,
        scale * theory,
        "k--",
        linewidth=2,
        label=r"$O(\log^4 T / T^3)$",
    )

    ax.set_xlabel("T (number of iterations)")
    ax.set_ylabel("KL divergence")
    ax.set_title(rf"$d={d}$, $k={k}$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"convergence_d{d}_k{k}.pdf")
    plt.savefig(path, dpi=150)
    plt.savefig(path.replace(".pdf", ".png"), dpi=150)
    plt.close()
    print(f"  Figure saved to {path}")


def plot_combined_figure(results: Dict, output_dir: str):
    """Create a combined figure with all configurations."""
    n_configs = len(results)
    cols = min(3, n_configs)
    rows = (n_configs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_configs == 1:
        axes = [axes]
    axes = np.atleast_1d(axes).flatten()

    for idx, ((d, k), data) in enumerate(sorted(results.items())):
        ax = axes[idx]
        t_arr = np.array(data["T"], dtype=float)
        kl_arr = np.array(data["kl"])

        ax.loglog(t_arr, kl_arr, "b-o", linewidth=2, markersize=5, label="Empirical")

        theory = np.array([compute_theoretical_rate(int(t), d) for t in t_arr])
        scale = kl_arr[-1] / theory[-1]
        ax.loglog(t_arr, scale * theory, "k--", linewidth=2, label=r"$O(\log^4 T / T^3)$")

        ax.set_xlabel("T")
        ax.set_ylabel("KL divergence")
        ax.set_title(rf"$d={d}$, $k={k}$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n_configs, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Sampling error of proposed sampler (cf. Figure 2 in paper)",
        fontsize=14,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "convergence_combined.pdf")
    plt.savefig(path, dpi=150)
    plt.savefig(path.replace(".pdf", ".png"), dpi=150)
    plt.close()
    print(f"\nCombined figure saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce numerical experiments from the diffusion convergence paper."
    )
    parser.add_argument(
        "--d_values",
        type=int,
        nargs="+",
        default=[10, 100, 500],
        help="Data dimensions to test.",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[10, 10, 100],
        help="Number of non-zero variance components.",
    )
    parser.add_argument(
        "--T_values",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 4000],
        help="Total iteration values to test.",
    )
    parser.add_argument(
        "--K", type=int, default=10, help="Number of rounds."
    )
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=10000,
        help="Monte Carlo samples for KL estimation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = ExperimentConfig(
        T_values=tuple(args.T_values),
        num_mc_samples=args.num_mc_samples,
        d_values=tuple(args.d_values),
        k_values=tuple(args.k_values),
        K=args.K,
    )

    print("=" * 60)
    print("Diffusion Model Convergence Experiment")
    print("=" * 60)
    print(f"Configurations: d={args.d_values}, k={args.k_values}")
    print(f"T values: {args.T_values}")
    print(f"K = {args.K}, MC samples = {args.num_mc_samples}")
    print("=" * 60)

    results = run_experiment_suite(config, output_dir=args.output_dir)

    print("\n" + "=" * 60)
    print("Experiment complete. Results summary:")
    print("=" * 60)
    for (d, k), data in sorted(results.items()):
        print(f"\nd={d}, k={k}:")
        for t, kl in zip(data["T"], data["kl"]):
            print(f"  T={t:5d}: KL={kl:.6e}")


if __name__ == "__main__":
    main()
