"""
Simulation experiments from Section 4 and Appendix C of the paper.

Runs all three experiments and saves figures to the output directory.

Usage:
    python experiments.py [--output-dir figures] [--no-complexity]

Experiments:
  1. Convergence with different (S, A) sizes  →  figure1a.pdf + figure1.pdf
  2. Convergence with different reward variances  →  figure1b.pdf
  3. Convergence with different transition kernels  →  figure2.pdf
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np

from complexity import compute_all_complexity, compute_convergence_rate
from config import CONFIG
from mdp import AverageRewardMDP
from mdp_factory import (
    make_mdp_varying_size,
    make_mdps_varying_reward,
    make_mdps_varying_kernel,
)
from plot import plot_figure1, plot_figure2, plot_rewards, plot_suboptimality
from policy_gradient import PPGResult, compute_step_size, projected_policy_gradient
from utils import uniform_policy


def run_ppg_on_mdp(
    mdp: AverageRewardMDP,
    n_iterations: int,
    eta: float,
    seed: int = 0,
    compute_bound: bool = False,
    n_complexity_samples: int = 50,
) -> Tuple[PPGResult, float, float]:
    """
    Run PPG on a single MDP and return the result, optimal reward, and step size.

    Parameters
    ----------
    mdp : AverageRewardMDP
    n_iterations : int
    eta : float
        Step size.  If <= 0, it is computed from L_2^Π.
    seed : int
        Seed for initial policy.
    compute_bound : bool
        Whether to compute the theoretical convergence bound (requires
        computing complexity constants, which is expensive for large MDPs).
    n_complexity_samples : int
        Number of policy samples for complexity constant estimation.

    Returns
    -------
    result : PPGResult
    rho_star : float
    eta_used : float
    """
    # Compute optimal policy and reward
    pi_star, rho_star = mdp.optimal_policy()

    # Compute step size from complexity constants if requested
    nu = None
    if eta <= 0 or compute_bound:
        complexity = compute_all_complexity(
            mdp, pi_star, n_samples=n_complexity_samples, seed=seed
        )
        if eta <= 0:
            eta = compute_step_size(complexity.L2, CONFIG.step_size_safety_factor)
        if compute_bound:
            c, nu = compute_convergence_rate(complexity.C_PL, mdp.S, complexity.L2)
            print(
                f"  L2={complexity.L2:.4f}, C_PL={complexity.C_PL:.4f}, "
                f"c={c:.4f}, nu={nu:.6f}, eta={eta:.6f}"
            )

    # Initialise with uniform policy
    pi_init = uniform_policy(mdp.S, mdp.A)

    result = projected_policy_gradient(
        mdp=mdp,
        pi_init=pi_init,
        eta=eta,
        n_iterations=n_iterations,
        rho_star=rho_star,
        nu=nu,
    )
    return result, rho_star, eta


# ---------------------------------------------------------------------------
# Experiment 1: Varying (S, A) sizes
# ---------------------------------------------------------------------------

def run_experiment1(output_dir: str) -> None:
    """
    Experiment 1 (Appendix C.1): Convergence for different (S, A) sizes.

    MDPs: (3,3), (9,9), (81,81)
    Iterations: 2000
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Varying (S, A) sizes")
    print("=" * 60)

    cfg = CONFIG.exp1
    results: Dict[str, PPGResult] = {}
    rho_stars: Dict[str, float] = {}

    for S, A in cfg.sizes:
        key = f"({S},{A})"
        print(f"\n  Running (S={S}, A={A}) ...")
        t0 = time.time()

        mdp = make_mdp_varying_size(S, A)
        result, rho_star, eta = run_ppg_on_mdp(
            mdp,
            n_iterations=cfg.n_iterations,
            eta=cfg.eta,
            seed=cfg.seed,
        )
        results[key] = result
        rho_stars[key] = rho_star

        print(
            f"  Done in {time.time()-t0:.1f}s | "
            f"ρ*={rho_star:.4f} | "
            f"final ρ={result.rewards[-1]:.4f} | "
            f"η={eta:.4f}"
        )

    plot_rewards(
        results,
        title="Figure 1(a): Varying $(|S|, |A|)$",
        output_path=os.path.join(output_dir, "figure1a.pdf"),
        rho_stars=rho_stars,
    )
    plot_suboptimality(
        results,
        title="Suboptimality: Varying $(|S|, |A|)$",
        output_path=os.path.join(output_dir, "figure1a_subopt.pdf"),
        plot_bounds=False,
    )
    return results, rho_stars


# ---------------------------------------------------------------------------
# Experiment 2: Varying reward variance
# ---------------------------------------------------------------------------

def run_experiment2(output_dir: str) -> None:
    """
    Experiment 2 (Appendix C.2): Convergence for different reward variances.

    MDP: S=16, A=16, fixed random transition kernel
    Reward variances: none, low, high, max
    Iterations: 2000
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Varying reward variance")
    print("=" * 60)

    cfg = CONFIG.exp2
    mdps = make_mdps_varying_reward(S=cfg.S, A=cfg.A, seed=cfg.seed)

    results: Dict[str, PPGResult] = {}
    rho_stars: Dict[str, float] = {}

    for key, mdp in mdps.items():
        print(f"\n  Running {key} ...")
        t0 = time.time()

        result, rho_star, eta = run_ppg_on_mdp(
            mdp,
            n_iterations=cfg.n_iterations,
            eta=cfg.eta,
            seed=cfg.seed,
        )
        results[key] = result
        rho_stars[key] = rho_star

        print(
            f"  Done in {time.time()-t0:.1f}s | "
            f"ρ*={rho_star:.4f} | "
            f"final ρ={result.rewards[-1]:.4f} | "
            f"η={eta:.4f}"
        )

    plot_rewards(
        results,
        title="Figure 1(b): Varying Reward Variance",
        output_path=os.path.join(output_dir, "figure1b.pdf"),
        rho_stars=rho_stars,
    )
    plot_suboptimality(
        results,
        title="Suboptimality: Varying Reward Variance",
        output_path=os.path.join(output_dir, "figure1b_subopt.pdf"),
        plot_bounds=False,
    )
    return results, rho_stars


# ---------------------------------------------------------------------------
# Experiment 3: Varying transition kernels
# ---------------------------------------------------------------------------

def run_experiment3(output_dir: str) -> None:
    """
    Experiment 3 (Appendix C.3): Convergence for different transition kernels.

    MDP: S=16, A=16, high-variance reward
    Kernels: uniform, non-uniform, deterministic
    Iterations: 3000
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Varying transition kernels")
    print("=" * 60)

    cfg = CONFIG.exp3
    mdps = make_mdps_varying_kernel(S=cfg.S, A=cfg.A, seed=cfg.seed)

    results: Dict[str, PPGResult] = {}
    rho_stars: Dict[str, float] = {}

    for key, mdp in mdps.items():
        print(f"\n  Running {key} ...")
        t0 = time.time()

        result, rho_star, eta = run_ppg_on_mdp(
            mdp,
            n_iterations=cfg.n_iterations,
            eta=cfg.eta,
            seed=cfg.seed,
        )
        results[key] = result
        rho_stars[key] = rho_star

        print(
            f"  Done in {time.time()-t0:.1f}s | "
            f"ρ*={rho_star:.4f} | "
            f"final ρ={result.rewards[-1]:.4f} | "
            f"η={eta:.4f}"
        )

    plot_figure2(
        results,
        output_path=os.path.join(output_dir, "figure2.pdf"),
        rho_stars=rho_stars,
    )
    plot_suboptimality(
        results,
        title=r"Suboptimality: Varying $C_p$",
        output_path=os.path.join(output_dir, "figure2_subopt.pdf"),
        plot_bounds=False,
    )
    return results, rho_stars


# ---------------------------------------------------------------------------
# Combined Figure 1
# ---------------------------------------------------------------------------

def run_all_experiments(output_dir: str) -> None:
    """Run all three experiments and produce all figures."""
    os.makedirs(output_dir, exist_ok=True)

    results1, rho_stars1 = run_experiment1(output_dir)
    results2, rho_stars2 = run_experiment2(output_dir)
    results3, rho_stars3 = run_experiment3(output_dir)

    # Combined Figure 1 (1a + 1b side by side)
    plot_figure1(
        results_exp1=results1,
        results_exp2=results2,
        output_path=os.path.join(output_dir, "figure1.pdf"),
        rho_stars_exp1=rho_stars1,
        rho_stars_exp2=rho_stars2,
    )

    print("\nAll experiments complete.")


# ---------------------------------------------------------------------------
# Complexity analysis utility
# ---------------------------------------------------------------------------

def print_complexity_analysis() -> None:
    """
    Print MDP complexity constants for all experimental MDPs.
    Useful for verifying that C_p, C_r, C_m values match the paper's claims.
    """
    print("\n" + "=" * 60)
    print("MDP Complexity Analysis")
    print("=" * 60)

    # Experiment 1 MDPs
    for S, A in [(3, 3), (9, 9)]:  # skip 81x81 for speed
        mdp = make_mdp_varying_size(S, A)
        pi_star, rho_star = mdp.optimal_policy()
        complexity = compute_all_complexity(mdp, pi_star, n_samples=50, seed=0)
        print(f"\n(S={S}, A={A}):")
        print(complexity)

    # Experiment 2 MDPs
    mdps2 = make_mdps_varying_reward(S=16, A=16, seed=42)
    for key, mdp in list(mdps2.items())[:2]:
        pi_star, rho_star = mdp.optimal_policy()
        complexity = compute_all_complexity(mdp, pi_star, n_samples=50, seed=0)
        print(f"\nReward variance={key}:")
        print(complexity)

    # Experiment 3 MDPs
    mdps3 = make_mdps_varying_kernel(S=16, A=16, seed=42)
    for key, mdp in mdps3.items():
        pi_star, rho_star = mdp.optimal_policy()
        complexity = compute_all_complexity(mdp, pi_star, n_samples=50, seed=0)
        print(f"\nKernel={key}:")
        print(complexity)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce experiments from 'Global Convergence of Policy "
                    "Gradient in Average Reward MDPs'"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=CONFIG.output_dir,
        help="Directory to save figures (default: figures/)",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a single experiment (1, 2, or 3). Default: run all.",
    )
    parser.add_argument(
        "--complexity",
        action="store_true",
        help="Print MDP complexity constants before running experiments.",
    )
    args = parser.parse_args()

    if args.complexity:
        print_complexity_analysis()

    if args.experiment == 1:
        os.makedirs(args.output_dir, exist_ok=True)
        run_experiment1(args.output_dir)
    elif args.experiment == 2:
        os.makedirs(args.output_dir, exist_ok=True)
        run_experiment2(args.output_dir)
    elif args.experiment == 3:
        os.makedirs(args.output_dir, exist_ok=True)
        run_experiment3(args.output_dir)
    else:
        run_all_experiments(args.output_dir)


if __name__ == "__main__":
    main()
