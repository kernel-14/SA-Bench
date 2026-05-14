## main.py
"""Entry point for reproducing all experiments and figures from
"Instance-dependent Convergence Theory for Diffusion Models".

This script orchestrates the full reproduction pipeline:
  - Figure 2 (Appendix A): Empirical KL divergence vs. T for three
    Gaussian target configurations (d=10/k=10, d=100/k=10, d=500/k=100).
  - Figure 1 (Section 1.1): Iteration complexity comparison across methods.
  - Figure 3 (Appendix B): TV distance vs. L for fixed T values.

Usage
-----
    python main.py                          # run all experiments and figures
    python main.py --experiment figure2     # only Figure 2
    python main.py --experiment figure1     # only Figure 1
    python main.py --experiment figure3     # only Figure 3
    python main.py --device cuda            # use GPU if available
    python main.py --c0 2.0 --c1 10.0      # override schedule constants

All hyperparameters default to values from config.yaml.
"""

import argparse
import math
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import Config
from experiments import Experiments
from metrics import Metrics
from plots import Plots


# ---------------------------------------------------------------------------
# Helper: CUDA availability check
# ---------------------------------------------------------------------------

def _resolve_device(requested_device: str) -> str:
    """Resolve the computation device, falling back to CPU if CUDA unavailable.

    Args:
        requested_device: Device string requested by the user ('cpu' or 'cuda').

    Returns:
        Resolved device string ('cpu' or 'cuda').
    """
    if requested_device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                warnings.warn(
                    "CUDA requested but not available. Falling back to CPU.",
                    UserWarning,
                    stacklevel=2,
                )
                return "cpu"
        except ImportError:
            warnings.warn(
                "PyTorch not found. Falling back to CPU.",
                UserWarning,
                stacklevel=2,
            )
            return "cpu"
    return requested_device


# ---------------------------------------------------------------------------
# Helper: build Config objects from parsed args + config.yaml defaults
# ---------------------------------------------------------------------------

def _build_configs(
    c0: float,
    c1: float,
    seed: int,
    device: str,
    figure_dir: str,
    K: int = 10,
    T_values: Optional[List[int]] = None,
) -> List[Config]:
    """Build the three paper configurations from Figure 2 (Appendix A).

    The three configurations are:
        (a) d=10,  k_active=10,  label='fig2a'
        (b) d=100, k_active=10,  label='fig2b'
        (c) d=500, k_active=100, label='fig2c'

    All share the same K, T_values, c0, c1, seed, device, figure_dir.

    Args:
        c0: Schedule constant (config.yaml: sampler.c0, default 2.0).
        c1: Schedule constant (config.yaml: sampler.c1, default 10.0).
        seed: Random seed (config.yaml: experiment.seed, default 42).
        device: Computation device (config.yaml: experiment.device, default 'cpu').
        figure_dir: Output directory (config.yaml: experiment.figure_dir, default 'figures').
        K: Number of sampler rounds (config.yaml: sampler.K, default 10).
        T_values: List of T values to sweep. If None, uses the default
            [50, 100, 200, 500, 1000, 2000, 5000] from config.yaml: T_values.

    Returns:
        List of three Config objects in order: fig2a, fig2b, fig2c.
    """
    if T_values is None:
        # Default T_values from config.yaml: T_values
        T_values = [50, 100, 200, 500, 1000, 2000, 5000]

    return Config.default_configs(
        K=K,
        T_values=T_values,
        c0=c0,
        c1=c1,
        seed=seed,
        device=device,
        figure_dir=figure_dir,
    )


# ---------------------------------------------------------------------------
# Helper: print summary statistics after run_all()
# ---------------------------------------------------------------------------

def _print_summary(
    results: Dict[str, Any],
    configs: List[Config],
) -> None:
    """Print a formatted summary of convergence results to stdout.

    For each configuration, reports:
      - Min and max KL divergence across T values.
      - Fitted constant C in the theoretical rate C * log^4(T) / T^3.
      - Log-log slope of KL vs. T (should approach -3 for large T).

    Args:
        results: Dict mapping config label -> result dict. Each result dict
            must have keys 'T_values' (List[int]) and 'kl_values' (List[float]).
        configs: List of Config objects in the same order as results.
    """
    print("\n" + "=" * 78)
    print("CONVERGENCE SUMMARY")
    print("=" * 78)

    # Header
    header: str = (
        f"{'Config':<18} | {'T_min':>6} | {'T_max':>6} | "
        f"{'KL_min':>12} | {'KL_max':>12} | {'C_fit':>12} | {'Slope':>7}"
    )
    print(header)
    print("-" * 78)

    for cfg in configs:
        key: str = cfg.label if cfg.label else f"d{cfg.d}_k{cfg.k_active}"

        if key not in results:
            print(f"  [{key}]: no results found.")
            continue

        result: Dict[str, Any] = results[key]
        T_values_raw: List[int] = result["T_values"]
        kl_values_raw: List[float] = result["kl_values"]

        # Filter valid (positive, finite, non-NaN) data points
        T_valid: List[int] = []
        kl_valid: List[float] = []
        for T_val, kl_val in zip(T_values_raw, kl_values_raw):
            if (
                not math.isnan(kl_val)
                and not math.isinf(kl_val)
                and kl_val > 0.0
                and T_val >= 2
            ):
                T_valid.append(T_val)
                kl_valid.append(kl_val)

        if len(T_valid) == 0:
            config_str: str = f"d={cfg.d}, k={cfg.k_active}"
            print(f"  {config_str:<16} | {'N/A':>6} | {'N/A':>6} | "
                  f"{'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>7}")
            continue

        T_min: int = min(T_valid)
        T_max: int = max(T_valid)
        kl_min: float = min(kl_valid)
        kl_max: float = max(kl_valid)

        # Fit theoretical rate C * log^4(T) / T^3
        C_fit: float = float("nan")
        if len(kl_valid) >= 2:
            try:
                C_fit, _ = Metrics.fit_theoretical_rate(T_valid, kl_valid)
            except Exception as exc:
                warnings.warn(
                    f"Could not fit theoretical rate for {key}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )

        # Log-log slope via linear regression: log(KL) ~ slope * log(T) + const
        # Theoretical slope is -3 (ignoring log^4(T) correction)
        slope: float = float("nan")
        if len(kl_valid) >= 3:
            try:
                log_T_arr: np.ndarray = np.log(
                    np.array(T_valid, dtype=np.float64)
                )
                log_kl_arr: np.ndarray = np.log(
                    np.array(kl_valid, dtype=np.float64)
                )
                coeffs: np.ndarray = np.polyfit(log_T_arr, log_kl_arr, 1)
                slope = float(coeffs[0])
            except Exception as exc:
                warnings.warn(
                    f"Could not compute log-log slope for {key}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )

        # Format row
        config_str = f"d={cfg.d}, k={cfg.k_active}"
        kl_min_str: str = f"{kl_min:.4e}" if not math.isnan(kl_min) else "NaN"
        kl_max_str: str = f"{kl_max:.4e}" if not math.isnan(kl_max) else "NaN"
        C_fit_str: str = f"{C_fit:.4e}" if not math.isnan(C_fit) else "NaN"
        slope_str: str = f"{slope:.3f}" if not math.isnan(slope) else "NaN"

        row: str = (
            f"{config_str:<18} | {T_min:>6} | {T_max:>6} | "
            f"{kl_min_str:>12} | {kl_max_str:>12} | {C_fit_str:>12} | {slope_str:>7}"
        )
        print(row)

    print("=" * 78)
    print(
        "Note: Slope should approach -3.0 for large T "
        "(theoretical rate O(log^4(T)/T^3) in KL)."
    )
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_argument_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    All defaults are taken from config.yaml values.

    Returns:
        Configured ArgumentParser instance.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Reproduce experiments from "
            "'Instance-dependent Convergence Theory for Diffusion Models'. "
            "Generates Figures 1, 2, and 3 from the paper."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Which experiment(s) to run
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "figure1", "figure2", "figure3"],
        help=(
            "Which experiment to run. "
            "'all' runs all three figures. "
            "'figure2' runs the convergence sweep (Appendix A). "
            "'figure1' generates the iteration complexity comparison (Section 1.1). "
            "'figure3' generates the TV distance comparison (Appendix B)."
        ),
    )

    # Device (config.yaml: experiment.device)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device. Falls back to CPU if CUDA is unavailable.",
    )

    # Output directory (config.yaml: experiment.figure_dir)
    parser.add_argument(
        "--figure_dir",
        type=str,
        default="figures",
        help="Output directory for saved figures (PDF and PNG).",
    )

    # Random seed (config.yaml: experiment.seed)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for constructing the target covariance diagonal "
            "(k_active values drawn from Unif(0, 10))."
        ),
    )

    # Schedule constant c0 (config.yaml: sampler.c0)
    parser.add_argument(
        "--c0",
        type=float,
        default=2.0,
        help=(
            "Schedule constant c0. Controls alpha_hat[T+1] = 1/T^c0. "
            "Must be positive. Paper: 'sufficiently large'."
        ),
    )

    # Schedule constant c1 (config.yaml: sampler.c1)
    parser.add_argument(
        "--c1",
        type=float,
        default=10.0,
        help=(
            "Schedule constant c1. Controls step size in the schedule recursion. "
            "Must be positive and c1/c0 should be sufficiently large. "
            "Paper: 'sufficiently large with c1/c0 sufficiently large'."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point: parse arguments, build configs, run experiments, plot.

    Execution flow:
        1. Parse CLI arguments.
        2. Resolve device (fall back to CPU if CUDA unavailable).
        3. Create output directory.
        4. Build three Config objects (fig2a, fig2b, fig2c).
        5. Instantiate Experiments and Plots.
        6. Dispatch to figure2 / figure1 / figure3 based on --experiment.
        7. Print summary statistics if figure2 was run.
    """
    # --- Step 1: Parse CLI arguments ---
    parser: argparse.ArgumentParser = _build_argument_parser()
    args: argparse.Namespace = parser.parse_args()

    # --- Step 2: Resolve device ---
    device: str = _resolve_device(args.device)

    # --- Step 3: Create output directory ---
    os.makedirs(args.figure_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(args.figure_dir)}")

    # --- Step 4: Build Config objects ---
    # T_values from config.yaml: T_values
    T_values: List[int] = [50, 100, 200, 500, 1000, 2000, 5000]

    # K from config.yaml: sampler.K
    K: int = 10

    configs: List[Config] = _build_configs(
        c0=args.c0,
        c1=args.c1,
        seed=args.seed,
        device=device,
        figure_dir=args.figure_dir,
        K=K,
        T_values=T_values,
    )

    # Print configuration summary
    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"  Experiment mode        : {args.experiment}")
    print(f"  Device                 : {device}")
    print(f"  Figure output dir      : {args.figure_dir}")
    print(f"  Random seed            : {args.seed}")
    print(f"  Schedule c0            : {args.c0}")
    print(f"  Schedule c1            : {args.c1}")
    print(f"  c1/c0 ratio            : {args.c1 / args.c0:.2f}")
    print(f"  Sampler rounds K       : {K}")
    print(f"  T values               : {T_values}")
    print(f"  Configurations         : {[cfg.label for cfg in configs]}")
    print("=" * 60 + "\n")

    # --- Step 5: Instantiate Experiments and Plots ---
    experiments: Experiments = Experiments(configs)
    plots: Plots = Plots(figure_dir=args.figure_dir)

    # results will be populated if figure2 is run
    results: Optional[Dict[str, Any]] = None

    # --- Step 6: Dispatch based on --experiment ---

    # ---- Figure 2: Empirical convergence (Appendix A) ----
    if args.experiment in ("figure2", "all"):
        print("\n" + "=" * 60)
        print("RUNNING FIGURE 2: Empirical Convergence Sweep")
        print("=" * 60)

        results = experiments.run_all()

        print("\nGenerating Figure 2...")
        plots.plot_figure2(results=results, configs=configs)
        print("Figure 2 saved.")

        # Print summary statistics
        _print_summary(results=results, configs=configs)

    # ---- Figure 1: Iteration complexity comparison (Section 1.1) ----
    if args.experiment in ("figure1", "all"):
        print("\n" + "=" * 60)
        print("GENERATING FIGURE 1: Iteration Complexity Comparison")
        print("=" * 60)

        # Parameters from config.yaml: figure1
        d_fig1: int = 100                    # config.yaml: figure1.d
        eps_fixed: float = 1.0               # config.yaml: figure1.eps_fixed
        L_min: float = 1.0                   # config.yaml: figure1.L_min
        L_max: float = 10000.0               # config.yaml: figure1.L_max
        L_num_points: int = 200              # config.yaml: figure1.L_num_points
        eps_min: float = 0.001               # config.yaml: figure1.eps_min
        eps_max: float = 0.5                 # config.yaml: figure1.eps_max
        eps_num_points: int = 200            # config.yaml: figure1.eps_num_points

        # L range for left subplot: [L_min, L_max] log-spaced
        L_values: np.ndarray = np.logspace(
            math.log10(L_min),
            math.log10(L_max),
            L_num_points,
        )

        # eps range for right subplot: [eps_min, eps_max] log-spaced
        eps_values: np.ndarray = np.logspace(
            math.log10(eps_min),
            math.log10(eps_max),
            eps_num_points,
        )

        print(f"  d = {d_fig1}")
        print(f"  L range: [{L_min:.1f}, {L_max:.1f}] ({L_num_points} points)")
        print(f"  eps range: [{eps_min:.4f}, {eps_max:.4f}] ({eps_num_points} points)")

        # Figure 1 left: T vs. L for eps = O(1)
        print("\nGenerating Figure 1 (left): T vs. L...")
        plots.plot_figure1_left(
            d=d_fig1,
            L_values=L_values,
            eps=eps_fixed,
        )
        print("Figure 1 (left) saved.")

        # Figure 1 right: T vs. eps for L = infinity
        print("\nGenerating Figure 1 (right): T vs. eps (L = infinity)...")
        plots.plot_figure1_right(
            d=d_fig1,
            eps_values=eps_values,
            L_inf=True,
        )
        print("Figure 1 (right) saved.")

    # ---- Figure 3: TV distance vs. L for fixed T (Appendix B) ----
    if args.experiment in ("figure3", "all"):
        print("\n" + "=" * 60)
        print("GENERATING FIGURE 3: TV Distance vs. L for Fixed T")
        print("=" * 60)

        # Parameters from config.yaml: figure3
        d_fig3: int = 100                    # config.yaml: figure3.d
        L_min_fig3: float = 1.0              # config.yaml: figure3.L_min
        L_max_fig3: float = 10000.0          # config.yaml: figure3.L_max
        L_num_fig3: int = 200                # config.yaml: figure3.L_num_points

        # T cases from config.yaml: figure3.T_cases
        # T_multiplier=1.0, T_exponent=1.0 -> T = d^1.0
        # T_multiplier=1.0, T_exponent=1.5 -> T = d^1.5
        # T_multiplier=1.0, T_exponent=2.0 -> T = d^2.0
        T_cases: List[Tuple[str, float]] = [
            ("T=O(d)",      float(d_fig3 ** 1.0)),   # config.yaml: figure3.T_cases[0]
            ("T=O(d^1.5)",  float(d_fig3 ** 1.5)),   # config.yaml: figure3.T_cases[1]
            ("T=O(d^2)",    float(d_fig3 ** 2.0)),   # config.yaml: figure3.T_cases[2]
        ]

        # L range for figure 3
        L_values_fig3: np.ndarray = np.logspace(
            math.log10(L_min_fig3),
            math.log10(L_max_fig3),
            L_num_fig3,
        )

        print(f"  d = {d_fig3}")
        print(f"  T cases: {[(label, T_val) for label, T_val in T_cases]}")
        print(f"  L range: [{L_min_fig3:.1f}, {L_max_fig3:.1f}] ({L_num_fig3} points)")

        print("\nGenerating Figure 3...")
        plots.plot_figure3(
            d=d_fig3,
            T_cases=T_cases,
            L_values=L_values_fig3,
        )
        print("Figure 3 saved.")

    # --- Final message ---
    print("\n" + "=" * 60)
    print("ALL REQUESTED EXPERIMENTS COMPLETE.")
    print(f"Figures saved to: {os.path.abspath(args.figure_dir)}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
