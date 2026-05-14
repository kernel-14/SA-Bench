## main.py
"""Entry point for reproducing experiments from:
    Murthy et al., "Global Convergence of Policy Gradient in Average Reward
    MDPs", ICLR 2024.

This module orchestrates all three experiments described in Section 4 and
Appendices C.1–C.3 of the paper, producing Figures 1(a), 1(b), and 2.

Usage:
    python main.py                    # Run all three experiments
    python main.py --exp 1            # Run only Experiment 1 (Figure 1a)
    python main.py --exp 2            # Run only Experiment 2 (Figure 1b)
    python main.py --exp 3            # Run only Experiment 3 (Figure 2)
    python main.py --seed 123         # Override random seed
    python main.py --output_dir out/  # Override output directory
    python main.py --eta 0.01         # Override step size multiplier

Experiments:
    1. Figure 1(a): Convergence vs state/action space size.
       MDPs: (S,A) in {(3,3), (9,9), (81,81)}, 2000 iterations.
    2. Figure 1(b): Convergence vs reward variance (C_r).
       MDP: (16,16), four reward variance levels, 2000 iterations.
    3. Figure 2: Convergence vs transition kernel type (C_p).
       MDP: (16,16), three kernel types, 3000 iterations.

All figures are saved to config.output_dir (default: 'results/').
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from config import Config
from experiments import Experiments
from plotter import Plotter


class Main:
    """Top-level orchestrator for all three PPG experiments.

    Delegates all computation to Experiments and all visualization to Plotter.
    Owns no algorithmic logic — its sole responsibility is wiring, timing,
    and producing human-readable console output.

    Attributes:
        config: Centralized configuration instance. All hyperparameters,
            MDP sizes, iteration counts, and step size settings are read
            from this object.
        experiments: Experiments instance that constructs MDPs and runs PPG.
        plotter: Plotter instance that renders results into figures.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the Main orchestrator.

        Instantiation order is critical:
        1. Config() — sets numpy random seed and creates output_dir.
        2. Experiments(config) — reads all hyperparameters from config.
        3. Plotter(config.output_dir) — only needs the output path.

        No computation happens in __init__. It is purely setup.

        Args:
            config: Optional pre-configured Config instance. If None,
                a fresh Config() is created with all defaults from
                config.yaml. Passing a pre-configured instance allows
                the CLI block to apply argument overrides before
                constructing Main, avoiding double-initialization.
        """
        # Step 1: Initialize configuration (or use provided instance)
        if config is None:
            self.config: Config = Config()
        else:
            self.config = config

        # Step 2: Initialize experiments (reads all hyperparameters from config)
        self.experiments: Experiments = Experiments(self.config)

        # Step 3: Initialize plotter (only needs output directory path)
        self.plotter: Plotter = Plotter(self.config.output_dir)

    def run_experiment1(self) -> None:
        """Run Experiment 1: Convergence vs State/Action Space Size (Figure 1a).

        Reproduces Figure 1(a) from Section 4 / Appendix C.1 of the paper.
        Tests three MDP sizes: (3,3), (9,9), (81,81) with the same structural
        construction (non-uniform kernel, max-variance reward) to isolate the
        effect of |S| and |A| on convergence speed.

        Expected result: Larger (|S|, |A|) → larger L_2^Π → slower convergence.
        The (3,3) MDP converges fastest; the (81,81) MDP converges slowest.

        Steps:
            1. Run PPG for all three MDP sizes via experiments.run_exp1()
            2. Log complexity metrics (C_m, C_p, C_r, κ_r, L_2^Π, η)
            3. Plot Figure 1(a) via plotter.plot_figure1a()
            4. Print summary of initial and final average rewards
        """
        print()
        print("=" * 70)
        print("EXPERIMENT 1: Convergence vs State/Action Space Size (Figure 1a)")
        print("=" * 70)
        print(f"  MDP sizes: {self.config.exp1_sizes}")
        print(f"  Iterations: {self.config.exp1_iterations}")
        print(f"  Kernel type: {self.config.exp1_kernel_type}")
        print(f"  Reward type: {self.config.exp1_reward_type}")
        print()

        # Step 1: Run PPG for all three MDP sizes
        t_start: float = time.perf_counter()
        results: Dict[Tuple[int, int], Dict] = self.experiments.run_exp1()
        t_elapsed: float = time.perf_counter() - t_start

        print(f"\n  [Exp1] Total runtime: {t_elapsed:.1f}s")

        # Step 2: Log complexity metrics for each MDP size
        print()
        print("  Complexity Metrics (Table 1/2 of the paper):")
        print("  " + "-" * 65)
        print(
            f"  {'Config':<16} {'C_m':>8} {'C_p':>8} {'C_r':>8} "
            f"{'kappa_r':>9} {'L2':>10} {'eta':>10}"
        )
        print("  " + "-" * 65)

        # Iterate in fixed order matching config.exp1_sizes
        for size_pair in self.config.exp1_sizes:
            key: Tuple[int, int] = (int(size_pair[0]), int(size_pair[1]))
            if key not in results:
                continue

            entry: Dict = results[key]
            complexity: Dict = entry.get('complexity', {})
            eta: float = entry.get('eta', float('nan'))

            c_m: float = complexity.get('C_m', float('nan'))
            c_p: float = complexity.get('C_p', float('nan'))
            c_r: float = complexity.get('C_r', float('nan'))
            kappa_r: float = complexity.get('kappa_r', float('nan'))
            l2: float = complexity.get('L2', float('nan'))

            config_label: str = f"|S|={key[0]},|A|={key[1]}"
            print(
                f"  {config_label:<16} {c_m:>8.4f} {c_p:>8.4f} {c_r:>8.4f} "
                f"{kappa_r:>9.4f} {l2:>10.4f} {eta:>10.6f}"
            )

        print("  " + "-" * 65)

        # Step 3: Plot Figure 1(a)
        print()
        print("  Generating Figure 1(a)...")
        self.plotter.plot_figure1a(results)

        # Step 4: Print summary of convergence
        print()
        print("  Convergence Summary:")
        print("  " + "-" * 55)
        print(f"  {'Config':<16} {'Initial rho':>14} {'Final rho':>12} {'Improvement':>13}")
        print("  " + "-" * 55)

        for size_pair in self.config.exp1_sizes:
            key = (int(size_pair[0]), int(size_pair[1]))
            if key not in results:
                continue

            entry = results[key]
            reward_history: List[float] = entry.get('reward_history', [])

            if len(reward_history) == 0:
                continue

            initial_rho: float = reward_history[0]
            final_rho: float = reward_history[-1]
            improvement: float = final_rho - initial_rho

            config_label = f"|S|={key[0]},|A|={key[1]}"
            print(
                f"  {config_label:<16} {initial_rho:>14.6f} {final_rho:>12.6f} "
                f"{improvement:>13.6f}"
            )

        print("  " + "-" * 55)
        print()

    def run_experiment2(self) -> None:
        """Run Experiment 2: Convergence vs Reward Variance / C_r (Figure 1b).

        Reproduces Figure 1(b) from Section 4 / Appendix C.2 of the paper.
        Tests four reward variance levels on a fixed (16,16) MDP with a
        shared randomly-generated transition kernel to isolate the effect
        of reward variance (C_r) on convergence speed.

        Expected result: Higher reward variance → larger C_r → larger L_2^Π
        → slower convergence. 'no_variance' converges fastest; 'max_variance'
        converges slowest.

        Steps:
            1. Run PPG for all four variance levels via experiments.run_exp2()
            2. Plot Figure 1(b) via plotter.plot_figure1b()
            3. Print summary of final rewards per variance level
        """
        print()
        print("=" * 70)
        print("EXPERIMENT 2: Convergence vs Reward Variance / C_r (Figure 1b)")
        print("=" * 70)
        S: int = self.config.exp2_size[0]
        A: int = self.config.exp2_size[1]
        print(f"  MDP size: (S={S}, A={A})")
        print(f"  Iterations: {self.config.exp2_iterations}")
        print(f"  Kernel type: {self.config.exp2_kernel_type}")
        print(f"  Variance levels: {list(self.config.exp2_reward_variants.keys())}")
        print()

        # Step 1: Run PPG for all four variance levels
        t_start: float = time.perf_counter()
        results: Dict[str, Dict] = self.experiments.run_exp2()
        t_elapsed: float = time.perf_counter() - t_start

        print(f"\n  [Exp2] Total runtime: {t_elapsed:.1f}s")

        # Step 2: Plot Figure 1(b)
        print()
        print("  Generating Figure 1(b)...")
        self.plotter.plot_figure1b(results)

        # Step 3: Print summary of convergence per variance level
        print()
        print("  Convergence Summary (ordered by variance level):")
        print("  " + "-" * 70)
        print(
            f"  {'Variance Level':<20} {'C_r':>8} {'Initial rho':>14} "
            f"{'Final rho':>12} {'Improvement':>13}"
        )
        print("  " + "-" * 70)

        # Fixed order: fastest to slowest convergence
        variance_order: List[str] = [
            'no_variance',
            'low_variance',
            'high_variance',
            'max_variance',
        ]

        for key in variance_order:
            if key not in results:
                continue

            entry: Dict = results[key]
            reward_history: List[float] = entry.get('reward_history', [])
            complexity: Dict = entry.get('complexity', {})
            label: str = entry.get(
                'label',
                self.config.exp2_reward_variants.get(key, {}).get('label', key)
            )

            if len(reward_history) == 0:
                continue

            initial_rho: float = reward_history[0]
            final_rho: float = reward_history[-1]
            improvement: float = final_rho - initial_rho
            c_r: float = complexity.get('C_r', float('nan'))

            print(
                f"  {label:<20} {c_r:>8.4f} {initial_rho:>14.6f} "
                f"{final_rho:>12.6f} {improvement:>13.6f}"
            )

        print("  " + "-" * 70)

        # Validate theoretical prediction: no_variance should have highest
        # final reward (fastest convergence)
        if 'no_variance' in results and 'max_variance' in results:
            rh_novar: List[float] = results['no_variance'].get('reward_history', [])
            rh_maxvar: List[float] = results['max_variance'].get('reward_history', [])
            if len(rh_novar) > 0 and len(rh_maxvar) > 0:
                if rh_novar[-1] >= rh_maxvar[-1]:
                    print(
                        "\n  ✓ Theoretical prediction confirmed: "
                        "no_variance converges faster than max_variance."
                    )
                else:
                    print(
                        "\n  ⚠ Note: max_variance achieved higher final reward "
                        "than no_variance. This may indicate the step size "
                        "needs tuning or more iterations are needed."
                    )

        print()

    def run_experiment3(self) -> None:
        """Run Experiment 3: Convergence vs Transition Kernel / C_p (Figure 2).

        Reproduces Figure 2 from Section 4 / Appendix C.3 of the paper.
        Tests three transition kernel types on a fixed (16,16) MDP with a
        shared high-variance reward function to isolate the effect of the
        transition kernel diameter (C_p) on convergence speed.

        Expected result: Higher C_p → larger L_2^Π → slower convergence.
        'uniform' (C_p ≈ 0) converges fastest; 'deterministic' (highest C_p)
        converges slowest.

        Steps:
            1. Run PPG for all three kernel types via experiments.run_exp3()
            2. Print C_p values for each kernel type
            3. Plot Figure 2 via plotter.plot_figure2()
            4. Print summary of final change in average reward per kernel type
        """
        print()
        print("=" * 70)
        print("EXPERIMENT 3: Convergence vs Transition Kernel / C_p (Figure 2)")
        print("=" * 70)
        S: int = self.config.exp3_size[0]
        A: int = self.config.exp3_size[1]
        print(f"  MDP size: (S={S}, A={A})")
        print(f"  Iterations: {self.config.exp3_iterations}")
        print(f"  Reward type: {self.config.exp3_reward_type}")
        print(f"  Kernel types: {list(self.config.exp3_kernel_variants.keys())}")
        print()

        # Step 1: Run PPG for all three kernel types
        t_start: float = time.perf_counter()
        results: Dict[str, Dict] = self.experiments.run_exp3()
        t_elapsed: float = time.perf_counter() - t_start

        print(f"\n  [Exp3] Total runtime: {t_elapsed:.1f}s")

        # Step 2: Print C_p values for each kernel type
        # This directly validates the theoretical claim that C_p drives
        # convergence speed (Theorem 1, Table 1/2 of the paper).
        print()
        print("  C_p Values by Kernel Type (Table 1/2 of the paper):")
        print("  " + "-" * 50)
        print(f"  {'Kernel Type':<20} {'C_p':>10} {'eta':>12}")
        print("  " + "-" * 50)

        kernel_order: List[str] = ['uniform', 'nonuniform', 'deterministic']
        kernel_labels: Dict[str, str] = {
            'uniform':       'Uniform',
            'nonuniform':    'Non-uniform',
            'deterministic': 'Deterministic',
        }

        for key in kernel_order:
            if key not in results:
                continue

            entry: Dict = results[key]
            c_p: float = entry.get('C_p', float('nan'))
            eta: float = entry.get('eta', float('nan'))
            label: str = kernel_labels.get(key, key)

            print(f"  {label:<20} {c_p:>10.4f} {eta:>12.6f}")

        print("  " + "-" * 50)

        # Validate theoretical prediction: uniform should have C_p ≈ 0
        if 'uniform' in results:
            c_p_uniform: float = results['uniform'].get('C_p', float('nan'))
            if c_p_uniform < 0.01:
                print(
                    f"\n  ✓ Uniform kernel has C_p ≈ {c_p_uniform:.4f} ≈ 0 "
                    "(actions don't affect transitions — trivial MDP)."
                )

        # Step 3: Plot Figure 2
        print()
        print("  Generating Figure 2...")
        self.plotter.plot_figure2(results)

        # Step 4: Print summary of convergence (change in average reward)
        print()
        print("  Convergence Summary (change in average reward = rho_final - rho_0):")
        print("  " + "-" * 60)
        print(
            f"  {'Kernel Type':<20} {'C_p':>8} {'rho_0':>10} "
            f"{'rho_final':>12} {'delta_rho':>12}"
        )
        print("  " + "-" * 60)

        for key in kernel_order:
            if key not in results:
                continue

            entry = results[key]
            reward_history: List[float] = entry.get('reward_history', [])
            c_p = entry.get('C_p', float('nan'))
            label = kernel_labels.get(key, key)

            if len(reward_history) == 0:
                continue

            rho_0: float = reward_history[0]
            rho_final: float = reward_history[-1]
            delta_rho: float = rho_final - rho_0

            print(
                f"  {label:<20} {c_p:>8.4f} {rho_0:>10.6f} "
                f"{rho_final:>12.6f} {delta_rho:>12.6f}"
            )

        print("  " + "-" * 60)

        # Validate theoretical prediction: uniform should converge fastest
        if 'uniform' in results and 'deterministic' in results:
            rh_uniform: List[float] = results['uniform'].get('reward_history', [])
            rh_det: List[float] = results['deterministic'].get('reward_history', [])
            if len(rh_uniform) > 0 and len(rh_det) > 0:
                delta_uniform: float = rh_uniform[-1] - rh_uniform[0]
                delta_det: float = rh_det[-1] - rh_det[0]
                if delta_uniform >= delta_det:
                    print(
                        "\n  ✓ Theoretical prediction confirmed: "
                        "uniform kernel converges faster than deterministic."
                    )
                else:
                    print(
                        "\n  ⚠ Note: deterministic kernel achieved larger "
                        "improvement than uniform. This may indicate the "
                        "step size needs tuning or more iterations are needed."
                    )

        print()

    def run_all(self) -> None:
        """Run all three experiments sequentially with total timing.

        Executes Experiments 1, 2, and 3 in order, producing Figures 1(a),
        1(b), and 2. Prints total elapsed time at the end.

        The 81×81 MDP in Experiment 1 is the most computationally expensive
        step (2000 iterations × O(S²) linear system solves). Expect this to
        dominate total runtime.

        Individual experiment timings are printed within each run_experiment*
        method. This method adds the overall total.
        """
        print()
        print("=" * 70)
        print("REPRODUCING: Global Convergence of Policy Gradient in Average")
        print("             Reward MDPs (Murthy et al., ICLR 2024)")
        print("=" * 70)
        print(f"  Random seed:  {self.config.random_seed}")
        print(f"  Output dir:   {self.config.output_dir}")
        print(f"  Step size multiplier: {self.config.step_size_multiplier}")
        print(f"  Step size fallback:   {self.config.step_size_fallback}")
        print(f"  Complexity samples:   {self.config.complexity_n_samples}")
        print()

        # Record overall start time
        t_total_start: float = time.perf_counter()

        # Run all three experiments
        self.run_experiment1()
        self.run_experiment2()
        self.run_experiment3()

        # Print total elapsed time
        t_total_elapsed: float = time.perf_counter() - t_total_start
        total_minutes: int = int(t_total_elapsed // 60)
        total_seconds: float = t_total_elapsed % 60.0

        print()
        print("=" * 70)
        print("ALL EXPERIMENTS COMPLETE")
        print("=" * 70)
        print(f"  Total runtime: {total_minutes}m {total_seconds:.1f}s")
        print(f"  Figures saved to: {os.path.abspath(self.config.output_dir)}")
        print()
        print("  Generated figures:")
        print(f"    - {os.path.join(self.config.output_dir, 'figure1a.png')}")
        print(f"    - {os.path.join(self.config.output_dir, 'figure1b.png')}")
        print(f"    - {os.path.join(self.config.output_dir, 'figure2.png')}")
        print()


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the main entry point.

    Defines the CLI interface for controlling which experiments to run
    and overriding key configuration values without modifying config.yaml.

    Returns:
        Parsed argument namespace with the following attributes:
            exp (str): Which experiment(s) to run. One of '1', '2', '3',
                or 'all'. Default: 'all'.
            seed (Optional[int]): Override config.random_seed. Default: None
                (use config.yaml value of 42).
            eta (Optional[float]): Override config.step_size_multiplier.
                Default: None (use config.yaml value of 0.5).
            output_dir (Optional[str]): Override config.output_dir. Default:
                None (use config.yaml value of 'results/').
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Reproduce experiments from: 'Global Convergence of Policy "
            "Gradient in Average Reward MDPs' (Murthy et al., ICLR 2024). "
            "Produces Figures 1(a), 1(b), and 2 from Section 4 of the paper."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                    # Run all experiments\n"
            "  python main.py --exp 1            # Run only Experiment 1\n"
            "  python main.py --seed 0 --exp 2   # Exp 2 with seed 0\n"
            "  python main.py --output_dir out/  # Save figures to out/\n"
            "  python main.py --eta 0.01         # Override step size multiplier\n"
        ),
    )

    parser.add_argument(
        '--exp',
        type=str,
        default='all',
        choices=['1', '2', '3', 'all'],
        help=(
            "Which experiment(s) to run. "
            "'1' = Figure 1(a) (state/action size), "
            "'2' = Figure 1(b) (reward variance), "
            "'3' = Figure 2 (transition kernel), "
            "'all' = all three (default: 'all')."
        ),
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help=(
            "Override the random seed for reproducibility. "
            "Default: None (uses config.yaml value of 42)."
        ),
    )

    parser.add_argument(
        '--eta',
        type=float,
        default=None,
        help=(
            "Override the step size multiplier (eta = multiplier / L2). "
            "Default: None (uses config.yaml value of 0.5). "
            "The paper requires eta < 1/L_2^Pi (Theorem 1)."
        ),
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help=(
            "Override the output directory for saved figures. "
            "Default: None (uses config.yaml value of 'results/')."
        ),
    )

    return parser.parse_args()


if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Step 1: Parse command-line arguments
    # -------------------------------------------------------------------------
    args: argparse.Namespace = _parse_args()

    # -------------------------------------------------------------------------
    # Step 2: Initialize Config with defaults from config.yaml
    # -------------------------------------------------------------------------
    config: Config = Config()

    # -------------------------------------------------------------------------
    # Step 3: Apply CLI overrides to config fields
    # -------------------------------------------------------------------------
    import numpy as np  # needed for re-seeding after override

    if args.seed is not None:
        config.random_seed = int(args.seed)
        # Re-seed numpy immediately so all downstream randomness uses the
        # overridden seed, not the default seed set in Config.__post_init__.
        np.random.seed(config.random_seed)
        print(f"[CLI] Overriding random seed: {config.random_seed}")

    if args.eta is not None:
        config.step_size_multiplier = float(args.eta)
        print(f"[CLI] Overriding step size multiplier: {config.step_size_multiplier}")

    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
        # Create the overridden output directory (Config.__post_init__ already
        # created the default directory; we need to create the new one too).
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"[CLI] Overriding output directory: {config.output_dir}")

    # -------------------------------------------------------------------------
    # Step 4: Instantiate Main with the (possibly overridden) config
    # -------------------------------------------------------------------------
    main: Main = Main(config=config)

    # -------------------------------------------------------------------------
    # Step 5: Dispatch to the requested experiment(s)
    # -------------------------------------------------------------------------
    try:
        if args.exp == '1':
            main.run_experiment1()
        elif args.exp == '2':
            main.run_experiment2()
        elif args.exp == '3':
            main.run_experiment3()
        elif args.exp == 'all':
            main.run_all()
        else:
            # This branch is unreachable due to argparse choices=['1','2','3','all'],
            # but included for defensive completeness.
            print(f"[ERROR] Unknown experiment: '{args.exp}'. "
                  "Choose from '1', '2', '3', or 'all'.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Experiment run interrupted by user (Ctrl+C).")
        sys.exit(130)

    except Exception as exc:
        print(f"\n[ERROR] Experiment '{args.exp}' failed with exception:")
        print(f"  {type(exc).__name__}: {exc}")
        print()
        print("Traceback:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
