## Code: main.py

```python
## main.py
"""Entry point for reproducing 'Conformal Prediction as Bayesian Quadrature'.

This script orchestrates all three experiments from the paper and generates
the corresponding tables and figures. It wires together configuration,
experiment runners, evaluation, and plotting without containing any
algorithmic logic of its own.

Experiments:
  1. Synthetic Binomial Data (Section 5.1) → Table 1, Figure 3, Figure 4
  2. Synthetic Heteroskedastic Data (Section 5.2) → Table 2
  3. False Negative Rate on MS-COCO (Section 5.3) → Table 3

Usage:
    # Run all experiments (MS-COCO data must be present):
    python main.py --exp all

    # Run only Experiment 1 (fully synthetic, no data download needed):
    python main.py --exp 1

    # Run Experiment 3 with custom data paths:
    python main.py --exp 3 --mscoco-scores-path data/mscoco/scores.npy \
                            --mscoco-labels-path data/mscoco/labels.npy

    # Debug mode (sequential, no parallelism):
    python main.py --exp 1 --no-parallel

    # Custom output directory:
    python main.py --exp all --output-dir my_results/

Config values used (from config.yaml):
    experiment.M = 10000
    experiment.seed = 42
    experiment.n_jobs = -1
    experiment.output_dir = "results/"
    cbq.n_mc_samples = 1000
    cbq.n_mc_figure = 100000
    exp1_synthetic_binomial.*
    exp2_synthetic_heteroskedastic.*
    exp3_mscoco.*
    plotting.*

References:
    Paper Section 5: Experiments.
    config.yaml: All hyperparameter values.
"""

import argparse
import os
import time
from typing import List

import numpy as np

from config import BinomialConfig, HeteroskedasticConfig, MSCOCOConfig
from data.mscoco_loader import MSCOCOLoader
from evaluation import (
    ExperimentResult,
    TrialResult,
    compute_experiment_results,
    print_table,
    summarize_to_dataframe,
)
from experiments.exp1_synthetic_binomial import (
    compute_L_plus_for_figure4,
    run_experiment_binomial,
)
from experiments.exp2_synthetic_heteroskedastic import run_experiment_heteroskedastic
from experiments.exp3_mscoco import run_experiment_mscoco
from plotting import plot_L_plus_density, plot_lambda_histograms


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment runner.

    Returns:
        argparse.Namespace with the following attributes:
          - exp: str, one of ['1', '2', '3', 'all']. Default 'all'.
          - no_parallel: bool. If True, overrides n_jobs=1 in all configs.
          - mscoco_scores_path: str. Path to MS-COCO scores .npy file.
          - mscoco_labels_path: str. Path to MS-COCO labels .npy file.
          - output_dir: str. Directory for saving tables and figures.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce experiments from "
            "'Conformal Prediction as Bayesian Quadrature'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --exp 1\n"
            "  python main.py --exp all --output-dir results/\n"
            "  python main.py --exp 3 "
            "--mscoco-scores-path data/mscoco/scores.npy "
            "--mscoco-labels-path data/mscoco/labels.npy\n"
        ),
    )

    parser.add_argument(
        "--exp",
        type=str,
        choices=["1", "2", "3", "all"],
        default="all",
        help=(
            "Which experiment(s) to run. "
            "'1' = Synthetic Binomial (Section 5.1), "
            "'2' = Synthetic Heteroskedastic (Section 5.2), "
            "'3' = MS-COCO FNR (Section 5.3), "
            "'all' = run all three in sequence. "
            "Default: 'all'."
        ),
    )

    parser.add_argument(
        "--no-parallel",
        action="store_true",
        default=False,
        help=(
            "Disable joblib parallelism (set n_jobs=1). "
            "Useful for debugging since parallel execution suppresses tracebacks. "
            "Default: False (use all available CPU cores)."
        ),
    )

    parser.add_argument(
        "--mscoco-scores-path",
        type=str,
        # Default from config.yaml: exp3_mscoco.scores_path
        default="data/mscoco/scores.npy",
        dest="mscoco_scores_path",
        help=(
            "Path to precomputed MS-COCO softmax scores .npy file. "
            "Expected shape: (N, C) float with values in [0, 1]. "
            "Download from: https://github.com/aangelopoulos/conformal-risk-control. "
            "Default: 'data/mscoco/scores.npy' (config.yaml: exp3_mscoco.scores_path)."
        ),
    )

    parser.add_argument(
        "--mscoco-labels-path",
        type=str,
        # Default from config.yaml: exp3_mscoco.labels_path
        default="data/mscoco/labels.npy",
        dest="mscoco_labels_path",
        help=(
            "Path to MS-COCO binary ground-truth labels .npy file. "
            "Expected shape: (N, C) binary with values in {0, 1}. "
            "Default: 'data/mscoco/labels.npy' (config.yaml: exp3_mscoco.labels_path)."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        # Default from config.yaml: experiment.output_dir
        default="results/",
        dest="output_dir",
        help=(
            "Directory for saving tables (CSV) and figures (PNG). "
            "Created automatically if it does not exist. "
            "Default: 'results/' (config.yaml: experiment.output_dir)."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Experiment 1: Synthetic Binomial Data (Section 5.1)
# ---------------------------------------------------------------------------


def run_experiment_1(args: argparse.Namespace) -> None:
    """Run Experiment 1 and produce Table 1, Figure 3, and Figure 4.

    Implements the synthetic binomial experiment from Section 5.1 of the paper.
    The true expected loss is 1 - λ (analytical), enabling direct verification
    of risk control guarantees.

    Config values (from config.yaml):
        exp1_synthetic_binomial.n_cal = 10
        exp1_synthetic_binomial.K = 4
        exp1_synthetic_binomial.alpha = 0.4
        exp1_synthetic_binomial.beta = 0.95
        exp1_synthetic_binomial.B = 1.0
        exp1_synthetic_binomial.risk_threshold = 0.6
        exp1_synthetic_binomial.figure4_lambdas = [0.7, 0.8, 0.9]
        cbq.n_mc_samples = 1000
        cbq.n_mc_figure = 100000
        experiment.M = 10000
        experiment.seed = 42
        plotting.figure3_path = "results/fig3_lambda_histograms.png"
        plotting.figure4_path = "results/fig4_L_plus_density.png"
        plotting.table1_path = "results/table1_synthetic_binomial.csv"

    Args:
        args: Parsed command-line arguments from parse_args().
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Synthetic Binomial Data (Section 5.1)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Instantiate BinomialConfig with values from config.yaml.
    # All defaults in BinomialConfig match config.yaml exactly.
    # ------------------------------------------------------------------
    config: BinomialConfig = BinomialConfig(
        exp_name="synthetic_binomial",
        # experiment.M = 10000
        M=10000,
        # exp1_synthetic_binomial.n_cal = 10
        n_cal=10,
        # exp1_synthetic_binomial.alpha = 0.4
        alpha=0.4,
        # exp1_synthetic_binomial.beta = 0.95
        beta=0.95,
        # exp1_synthetic_binomial.B = 1.0
        B=1.0,
        # cbq.n_mc_samples = 1000
        n_mc_samples=1000,
        # cbq.n_mc_figure = 100000
        n_mc_figure=100000,
        # experiment.seed = 42
        seed=42,
        # exp1_synthetic_binomial.lambda_grid: start=0.0, stop=1.0, num=500
        lambda_grid=np.linspace(0.0, 1.0, 500),
        # experiment.n_jobs = -1 (all cores), overridden by --no-parallel
        n_jobs=-1,
        # exp1_synthetic_binomial.K = 4
        K=4,
    )

    # Override n_jobs if --no-parallel flag is set.
    if args.no_parallel:
        config.n_jobs = 1
        print("  [INFO] Parallelism disabled (--no-parallel). Running sequentially.")

    # Attach figure4_lambdas from config.yaml (exp1_synthetic_binomial.figure4_lambdas).
    # BinomialConfig does not have this field by default; we add it dynamically
    # since it is only used in compute_L_plus_for_figure4.
    config.figure4_lambdas = [0.7, 0.8, 0.9]  # type: ignore[attr-defined]

    print(
        f"  Config: M={config.M}, n_cal={config.n_cal}, K={config.K}, "
        f"α={config.alpha}, β={config.beta}, B={config.B}, "
        f"n_mc_samples={config.n_mc_samples}, seed={config.seed}"
    )

    # ------------------------------------------------------------------
    # Step 2: Run all M=10,000 trials in parallel.
    # ------------------------------------------------------------------
    print(f"\n  Running {config.M} trials...")
    t0: float = time.time()
    trial_results: List[TrialResult] = run_experiment_binomial(config)
    elapsed: float = time.time() - t0
    print(f"  Completed {config.M} trials in {elapsed:.1f}s.")

    # ------------------------------------------------------------------
    # Step 3: Aggregate results and produce Table 1.
    # ------------------------------------------------------------------
    print("\n  Computing evaluation statistics...")
    exp_results: List[ExperimentResult] = compute_experiment_results(
        trial_results=trial_results,
        config=config,
    )

    df = summarize_to_dataframe(exp_results)

    # Print Table 1 to stdout.
    print_table(df, title="Table 1: Synthetic Binomial Results (Section 5.1)")

    # Save Table 1 as CSV.
    # plotting.table1_path = "results/table1_synthetic_binomial.csv"
    table1_path: str = os.path.join(args.output_dir, "table1_synthetic_binomial.csv")
    df.to_csv(table1_path, index=False)
    print(f"  Table 1 saved to: {table1_path}")

    # Print mean lambda statistics (mentioned in paper Section 5.1 text).
    # "CRC mean 0.3363 ± 0.0007, ours 0.1758 ± 0.0006"
    for result in exp_results:
        se: float = result.std_lambda / np.sqrt(config.M) if config.M > 0 else 0.0
        print(
            f"  {result.method_name}: mean λ = {result.mean_lambda:.4f} "
            f"± {se:.4f} (std/sqrt(M))"
        )

    # ------------------------------------------------------------------
    # Step 4: Generate Figure 3 — Lambda histograms.
    # alpha_threshold = 0.6 (config.yaml: exp1_synthetic_binomial.risk_threshold)
    # Risk exceeds α=0.4 iff λ < 0.6 (true risk = 1 - λ).
    # ------------------------------------------------------------------
    print("\n  Generating Figure 3 (lambda histograms)...")

    # Extract lambda arrays from trial results.
    lambda_crc: np.ndarray = np.array(
        [r.lambda_crc for r in trial_results], dtype=float
    )
    lambda_hpd: np.ndarray = np.array(
        [r.lambda_hpd for r in trial_results], dtype=float
    )

    # plotting.figure3_path = "results/fig3_lambda_histograms.png"
    fig3_path: str = os.path.join(args.output_dir, "fig3_lambda_histograms.png")

    plot_lambda_histograms(
        lambda_crc=lambda_crc,
        lambda_hpd=lambda_hpd,
        # exp1_synthetic_binomial.risk_threshold = 0.6
        alpha_threshold=0.6,
        # exp1_synthetic_binomial.alpha = 0.4
        alpha=config.alpha,
        save_path=fig3_path,
        # plotting.hist_bins = 50
        hist_bins=50,
        # plotting.hist_alpha = 0.7
        hist_alpha=0.7,
        # plotting.dpi = 150
        dpi=150,
    )
    print(f"  Figure 3 saved to: {fig3_path}")

    # ------------------------------------------------------------------
    # Step 5: Generate Figure 4 — L⁺ density plots.
    # Uses 100,000 Dirichlet samples (config.yaml: cbq.n_mc_figure = 100000)
    # for λ ∈ {0.7, 0.8, 0.9} (config.yaml: exp1_synthetic_binomial.figure4_lambdas).
    # ------------------------------------------------------------------
    print("\n  Generating Figure 4 (L+ density plots, 100,000 Dirichlet samples)...")

    losses_dict = compute_L_plus_for_figure4(config)

    # plotting.figure4_path = "results/fig4_L_plus_density.png"
    fig4_path: str = os.path.join(args.output_dir, "fig4_L_plus_density.png")

    plot_L_plus_density(
        losses_dict=losses_dict,
        save_path=fig4_path,
        # exp1_synthetic_binomial.alpha = 0.4 (reference line in Figure 4)
        alpha_ref=config.alpha,
        # plotting.dpi = 150
        dpi=150,
    )
    print(f"  Figure 4 saved to: {fig4_path}")

    print("\n  Experiment 1 complete.")


# ---------------------------------------------------------------------------
# Experiment 2: Synthetic Heteroskedastic Data (Section 5.2)
# ---------------------------------------------------------------------------


def run_experiment_2(args: argparse.Namespace) -> None:
    """Run Experiment 2 and produce Table 2.

    Implements the synthetic heteroskedastic regression experiment from
    Section 5.2 of the paper. The true risk is computed via numerical
    integration of E_X[2·Φ(-λ/X)] over X ~ Uniform[0, 4].

    Config values (from config.yaml):
        exp2_synthetic_heteroskedastic.n_cal = 200
        exp2_synthetic_heteroskedastic.x_low = 0.0
        exp2_synthetic_heteroskedastic.x_high = 4.0
        exp2_synthetic_heteroskedastic.alpha = 0.1
        exp2_synthetic_heteroskedastic.beta = 0.95
        exp2_synthetic_heteroskedastic.B = 1.0
        exp2_synthetic_heteroskedastic.n_quad_true_risk = 1000
        cbq.n_mc_samples = 1000
        experiment.M = 10000
        experiment.seed = 42
        plotting.table2_path = "results/table2_synthetic_heteroskedastic.csv"

    Args:
        args: Parsed command-line arguments from parse_args().
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Synthetic Heteroskedastic Data (Section 5.2)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Instantiate HeteroskedasticConfig with values from config.yaml.
    # ------------------------------------------------------------------
    config: HeteroskedasticConfig = HeteroskedasticConfig(
        exp_name="synthetic_heteroskedastic",
        # experiment.M = 10000
        M=10000,
        # exp2_synthetic_heteroskedastic.n_cal = 200
        n_cal=200,
        # exp2_synthetic_heteroskedastic.alpha = 0.1
        alpha=0.1,
        # exp2_synthetic_heteroskedastic.beta = 0.95
        beta=0.95,
        # exp2_synthetic_heteroskedastic.B = 1.0
        B=1.0,
        # cbq.n_mc_samples = 1000
        n_mc_samples=1000,
        # cbq.n_mc_figure = 100000 (inherited from ExperimentConfig, unused here)
        n_mc_figure=100000,
        # experiment.seed = 42
        seed=42,
        # exp2_synthetic_heteroskedastic.lambda_grid: start=0.0, stop=20.0, num=1000
        # Must extend to 20 to cover RCPS solution (~7.15 based on Table 2 mean PI/2).
        lambda_grid=np.linspace(0.0, 20.0, 1000),
        # experiment.n_jobs = -1 (all cores), overridden by --no-parallel
        n_jobs=-1,
        # exp2_synthetic_heteroskedastic.x_low = 0.0
        x_low=0.0,
        # exp2_synthetic_heteroskedastic.x_high = 4.0
        x_high=4.0,
        # exp2_synthetic_heteroskedastic.n_quad_true_risk = 1000
        n_quad_true_risk=1000,
    )

    # Override n_jobs if --no-parallel flag is set.
    if args.no_parallel:
        config.n_jobs = 1
        print("  [INFO] Parallelism disabled (--no-parallel). Running sequentially.")

    print(
        f"  Config: M={config.M}, n_cal={config.n_cal}, "
        f"X~U[{config.x_low}, {config.x_high}], Y|X~N(0,X²), "
        f"α={config.alpha}, β={config.beta}, B={config.B}, "
        f"n_mc_samples={config.n_mc_samples}, seed={config.seed}"
    )
    print(
        f"  Lambda grid: [{config.lambda_grid[0]:.1f}, {config.lambda_grid[-1]:.1f}] "
        f"with {len(config.lambda_grid)} points."
    )

    # ------------------------------------------------------------------
    # Step 2: Run all M=10,000 trials in parallel.
    # ------------------------------------------------------------------
    print(f"\n  Running {config.M} trials...")
    t0: float = time.time()
    trial_results: List[TrialResult] = run_experiment_heteroskedastic(config)
    elapsed: float = time.time() - t0
    print(f"  Completed {config.M} trials in {elapsed:.1f}s.")

    # ------------------------------------------------------------------
    # Step 3: Aggregate results and produce Table 2.
    # Table 2 has an extra column "Mean Prediction Interval Length" (= 2λ).
    # This is populated via the 'interval_length_{method}' keys in
    # TrialResult.extra, aggregated by compute_experiment_results into
    # ExperimentResult.mean_extra['mean_interval_length'].
    # ------------------------------------------------------------------
    print("\n  Computing evaluation statistics...")
    exp_results: List[ExperimentResult] = compute_experiment_results(
        trial_results=trial_results,
        config=config,
    )

    df = summarize_to_dataframe(exp_results)

    # Print Table 2 to stdout.
    print_table(
        df,
        title="Table 2: Synthetic Heteroskedastic Results (Section 5.2)",
    )

    # Save Table 2 as CSV.
    # plotting.table2_path = "results/table2_synthetic_heteroskedastic.csv"
    table2_path: str = os.path.join(
        args.output_dir, "table2_synthetic_heteroskedastic.csv"
    )
    df.to_csv(table2_path, index=False)
    print(f"  Table 2 saved to: {table2_path}")

    print("\n  Experiment 2 complete.")


# ---------------------------------------------------------------------------
# Experiment 3: False Negative Rate on MS-COCO (Section 5.3)
# ---------------------------------------------------------------------------


def run_experiment_3(args: argparse.Namespace) -> None:
    """Run Experiment 3 and produce Table 3.

    Implements the MS-COCO false negative rate experiment from Section 5.3
    of the paper, mirroring Angelopoulos & Bates (2023, Section 5.1).

    Requires precomputed MS-COCO softmax scores and binary labels. If the
    data files are not found, this function prints a helpful skip message
    and returns without crashing (allowing --exp all to still produce
    results for experiments 1 and 2).

    Config values (from config.yaml):
        exp3_mscoco.n_cal = 1000
        exp3_mscoco.n_test = 3952
        exp3_mscoco.alpha = 0.1
        exp3_mscoco.beta = 0.95
        exp3_mscoco.B = 1.0
        exp3_mscoco.scores_path = "data/mscoco/scores.npy"
        exp3_mscoco.labels_path = "data/mscoco/labels.npy"
        cbq.n_mc_samples = 1000
        experiment.M = 10000
        experiment.seed = 42
        plotting.table3_path = "results/table3_mscoco.csv"

    Args:
        args: Parsed command-line arguments from parse_args().
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: False Negative Rate on MS-COCO (Section 5.3)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Validate data file paths before doing any expensive work.
    # If files are missing, print a clear skip message and return.
    # This allows --exp all to still produce results for experiments 1 and 2.
    # ------------------------------------------------------------------
    scores_path: str = args.mscoco_scores_path
    labels_path: str = args.mscoco_labels_path

    if not os.path.exists(scores_path):
        print(
            f"\n  [SKIP] MS-COCO scores file not found at: {scores_path}\n"
            f"  To run Experiment 3, download the precomputed scores from:\n"
            f"    https://github.com/aangelopoulos/conformal-risk-control\n"
            f"  and place the file at: {scores_path}\n"
            f"  Then re-run with: python main.py --exp 3 "
            f"--mscoco-scores-path {scores_path}"
        )
        return

    if not os.path.exists(labels_path):
        print(
            f"\n  [SKIP] MS-COCO labels file not found at: {labels_path}\n"
            f"  To run Experiment 3, download the precomputed labels from:\n"
            f"    https://github.com/aangelopoulos/conformal-risk-control\n"
            f"  and place the file at: {labels_path}\n"
            f"  Then re-run with: python main.py --exp 3 "
            f"--mscoco-labels-path {labels_path}"
        )
        return

    # ------------------------------------------------------------------
    # Step 2: Instantiate MSCOCOConfig with values from config.yaml.
    # ------------------------------------------------------------------
    config: MSCOCOConfig = MSCOCOConfig(
        exp_name="mscoco",
        # experiment.M = 10000
        M=10000,
        # exp3_mscoco.n_cal = 1000
        n_cal=1000,
        # exp3_mscoco.alpha = 0.1
        alpha=0.1,
        # exp3_mscoco.beta = 0.95
        beta=0.95,
        # exp3_mscoco.B = 1.0
        B=1.0,
        # cbq.n_mc_samples = 1000
        n_mc_samples=1000,
        # cbq.n_mc_figure = 100000 (inherited from ExperimentConfig, unused here)
        n_mc_figure=100000,
        # experiment.seed = 42
        seed=42,
        # exp3_mscoco.lambda_grid: start=0.0, stop=1.0, num=500
        # Lambda is a tolerance parameter in [0, 1]; score threshold = 1 - λ.
        lambda_grid=np.linspace(0.0, 1.0, 500),
        # experiment.n_jobs = -1 (all cores), overridden by --no-parallel
        n_jobs=-1,
        # exp3_mscoco.n_test = 3952
        n_test=3952,
        # exp3_mscoco.scores_path
        scores_path=scores_path,
        # exp3_mscoco.labels_path
        labels_path=labels_path,
    )

    # Override n_jobs if --no-parallel flag is set.
    if args.no_parallel:
        config.n_jobs = 1
        print("  [INFO] Parallelism disabled (--no-parallel). Running sequentially.")

    print(
        f"  Config: M={config.M}, n_cal={config.n_cal}, n_test={config.n_test}, "
        f"α={config.alpha}, β={config.beta}, B={config.B}, "
        f"n_mc_samples={config.n_mc_samples}, seed={config.seed}"
    )
    print(f"  Scores: {scores_path}")
    print(f"  Labels: {labels_path}")

    # ------------------------------------------------------------------
    # Step 3: Load MS-COCO data once in the main process.
    # The loaded arrays are passed to run_experiment_mscoco, which
    # distributes them to parallel workers. Loading once avoids repeated
    # disk I/O in each worker process.
    # ------------------------------------------------------------------
    print("\n  Loading MS-COCO data...")
    loader: MSCOCOLoader = MSCOCOLoader(
        scores_path=scores_path,
        labels_path=labels_path,
    )

    try:
        loader.load()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  [ERROR] Failed to load MS-COCO data: {exc}")
        print(
            "  Download from: https://github.com/aangelopoulos/conformal-risk-control"
        )
        return

    # Validate pool size: must have at least n_cal + n_test = 4952 examples.
    assert loader.scores is not None  # guaranteed by loader.load() success
    n_total: int = loader.scores.shape[0]
    n_required: int = config.n_cal + config.n_test  # 1000 + 3952 = 4952
    if n_total < n_required:
        print(
            f"\n  [ERROR] Dataset too small: N={n_total} < "
            f"n_cal + n_test = {n_required}. "
            f"Cannot run Experiment 3."
        )
        return

    print(
        f"  Loaded: N={n_total} examples, C={loader.scores.shape[1]} classes."
    )

    # ------------------------------------------------------------------
    # Step 4: Run all M=10,000 trials in parallel.
    # ------------------------------------------------------------------
    print(f"\n  Running {config.M} trials...")
    t0: float = time.time()
    trial_results: List[TrialResult] = run_experiment_mscoco(config, loader)
    elapsed: float = time.time() - t0
    print(f"  Completed {config.M} trials in {elapsed:.1f}s.")

    # ------------------------------------------------------------------
    # Step 5: Aggregate results and produce Table 3.
    # Table 3 has columns: Method, Relative Freq., Pred. Set Size.
    # The 'pred_set_size_{method}' keys in TrialResult.extra are aggregated
    # by compute_experiment_results into ExperimentResult.mean_extra
    # ['mean_pred_set_size'], which summarize_to_dataframe detects and
    # includes as the "Pred. Set Size" column.
    # ------------------------------------------------------------------
    print("\n  Computing evaluation statistics...")
    exp_results: List[ExperimentResult] = compute_experiment_results(
        trial_results=trial_results,
        config=config,
    )

    df = summarize_to_dataframe(exp_results)

    # Print Table 3 to stdout.
    print_table(
        df,
        title="Table 3: MS-COCO False Negative Rate Results (Section 5.3)",
    )

    # Save Table 3 as CSV.
    # plotting.table3_path = "results/table3_mscoco.csv"
    table3_path: str = os.path.join(args.output_dir, "table3_mscoco.csv")
    df.to_csv(table3_path, index=False)
    print(f"  Table 3 saved to: {table3_path}")

    print("\n  Experiment 3 complete.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate all experiments and generate tables and figures.

    Dispatches to the appropriate experiment runner(s) based on the --exp
    argument. Creates the output directory if it does not exist. Prints
    timing information for each experiment and the total run.

    Config values used (from config.yaml):
        experiment.output_dir = "results/"  (default, overridable via --output-dir)
    """
    # ------------------------------------------------------------------
    # Step 1: Parse command-line arguments.
    # ------------------------------------------------------------------
    args: argparse.Namespace = parse_args()

    # ------------------------------------------------------------------
    # Step 2: Create output directory.
    # exist_ok=True prevents errors if the directory already exists from
    # a previous run. All tables and figures are saved here.
    # ------------------------------------------------------------------