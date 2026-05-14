"""
main.py – Experiment orchestration for "Conformal Prediction as Bayesian Quadrature".

This module contains the `ExperimentRunner` class that executes all three
experiments (binomial, heteroskedastic, MS‑COCO) and prints the results in
the format of Tables 5.1, 5.2, and 5.3 from the paper.

Usage:
    python main.py [--config CONFIG_PATH]

If no config path is given, the default `config.yaml` in the current
directory is used.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tqdm

from config import Config, load_config
from data import (
    BinomialDataGenerator,
    CocoDataLoader,
    HeteroDataGenerator,
)
from evaluation import EvaluationMetrics
from losses import BinomialLoss, FNRLoss, MiscoverageLoss
from methods import BQCMethod, CRCMethod, RCPSMethod
from utils import build_lambda_grid


# ---------------------------------------------------------------------------
# Utility: table printing
# ---------------------------------------------------------------------------

def _float_to_percent(x: float) -> str:
    """Format a float as a percentage string with two decimals."""
    return f"{100 * x:.2f}%"


def _format_ci(lower: float, upper: float) -> str:
    """Format a Clopper‑Pearson confidence interval as a string."""
    return f"[{_float_to_percent(lower)}, {_float_to_percent(upper)}]"


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Orchestrates the three experiments from the paper.

    Parameters
    ----------
    config : Config
        The configuration dataclass (loaded from YAML).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        # Pre‑build the λ grid used by all experiments.
        self.lambda_grid = build_lambda_grid(
            config.lambda_min,
            config.lambda_max,
            config.lambda_grid_size,
        )
        # Global RNG for master seed only.
        self.master_rng = np.random.default_rng(config.seed)
        # Evaluation metric aggregator.
        self.evaluator = EvaluationMetrics()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_all(self) -> None:
        """Run all experiments sequentially and print results."""
        t0 = time.perf_counter()
        print("\n=== Binomial experiment ===")
        self._run_binomial()
        t1 = time.perf_counter()
        print(f"\nTime: {t1 - t0:.1f} s")

        print("\n=== Heteroskedastic experiment ===")
        self._run_heteroskedastic()
        t2 = time.perf_counter()
        print(f"\nTime: {t2 - t1:.1f} s")

        if not self.config.skip_coco:
            print("\n=== MS‑COCO experiment ===")
            self._run_coco()
            t3 = time.perf_counter()
            print(f"\nTime: {t3 - t2:.1f} s")
        else:
            print("\n[COCO experiment skipped]")

    # ------------------------------------------------------------------
    # Binomial experiment (Table 5.1)
    # ------------------------------------------------------------------

    def _run_binomial(self) -> None:
        cfg = self.config
        alpha = cfg.alpha          # 0.4
        B = cfg.B
        beta = cfg.confidence_beta
        delta = cfg.rcps_delta
        n_trials = cfg.num_trials

        results: List[Dict[str, Any]] = []

        for trial in tqdm.tqdm(range(n_trials), desc="Binomial trial", unit="trial"):
            # Per‑trial seed ensures reproducibility and independence.
            trial_seed = cfg.seed + trial
            rng = np.random.default_rng(trial_seed)

            # Generate calibration data.
            gen = BinomialDataGenerator(
                n_calibration=cfg.n_binomial,
                K=cfg.K_binomial,
                rng=rng,
            )
            cal_data, _ = gen.generate()  # shape (n_cal, K)

            # Loss matrix (n_cal × n_grid)
            loss_fn = BinomialLoss(K=cfg.K_binomial)
            # Vectorised: for each λ, for each calibration point, compute mean exceedance.
            # cal_data shape (n_cal, K) -> expanded to (n_cal, 1, K) and compare to grid (1, G, 1)
            exceeds = (cal_data[:, None, :] > self.lambda_grid[None, :, None])  # (n_cal, G, K)
            loss_matrix = exceeds.mean(axis=2)  # (n_cal, G)

            # Instantiate decision rules.
            crc = CRCMethod(alpha, B)
            rcps = RCPSMethod(alpha, B, delta)
            bqc = BQCMethod(alpha, beta, B,
                            n_dir_samples=cfg.num_dirichlet_samples,
                            rng=rng)

            # For each method, get lambda_hat.
            for method_name, method_obj in [("CRC", crc),
                                             ("RCPS", rcps),
                                             ("BQC", bqc)]:
                lam, _ = method_obj(loss_matrix, self.lambda_grid)
                # True risk in binomial case: 1 − λ.
                true_risk = 1.0 - lam
                risk_exceeded = (true_risk > alpha)

                results.append({
                    "method": method_name,
                    "trial": trial,
                    "lambda_hat": lam,
                    "risk_exceeded": risk_exceeded,
                    "interval_length": None,  # no size metric for binomial
                })

        # Aggregate and print.
        self._print_table(results, "Decision Rule", include_length=False)

    # ------------------------------------------------------------------
    # Heteroskedastic experiment (Table 5.2)
    # ------------------------------------------------------------------

    def _run_heteroskedastic(self) -> None:
        cfg = self.config
        alpha = cfg.hetero_alpha     # 0.1
        B = cfg.B
        beta = cfg.confidence_beta
        delta = cfg.rcps_delta
        n_trials = cfg.num_trials

        results: List[Dict[str, Any]] = []

        for trial in tqdm.tqdm(range(n_trials), desc="Hetero trial", unit="trial"):
            trial_seed = cfg.seed + trial
            rng = np.random.default_rng(trial_seed)

            gen = HeteroDataGenerator(
                n_calibration=cfg.n_hetero,
                n_test=cfg.hetero_test_size,
                rng=rng,
            )
            (X_cal, Y_cal), (X_test, Y_test) = gen.generate()

            # Loss matrix: miscoverage indicator.
            # (n_cal, G) matrix: 1 if |Y| > λ
            loss_matrix = (np.abs(Y_cal[:, None]) > self.lambda_grid[None, :]).astype(np.float64)

            # Decision rules.
            crc = CRCMethod(alpha, B)
            rcps = RCPSMethod(alpha, B, delta)
            bqc = BQCMethod(alpha, beta, B,
                            n_dir_samples=cfg.num_dirichlet_samples,
                            rng=rng)

            for method_name, method_obj in [("CRC", crc),
                                             ("RCPS", rcps),
                                             ("BQC", bqc)]:
                lam, _ = method_obj(loss_matrix, self.lambda_grid)

                # Estimate true risk on large test set.
                test_losses = (np.abs(Y_test) > lam).astype(np.float64)
                true_risk = np.mean(test_losses)
                risk_exceeded = (true_risk > alpha)

                results.append({
                    "method": method_name,
                    "trial": trial,
                    "lambda_hat": lam,
                    "risk_exceeded": risk_exceeded,
                    "interval_length": 2.0 * lam,
                })

        self._print_table(results, "Decision Rule", include_length=True,
                          length_label="Mean Prediction Interval Length")

    # ------------------------------------------------------------------
    # MS‑COCO multilabel experiment (Table 5.3)
    # ------------------------------------------------------------------

    def _run_coco(self) -> None:
        cfg = self.config
        alpha = cfg.coco_alpha      # 0.1
        B = cfg.B
        beta = cfg.confidence_beta
        delta = cfg.rcps_delta
        n_trials = cfg.num_trials

        # Load the full COCO validation set and pre‑compute probabilities once.
        print("Loading COCO dataset and model (this may take a while)...")
        coco_loader = CocoDataLoader(
            data_root=cfg.coco_data_root,
            annotation_file=cfg.coco_annotation_file,
            model_path=cfg.coco_model_path,
            n_calibration=cfg.n_coco_cal,
            n_test=cfg.n_coco_test,
            rng=self.master_rng,
        )

        results: List[Dict[str, Any]] = []

        for trial in tqdm.tqdm(range(n_trials), desc="COCO trial", unit="trial"):
            trial_seed = cfg.seed + trial
            rng = np.random.default_rng(trial_seed)

            # Obtain random calibration/test split for this trial.
            (probs_cal, labels_cal), (probs_test, labels_test) = coco_loader.generate(rng=rng)

            # --- Loss matrix ---
            # For each λ, compute false negative rate loss.
            # We vectorise over calibration points and λ.
            n_cal, n_classes = probs_cal.shape
            G = len(self.lambda_grid)

            # mask: (n_cal, G, n_classes) boolean: probs >= 1 - λ
            thresholds = 1.0 - self.lambda_grid   # shape (G,)
            mask_cal = probs_cal[:, None, :] >= thresholds[None, :, None]  # (n_cal, G, n_classes)

            # Ground truth label counts per image: (n_cal, 1)
            label_counts = labels_cal.sum(axis=1, keepdims=True)  # (n_cal, 1)
            # Prevent division by zero (will be clipped later).
            safe_counts = np.maximum(label_counts, 1.0)

            # Intersection sizes: (mask_cal & labels_cal[:, None, :]).sum(axis=2) -> (n_cal, G)
            intersection = (
                mask_cal.astype(np.float64) * labels_cal[:, None, :].astype(np.float64)
            ).sum(axis=2)

            # Recall per image: (n_cal, G)
            recall = intersection / safe_counts
            # Loss = 1 - recall, for images with no labels loss is 0 (already safe)
            loss_matrix = 1.0 - recall
            # But for empty labels, recall is defined as 0/1 = 0, so loss = 1. We need to set to 0.
            empty_mask = (label_counts == 0).flatten()  # shape (n_cal,)
            loss_matrix[empty_mask, :] = 0.0

            # Decision rules.
            crc = CRCMethod(alpha, B)
            rcps = RCPSMethod(alpha, B, delta)
            bqc = BQCMethod(alpha, beta, B,
                            n_dir_samples=cfg.num_dirichlet_samples,
                            rng=rng)

            for method_name, method_obj in [("CRC", crc),
                                             ("RCPS", rcps),
                                             ("BQC", bqc)]:
                lam, _ = method_obj(loss_matrix, self.lambda_grid)

                # --- True risk on test set ---
                threshold_test = 1.0 - lam
                mask_test = probs_test >= threshold_test      # (n_test, n_classes)
                label_counts_test = labels_test.sum(axis=1, keepdims=True)
                safe_counts_test = np.maximum(label_counts_test, 1.0)

                intersection_test = (
                    mask_test.astype(np.float64) * labels_test.astype(np.float64)
                ).sum(axis=1, keepdims=True)

                recall_test = intersection_test / safe_counts_test
                test_losses = 1.0 - recall_test
                empty_mask_test = (label_counts_test == 0).flatten()
                test_losses[empty_mask_test] = 0.0

                true_risk = np.mean(test_losses)
                risk_exceeded = (true_risk > alpha)

                # Prediction set size = number of classes predicted per image.
                set_sizes = mask_test.sum(axis=1).astype(np.float64)  # (n_test,)
                mean_set_size = float(np.mean(set_sizes))

                results.append({
                    "method": method_name,
                    "trial": trial,
                    "lambda_hat": lam,
                    "risk_exceeded": risk_exceeded,
                    "interval_length": mean_set_size,  # reuse field for set size
                })

        self._print_table(results, "Method", include_length=True,
                          length_label="Pred. Set Size")

    # ------------------------------------------------------------------
    # Table printing helper
    # ------------------------------------------------------------------

    def _print_table(
        self,
        results: List[Dict[str, Any]],
        method_column: str,
        include_length: bool,
        length_label: str = "Mean Length",
    ) -> None:
        """
        Aggregate results per method and print a formatted table.

        Parameters
        ----------
        results : list of dict
            Per‑trial records.  Each must contain "method", "risk_exceeded",
            and "interval_length" (if `include_length` is True).
        method_column : str
            Header for the method name column.
        include_length : bool
            Whether to print a column for the complexity metric.
        length_label : str
            Header for the complexity metric column.
        """
        # Group by method
        method_dict: Dict[str, List[Dict[str, Any]]] = {}
        for rec in results:
            m = rec["method"]
            if m not in method_dict:
                method_dict[m] = []
            method_dict[m].append(rec)

        # Print table
        col_width = 12
        header = f"{method_column:<{col_width}}\tRelative Freq.\t95% CI"
        if include_length:
            header += f"\t{length_label}"
        print(header)
        print("-" * len(header.expandtabs()))

        for method, recs in method_dict.items():
            n_total = len(recs)
            n_fail = sum(1 for r in recs if r["risk_exceeded"])
            freq = n_fail / n_total
            ci_low, ci_up = self.evaluator.clopper_pearson_ci(n_fail, n_total, alpha=0.05)
            line = f"{method:<{col_width}}\t{_float_to_percent(freq)}\t{_format_ci(ci_low, ci_up)}"
            if include_length:
                lengths = [r["interval_length"] for r in recs if r["interval_length"] is not None]
                avg_len = np.mean(lengths) if lengths else float("nan")
                line += f"\t{avg_len:.4f}"
            print(line)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce experiments from 'Conformal Prediction as Bayesian Quadrature'."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    runner = ExperimentRunner(config)
    runner.run_all()


if __name__ == "__main__":
    main()
