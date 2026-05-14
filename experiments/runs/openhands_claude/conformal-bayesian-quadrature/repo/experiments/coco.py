"""
Section 5.3: False Negative Rate on MS-COCO.

Mirrors the experimental setup of Angelopoulos & Bates (2023, Section 5.1).
Uses pre-computed softmax scores from a ResNet-101 model on MS-COCO.

Each random split: 1000 calibration examples, 3952 test examples.
Loss: false negative rate (FNR) for multilabel classification.
Target: alpha=0.1, beta=0.95.

Reproduces Table 3: relative frequency of trials exceeding risk threshold
and average prediction set size.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

from config import COCOConfig
from data.coco_loader import load_coco_data, fnr_loss, prediction_set_size
from methods import lambda_crc, lambda_rcps_hoeffding, lambda_bq_hpd
from utils import compute_failure_rate, format_results_table


def run_single_trial(
    cfg: COCOConfig,
    scores: np.ndarray,
    labels: np.ndarray,
    lambda_grid: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Run one trial of the COCO experiment.

    Randomly splits the dataset into calibration (n_calib) and test (n_test)
    sets, applies each method to select lambda, and returns the chosen lambda
    and test set metrics.
    """
    N = len(scores)
    indices = rng.permutation(N)
    calib_idx = indices[: cfg.n_calib]
    test_idx = indices[cfg.n_calib : cfg.n_calib + cfg.n_test]

    scores_calib = scores[calib_idx]
    labels_calib = labels[calib_idx]
    scores_test = scores[test_idx]
    labels_test = labels[test_idx]

    def loss_fn(lam: float) -> np.ndarray:
        return fnr_loss(scores_calib, labels_calib, lam)

    lam_crc = lambda_crc(loss_fn, lambda_grid, cfg.alpha, cfg.B)
    lam_rcps = lambda_rcps_hoeffding(loss_fn, lambda_grid, cfg.alpha, cfg.beta, cfg.B)
    lam_hpd = lambda_bq_hpd(
        loss_fn, lambda_grid, cfg.alpha, cfg.beta, cfg.B,
        n_dirichlet=cfg.n_dirichlet_decision, rng=rng,
    )

    # Compute test risk (true FNR on test set) for each chosen lambda
    test_risk_crc = float(np.mean(fnr_loss(scores_test, labels_test, lam_crc)))
    test_risk_rcps = float(np.mean(fnr_loss(scores_test, labels_test, lam_rcps)))
    test_risk_hpd = float(np.mean(fnr_loss(scores_test, labels_test, lam_hpd)))

    # Prediction set sizes on test set
    set_size_crc = prediction_set_size(scores_test, lam_crc)
    set_size_rcps = prediction_set_size(scores_test, lam_rcps)
    set_size_hpd = prediction_set_size(scores_test, lam_hpd)

    return {
        "lam_crc": lam_crc,
        "lam_rcps": lam_rcps,
        "lam_hpd": lam_hpd,
        "test_risk_crc": test_risk_crc,
        "test_risk_rcps": test_risk_rcps,
        "test_risk_hpd": test_risk_hpd,
        "set_size_crc": set_size_crc,
        "set_size_rcps": set_size_rcps,
        "set_size_hpd": set_size_hpd,
    }


def run_experiment(cfg: COCOConfig) -> Dict:
    """
    Run the full COCO experiment (M=10,000 trials).
    """
    scores, labels = load_coco_data(cfg.data_path)
    print(f"  Loaded COCO data: {scores.shape[0]} examples, {scores.shape[1]} classes")

    rng = np.random.default_rng(cfg.seed)
    lambda_grid = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.lambda_steps)

    results_list = []
    for trial in range(cfg.M):
        result = run_single_trial(cfg, scores, labels, lambda_grid, rng)
        results_list.append(result)

        if (trial + 1) % 500 == 0:
            print(f"  Completed {trial + 1}/{cfg.M} trials")

    # Aggregate results
    keys = results_list[0].keys()
    aggregated = {k: np.array([r[k] for r in results_list]) for k in keys}

    exceeded = {
        "crc": aggregated["test_risk_crc"] > cfg.alpha,
        "rcps": aggregated["test_risk_rcps"] > cfg.alpha,
        "hpd": aggregated["test_risk_hpd"] > cfg.alpha,
    }

    return {
        "lambdas": {
            "crc": aggregated["lam_crc"],
            "rcps": aggregated["lam_rcps"],
            "hpd": aggregated["lam_hpd"],
        },
        "test_risks": {
            "crc": aggregated["test_risk_crc"],
            "rcps": aggregated["test_risk_rcps"],
            "hpd": aggregated["test_risk_hpd"],
        },
        "set_sizes": {
            "crc": aggregated["set_size_crc"],
            "rcps": aggregated["set_size_rcps"],
            "hpd": aggregated["set_size_hpd"],
        },
        "exceeded": exceeded,
    }


def run_and_report(cfg: COCOConfig, output_dir: str) -> None:
    """
    Run the full COCO experiment and print/save results.
    """
    print("=" * 60)
    print("Experiment 5.3: False Negative Rate on MS-COCO")
    print("=" * 60)

    results = run_experiment(cfg)
    exceeded = results["exceeded"]
    set_sizes = results["set_sizes"]

    method_names = ["CRC", "RCPS", "Ours (β = 0.95)"]
    method_keys = ["crc", "rcps", "hpd"]

    freqs = []
    cis = []
    mean_set_sizes = []
    for key in method_keys:
        freq, ci = compute_failure_rate(exceeded[key])
        freqs.append(freq)
        cis.append(ci)
        mean_set_sizes.append(float(np.mean(set_sizes[key])))

    print("\nTable 3: Results on MS-COCO")
    print(format_results_table(
        method_names, freqs, cis,
        extra_cols={"Mean Pred. Set Size": mean_set_sizes},
    ))

    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "coco_results.npz"),
        lambdas_crc=results["lambdas"]["crc"],
        lambdas_rcps=results["lambdas"]["rcps"],
        lambdas_hpd=results["lambdas"]["hpd"],
        test_risks_crc=results["test_risks"]["crc"],
        test_risks_rcps=results["test_risks"]["rcps"],
        test_risks_hpd=results["test_risks"]["hpd"],
        set_sizes_crc=set_sizes["crc"],
        set_sizes_rcps=set_sizes["rcps"],
        set_sizes_hpd=set_sizes["hpd"],
        exceeded_crc=exceeded["crc"],
        exceeded_rcps=exceeded["rcps"],
        exceeded_hpd=exceeded["hpd"],
    )
    print(f"\nResults saved to {output_dir}/coco_results.npz")
