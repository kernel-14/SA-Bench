"""
Section 5.1: Synthetic Binomial Data experiment.

Loss distribution: ell(z_i, lambda) = (1/K) * sum_{k=1}^K 1{V_ik > lambda}
where V_ik ~ Uniform(0, 1).

True expected loss = 1 - lambda.
Risk exceeds alpha=0.4 when lambda < 0.6.

Reproduces:
  - Table 1: relative frequency of trials exceeding risk threshold
  - Figure 3: histogram of chosen lambda for CRC and BQ-HPD
  - Figure 4: histogram of L+ for lambda in {0.7, 0.8, 0.9}
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from config import SyntheticBinomialConfig
from methods import lambda_crc, lambda_rcps_hoeffding, lambda_bq_hpd, compute_L_plus_samples
from utils import compute_failure_rate, format_results_table


def binomial_loss(
    n: int,
    K: int,
    lam: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample calibration losses for the synthetic binomial experiment.

    ell(z_i, lambda) = (1/K) * sum_{k=1}^K 1{V_ik > lambda}

    Args:
        n: number of calibration samples.
        K: number of Bernoulli trials per sample.
        lam: threshold parameter.
        rng: random generator.

    Returns:
        1-D array of losses, shape (n,).
    """
    V = rng.uniform(0.0, 1.0, size=(n, K))
    return np.mean(V > lam, axis=1)


def run_single_trial(
    cfg: SyntheticBinomialConfig,
    lambda_grid: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Run one trial of the synthetic binomial experiment.

    Generates calibration data, applies each method to select lambda,
    and returns the chosen lambda for each method.
    """
    # Pre-generate all calibration data for this trial (V matrix)
    V_calib = rng.uniform(0.0, 1.0, size=(cfg.n, cfg.K))

    def loss_fn(lam: float) -> np.ndarray:
        return np.mean(V_calib > lam, axis=1)

    lam_crc = lambda_crc(loss_fn, lambda_grid, cfg.alpha, cfg.B)
    lam_rcps = lambda_rcps_hoeffding(loss_fn, lambda_grid, cfg.alpha, cfg.beta, cfg.B)
    lam_hpd = lambda_bq_hpd(
        loss_fn, lambda_grid, cfg.alpha, cfg.beta, cfg.B,
        n_dirichlet=cfg.n_dirichlet_decision, rng=rng,
    )

    return {
        "crc": lam_crc,
        "rcps": lam_rcps,
        "hpd": lam_hpd,
    }


def run_experiment(cfg: SyntheticBinomialConfig) -> Dict:
    """
    Run the full synthetic binomial experiment (M=10,000 trials).

    Returns a dict with chosen lambdas and failure indicators for each method.
    """
    rng = np.random.default_rng(cfg.seed)
    lambda_grid = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.lambda_steps)

    lambdas = {"crc": [], "rcps": [], "hpd": []}

    for trial in range(cfg.M):
        result = run_single_trial(cfg, lambda_grid, rng)
        for method, lam in result.items():
            lambdas[method].append(lam)

        if (trial + 1) % 1000 == 0:
            print(f"  Completed {trial + 1}/{cfg.M} trials")

    lambdas = {k: np.array(v) for k, v in lambdas.items()}

    # A trial exceeds the risk threshold when lambda < 0.6 (true risk = 1-lambda > 0.4)
    risk_threshold_lambda = 1.0 - cfg.alpha  # 0.6
    exceeded = {k: lambdas[k] < risk_threshold_lambda for k in lambdas}

    return {"lambdas": lambdas, "exceeded": exceeded}


def compute_L_plus_histogram(
    cfg: SyntheticBinomialConfig,
    lambda_vals: list[float],
    rng: np.random.Generator,
) -> Dict[float, np.ndarray]:
    """
    Compute L+ samples for Figure 4 using 100,000 Dirichlet samples.

    For each lambda, generates a single calibration set and computes L+.
    """
    results = {}
    for lam in lambda_vals:
        V_calib = rng.uniform(0.0, 1.0, size=(cfg.n, cfg.K))
        losses = np.mean(V_calib > lam, axis=1)
        L_plus = compute_L_plus_samples(losses, cfg.B, cfg.n_dirichlet_histogram, rng)
        results[lam] = L_plus
    return results


def plot_lambda_histograms(
    lambdas: Dict[str, np.ndarray],
    alpha: float,
    output_dir: str,
) -> None:
    """
    Figure 3: Histogram of chosen lambda for CRC and BQ-HPD across M trials.
    """
    risk_threshold = 1.0 - alpha  # lambda < this => risk exceeds alpha

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (method, title) in zip(
        axes,
        [("crc", "CRC ($\\lambda_{\\mathrm{crc}}$)"),
         ("hpd", "BQ-HPD ($\\lambda_{\\mathrm{hpd}}^{0.95}$)")],
    ):
        lams = lambdas[method]
        bins = np.linspace(0, 1, 51)
        counts, edges = np.histogram(lams, bins=bins)

        # Color bars red where lambda < risk_threshold (risk exceeds alpha)
        for i, (left, right, count) in enumerate(zip(edges[:-1], edges[1:], counts)):
            color = "red" if right <= risk_threshold else "steelblue"
            ax.bar(left, count, width=right - left, color=color, align="edge",
                   edgecolor="white", linewidth=0.3)

        ax.axvline(risk_threshold, color="black", linestyle="--", linewidth=1.5,
                   label=f"$\\lambda = {risk_threshold}$ (risk threshold)")
        ax.set_xlabel("$\\lambda$", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "figure3_lambda_histograms.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "figure3_lambda_histograms.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_L_plus_histogram(
    L_plus_by_lambda: Dict[float, np.ndarray],
    alpha: float,
    output_dir: str,
) -> None:
    """
    Figure 4: Probability density of L+ for lambda in {0.7, 0.8, 0.9}.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for (lam, L_plus), color in zip(L_plus_by_lambda.items(), colors):
        ax.hist(L_plus, bins=100, density=True, alpha=0.6, color=color,
                label=f"$\\lambda = {lam}$")

    ax.axvline(alpha, color="black", linestyle="--", linewidth=1.5,
               label=f"$\\alpha = {alpha}$")
    ax.set_xlabel("$L^+$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of $L^+$ for different $\\lambda$", fontsize=13)
    ax.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "figure4_L_plus_histogram.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "figure4_L_plus_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close()


def run_and_report(cfg: SyntheticBinomialConfig, output_dir: str) -> None:
    """
    Run the full synthetic binomial experiment and print/save results.
    """
    print("=" * 60)
    print("Experiment 5.1: Synthetic Binomial Data")
    print("=" * 60)

    results = run_experiment(cfg)
    lambdas = results["lambdas"]
    exceeded = results["exceeded"]

    method_names = ["CRC", "RCPS", "Ours (β = 0.95)"]
    method_keys = ["crc", "rcps", "hpd"]

    freqs = []
    cis = []
    for key in method_keys:
        freq, ci = compute_failure_rate(exceeded[key])
        freqs.append(freq)
        cis.append(ci)

    print("\nTable 1: Relative frequency of trials exceeding risk threshold")
    print(format_results_table(method_names, freqs, cis))

    print("\nMean lambda values:")
    for name, key in zip(method_names, method_keys):
        mean_lam = np.mean(lambdas[key])
        se_lam = np.std(lambdas[key]) / np.sqrt(cfg.M)
        print(f"  {name}: {mean_lam:.4f} ± {se_lam:.4f}")

    # Figure 3
    plot_lambda_histograms(lambdas, cfg.alpha, output_dir)
    print(f"\nFigure 3 saved to {output_dir}/figure3_lambda_histograms.{{pdf,png}}")

    # Figure 4: L+ histogram
    rng_fig4 = np.random.default_rng(cfg.seed + 1)
    L_plus_by_lambda = compute_L_plus_histogram(cfg, cfg.lambda_histogram, rng_fig4)
    plot_L_plus_histogram(L_plus_by_lambda, cfg.alpha, output_dir)
    print(f"Figure 4 saved to {output_dir}/figure4_L_plus_histogram.{{pdf,png}}")

    # Save numerical results
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "synthetic_binomial_results.npz"),
        lambdas_crc=lambdas["crc"],
        lambdas_rcps=lambdas["rcps"],
        lambdas_hpd=lambdas["hpd"],
        exceeded_crc=exceeded["crc"],
        exceeded_rcps=exceeded["rcps"],
        exceeded_hpd=exceeded["hpd"],
    )
