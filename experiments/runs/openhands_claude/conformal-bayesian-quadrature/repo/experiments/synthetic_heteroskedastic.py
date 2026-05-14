"""
Section 5.2: Synthetic Heteroskedastic Data experiment.

Data generating process:
  X ~ Uniform(0, 4)
  Y | X ~ N(0, X^2)

Prediction intervals: [-lambda, lambda].
Loss: miscoverage = 1{|Y| > lambda} (binary, B=1).
Target: alpha=0.1 (90% coverage), beta=0.95.

Reproduces Table 2: relative frequency of trials exceeding risk threshold
and mean prediction interval length.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from config import SyntheticHeteroskedasticConfig
from methods import lambda_crc, lambda_rcps_hoeffding, lambda_bq_hpd
from utils import compute_failure_rate, format_results_table


def generate_calibration_data(
    n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate calibration data for the heteroskedastic experiment.

    X ~ Uniform(0, 4), Y | X ~ N(0, X^2).

    Returns:
        X: shape (n,)
        Y: shape (n,)
    """
    X = rng.uniform(0.0, 4.0, size=n)
    Y = rng.normal(0.0, X)
    return X, Y


def miscoverage_loss(Y: np.ndarray, lam: float) -> np.ndarray:
    """
    Miscoverage loss: 1{|Y| > lambda}.

    Args:
        Y: array of observations.
        lam: half-width of prediction interval.

    Returns:
        Binary loss array.
    """
    return (np.abs(Y) > lam).astype(np.float32)


def compute_true_risk_heteroskedastic_analytical(lam: float, n_quad: int = 10_000) -> float:
    """
    Compute true risk Pr(|Y| > lambda) analytically via numerical integration.

    Pr(|Y| > lambda) = E_X[2 * Phi(-lambda / X)]
                     = (1/4) * integral_0^4 2 * Phi(-lambda / x) dx

    Uses Gaussian quadrature for accurate integration.

    Args:
        lam: half-width of prediction interval.
        n_quad: number of quadrature points.

    Returns:
        True risk value.
    """
    if lam <= 0:
        return 1.0
    x_vals = np.linspace(1e-6, 4.0, n_quad)
    integrand = 2.0 * stats.norm.cdf(-lam / x_vals)
    return float(np.trapz(integrand, x_vals) / 4.0)


def run_single_trial(
    cfg: SyntheticHeteroskedasticConfig,
    lambda_grid: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Run one trial of the heteroskedastic experiment.

    Returns chosen lambda for each method.
    """
    _, Y_calib = generate_calibration_data(cfg.n, rng)

    def loss_fn(lam: float) -> np.ndarray:
        return miscoverage_loss(Y_calib, lam)

    # SCP and CRC are equivalent for miscoverage loss (Section 4.6)
    lam_scp_crc = lambda_crc(loss_fn, lambda_grid, cfg.alpha, cfg.B)
    lam_rcps = lambda_rcps_hoeffding(loss_fn, lambda_grid, cfg.alpha, cfg.beta, cfg.B)
    lam_hpd = lambda_bq_hpd(
        loss_fn, lambda_grid, cfg.alpha, cfg.beta, cfg.B,
        n_dirichlet=cfg.n_dirichlet_decision, rng=rng,
    )

    return {
        "scp_crc": lam_scp_crc,
        "rcps": lam_rcps,
        "hpd": lam_hpd,
    }


def compute_true_risk_for_lambda(lam: float) -> float:
    """
    Compute the true risk for a given lambda in the heteroskedastic setting.
    """
    return compute_true_risk_heteroskedastic_analytical(lam)


def run_experiment(cfg: SyntheticHeteroskedasticConfig) -> Dict:
    """
    Run the full heteroskedastic experiment (M=10,000 trials).
    """
    rng = np.random.default_rng(cfg.seed)
    lambda_grid = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.lambda_steps)

    lambdas = {"scp_crc": [], "rcps": [], "hpd": []}

    for trial in range(cfg.M):
        result = run_single_trial(cfg, lambda_grid, rng)
        for method, lam in result.items():
            lambdas[method].append(lam)

        if (trial + 1) % 1000 == 0:
            print(f"  Completed {trial + 1}/{cfg.M} trials")

    lambdas = {k: np.array(v) for k, v in lambdas.items()}

    # Compute true risk for each chosen lambda
    # True risk = Pr(|Y| > lambda) = E_X[2*Phi(-lambda/X)]
    # A trial exceeds the risk threshold when true_risk(lambda) > alpha
    print("  Computing true risks for each trial...")
    exceeded = {}
    for key, lam_arr in lambdas.items():
        true_risks = np.array([compute_true_risk_for_lambda(lam) for lam in lam_arr])
        exceeded[key] = true_risks > cfg.alpha

    # Prediction interval length = 2 * lambda
    interval_lengths = {k: 2.0 * lambdas[k] for k in lambdas}

    return {
        "lambdas": lambdas,
        "exceeded": exceeded,
        "interval_lengths": interval_lengths,
    }


def run_and_report(cfg: SyntheticHeteroskedasticConfig, output_dir: str) -> None:
    """
    Run the full heteroskedastic experiment and print/save results.
    """
    print("=" * 60)
    print("Experiment 5.2: Synthetic Heteroskedastic Data")
    print("=" * 60)

    results = run_experiment(cfg)
    lambdas = results["lambdas"]
    exceeded = results["exceeded"]
    interval_lengths = results["interval_lengths"]

    method_names = ["Split Conformal Prediction / CRC", "RCPS", "Ours (β = 0.95)"]
    method_keys = ["scp_crc", "rcps", "hpd"]

    freqs = []
    cis = []
    mean_lengths = []
    for key in method_keys:
        freq, ci = compute_failure_rate(exceeded[key])
        freqs.append(freq)
        cis.append(ci)
        mean_lengths.append(float(np.mean(interval_lengths[key])))

    print("\nTable 2: Relative frequency of trials exceeding risk threshold")
    print(format_results_table(
        method_names, freqs, cis,
        extra_cols={"Mean Pred. Interval Length": mean_lengths},
    ))

    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "synthetic_heteroskedastic_results.npz"),
        lambdas_scp_crc=lambdas["scp_crc"],
        lambdas_rcps=lambdas["rcps"],
        lambdas_hpd=lambdas["hpd"],
        exceeded_scp_crc=exceeded["scp_crc"],
        exceeded_rcps=exceeded["rcps"],
        exceeded_hpd=exceeded["hpd"],
        interval_lengths_scp_crc=interval_lengths["scp_crc"],
        interval_lengths_rcps=interval_lengths["rcps"],
        interval_lengths_hpd=interval_lengths["hpd"],
    )
    print(f"\nResults saved to {output_dir}/synthetic_heteroskedastic_results.npz")
