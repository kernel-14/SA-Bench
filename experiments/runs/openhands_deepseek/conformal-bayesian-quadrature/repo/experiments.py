"""Experiment runners for all three paper experiments (Section 5).

Section 5.1 - Synthetic Binomial Data
Section 5.2 - Synthetic Heteroskedastic Data
Section 5.3 - MS-COCO False Negative Rate Control
"""

import numpy as np
from typing import Dict, Optional, Tuple
import time

from config import (
    BinomialConfig,
    HeteroskedasticConfig,
    MSCOCOConfig,
    ExperimentConfig,
)
from bayesian_quadrature import (
    compute_lplus_distribution,
    compute_critical_value,
    select_lambda_hpd,
    compute_lplus_for_lambda,
)
from conformal_methods import (
    select_lambda_crc,
    select_lambda_scp,
    compute_miscoverage_losses,
)
from rcps import select_lambda_rcps
from data import (
    generate_binomial_losses_multilambda,
    expected_binomial_loss,
    generate_heteroskedastic_data,
    compute_heteroskedastic_scores,
    compute_miscoverage_losses_heteroskedastic,
    compute_miscoverage_risk,
    compute_prediction_interval_length,
    load_coco_dummy,
)
from utils import (
    compute_exceedance_rate,
    compute_risk_statistics,
    compute_prediction_set_statistics,
    format_frequency_table,
    clopper_pearson_ci,
)


def make_lambda_grid(low: float, high: float, n: int) -> np.ndarray:
    """Create a grid of λ values for selection."""
    return np.linspace(low, high, n)


def run_binomial_experiment(
    config: Optional[BinomialConfig] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Run the synthetic binomial experiment (Section 5.1).

    n = 10, K = 4, α = 0.4, B = 1.0, β = 0.95.
    M = 10000 random trials.

    Loss: ℓ(z_i, λ) = (1/K) Σ_k 1{V_ik > λ}, V_ik ~ U(0,1).
    E[ℓ] = 1 - λ, so risk > α when λ < 1 - α = 0.6.
    """
    if config is None:
        config = BinomialConfig()

    rng = np.random.default_rng(seed)
    trial_seeds = rng.integers(0, 2**31, size=config.M_trials)

    lambda_grid = make_lambda_grid(0.0, 1.0, config.lambda_grid_size)

    crc_lambdas = np.empty(config.M_trials)
    rcps_lambdas = np.empty(config.M_trials)
    hpd_lambdas = np.empty(config.M_trials)

    crc_exceeded = np.zeros(config.M_trials, dtype=bool)
    rcps_exceeded = np.zeros(config.M_trials, dtype=bool)
    hpd_exceeded = np.zeros(config.M_trials, dtype=bool)

    for trial in range(config.M_trials):
        t_rng = np.random.default_rng(int(trial_seeds[trial]))
        losses_by_lambda = generate_binomial_losses_multilambda(
            n=config.n_calibration,
            K=config.K,
            lambdas=lambda_grid,
            rng=t_rng,
        )

        lam_crc = select_lambda_crc(losses_by_lambda, B=config.B, alpha=config.alpha)
        crc_lambdas[trial] = lam_crc
        crc_exceeded[trial] = expected_binomial_loss(lam_crc) > config.alpha

        lam_rcps = select_lambda_rcps(
            losses_by_lambda, B=config.B, alpha=config.alpha, delta=config.rcps_delta
        )
        rcps_lambdas[trial] = lam_rcps
        rcps_exceeded[trial] = expected_binomial_loss(lam_rcps) > config.alpha

        lam_hpd = select_lambda_hpd(
            losses_by_lambda,
            B=config.B,
            alpha=config.alpha,
            beta=config.beta,
            n_samples=1000,
            rng=t_rng,
        )
        hpd_lambdas[trial] = lam_hpd
        hpd_exceeded[trial] = expected_binomial_loss(lam_hpd) > config.alpha

    crc_lower, crc_freq, crc_upper = clopper_pearson_ci(
        int(np.sum(crc_exceeded)), config.M_trials
    )
    rcps_lower, rcps_freq, rcps_upper = clopper_pearson_ci(
        int(np.sum(rcps_exceeded)), config.M_trials
    )
    hpd_lower, hpd_freq, hpd_upper = clopper_pearson_ci(
        int(np.sum(hpd_exceeded)), config.M_trials
    )

    results = {
        "CRC": {
            "relative_freq": crc_freq,
            "ci_lower": crc_lower,
            "ci_upper": crc_upper,
        },
        "RCPS": {
            "relative_freq": rcps_freq,
            "ci_lower": rcps_lower,
            "ci_upper": rcps_upper,
        },
        "Ours (beta=0.95)": {
            "relative_freq": hpd_freq,
            "ci_lower": hpd_lower,
            "ci_upper": hpd_upper,
        },
    }

    crc_stats = compute_risk_statistics(crc_lambdas, expected_binomial_loss)
    hpd_stats = compute_risk_statistics(hpd_lambdas, expected_binomial_loss)

    if verbose:
        print("=" * 60)
        print("Synthetic Binomial Experiment (Section 5.1)")
        print("=" * 60)
        print(format_frequency_table(
            results, ["CRC", "RCPS", "Ours (beta=0.95)"]
        ))
        print(f"\nCRC mean risk: {crc_stats['mean_risk']:.6f} ± {crc_stats['ste_risk']:.6f}")
        print(f"HPD mean risk: {hpd_stats['mean_risk']:.6f} ± {hpd_stats['ste_risk']:.6f}")

    return {
        "results": results,
        "crc_stats": crc_stats,
        "hpd_stats": hpd_stats,
        "crc_lambdas": crc_lambdas,
        "rcps_lambdas": rcps_lambdas,
        "hpd_lambdas": hpd_lambdas,
    }


def run_heteroskedastic_experiment(
    config: Optional[HeteroskedasticConfig] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Run the synthetic heteroskedastic experiment (Section 5.2).

    n = 200, α = 0.1, β = 0.95.
    M = 10000 random trials.

    X ~ U[0, 4], Y|X ~ N(0, X²).
    Prediction interval: [-λ, λ].
    Loss is miscoverage: ℓ = 1{|Y| > λ}, B = 1.
    """
    if config is None:
        config = HeteroskedasticConfig()

    rng = np.random.default_rng(seed)
    trial_seeds = rng.integers(0, 2**31, size=config.M_trials)

    lambda_grid = make_lambda_grid(0.5, 20.0, config.lambda_grid_size)

    scp_lambdas = np.empty(config.M_trials)
    rcps_lambdas = np.empty(config.M_trials)
    hpd_lambdas = np.empty(config.M_trials)

    scp_exceeded = np.zeros(config.M_trials, dtype=bool)
    rcps_exceeded = np.zeros(config.M_trials, dtype=bool)
    hpd_exceeded = np.zeros(config.M_trials, dtype=bool)

    scp_intervals = np.empty(config.M_trials)
    rcps_intervals = np.empty(config.M_trials)
    hpd_intervals = np.empty(config.M_trials)

    n_test = 5000

    for trial in range(config.M_trials):
        t_rng = np.random.default_rng(int(trial_seeds[trial]))

        X_cal, Y_cal, X_test, Y_test = generate_heteroskedastic_data(
            n_cal=config.n_calibration,
            n_test=n_test,
            x_range=config.x_range,
            rng=t_rng,
        )

        cal_losses_by_lambda = compute_miscoverage_losses_heteroskedastic(
            Y_cal, lambda_grid
        )

        lam_scp = select_lambda_scp(compute_heteroskedastic_scores(Y_cal), config.alpha)
        scp_lambdas[trial] = lam_scp
        scp_exceeded[trial] = compute_miscoverage_risk(Y_test, lam_scp) > config.alpha
        scp_intervals[trial] = compute_prediction_interval_length(lam_scp)

        lam_rcps = select_lambda_rcps(
            cal_losses_by_lambda,
            B=1.0,
            alpha=config.alpha,
            delta=config.rcps_delta,
        )
        rcps_lambdas[trial] = lam_rcps
        rcps_exceeded[trial] = compute_miscoverage_risk(Y_test, lam_rcps) > config.alpha
        rcps_intervals[trial] = compute_prediction_interval_length(lam_rcps)

        lam_hpd = select_lambda_hpd(
            cal_losses_by_lambda,
            B=1.0,
            alpha=config.alpha,
            beta=config.beta,
            n_samples=1000,
            rng=t_rng,
        )
        hpd_lambdas[trial] = lam_hpd
        hpd_exceeded[trial] = compute_miscoverage_risk(Y_test, lam_hpd) > config.alpha
        hpd_intervals[trial] = compute_prediction_interval_length(lam_hpd)

    scp_lower, scp_freq, scp_upper = clopper_pearson_ci(
        int(np.sum(scp_exceeded)), config.M_trials
    )
    rcps_lower, rcps_freq, rcps_upper = clopper_pearson_ci(
        int(np.sum(rcps_exceeded)), config.M_trials
    )
    hpd_lower, hpd_freq, hpd_upper = clopper_pearson_ci(
        int(np.sum(hpd_exceeded)), config.M_trials
    )

    scp_mean_int = float(np.mean(scp_intervals))
    rcps_mean_int = float(np.mean(rcps_intervals))
    hpd_mean_int = float(np.mean(hpd_intervals))

    results = {
        "Split Conformal / CRC": {
            "relative_freq": scp_freq,
            "ci_lower": scp_lower,
            "ci_upper": scp_upper,
            "mean_pred_size": scp_mean_int,
        },
        "RCPS": {
            "relative_freq": rcps_freq,
            "ci_lower": rcps_lower,
            "ci_upper": rcps_upper,
            "mean_pred_size": rcps_mean_int,
        },
        "Ours (beta=0.95)": {
            "relative_freq": hpd_freq,
            "ci_lower": hpd_lower,
            "ci_upper": hpd_upper,
            "mean_pred_size": hpd_mean_int,
        },
    }

    if verbose:
        print("=" * 60)
        print("Synthetic Heteroskedastic Experiment (Section 5.2)")
        print("=" * 60)
        print(format_frequency_table(
            results, ["Split Conformal / CRC", "RCPS", "Ours (beta=0.95)"]
        ))

    return {
        "results": results,
        "scp_lambdas": scp_lambdas,
        "rcps_lambdas": rcps_lambdas,
        "hpd_lambdas": hpd_lambdas,
        "scp_intervals": scp_intervals,
        "rcps_intervals": rcps_intervals,
        "hpd_intervals": hpd_intervals,
    }


def run_mscoco_experiment(
    config: Optional[MSCOCOConfig] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Run the MS-COCO multilabel classification experiment (Section 5.3).

    1000 calibration, 3952 test, 80 classes.
    α = 0.1, β = 0.95, B = 1.0.

    Controls false negative rate using score threshold λ.
    """
    if config is None:
        config = MSCOCOConfig()

    rng = np.random.default_rng(seed)
    trial_seeds = rng.integers(0, 2**31, size=config.M_trials)

    lambda_grid = make_lambda_grid(0.1, 0.9, config.lambda_grid_size)

    crc_lambdas = np.empty(config.M_trials)
    rcps_lambdas = np.empty(config.M_trials)
    hpd_lambdas = np.empty(config.M_trials)

    crc_exceeded = np.zeros(config.M_trials, dtype=bool)
    rcps_exceeded = np.zeros(config.M_trials, dtype=bool)
    hpd_exceeded = np.zeros(config.M_trials, dtype=bool)

    crc_set_sizes = np.empty(config.M_trials)
    rcps_set_sizes = np.empty(config.M_trials)
    hpd_set_sizes = np.empty(config.M_trials)

    for trial in range(config.M_trials):
        t_rng = np.random.default_rng(int(trial_seeds[trial]))
        dataset = load_coco_dummy(
            n_cal=config.n_calibration,
            n_test=config.n_test,
            num_classes=config.num_classes,
            seed=int(trial_seeds[trial]),
        )

        cal_losses_by_lambda = dataset.compute_fnr_losses_multilambda(
            dataset.cal_scores, dataset.cal_labels, lambda_grid
        )

        lam_crc = select_lambda_crc(cal_losses_by_lambda, B=config.B, alpha=config.alpha)
        crc_lambdas[trial] = lam_crc
        crc_test_losses = dataset.compute_fnr_losses(
            dataset.test_scores, dataset.test_labels, lam_crc
        )
        crc_exceeded[trial] = np.mean(crc_test_losses) > config.alpha
        crc_set_sizes[trial] = dataset.compute_prediction_set_size(
            dataset.test_scores, lam_crc
        )

        lam_rcps = select_lambda_rcps(
            cal_losses_by_lambda,
            B=config.B,
            alpha=config.alpha,
            delta=config.rcps_delta,
        )
        rcps_lambdas[trial] = lam_rcps
        rcps_test_losses = dataset.compute_fnr_losses(
            dataset.test_scores, dataset.test_labels, lam_rcps
        )
        rcps_exceeded[trial] = np.mean(rcps_test_losses) > config.alpha
        rcps_set_sizes[trial] = dataset.compute_prediction_set_size(
            dataset.test_scores, lam_rcps
        )

        lam_hpd = select_lambda_hpd(
            cal_losses_by_lambda,
            B=config.B,
            alpha=config.alpha,
            beta=config.beta,
            n_samples=1000,
            rng=t_rng,
        )
        hpd_lambdas[trial] = lam_hpd
        hpd_test_losses = dataset.compute_fnr_losses(
            dataset.test_scores, dataset.test_labels, lam_hpd
        )
        hpd_exceeded[trial] = np.mean(hpd_test_losses) > config.alpha
        hpd_set_sizes[trial] = dataset.compute_prediction_set_size(
            dataset.test_scores, lam_hpd
        )

    crc_lower, crc_freq, crc_upper = clopper_pearson_ci(
        int(np.sum(crc_exceeded)), config.M_trials
    )
    rcps_lower, rcps_freq, rcps_upper = clopper_pearson_ci(
        int(np.sum(rcps_exceeded)), config.M_trials
    )
    hpd_lower, hpd_freq, hpd_upper = clopper_pearson_ci(
        int(np.sum(hpd_exceeded)), config.M_trials
    )

    crc_mean_size = float(np.mean(crc_set_sizes))
    rcps_mean_size = float(np.mean(rcps_set_sizes))
    hpd_mean_size = float(np.mean(hpd_set_sizes))

    results = {
        "CRC": {
            "relative_freq": crc_freq,
            "ci_lower": crc_lower,
            "ci_upper": crc_upper,
            "mean_pred_size": crc_mean_size,
        },
        "RCPS": {
            "relative_freq": rcps_freq,
            "ci_lower": rcps_lower,
            "ci_upper": rcps_upper,
            "mean_pred_size": rcps_mean_size,
        },
        "Ours (beta=0.95)": {
            "relative_freq": hpd_freq,
            "ci_lower": hpd_lower,
            "ci_upper": hpd_upper,
            "mean_pred_size": hpd_mean_size,
        },
    }

    if verbose:
        print("=" * 60)
        print("MS-COCO Experiment (Section 5.3)")
        print("=" * 60)
        print(format_frequency_table(
            results, ["CRC", "RCPS", "Ours (beta=0.95)"]
        ))

    return {
        "results": results,
        "crc_lambdas": crc_lambdas,
        "rcps_lambdas": rcps_lambdas,
        "hpd_lambdas": hpd_lambdas,
        "crc_set_sizes": crc_set_sizes,
        "rcps_set_sizes": rcps_set_sizes,
        "hpd_set_sizes": hpd_set_sizes,
    }


def run_all_experiments(
    config: Optional[ExperimentConfig] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Run all three paper experiments.

    Args:
        config: Experiment configuration.
        seed: Master random seed.
        verbose: Whether to print results.

    Returns:
        Dict with results from all three experiments.
    """
    if config is None:
        config = ExperimentConfig()

    if verbose:
        print("=" * 70)
        print("Conformal Prediction as Bayesian Quadrature")
        print("Paper Reproduction Experiments")
        print("=" * 70)
        print()

    binomial_results = run_binomial_experiment(
        config=config.binomial, seed=seed, verbose=verbose
    )
    print()

    hetero_results = run_heteroskedastic_experiment(
        config=config.heteroskedastic, seed=seed, verbose=verbose
    )
    print()

    coco_results = run_mscoco_experiment(
        config=config.ms_coco, seed=seed, verbose=verbose
    )

    return {
        "binomial": binomial_results,
        "heteroskedastic": hetero_results,
        "ms_coco": coco_results,
    }


def run_lplus_diagnostic_plot(
    config: Optional[BinomialConfig] = None,
    seed: int = 42,
) -> Dict:
    """Generate L⁺ diagnostic data for λ ∈ {0.7, 0.8, 0.9} (Figure 4).

    Uses 100,000 Dirichlet samples as described in the paper.

    Args:
        config: Binomial experiment config.
        seed: Random seed.

    Returns:
        Dict with L⁺ samples and statistics for each λ.
    """
    if config is None:
        config = BinomialConfig()

    rng = np.random.default_rng(seed)
    lambda_values = [0.7, 0.8, 0.9]

    results = {}
    for lam in lambda_values:
        losses = generate_binomial_losses_multilambda(
            n=config.n_calibration,
            K=config.K,
            lambdas=np.array([lam]),
            rng=rng,
        )[lam]

        lplus_samples = compute_lplus_distribution(
            losses, B=config.B, n_samples=100000, rng=rng
        )
        results[lam] = {
            "lplus_samples": lplus_samples,
            "mean": float(np.mean(lplus_samples)),
            "median": float(np.median(lplus_samples)),
            "q025": float(np.quantile(lplus_samples, 0.025)),
            "q975": float(np.quantile(lplus_samples, 0.975)),
            "prob_leq_alpha": float(np.mean(lplus_samples <= config.alpha)),
        }

    return results


if __name__ == "__main__":
    run_all_experiments(verbose=True)
