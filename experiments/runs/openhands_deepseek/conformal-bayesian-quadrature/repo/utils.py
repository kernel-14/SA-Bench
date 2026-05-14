"""Evaluation metrics, statistics, and plotting utilities."""

import numpy as np
from typing import Dict, Tuple, List, Optional


def clopper_pearson_ci(
    successes: int,
    trials: int,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Compute Clopper-Pearson confidence interval for a binomial proportion.

    Args:
        successes: Number of successes (e.g., risk exceeded).
        trials: Total number of trials.
        alpha: Significance level (for 95% CI, alpha=0.05).

    Returns:
        (lower, mean, upper): Confidence interval bounds and point estimate.
    """
    from scipy.stats import beta

    if trials == 0:
        return 0.0, 0.0, 0.0

    mean = successes / trials

    if successes == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2, successes, trials - successes + 1)

    if successes == trials:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha / 2, successes + 1, trials - successes)

    return lower, mean, upper


def compute_exceedance_rate(
    lambdas: np.ndarray,
    true_risk_fn,
    alpha: float,
) -> Dict[str, float]:
    """Compute the fraction of trials where risk exceeds α.

    Args:
        lambdas: Array of selected λ values across trials.
        true_risk_fn: Function mapping λ to true risk.
        alpha: Target risk threshold.

    Returns:
        Dict with 'relative_freq' and optional CI info.
    """
    risks = np.array([true_risk_fn(lam) for lam in lambdas])
    exceeded = np.sum(risks > alpha)
    total = len(lambdas)

    lower, mean, upper = clopper_pearson_ci(int(exceeded), total)

    return {
        "relative_freq": mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "exceeded_count": int(exceeded),
        "total_trials": total,
    }


def compute_risk_statistics(
    lambdas: np.ndarray,
    true_risk_fn,
) -> Dict[str, float]:
    """Compute risk statistics across trials.

    Args:
        lambdas: Array of selected λ values.
        true_risk_fn: Function mapping λ to true risk.

    Returns:
        Dict with mean_risk, std_risk, mean_lambda, std_lambda, etc.
    """
    risks = np.array([true_risk_fn(lam) for lam in lambdas])

    return {
        "mean_risk": float(np.mean(risks)),
        "std_risk": float(np.std(risks)),
        "ste_risk": float(np.std(risks) / np.sqrt(len(risks))),
        "mean_lambda": float(np.mean(lambdas)),
        "std_lambda": float(np.std(lambdas)),
        "min_lambda": float(np.min(lambdas)),
        "max_lambda": float(np.max(lambdas)),
        "median_lambda": float(np.median(lambdas)),
    }


def compute_prediction_set_statistics(
    set_sizes: np.ndarray,
) -> Dict[str, float]:
    """Compute statistics about prediction set sizes.

    Args:
        set_sizes: Array of prediction set sizes across trials.

    Returns:
        Dict with mean, std, median set size.
    """
    return {
        "mean_size": float(np.mean(set_sizes)),
        "std_size": float(np.std(set_sizes)),
        "median_size": float(np.median(set_sizes)),
        "min_size": float(np.min(set_sizes)),
        "max_size": float(np.max(set_sizes)),
    }


def format_frequency_table(
    results: Dict[str, Dict],
    method_names: List[str],
) -> str:
    """Format results as a frequency table (like Tables 1, 2, 3 in paper).

    Args:
        results: Dict mapping method name to result dict with 'relative_freq',
                 'ci_lower', 'ci_upper', and optionally 'mean_pred_size'.
        method_names: List of method names in order.

    Returns:
        Formatted string table.
    """
    lines = []
    header = f"{'Method':<20} {'Rel. Freq.':>10} {'95% CI':>20}"
    extra_col = "mean_pred_size" in next(iter(results.values()))
    if extra_col:
        header += f" {'Pred. Size':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for name in method_names:
        r = results[name]
        freq = r["relative_freq"]
        ci_str = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
        line = f"{name:<20} {freq*100:>9.2f}% {ci_str:>20}"
        if extra_col:
            line += f" {r.get('mean_pred_size', float('nan')):>12.2f}"
        lines.append(line)

    return "\n".join(lines)


def histogram_data(
    values: np.ndarray,
    bins: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histogram data for visualization.

    Args:
        values: Array of values.
        bins: Number of bins.

    Returns:
        (bin_edges, counts).
    """
    counts, bin_edges = np.histogram(values, bins=bins)
    return bin_edges, counts


def posterior_summary(lplus_samples: np.ndarray) -> Dict[str, float]:
    """Summarize L⁺ posterior distribution.

    Args:
        lplus_samples: Monte Carlo samples of L⁺.

    Returns:
        Dict with mean, median, std, and various quantiles.
    """
    return {
        "mean": float(np.mean(lplus_samples)),
        "std": float(np.std(lplus_samples)),
        "median": float(np.median(lplus_samples)),
        "q025": float(np.quantile(lplus_samples, 0.025)),
        "q05": float(np.quantile(lplus_samples, 0.05)),
        "q10": float(np.quantile(lplus_samples, 0.10)),
        "q90": float(np.quantile(lplus_samples, 0.90)),
        "q95": float(np.quantile(lplus_samples, 0.95)),
        "q975": float(np.quantile(lplus_samples, 0.975)),
    }
