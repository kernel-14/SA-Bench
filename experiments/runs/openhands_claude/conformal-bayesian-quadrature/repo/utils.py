"""
Statistical utility functions: Clopper-Pearson confidence intervals,
true risk computation for synthetic experiments, and result formatting.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Tuple


def clopper_pearson_ci(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Exact Clopper-Pearson confidence interval for a binomial proportion.

    Args:
        k: number of successes.
        n: number of trials.
        confidence: confidence level (e.g. 0.95 for 95% CI).

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    alpha = 1.0 - confidence
    lower = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    upper = stats.beta.ppf(1.0 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lower), float(upper)


def true_risk_binomial(lam: float) -> float:
    """
    True expected loss for the synthetic binomial experiment (Section 5.1).

    The loss is ell(z, lambda) = (1/K) * sum_k 1{V_k > lambda} where V_k ~ Uniform(0,1).
    The expected loss is E[1{V > lambda}] = 1 - lambda.

    Args:
        lam: threshold parameter.

    Returns:
        True expected loss = 1 - lambda.
    """
    return 1.0 - lam


def true_risk_heteroskedastic(lam: float, n_mc: int = 100_000, seed: int = 0) -> float:
    """
    True expected miscoverage loss for the synthetic heteroskedastic experiment
    (Section 5.2) via Monte Carlo integration.

    X ~ Uniform(0, 4), Y | X ~ N(0, X^2).
    Loss = 1{|Y| > lambda} = 1{Y not in [-lambda, lambda]}.
    True risk = Pr(|Y| > lambda) = E_X[2 * Phi(-lambda / X)].

    Args:
        lam: half-width of prediction interval.
        n_mc: number of Monte Carlo samples for integration.
        seed: random seed.

    Returns:
        Estimated true risk.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 4.0, size=n_mc)
    Y = rng.normal(0.0, X)
    return float(np.mean(np.abs(Y) > lam))


def compute_failure_rate(
    exceeded: np.ndarray,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute the relative frequency of trials exceeding the risk threshold
    and its 95% Clopper-Pearson confidence interval.

    Args:
        exceeded: boolean array of shape (M,) indicating whether each trial
                  exceeded the risk threshold.

    Returns:
        (relative_freq, (ci_lower, ci_upper))
    """
    M = len(exceeded)
    k = int(np.sum(exceeded))
    freq = k / M
    ci = clopper_pearson_ci(k, M, confidence=0.95)
    return freq, ci


def format_results_table(
    method_names: list[str],
    relative_freqs: list[float],
    cis: list[Tuple[float, float]],
    extra_cols: dict[str, list] | None = None,
) -> str:
    """
    Format experiment results as a text table matching the paper's Tables 1-3.
    """
    header = f"{'Decision Rule':<35} {'Relative Freq.':>15} {'95% CI':>25}"
    if extra_cols:
        for col_name in extra_cols:
            header += f"  {col_name:>25}"
    lines = [header, "-" * len(header)]

    for i, (name, freq, ci) in enumerate(zip(method_names, relative_freqs, cis)):
        row = f"{name:<35} {freq * 100:>14.2f}%  [{ci[0] * 100:.2f}%, {ci[1] * 100:.2f}%]"
        if extra_cols:
            for col_vals in extra_cols.values():
                row += f"  {col_vals[i]:>25.4f}"
        lines.append(row)

    return "\n".join(lines)
