"""
Utility functions for the Conformal Prediction as Bayesian Quadrature experiments.
"""

import numpy as np
from scipy import stats


def clopper_pearson_ci(k, n, confidence=0.95):
    """
    Compute Clopper-Pearson confidence interval for a binomial proportion.
    
    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI).
    
    Returns
    -------
    lower : float
        Lower bound of confidence interval.
    upper : float
        Upper bound of confidence interval.
    """
    alpha_ci = 1 - confidence
    lower = stats.beta.ppf(alpha_ci / 2, k, n - k + 1) if k > 0 else 0.0
    upper = stats.beta.ppf(1 - alpha_ci / 2, k + 1, n - k) if k < n else 1.0
    return lower, upper


def compute_exceedance_stats(values, threshold, M, confidence=0.95):
    """
    Compute exceedance statistics with confidence intervals.
    
    Parameters
    ----------
    values : array-like of shape (M,)
        Values to compare against threshold.
    threshold : float
        Threshold value.
    M : int
        Total number of trials.
    confidence : float
        Confidence level for CI.
    
    Returns
    -------
    dict with keys:
        - relative_freq: fraction of values exceeding threshold
        - ci_low: lower bound of CI
        - ci_high: upper bound of CI
        - n_exceed: number of values exceeding threshold
    """
    values = np.asarray(values)
    exceed = values > threshold
    n_exceed = int(np.sum(exceed))
    freq = n_exceed / M
    ci_low, ci_high = clopper_pearson_ci(n_exceed, M, confidence)
    
    return {
        "relative_freq": freq,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_exceed": n_exceed,
    }


def format_table_row(name, freq, ci_low, ci_high, extra=None):
    """Format a table row for printing."""
    freq_pct = freq * 100
    ci_low_pct = ci_low * 100
    ci_high_pct = ci_high * 100
    row = f"{name:<40} {freq_pct:.2f}%{'':<10} [{ci_low_pct:.2f}%, {ci_high_pct:.2f}%]"
    if extra is not None:
        row += f"  {extra}"
    return row
