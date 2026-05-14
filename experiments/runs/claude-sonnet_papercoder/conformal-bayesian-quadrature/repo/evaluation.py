## evaluation.py
"""Statistical evaluation and result aggregation for CBQ experiments.

This module converts raw per-trial outputs from the experiment runners into
the summary statistics that appear in Tables 1, 2, and 3 of the paper. It
is a pure aggregation layer with no stochastic operations.

Key responsibilities:
  - TrialResult: atomic per-trial output (λ chosen + risk incurred, all methods)
  - ExperimentResult: aggregated statistics across M trials for one method
  - clopper_pearson_ci: exact Clopper-Pearson binomial proportion CI
  - compute_experiment_results: aggregate M TrialResults into 3 ExperimentResults
  - summarize_to_dataframe: format results as a pandas DataFrame (paper table style)
  - print_table: print formatted table to stdout

References:
    Paper Tables 1, 2, 3: failure rates and Clopper-Pearson CIs.
    config.yaml: evaluation.ci_confidence = 0.95, experiment.M = 10000.
    Paper text (Section 5.1): "CRC mean 0.3363 ± 0.0007" — ± is std/sqrt(M).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta as scipy_beta

from config import ExperimentConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Atomic output from a single Monte Carlo trial.

    Stores the decision rule output (chosen λ) and the true risk incurred
    for all three methods simultaneously. Using a single dataclass per trial
    ensures that all methods operate on the same calibration data draw.

    Attributes:
        lambda_crc: λ chosen by Conformal Risk Control (Proposition 3.2).
            May be np.inf if no λ in the grid satisfies the CRC criterion.
        lambda_rcps: λ chosen by RCPS with Hoeffding UCB (Bates et al., 2021).
            May be np.inf if no λ satisfies the UCB criterion.
        lambda_hpd: λ chosen by CBQ-HPD at β=0.95 (Corollary 4.4).
            May be np.inf if no λ achieves Pr(L⁺ ≤ α) ≥ β.
        risk_crc: True risk incurred by lambda_crc in this trial.
            Exp1: 1 − lambda_crc (analytical).
            Exp2: E_X[2Φ(−λ/X)] via numerical integration.
            Exp3: empirical FNR on held-out test set.
        risk_rcps: True risk incurred by lambda_rcps. Same computation as
            risk_crc but for the RCPS-chosen threshold.
        risk_hpd: True risk incurred by lambda_hpd. Same computation.
        extra: Method-specific supplementary values. Keys per experiment:
            Exp1: {} (empty)
            Exp2: {"interval_length_crc": float, "interval_length_rcps": float,
                   "interval_length_hpd": float}  — each = 2 × λ
            Exp3: {"pred_set_size_crc": float, "pred_set_size_rcps": float,
                   "pred_set_size_hpd": float}  — mean prediction set size on test set
    """

    lambda_crc: float = np.inf
    lambda_rcps: float = np.inf
    lambda_hpd: float = np.inf
    risk_crc: float = 0.0
    risk_rcps: float = 0.0
    risk_hpd: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Aggregated statistics for one method across all M trials.

    Three ExperimentResult objects are produced per experiment (one per
    method: CRC, RCPS, CBQ-HPD). These map directly to rows in the paper's
    Tables 1, 2, and 3.

    Attributes:
        method_name: Display name for the method. One of:
            "CRC", "RCPS", "CBQ-HPD (β=0.95)".
        config_name: Experiment identifier from ExperimentConfig.exp_name.
            E.g., "synthetic_binomial", "synthetic_heteroskedastic", "mscoco".
        failure_rate: Fraction of trials where true risk exceeded α.
            k / M where k = #{trials : risk > α}. This is the "Relative Freq."
            column in the paper's tables.
        ci_low: Lower bound of 95% Clopper-Pearson CI for failure_rate.
            Expressed as a proportion in [0, 1] (not a percentage).
        ci_high: Upper bound of 95% Clopper-Pearson CI for failure_rate.
            Expressed as a proportion in [0, 1].
        mean_lambda: Mean of chosen λ values across M trials (finite values
            only — np.inf entries are excluded from the mean).
        std_lambda: Population standard deviation of chosen λ values (ddof=0,
            finite values only). Standard error = std_lambda / sqrt(M).
        mean_extra: Averaged extra fields across all trials. Keys:
            Exp2: {"mean_interval_length": float}
            Exp3: {"mean_pred_set_size": float}
            Exp1: {} (empty)
    """

    method_name: str = ""
    config_name: str = ""
    failure_rate: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    mean_lambda: float = 0.0
    std_lambda: float = 0.0
    mean_extra: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Clopper-Pearson confidence interval
# ---------------------------------------------------------------------------


def clopper_pearson_ci(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute the exact Clopper-Pearson CI for a binomial proportion k/n.

    The Clopper-Pearson interval inverts two one-sided binomial tests. It is
    the exact method stated in the paper: "Error bars are computed as 95%
    Clopper-Pearson confidence intervals for binomial proportions."

    Mathematical basis:
        lower = Beta_{α/2}(k, n−k+1)     [α/2 quantile of Beta(k, n−k+1)]
        upper = Beta_{1−α/2}(k+1, n−k)   [1−α/2 quantile of Beta(k+1, n−k)]

    where α = 1 − confidence = 0.05 for 95% CI.

    Edge cases:
        k = 0: lower = 0.0 exactly (no failures observed).
               upper = Beta_{1−α/2}(1, n) = 1 − (α/2)^{1/n}.
        k = n: upper = 1.0 exactly (all trials failed).
               lower = Beta_{α/2}(n, 1) = (α/2)^{1/n}.

    Verification against paper (Table 1):
        CRC: k ≈ 2120, n = 10000 → CI ≈ [20.40%, 22.01%]
        RCPS: k = 0, n = 10000 → CI = [0.00%, 0.04%]
        CBQ-HPD: k ≈ 3, n = 10000 → CI ≈ [0.01%, 0.09%]

    Config reference: evaluation.ci_confidence = 0.95 (config.yaml).

    Args:
        k: Number of "successes" (failures in our context — trials where
            risk > α). Must satisfy 0 ≤ k ≤ n.
        n: Total number of trials. Must be > 0. Per config.yaml:
            experiment.M = 10000.
        confidence: Desired confidence level. Default 0.95 per config.yaml
            (evaluation.ci_confidence = 0.95). Must be in (0, 1).

    Returns:
        Tuple (lower, upper) where both values are proportions in [0, 1].
        Multiply by 100 to convert to percentages for display.

    Raises:
        ValueError: If k < 0, k > n, n <= 0, or confidence not in (0, 1).

    Example:
        >>> # RCPS: k=0, n=10000 → [0.00%, 0.04%]
        >>> lo, hi = clopper_pearson_ci(0, 10000)
        >>> abs(lo) < 1e-10
        True
        >>> abs(hi * 100 - 0.04) < 0.01
        True
        >>> # CRC: k=2120, n=10000 → [20.40%, 22.01%]
        >>> lo2, hi2 = clopper_pearson_ci(2120, 10000)
        >>> abs(lo2 * 100 - 20.40) < 0.05
        True
        >>> abs(hi2 * 100 - 22.01) < 0.05
        True
    """
    # Input validation.
    if n <= 0:
        raise ValueError(f"n must be positive, got n={n}.")
    if k < 0 or k > n:
        raise ValueError(
            f"k must satisfy 0 ≤ k ≤ n, got k={k}, n={n}."
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(
            f"confidence must be in (0, 1), got confidence={confidence}."
        )

    # α for the CI (not the risk level α from the paper — different usage here).
    alpha_ci: float = 1.0 - confidence

    # ------------------------------------------------------------------
    # Edge case: k = 0 (no failures observed)
    # ------------------------------------------------------------------
    if k == 0:
        lower: float = 0.0
        # upper = Beta_{1 − α/2}(k+1, n−k) = Beta_{1 − α/2}(1, n)
        upper: float = float(scipy_beta.ppf(1.0 - alpha_ci / 2.0, 1, n))
        return lower, upper

    # ------------------------------------------------------------------
    # Edge case: k = n (all trials failed)
    # ------------------------------------------------------------------
    if k == n:
        # lower = Beta_{α/2}(k, n−k+1) = Beta_{α/2}(n, 1)
        lower = float(scipy_beta.ppf(alpha_ci / 2.0, n, 1))
        upper = 1.0
        return lower, upper

    # ------------------------------------------------------------------
    # General case: 0 < k < n
    # ------------------------------------------------------------------
    # lower = α/2 quantile of Beta(k, n−k+1)
    lower = float(scipy_beta.ppf(alpha_ci / 2.0, k, n - k + 1))

    # upper = 1 − α/2 quantile of Beta(k+1, n−k)
    upper = float(scipy_beta.ppf(1.0 - alpha_ci / 2.0, k + 1, n - k))

    # Clamp to [0, 1] to handle any tiny numerical overshoot.
    lower = float(np.clip(lower, 0.0, 1.0))
    upper = float(np.clip(upper, 0.0, 1.0))

    return lower, upper


# ---------------------------------------------------------------------------
# Aggregation: M TrialResults → 3 ExperimentResults
# ---------------------------------------------------------------------------


def compute_experiment_results(
    trial_results: List[TrialResult],
    config: ExperimentConfig,
) -> List[ExperimentResult]:
    """Aggregate M TrialResult objects into three ExperimentResult objects.

    Produces one ExperimentResult per method (CRC, RCPS, CBQ-HPD), computing
    failure rates, Clopper-Pearson CIs, lambda statistics, and averaged extra
    fields. This function generates the numbers that appear in Tables 1–3.

    Failure criterion: risk > α (strict inequality), matching the paper's
    language "exceeding the target risk threshold α".

    Lambda statistics: computed over finite λ values only (np.inf entries
    are excluded). np.inf occurs when no λ in the grid satisfies the
    decision rule criterion — the corresponding risk is 0 (no failure).

    Extra field aggregation:
        Exp2: collects "interval_length_{method}" from each trial's extra dict
              and stores the mean as "mean_interval_length" in mean_extra.
        Exp3: collects "pred_set_size_{method}" and stores as "mean_pred_set_size".
        Exp1: extra dicts are empty; mean_extra is {}.

    Args:
        trial_results: List of M TrialResult objects, one per trial. Length
            must equal config.M (= 10000 per config.yaml).
        config: Experiment configuration providing config.alpha (risk threshold)
            and config.M (total trials). The alpha values per config.yaml are:
            - BinomialConfig.alpha = 0.4
            - HeteroskedasticConfig.alpha = 0.1
            - MSCOCOConfig.alpha = 0.1

    Returns:
        List of three ExperimentResult objects in order:
        [CRC result, RCPS result, CBQ-HPD result].

    Raises:
        ValueError: If trial_results is empty.

    Example:
        >>> from config import BinomialConfig
        >>> cfg = BinomialConfig()
        >>> # Simulate 3 trials: CRC fails 1/3, RCPS never fails, HPD never fails
        >>> trials = [
        ...     TrialResult(lambda_crc=0.5, lambda_rcps=0.7, lambda_hpd=0.65,
        ...                 risk_crc=0.5, risk_rcps=0.3, risk_hpd=0.35),
        ...     TrialResult(lambda_crc=0.7, lambda_rcps=0.8, lambda_hpd=0.75,
        ...                 risk_crc=0.3, risk_rcps=0.2, risk_hpd=0.25),
        ...     TrialResult(lambda_crc=0.6, lambda_rcps=0.9, lambda_hpd=0.8,
        ...                 risk_crc=0.4, risk_rcps=0.1, risk_hpd=0.2),
        ... ]
        >>> cfg.M = 3
        >>> results = compute_experiment_results(trials, cfg)
        >>> len(results)
        3
        >>> results[0].method_name
        'CRC'
        >>> results[0].failure_rate  # 1 trial has risk_crc=0.5 > alpha=0.4
        0.3333333333333333
    """
    if not trial_results:
        raise ValueError("trial_results must not be empty.")

    M: int = config.M
    alpha: float = config.alpha

    # ------------------------------------------------------------------
    # Step 1: Extract per-method arrays from the list of TrialResults.
    # ------------------------------------------------------------------
    lambdas_crc: np.ndarray = np.array(
        [r.lambda_crc for r in trial_results], dtype=float
    )
    lambdas_rcps: np.ndarray = np.array(
        [r.lambda_rcps for r in trial_results], dtype=float
    )
    lambdas_hpd: np.ndarray = np.array(
        [r.lambda_hpd for r in trial_results], dtype=float
    )
    risks_crc: np.ndarray = np.array(
        [r.risk_crc for r in trial_results], dtype=float
    )
    risks_rcps: np.ndarray = np.array(
        [r.risk_rcps for r in trial_results], dtype=float
    )
    risks_hpd: np.ndarray = np.array(
        [r.risk_hpd for r in trial_results], dtype=float
    )

    # ------------------------------------------------------------------
    # Step 2: Compute failure counts (strict inequality: risk > alpha).
    # ------------------------------------------------------------------
    k_crc: int = int(np.sum(risks_crc > alpha))
    k_rcps: int = int(np.sum(risks_rcps > alpha))
    k_hpd: int = int(np.sum(risks_hpd > alpha))

    # ------------------------------------------------------------------
    # Step 3: Compute failure rates.
    # ------------------------------------------------------------------
    failure_rate_crc: float = k_crc / M
    failure_rate_rcps: float = k_rcps / M
    failure_rate_hpd: float = k_hpd / M

    # ------------------------------------------------------------------
    # Step 4: Compute Clopper-Pearson CIs (95% per config.yaml).
    # ------------------------------------------------------------------
    _CI_CONFIDENCE: float = 0.95  # evaluation.ci_confidence from config.yaml

    ci_low_crc, ci_high_crc = clopper_pearson_ci(k_crc, M, _CI_CONFIDENCE)
    ci_low_rcps, ci_high_rcps = clopper_pearson_ci(k_rcps, M, _CI_CONFIDENCE)
    ci_low_hpd, ci_high_hpd = clopper_pearson_ci(k_hpd, M, _CI_CONFIDENCE)

    # ------------------------------------------------------------------
    # Step 5: Compute lambda statistics (finite values only).
    # np.inf entries arise when no λ in the grid satisfies the criterion.
    # These correspond to risk = 0 (no failure) and are excluded from
    # mean/std to avoid distorting the statistics.
    # ------------------------------------------------------------------
    def _lambda_stats(lambdas: np.ndarray) -> Tuple[float, float]:
        """Compute mean and population std of finite lambda values."""
        finite_mask: np.ndarray = np.isfinite(lambdas)
        finite_lambdas: np.ndarray = lambdas[finite_mask]
        if len(finite_lambdas) == 0:
            return np.inf, 0.0
        mean_val: float = float(np.mean(finite_lambdas))
        std_val: float = float(np.std(finite_lambdas, ddof=0))
        return mean_val, std_val

    mean_lambda_crc, std_lambda_crc = _lambda_stats(lambdas_crc)
    mean_lambda_rcps, std_lambda_rcps = _lambda_stats(lambdas_rcps)
    mean_lambda_hpd, std_lambda_hpd = _lambda_stats(lambdas_hpd)

    # ------------------------------------------------------------------
    # Step 6: Aggregate extra fields.
    # Use defensive .get() with None fallback to handle Exp1 (empty extra).
    # ------------------------------------------------------------------
    mean_extra_crc: Dict[str, float] = _aggregate_extra(
        trial_results, method_suffix="crc"
    )
    mean_extra_rcps: Dict[str, float] = _aggregate_extra(
        trial_results, method_suffix="rcps"
    )
    mean_extra_hpd: Dict[str, float] = _aggregate_extra(
        trial_results, method_suffix="hpd"
    )

    # ------------------------------------------------------------------
    # Step 7: Construct and return ExperimentResult objects.
    # ------------------------------------------------------------------
    config_name: str = config.exp_name

    results: List[ExperimentResult] = [
        ExperimentResult(
            method_name="CRC",
            config_name=config_name,
            failure_rate=failure_rate_crc,
            ci_low=ci_low_crc,
            ci_high=ci_high_crc,
            mean_lambda=mean_lambda_crc,
            std_lambda=std_lambda_crc,
            mean_extra=mean_extra_crc,
        ),
        ExperimentResult(
            method_name="RCPS",
            config_name=config_name,
            failure_rate=failure_rate_rcps,
            ci_low=ci_low_rcps,
            ci_high=ci_high_rcps,
            mean_lambda=mean_lambda_rcps,
            std_lambda=std_lambda_rcps,
            mean_extra=mean_extra_rcps,
        ),
        ExperimentResult(
            method_name="CBQ-HPD (β=0.95)",
            config_name=config_name,
            failure_rate=failure_rate_hpd,
            ci_low=ci_low_hpd,
            ci_high=ci_high_hpd,
            mean_lambda=mean_lambda_hpd,
            std_lambda=std_lambda_hpd,
            mean_extra=mean_extra_hpd,
        ),
    ]

    return results


def _aggregate_extra(
    trial_results: List[TrialResult],
    method_suffix: str,
) -> Dict[str, float]:
    """Aggregate per-trial extra fields for one method into mean values.

    Inspects the ``extra`` dict of each TrialResult for keys ending in
    ``method_suffix`` (e.g., "interval_length_crc", "pred_set_size_rcps")
    and computes the mean across all trials where the key is present.

    The output dict uses canonical keys without the method suffix:
      - "interval_length_{suffix}" → "mean_interval_length"
      - "pred_set_size_{suffix}"   → "mean_pred_set_size"

    Args:
        trial_results: List of M TrialResult objects.
        method_suffix: One of "crc", "rcps", "hpd". Used to look up the
            correct key in each trial's extra dict.

    Returns:
        Dict mapping canonical key names to their mean values across trials.
        Returns {} if no relevant keys are found (Exp1 case).
    """
    # Collect values for each known extra key pattern.
    interval_lengths: List[float] = []
    pred_set_sizes: List[float] = []

    interval_key: str = f"interval_length_{method_suffix}"
    pred_set_key: str = f"pred_set_size_{method_suffix}"

    for r in trial_results:
        il_val: Optional[Any] = r.extra.get(interval_key)
        if il_val is not None:
            interval_lengths.append(float(il_val))

        ps_val: Optional[Any] = r.extra.get(pred_set_key)
        if ps_val is not None:
            pred_set_sizes.append(float(ps_val))

    mean_extra: Dict[str, float] = {}

    if interval_lengths:
        mean_extra["mean_interval_length"] = float(np.mean(interval_lengths))

    if pred_set_sizes:
        mean_extra["mean_pred_set_size"] = float(np.mean(pred_set_sizes))

    return mean_extra


# ---------------------------------------------------------------------------
# DataFrame formatting
# ---------------------------------------------------------------------------


def summarize_to_dataframe(
    results: List[ExperimentResult],
) -> pd.DataFrame:
    """Convert ExperimentResult objects into a pandas DataFrame (paper table style).

    Produces a DataFrame that mirrors the paper's Tables 1, 2, and 3.
    Column names and formatting match the paper exactly:

    Table 1 (synthetic_binomial):
        ["Decision Rule", "Relative Freq.", "95% CI"]

    Table 2 (synthetic_heteroskedastic):
        ["Decision Rule", "Relative Freq.", "95% CI", "Mean Prediction Interval Length"]

    Table 3 (mscoco):
        ["Method", "Relative Freq.", "Pred. Set Size"]

    The first column name is inferred from the experiment type:
        - "mscoco" → "Method" (Table 3 header)
        - all others → "Decision Rule" (Tables 1 and 2 header)

    All numeric values are pre-formatted as strings to avoid float precision
    issues in display. The "Relative Freq." column uses 2 decimal places
    (e.g., "21.20%"). The CI column uses the format "[XX.XX%, YY.YY%]".

    Args:
        results: List of ExperimentResult objects, typically three (one per
            method: CRC, RCPS, CBQ-HPD). All must have the same config_name.

    Returns:
        A pandas DataFrame with one row per method and columns matching the
        paper's table format. All values are strings for clean display.

    Raises:
        ValueError: If results is empty.

    Example:
        >>> from config import BinomialConfig
        >>> cfg = BinomialConfig()
        >>> cfg.M = 10000
        >>> r1 = ExperimentResult("CRC", "synthetic_binomial",
        ...                       0.2120, 0.2040, 0.2201, 0.3363, 0.07, {})
        >>> r2 = ExperimentResult("RCPS", "synthetic_binomial",
        ...                       0.0000, 0.0000, 0.0004, 0.9, 0.05, {})
        >>> r3 = ExperimentResult("CBQ-HPD (β=0.95)", "synthetic_binomial",
        ...                       0.0003, 0.0001, 0.0009, 0.7, 0.06, {})
        >>> df = summarize_to_dataframe([r1, r2, r3])
        >>> list(df.columns)
        ['Decision Rule', 'Relative Freq.', '95% CI']
        >>> df.iloc[0]["Relative Freq."]
        '21.20%'
    """
    if not results:
        raise ValueError("results must not be empty.")

    # Determine the first column name based on experiment type.
    config_name: str = results[0].config_name
    first_col: str = "Method" if config_name == "mscoco" else "Decision Rule"

    # Determine which extra columns are present by inspecting the first result.
    # All results in the list should have the same extra key structure.
    has_interval_length: bool = any(
        "mean_interval_length" in r.mean_extra for r in results
    )
    has_pred_set_size: bool = any(
        "mean_pred_set_size" in r.mean_extra for r in results
    )

    # Build rows.
    rows: List[Dict[str, str]] = []
    for result in results:
        row: Dict[str, str] = {
            first_col: result.method_name,
            "Relative Freq.": f"{result.failure_rate * 100:.2f}%",
            "95% CI": (
                f"[{result.ci_low * 100:.2f}%, {result.ci_high * 100:.2f}%]"
            ),
        }

        # Table 2 extra column: Mean Prediction Interval Length.
        if has_interval_length:
            mean_il: float = result.mean_extra.get("mean_interval_length", float("nan"))
            row["Mean Prediction Interval Length"] = f"{mean_il:.2f}"

        # Table 3 extra column: Pred. Set Size.
        if has_pred_set_size:
            mean_ps: float = result.mean_extra.get("mean_pred_set_size", float("nan"))
            row["Pred. Set Size"] = f"{mean_ps:.2f}"

        rows.append(row)

    # Construct DataFrame with explicit column ordering.
    columns: List[str] = [first_col, "Relative Freq.", "95% CI"]
    if has_interval_length:
        columns.append("Mean Prediction Interval Length")
    if has_pred_set_size:
        columns.append("Pred. Set Size")

    df: pd.DataFrame = pd.DataFrame(rows, columns=columns)
    return df


# ---------------------------------------------------------------------------
# Table display
# ---------------------------------------------------------------------------


def print_table(df: pd.DataFrame, title: str = "") -> None:
    """Print a formatted table to stdout matching the paper's presentation.

    Outputs a separator line, the title (centered), another separator, the
    DataFrame contents (without row index), and a final separator. This
    format is used for all three tables in the paper.

    Args:
        df: DataFrame produced by summarize_to_dataframe. All values should
            be pre-formatted strings for clean display.
        title: Optional title string printed above the table. Examples:
            "Table 1: Synthetic Binomial Results (Section 5.1)"
            "Table 2: Synthetic Heteroskedastic Results (Section 5.2)"
            "Table 3: MS-COCO False Negative Rate Results (Section 5.3)"

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "Method": ["CRC", "RCPS"],
        ...     "Relative Freq.": ["21.20%", "0.00%"],
        ...     "95% CI": ["[20.40%, 22.01%]", "[0.00%, 0.04%]"],
        ... })
        >>> print_table(df, "Table 1: Synthetic Binomial")
        ======================================================================
                              Table 1: Synthetic Binomial
        ======================================================================
        Method Relative Freq.              95% CI
           CRC         21.20%  [20.40%, 22.01%]
          RCPS          0.00%    [0.00%, 0.04%]
        ======================================================================
    """
    _SEP_WIDTH: int = 70
    separator: str = "=" * _SEP_WIDTH

    print(separator)
    if title:
        print(title.center(_SEP_WIDTH))
        print(separator)

    # to_string with index=False suppresses the row index, matching the
    # paper's table format. justify="right" aligns numeric columns cleanly.
    print(df.to_string(index=False, justify="right"))
    print(separator)
