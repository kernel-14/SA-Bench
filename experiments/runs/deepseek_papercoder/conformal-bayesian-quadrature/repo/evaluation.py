# evaluation.py
"""
Evaluation metrics for "Conformal Prediction as Bayesian Quadrature".

Aggregates per‑trial results from the three decision rules and computes
the statistics reported in the paper's tables:
    - Relative frequency of exceeding the target risk α
    - 95% Clopper‑Pearson confidence intervals
    - Mean prediction set size / interval length (when applicable)

Usage:
    from evaluation import EvaluationMetrics

    em = EvaluationMetrics()
    stats = em.compute(all_trials)   # all_trials is a list of dicts
    print(stats)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Re‑use the utility implementation of Clopper‑Pearson CI.
# This keeps the statistical code in one place, while providing a
# thin wrapper that matches the class interface defined in the design.
from utils import clopper_pearson_ci as _clopper_pearson_ci


class EvaluationMetrics:
    """
    Aggregator for conformal prediction trial results.

    All methods operate on lists of dictionaries, where each dictionary
    represents the outcome of a single trial for a single method and
    contains at least the keys:
        - "method"           : str (e.g., "CRC", "RCPS", "BQC")
        - "risk_exceeds"     : bool (True if the true risk > α)
        - "interval_length"  : float | None (prediction set size or
                                interval length, omitted or None for
                                experiments without a size measure)
    """

    def clopper_pearson_ci(
        self,
        successes: int,
        n: int,
        alpha: float = 0.05,
    ) -> Tuple[float, float]:
        """
        Compute a two‑sided exact Clopper‑Pearson confidence interval
        for a binomial proportion.

        This is a thin wrapper around the implementation in `utils.py`
        for consistent interface.

        Args:
            successes: Number of observed successes (trials where risk > α).
            n: Total number of trials.
            alpha: Significance level (0.05 → 95% CI).

        Returns:
            (lower, upper) bounds as floats.
        """
        return _clopper_pearson_ci(successes, n, confidence=1.0 - alpha)

    def mean_interval_length(self, method_results: List[Dict[str, Any]]) -> Optional[float]:
        """
        Compute the average prediction set size (or interval length) over a
        list of trial dictionaries.

        Only entries whose ``"interval_length"`` key is not ``None`` are
        included.  If no valid length is found, returns ``None``.

        Args:
            method_results: List of dicts, each corresponding to one trial
                for a single decision rule.

        Returns:
            The arithmetic mean of the non‑None interval lengths, or None
            if the list is empty or contains only None values.
        """
        lengths = [
            d["interval_length"]
            for d in method_results
            if d.get("interval_length") is not None
        ]
        if not lengths:
            return None
        return float(np.mean(lengths))

    def compute(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate trial results across all methods.

        For each method found in the results, the following statistics are
        computed:
            - ``relative_freq`` : fraction of trials where true risk exceeded α
            - ``ci_lower``, ``ci_upper`` : 95% Clopper‑Pearson CI for that fraction
            - ``mean_length``   : average interval length (or set size) if
              available; otherwise None

        Args:
            results: A list of dictionaries, each containing the outcome of
                one trial for one method.  Must include keys ``"method"``,
                ``"risk_exceeds"``, and optionally ``"interval_length"``.

        Returns:
            A nested dictionary of the form::

                {
                    "CRC": {
                        "relative_freq": float,
                        "ci_lower": float,
                        "ci_upper": float,
                        "mean_length": Optional[float],
                    },
                    ...
                }
        """
        # Group trials by method name
        method_trials: Dict[str, List[Dict[str, Any]]] = {}
        for trial in results:
            m = trial["method"]
            if m not in method_trials:
                method_trials[m] = []
            method_trials[m].append(trial)

        output: Dict[str, Dict[str, Any]] = {}

        for method, trials in method_trials.items():
            if not trials:
                output[method] = {
                    "relative_freq": 0.0,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "mean_length": None,
                }
                continue

            n_total = len(trials)
            n_fail = sum(1 for t in trials if t["risk_exceeds"])
            freq = n_fail / n_total

            # 95% Clopper‑Pearson confidence interval
            ci_lower, ci_upper = self.clopper_pearson_ci(
                n_fail, n_total, alpha=0.05
            )

            # Mean prediction set size / interval length (if available)
            mean_len = self.mean_interval_length(trials)

            output[method] = {
                "relative_freq": freq,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "mean_length": mean_len,
            }

        return output

