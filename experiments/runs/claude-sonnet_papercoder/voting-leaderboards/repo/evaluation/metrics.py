```python
## evaluation/metrics.py
"""Centralized metric computation for the adversarial leaderboard manipulation paper.

This module provides the Metrics class, a stateless collection of static methods
that compute all evaluation metrics used across the three experimental components:

  1. De-anonymization accuracy (Section 2.3, Tables 2, 3, 7, Figure 3)
  2. Leaderboard manipulation simulation results (Section 3, Tables 4, 5, 8, 9)
  3. Mitigation effectiveness (Section 4.3, Figures 4, 5, 6)

All methods are @staticmethod — no instance state is needed. The module depends
only on data_structures.py (for SimulationResult) and standard scientific Python
libraries (numpy, pandas, scikit-learn).

Paper alignment:
  - Section 2.3: "We evaluate the classifier using the average test accuracy
    across all prompts."
  - Section 3.1: Tables 4 and 5 report n_votes and n_interactions to achieve
    each target rank.
  - Section 4.3: Figures 4, 5, 6 show detection rate vs. votes and utility
    loss vs. noise scale.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from data_structures import SimulationResult

# ---------------------------------------------------------------------------
# Sentinel string for unreached / impossible target ranks in simulation tables.
# The paper uses "N/A" in Tables 4, 5, 8, 9 for these cells.
# Using a string (not np.nan) preserves the paper's exact table format in CSV.
# ---------------------------------------------------------------------------
_NA_STRING: str = "N/A"

# ---------------------------------------------------------------------------
# Valid metric identifiers for summarize_simulation_table.
# ---------------------------------------------------------------------------
_VALID_METRICS: frozenset = frozenset({"n_votes", "n_interactions"})

# ---------------------------------------------------------------------------
# Column names for detection rate DataFrames (Figures 4, 5).
# ---------------------------------------------------------------------------
_COL_VOTE_COUNT: str = "vote_count"
_COL_ADVERSARY_TYPE: str = "adversary_type"
_COL_DETECTION_RATE: str = "detection_rate"

# ---------------------------------------------------------------------------
# Column names for utility/noise DataFrame (Figure 6).
# ---------------------------------------------------------------------------
_COL_NOISE_SCALE: str = "noise_scale"
_COL_AVG_RANK_CHANGE: str = "avg_rank_change"


class Metrics:
    """Stateless utility class for computing all evaluation metrics.

    All methods are @staticmethod — instantiation is not required. Callers
    use Metrics.method_name(...) directly.

    This class centralizes metric computation to ensure consistent handling
    of edge cases (empty inputs, unachieved simulation targets, missing dict
    keys) across all three experimental components.

    Example:
        >>> from data_structures import SimulationResult
        >>> result = SimulationResult("llama-13b", 128, True, 126, 10000)
        >>> df = Metrics.summarize_simulation_table(
        ...     [result],
        ...     row_models=["llama-13b"],
        ...     col_ranks=[128],
        ...     metric="n_votes",
        ... )
        >>> df.loc["llama-13b", "Target rank: 128"]
        126
    """

    # -----------------------------------------------------------------------
    # De-anonymization metrics
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_detection_accuracy(
        y_true: List[int],
        y_pred: List[int],
    ) -> float:
        """Compute binary classification accuracy for the training-based detector.

        Thin wrapper around sklearn.metrics.accuracy_score for the binary
        classification task: target model (class 1) vs. all other models (class 0).

        Paper alignment: Section 2.3 — "We evaluate the classifier using the
        average test accuracy across all prompts." With the 80/20 split on 100
        samples, the test set has exactly 20 samples per (prompt, model) pair.

        Args:
            y_true: Ground-truth binary labels. Each element is 0 (non-target
                model) or 1 (target model). Length equals the test set size
                (typically 20 per the paper's 80/20 split on 100 samples).
            y_pred: Predicted binary labels from the logistic regression
                classifier. Same length as y_true.

        Returns:
            Accuracy as a float in [0.0, 1.0]. NOT multiplied by 100 —
            percentage conversion is handled by callers (TrainingBasedDetector
            multiplies by 100 when building DataFrames for Table 3 / Figure 3).

        Raises:
            ValueError: If y_true and y_pred have different lengths (raised
                by sklearn.metrics.accuracy_score).

        Example:
            >>> Metrics.compute_detection_accuracy([1, 0, 1, 0], [1, 0, 0, 0])
            0.75
            >>> Metrics.compute_detection_accuracy([1, 1, 0, 0], [1, 1, 0, 0])
            1.0
            >>> Metrics.compute_detection_accuracy([], [])
            0.0
        """
        # Handle empty inputs gracefully — sklearn raises for empty arrays.
        if len(y_true) == 0 and len(y_pred) == 0:
            return 0.0

        return float(accuracy_score(y_true, y_pred))

    @staticmethod
    def compute_average_accuracy_per_category(
        results: Dict[str, List[float]],
    ) -> Dict[str, float]:
        """Average per-prompt test accuracies across all prompts within each category.

        Computes the mean accuracy over all prompts in each category for a
        single target model. This produces the single accuracy value per
        (model, category) cell shown in Figure 3 of the paper.

        Paper alignment: Section 2.4.2 — "We evaluate the classifier using
        the average test accuracy across all prompts." Figure 3 shows one
        accuracy value per (model, category) cell, which is this average.

        Args:
            results: Dict mapping category name (e.g., "english", "math") to
                a list of per-prompt accuracy floats in [0.0, 1.0]. Each list
                has length n_prompts_per_category (200 per config.yaml). Values
                are in [0.0, 1.0] range (not percentages).

        Returns:
            Dict mapping category name to mean accuracy as a float in [0.0, 1.0].
            Returns np.nan for categories with empty accuracy lists (defensive
            handling for incomplete data collection).

        Example:
            >>> results = {"english": [0.95, 0.97, 0.93], "math": [0.99, 1.0]}
            >>> Metrics.compute_average_accuracy_per_category(results)
            {'english': 0.9833..., 'math': 0.995}
            >>> Metrics.compute_average_accuracy_per_category({"empty_cat": []})
            {'empty_cat': nan}
        """
        averaged: Dict[str, float] = {}

        for category_name, accuracy_list in results.items():
            if not accuracy_list:
                # Empty list: return nan as a sentinel for missing data.
                averaged[category_name] = float("nan")
            else:
                averaged[category_name] = float(np.mean(accuracy_list))

        return averaged

    # -----------------------------------------------------------------------
    # Simulation result metrics
    # -----------------------------------------------------------------------

    @staticmethod
    def summarize_simulation_table(
        results: List[SimulationResult],
        row_models: List[str],
        col_ranks: List[int],
        metric: str = "n_votes",
    ) -> pd.DataFrame:
        """Format SimulationResult objects into the paper's table format.

        Builds a DataFrame matching Tables 4, 5, 8, and 9 of the paper, where:
          - Rows are target models (or detector accuracy levels for Table 8,
            or non-target strategies for Table 9 — the caller passes the
            appropriate row labels via row_models).
          - Columns are target ranks.
          - Cell values are n_votes or n_interactions (selected by metric).
          - Cells where the target rank was not achieved are "N/A".

        Paper alignment:
          - Table 4(a): metric="n_votes", high-ranked models.
          - Table 4(b): metric="n_interactions", high-ranked models.
          - Table 5(a): metric="n_votes", low-ranked models.
          - Table 5(b): metric="n_interactions", low-ranked models.
          - Table 8(a): metric="n_votes", ablation on detector accuracy.
          - Table 8(b): metric="n_interactions", ablation on detector accuracy.
          - Table 9: metric="n_interactions", ablation on non-target strategy.

        Args:
            results: List of SimulationResult objects from AttackSimulator.
                Each provides target_model, target_rank, achieved, n_votes,
                and n_interactions. Multiple results for the same
                (target_model, target_rank) pair are handled by taking the
                last one (results are assumed to be unique per pair).
            row_models: Ordered list of model name strings (or other row
                identifiers) that determine the row order and index of the
                output DataFrame. Rows not present in results are filled with
                "N/A". The caller passes config.simulation.high_ranked_targets
                names or similar.
            col_ranks: Ordered list of target rank integers that determine
                the column order. Columns not present in results are filled
                with "N/A". Column headers are formatted as "Target rank: {rank}".
            metric: Which metric to report in cell values. Must be one of
                "n_votes" (default, for (a) sub-tables) or "n_interactions"
                (for (b) sub-tables). Raises ValueError for unknown values.

        Returns:
            pd.DataFrame with:
              - Index: row_models strings (e.g., model names).
              - Columns: ["Target rank: {r}" for r in col_ranks].
              - Values: int (n_votes or n_interactions) for achieved results,
                or the string "N/A" for unachieved/impossible targets.
            The DataFrame has mixed types (int and str) in cells, which is
            intentional to match the paper's table format exactly.

        Raises:
            ValueError: If metric is not "n_votes" or "n_interactions".

        Example:
            >>> results = [
            ...     SimulationResult("llama-13b", 128, True, 126, 10000),
            ...     SimulationResult("llama-13b", 127, True, 255, 15000),
            ...     SimulationResult("llama-13b", 79, False, 0, 500000),
            ... ]
            >>> df = Metrics.summarize_simulation_table(
            ...     results,
            ...     row_models=["llama-13b"],
            ...     col_ranks=[79, 127, 128],
            ...     metric="n_votes",
            ... )
            >>> df.loc["llama-13b", "Target rank: 128"]
            126
            >>> df.loc["llama-13b", "Target rank: 79"]
            'N/A'
        """
        # --- Validate metric parameter ---
        if metric not in _VALID_METRICS:
            raise ValueError(
                f"summarize_simulation_table: metric='{metric}' is invalid. "
                f"Must be one of {sorted(_VALID_METRICS)}."
            )

        # --- Build lookup: (target_model, target_rank) -> SimulationResult ---
        # If multiple results exist for the same pair, the last one wins.
        result_lookup: Dict[tuple, SimulationResult] = {}
        for result in results:
            key: tuple = (result.target_model, result.target_rank)
            result_lookup[key] = result

        # --- Build column headers ---
        col_headers: List[str] = [f"Target rank: {r}" for r in col_ranks]

        # --- Build table data ---
        # Structure: table_data[row_model][col_header] = value (int or "N/A")
        table_data: Dict[str, Dict[str, Union[int, str]]] = {}

        for row_model in row_models:
            row_dict: Dict[str, Union[int, str]] = {}

            for rank, col_header in zip(col_ranks, col_headers):
                key = (row_model, rank)
                result: Optional[SimulationResult] = result_lookup.get(key)

                if result is None:
                    # No simulation result for this (model, rank) pair.
                    # This can happen when the target rank equals the current
                    # rank (no movement needed) or when the experiment was
                    # not run for this combination.
                    row_dict[col_header] = _NA_STRING

                elif not result.achieved:
                    # Target rank was not reached within max_interactions.
                    # Paper tables show "N/A" for these cells.
                    row_dict[col_header] = _NA_STRING

                else:
                    # Target rank was achieved — report the requested metric.
                    if metric == "n_votes":
                        row_dict[col_header] = result.n_votes
                    else:
                        # metric == "n_interactions"
                        row_dict[col_header] = result.n_interactions

            table_data[row_model] = row_dict

        # --- Construct DataFrame ---
        if not table_data:
            # No rows — return empty DataFrame with correct column structure.
            return pd.DataFrame(columns=col_headers)

        df: pd.DataFrame = pd.DataFrame.from_dict(
            table_data,
            orient="index",
            columns=col_headers,
        )

        # Ensure the index name is set for clarity in CSV output.
        df.index.name = "Target model"

        # Ensure row order matches the requested row_models order.
        # Rows not in table_data (shouldn't happen, but defensive) are dropped.
        available_rows: List[str] = [m for m in row_models if m in df.index]
        df = df.loc[available_rows]

        return df

    # -----------------------------------------------------------------------
    # Mitigation effectiveness metrics
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_detection_rate_curve(
        detection_results: Dict[str, Any],
    ) -> pd.DataFrame:
        """Convert raw trial-level detection results into a tidy DataFrame.

        Transforms the nested dict produced by MaliciousUserDetector.evaluate_scenario1
        or PerturbedLeaderboard.evaluate_scenario2 into a tidy DataFrame with
        one row per (adversary_type, vote_count) combination. This format is
        consumed directly by Visualizer.plot_detection_rate_vs_votes to produce
        Figures 4 and 5.

        Paper alignment:
          - Figure 4: Detection rate vs. number of malicious votes for naive
            vs. informed adversary (Scenario 1).
          - Figure 5: Detection rate vs. number of malicious votes for different
            noise scales (Scenario 2, one curve per noise scale).

        Expected input schema (from MaliciousUserDetector.evaluate_scenario1):
            {
                "vote_counts": [10, 20, 50, 100, 200, 500, 1000],
                "results": {
                    "naive": {10: [True, False, ...], 20: [...], ...},
                    "informed": {10: [...], 20: [...], ...}
                }
            }

        Expected input schema (from PerturbedLeaderboard.evaluate_scenario2,
        for Figure 5 — detection rate curves per noise scale):
            {
                "vote_counts": [10, 20, 50, 100, 200, 500, 1000],
                "results": {
                    "noise_0.1": {10: [True, False, ...], ...},
                    "noise_0.5": {...},
                    ...
                }
            }

        Args:
            detection_results: Dict with keys "vote_counts" (List[int]) and
                "results" (Dict[str, Dict[int, List[bool]]]). The outer key
                of "results" is the adversary type label (e.g., "naive",
                "informed", or "noise_1.0"). The inner key is the vote count
                (int). The value is a list of n_trials boolean detection
                outcomes.

        Returns:
            pd.DataFrame with columns:
              - "vote_count" (int): Number of malicious votes in the sequence.
              - "adversary_type" (str): Label for the adversary type or noise
                scale (e.g., "naive", "informed", "noise_1.0").
              - "detection_rate" (float): Fraction of n_trials where the
                adversary was detected. In [0.0, 1.0].
            Sorted by ["adversary_type", "vote_count"] for consistent ordering.
            Returns an empty DataFrame with the correct columns if the input
            is empty or malformed.

        Example:
            >>> results = {
            ...     "vote_counts": [10, 50],
            ...     "results": {
            ...         "naive": {10: [True, False, True], 50: [True, True, True]},
            ...         "informed": {10: [False, False, True], 50: [True, False, True]},
            ...     }
            ... }
            >>> df = Metrics.compute_detection_rate_curve(results)
            >>> df.shape
            (4, 3)
            >>> df.columns.tolist()
            ['vote_count', 'adversary_type', 'detection_rate']
            >>> df.loc[df['adversary_type'] == 'naive'].iloc[0]['detection_rate']
            0.6666...
        """
        # --- Validate and extract input ---
        if not detection_results:
            return pd.DataFrame(
                columns=[_COL_VOTE_COUNT, _COL_ADVERSARY_TYPE, _COL_DETECTION_RATE]
            )

        vote_counts: List[int] = list(
            detection_results.get("vote_counts", [])
        )
        results_by_type: Dict[str, Dict[int, List[bool]]] = dict(
            detection_results.get("results", {})
        )

        if not results_by_type:
            return pd.DataFrame(
                columns=[_COL_VOTE_COUNT, _COL_ADVERSARY_TYPE, _COL_DETECTION_RATE]
            )

        # --- Build rows ---
        rows: List[Dict[str, Any]] = []

        for adversary_type, vote_count_results in results_by_type.items():
            # Determine the set of vote counts to iterate over.
            # Use the union of the configured vote_counts and the keys actually
            # present in vote_count_results to handle partial results gracefully.
            all_vote_counts: List[int] = sorted(
                set(vote_counts) | set(vote_count_results.keys())
            )

            for vote_count in all_vote_counts:
                trial_outcomes: Optional[List[bool]] = vote_count_results.get(
                    vote_count
                )

                if trial_outcomes is None or len(trial_outcomes) == 0:
                    # Missing data for this (adversary_type, vote_count) pair.
                    detection_rate: float = float("nan")
                else:
                    # Detection rate = fraction of trials where detection occurred.
                    detection_rate = float(np.mean(trial_outcomes))

                rows.append(
                    {
                        _COL_VOTE_COUNT: int(vote_count),
                        _COL_ADVERSARY_TYPE: str(adversary_type),
                        _COL_DETECTION_RATE: detection_rate,
                    }
                )

        # --- Construct and sort DataFrame ---
        if not rows:
            return pd.DataFrame(
                columns=[_COL_VOTE_COUNT, _COL_ADVERSARY_TYPE, _COL_DETECTION_RATE]
            )

        df: pd.DataFrame = pd.DataFrame(rows)

        # Sort for consistent ordering across runs.
        df = df.sort_values(
            by=[_COL_ADVERSARY_TYPE, _COL_VOTE_COUNT],
            ascending=[True, True],
        ).reset_index(drop=True)

        return df

    @staticmethod
    def compute_utility_noise_curve(
        scenario2_results: Dict[str, Any],
    ) -> pd.DataFrame:
        """Convert PerturbedLeaderboard.evaluate_scenario2 results into a tidy DataFrame.

        Transforms the nested dict from evaluate_scenario2 into a tidy DataFrame
        with one row per (noise_scale, vote_count) combination. This single
        DataFrame supports both Figure 5 (detection rate vs. votes, one curve
        per noise scale) and Figure 6 (utility loss vs. noise scale) by
        providing all relevant columns.

        Paper alignment:
          - Figure 5: Detection rate vs. number of malicious votes for different
            noise scales. Visualizer slices by noise_scale to get one curve per
            noise scale.
          - Figure 6: Average absolute rank change (utility loss) vs. noise scale.
            Visualizer aggregates avg_rank_change per noise_scale.
          - Section 4.3: "we measure utility as the average absolute change in
            the ranking of any item."

        Expected input schema (from PerturbedLeaderboard.evaluate_scenario2):
            {
                "noise_scales": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                "vote_counts": [10, 20, 50, 100, 200, 500, 1000],
                "detection_rates": {
                    0.1: {10: 0.05, 20: 0.08, 50: 0.12, ...},
                    0.5: {10: 0.10, 20: 0.18, ...},
                    ...
                },
                "utility_losses": {
                    0.1: 0.3,
                    0.5: 1.2,
                    1.0: 2.8,
                    ...
                }
            }

        Note: detection_rates values are already aggregated floats (mean over
        n_trials), not raw boolean lists. This differs from compute_detection_rate_curve
        which receives raw boolean lists. The evaluate_scenario2 method pre-aggregates
        for efficiency.

        Args:
            scenario2_results: Dict with keys:
              - "noise_scales": List[float] of noise scale values tested.
              - "vote_counts": List[int] of vote count values tested.
              - "detection_rates": Dict[float, Dict[int, float]] mapping
                noise_scale -> vote_count -> detection_rate (float in [0,1]).
              - "utility_losses": Dict[float, float] mapping noise_scale ->
                average absolute rank change (utility loss metric).

        Returns:
            pd.DataFrame with columns:
              - "noise_scale" (float): The Gaussian noise standard deviation
                added to BT ratings.
              - "vote_count" (int): Number of malicious votes in the sequence.
              - "detection_rate" (float): Fraction of trials where the adversary
                was detected. In [0.0, 1.0]. np.nan if missing.
              - "avg_rank_change" (float): Average absolute rank change (utility
                loss) for this noise scale. Same value repeated for all vote
                counts at a given noise scale. np.nan if missing.
            Sorted by ["noise_scale", "vote_count"] for consistent ordering.
            Returns an empty DataFrame with the correct columns if the input
            is empty or malformed.

        Example:
            >>> results = {
            ...     "noise_scales": [0.5, 1.0],
            ...     "vote_counts": [10, 50],
            ...     "detection_rates": {0.5: {10: 0.1, 50: 0.3}, 1.0: {10: 0.2, 50: 0.5}},
            ...     "utility_losses": {0.5: 1.2, 1.0: 2.8},
            ... }
            >>> df = Metrics.compute_utility_noise_curve(results)
            >>> df.shape
            (4, 4)
            >>> df.columns.tolist()
            ['noise_scale', 'vote_count', 'detection_rate', 'avg_rank_change']
            >>> df.loc[df['noise_scale'] == 1.0].iloc[0]['avg_rank_change']
            2.8
        """
        # Define output column names.
        output_columns: List[str] = [
            _COL_NOISE_SCALE,
            _COL_VOTE_COUNT,
            _COL_DETECTION_RATE,
            _COL_AVG_RANK_CHANGE,
        ]

        # --- Validate and extract input ---
        if not scenario2_results:
            return pd.DataFrame(columns=output_columns)

        noise_scales: List[float] = list(
            scenario2_results.get("noise_scales", [])
        )
        vote_counts: List[int] = list(
            scenario2_results.get("vote_counts", [])
        )
        detection_rates_nested: Dict[float, Dict[int, float]] = dict(
            scenario2_results.get("detection_rates", {})
        )
        utility_losses: Dict[float, float] = dict(
            scenario2_results.get("utility_losses", {})
        )

        # If noise_scales or vote_counts are empty, try to infer from the
        # detection_rates dict keys.
        if not noise_scales and detection_rates_nested:
            noise_scales = sorted(detection_rates_nested.keys())

        if not vote_counts and detection_rates_nested:
            # Collect all vote count keys across all noise scales.
            all_vc: set = set()
            for vc_dict in detection_rates_nested.values():
                all_vc.update(vc_dict.keys())
            vote_counts = sorted(all_vc)

        if not noise_scales:
            return pd.DataFrame(columns=output_columns)

        # --- Build rows ---
        rows: List[Dict[str, Any]] = []

        for noise_scale in noise_scales:
            # Utility loss for this noise scale (scalar, same for all vote counts).
            utility_loss: float = float(
                utility_losses.get(noise_scale, float("nan"))
            )

            # Detection rates for this noise scale at each vote count.
            dr_for_scale: Dict[int, float] = dict(
                detection_rates_nested.get(noise_scale, {})
            )

            # Determine vote counts to iterate over for this noise scale.
            # Use the union of configured vote_counts and actual keys present.
            all_vote_counts_for_scale: List[int] = sorted(
                set(vote_counts) | set(dr_for_scale.keys())
            )

            if not all_vote_counts_for_scale:
                # No vote counts available — add a single row with NaN detection rate.
                rows.append(
                    {
                        _COL_NOISE_SCALE: float(noise_scale),
                        _COL_VOTE_COUNT: int(0),
                        _COL_DETECTION_RATE: float("nan"),
                        _COL_AVG_RANK_CHANGE: utility_loss,
                    }
                )
                continue

            for vote_count in all_vote_counts_for_scale:
                detection_rate: float = float(
                    dr_for_scale.get(vote_count, float("nan"))
                )

                rows.append(
                    {
                        _COL_NOISE_SCALE: float(noise_scale),
                        _COL_VOTE_COUNT: int(vote_count),
                        _COL_DETECTION_RATE: detection_rate,
                        _COL_AVG_RANK_CHANGE: utility_loss,
                    }
                )

        # --- Construct and sort DataFrame ---
        if not rows:
            return pd.DataFrame(columns=output_columns)

        df: pd.DataFrame = pd.DataFrame(rows, columns=output_columns)

        # Sort for consistent ordering across runs.
        df = df.sort_values(
            by=[_COL_NOISE_SCALE, _COL_VOTE_COUNT],
            ascending=[True, True],
        ).reset_index(drop=True)

        return df

    # -----------------------------------------------------------------------
    # Summary helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def summarize_deanonymization(df: pd.DataFrame) -> pd.DataFrame:
        """Compute summary statistics for a de-anonymization accuracy DataFrame.

        Adds summary rows (mean, min, max) to a de-anonymization results
        DataFrame (e.g., from IdentityProbingDetector.evaluate_all or
        TrainingBasedDetector.evaluate_all_models_categories). Useful for
        quick inspection of overall detector performance.

        Args:
            df: DataFrame with model names as index and prompt categories or
                feature types as columns. Values are accuracy percentages
                (float, 0.0–100.0). Produced by IdentityProbingDetector.evaluate_all
                or TrainingBasedDetector.evaluate_all_models_categories.

        Returns:
            A copy of the input DataFrame with three additional rows appended:
              - "Mean": column-wise mean accuracy across all models.
              - "Min": column-wise minimum accuracy.
              - "Max": column-wise maximum accuracy.
            All summary values are rounded to 1 decimal place to match the
            paper's table format. Returns the input DataFrame unchanged if it
            is empty.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame(
            ...     {"Who are you?": [99.3, 97.2, 92.7]},
            ...     index=["claude-3-5-sonnet", "gemini-1.5-pro", "gpt-4o-mini"],
            ... )
            >>> summary = Metrics.summarize_deanonymization(df)
            >>> summary.loc["Mean", "Who are you?"]
            96.4
        """
        if df.empty:
            return df.copy()

        # Compute summary statistics on numeric columns only.
        # Use skipna=True to handle any np.nan values from incomplete collection.
        numeric_df: pd.DataFrame = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return df.copy()

        mean_row: pd.Series = numeric_df.mean(skipna=True).round(1)
        min_row: pd.Series = numeric_df.min(skipna=True).round(1)
        max_row: pd.Series = numeric_df.max(skipna=True).round(1)

        # Build summary DataFrame with the same columns as the input.
        summary_rows: pd.DataFrame = pd.DataFrame(
            [mean_row, min_row, max_row],
            index=["Mean", "Min", "Max"],
        )

        # Concatenate the original DataFrame with the summary rows.
        result: pd.DataFrame = pd.concat([df, summary_rows], axis=0)

        return result

    @staticmethod
    def summarize_simulation(
        results: List[SimulationResult],
    ) -> pd.DataFrame:
        """Build a flat summary DataFrame from a list of SimulationResult objects.

        Converts each SimulationResult into a row with all fields as columns.
        Useful for exploratory analysis and for passing to CostModel.summarize.

        Args:
            results: List of SimulationResult objects from AttackSimulator.
                May be empty (returns empty DataFrame with correct columns).

        Returns:
            pd.DataFrame with one row per SimulationResult and columns:
              - "target_model" (str)
              - "target_rank" (int)
              - "achieved" (bool)
              - "n_votes" (int)
              - "n_interactions" (int)
              - "detection_accuracy" (float)
              - "non_target_strategy" (str)
              - "direction" (str)
            Rows are sorted by ["target_model", "target_rank"] for consistent
            ordering. Returns an empty DataFrame with the correct columns if
            results is empty.

        Example:
            >>> results = [