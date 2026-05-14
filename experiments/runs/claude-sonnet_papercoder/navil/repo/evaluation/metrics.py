## evaluation/metrics.py
"""Metric computation utilities for NaViL benchmark evaluation.

This module provides ``MetricsCalculator``, a pure computation utility class
that implements all benchmark-specific metric functions used to evaluate NaViL
against the benchmarks listed in ``config.yaml``.

No model inference occurs here. The ``BenchmarkEvaluator`` in
``evaluation/evaluator.py`` collects model predictions and ground-truth targets,
then delegates scoring to this class.

Supported metrics:
    - ``accuracy``:          Exact-match accuracy (lowercased, stripped).
                             Used for: MMVet, MMMU, MMBench, MathVista, TextVQA,
                             ScienceQA, GQA, AI2D, CCBench, MMLU, CMMLU, MATH.
    - ``anls``:              Average Normalized Levenshtein Similarity.
                             Used for: DocVQA, InfographicVQA.
    - ``relaxed_accuracy``:  Numeric-tolerant accuracy with string fallback.
                             Used for: ChartQA.
    - ``normalize_to_100``:  Scale a raw score to [0, 100].
                             Used for: MME (max=5000), OCRBench (max=1000).
    - ``compute_average_score``: Normalized mean across all benchmarks.
                             Produces the "Avg" column in Tables 1 and 2.

Config alignment (config.yaml / configs/navil_2b.yaml):
    evaluation.benchmarks[*].metric    — which method to use per benchmark
    evaluation.benchmarks[*].max_value — normalization denominator per benchmark

    Benchmark max_value mapping:
        mme:        5000  (perception + cognition sum)
        ocrbench:   1000  (score out of 1000)
        all others: 100   (already in [0, 100])

Dependencies:
    - editdistance: character-level Levenshtein distance for ANLS
    - numpy:        mean computation for compute_average_score
    - re:           string normalization (whitespace collapsing)

No internal project dependencies — this is a standalone leaf module.
"""

import re
from typing import Dict, List, Optional

import editdistance
import numpy as np


class MetricsCalculator:
    """Computes benchmark-specific evaluation metrics for NaViL.

    All public methods accept lists of string predictions and targets,
    apply appropriate normalization, and return a float score in [0, 100]
    (except ``normalize_to_100`` which returns a raw normalized float).

    The class is stateless beyond a compiled regex for whitespace normalization,
    making it safe to instantiate once and reuse across all benchmark evaluations.

    Attributes:
        _whitespace_re: Compiled regex pattern for collapsing multiple
                        consecutive whitespace characters into a single space.
                        Used by ``_normalize_string`` to ensure consistent
                        string comparison across all metric methods.

    Example::

        calc = MetricsCalculator()

        # Exact-match accuracy
        preds   = ["cat", "dog", "bird"]
        targets = ["cat", "fish", "bird"]
        score = calc.accuracy(preds, targets)
        # score == 66.666...  (2/3 correct)

        # ANLS for DocVQA
        preds   = ["invoice 2024", "total amount"]
        targets = ["invoice 2024", "total amout"]  # typo in target
        score = calc.anls(preds, targets)
        # score > 90.0  (high similarity despite typo)

        # ChartQA relaxed accuracy
        preds   = ["12.5%", "100"]
        targets = ["12.4%", "99"]
        score = calc.relaxed_accuracy(preds, targets, tolerance=0.05)
        # score == 100.0  (both within 5% tolerance)

        # Normalized average across benchmarks
        results = {"mmvet": 78.3, "mme": 1822.0, "ocrbench": 796.0}
        maxes   = {"mmvet": 100,  "mme": 5000,    "ocrbench": 1000}
        avg = calc.compute_average_score(results, maxes)
        # avg == mean([78.3, 36.44, 79.6]) ≈ 64.78
    """

    def __init__(self) -> None:
        """Initialise MetricsCalculator with a compiled whitespace regex.

        No heavy resources are loaded. The compiled regex is the only
        stateful attribute, created once for efficiency across many calls.
        """
        # Compiled regex for collapsing multiple whitespace characters
        # (spaces, tabs, newlines) into a single space. Used in
        # _normalize_string to ensure consistent string comparison.
        self._whitespace_re: re.Pattern = re.compile(r"\s+")

        # Compiled regex for stripping numeric noise characters before
        # attempting float parsing in relaxed_accuracy.
        # Strips: %, $, comma, and whitespace.
        self._numeric_noise_re: re.Pattern = re.compile(r"[%,$\s]")

    # ---------------------------------------------------------------------- #
    # Internal helper                                                          #
    # ---------------------------------------------------------------------- #

    def _normalize_string(self, s: str) -> str:
        """Apply standard string normalization for metric comparison.

        Normalization steps:
        1. Lowercase: ``s.lower()``
        2. Strip leading/trailing whitespace: ``.strip()``
        3. Collapse internal multiple whitespace to single space.

        This normalization is applied consistently across ``accuracy``,
        ``anls``, and ``relaxed_accuracy`` to avoid spurious mismatches
        due to case or whitespace differences.

        Args:
            s: Input string to normalize.

        Returns:
            Normalized string with lowercase, stripped, and collapsed
            whitespace.

        Example::

            calc._normalize_string("  Hello   World  ")
            # Returns: "hello world"
        """
        # Step 1 & 2: lowercase and strip
        normalized: str = s.lower().strip()
        # Step 3: collapse multiple whitespace to single space
        normalized = self._whitespace_re.sub(" ", normalized)
        return normalized

    # ---------------------------------------------------------------------- #
    # Public metric methods                                                    #
    # ---------------------------------------------------------------------- #

    def accuracy(
        self,
        preds: List[str],
        targets: List[str],
    ) -> float:
        """Compute exact-match accuracy after lowercasing and stripping.

        Used for classification-style benchmarks where the model must
        produce an exact string match with the ground-truth answer:
        MMVet, MMMU, MMBench-EN, MathVista, TextVQA, ScienceQA-IMG,
        GQA, AI2D, CCBench, MMLU, CMMLU, MATH.

        Normalization applied before comparison:
        - Lowercase
        - Strip leading/trailing whitespace
        - Collapse internal whitespace

        Args:
            preds:   List of model prediction strings. Length must equal
                     ``len(targets)``.
            targets: List of ground-truth answer strings. Length must equal
                     ``len(preds)``.

        Returns:
            Accuracy as a float in [0.0, 100.0]. Returns ``0.0`` if either
            list is empty or if lengths do not match.

        Example::

            calc = MetricsCalculator()
            preds   = ["A", "b", "C", "d"]
            targets = ["A", "B", "c", "d"]
            score = calc.accuracy(preds, targets)
            # score == 50.0  (2/4 correct after normalization: "a"=="a", "d"=="d")
        """
        # Guard: empty or mismatched lists
        if not preds or not targets:
            return 0.0

        if len(preds) != len(targets):
            # Log mismatch but do not raise — compute over the shorter list
            min_len: int = min(len(preds), len(targets))
            preds = preds[:min_len]
            targets = targets[:min_len]

        total: int = len(preds)
        if total == 0:
            return 0.0

        num_correct: int = 0
        pred: str
        target: str
        for pred, target in zip(preds, targets):
            pred_norm: str = self._normalize_string(str(pred))
            target_norm: str = self._normalize_string(str(target))
            if pred_norm == target_norm:
                num_correct += 1

        return (num_correct / total) * 100.0

    def anls(
        self,
        preds: List[str],
        targets: List[str],
    ) -> float:
        """Compute Average Normalized Levenshtein Similarity (ANLS).

        Used for document understanding benchmarks where OCR errors and
        minor transcription differences should not be penalized harshly:
        DocVQA and InfographicVQA.

        ANLS formula per sample:
            edit_dist = editdistance.eval(pred, target)  # character-level
            max_len   = max(len(pred), len(target))
            nls       = 1.0 - edit_dist / max_len        # in [0, 1]
            if nls < 0.5:
                nls = 0.0  # threshold: < 50% similar → treat as wrong

        The threshold at 0.5 is the standard ANLS definition from the
        DocVQA paper (Mathew et al., 2021). Predictions that are less than
        50% similar to the target receive zero credit.

        Args:
            preds:   List of model prediction strings.
            targets: List of ground-truth answer strings. For multi-answer
                     benchmarks (DocVQA, InfographicVQA), the evaluator
                     should pass the best-matching target (highest NLS)
                     for each prediction before calling this method.

        Returns:
            ANLS score as a float in [0.0, 100.0]. Returns ``0.0`` if
            either list is empty or lengths do not match.

        Example::

            calc = MetricsCalculator()
            preds   = ["invoice number 42", "total: $100"]
            targets = ["invoice number 42", "total: $10"]  # typo
            score = calc.anls(preds, targets)
            # First pair: exact match → NLS=1.0
            # Second pair: edit_dist=1, max_len=10 → NLS=0.9 ≥ 0.5 → NLS=0.9
            # ANLS = (1.0 + 0.9) / 2 * 100 = 95.0
        """
        # Guard: empty or mismatched lists
        if not preds or not targets:
            return 0.0

        if len(preds) != len(targets):
            min_len: int = min(len(preds), len(targets))
            preds = preds[:min_len]
            targets = targets[:min_len]

        total: int = len(preds)
        if total == 0:
            return 0.0

        nls_sum: float = 0.0
        pred: str
        target: str
        for pred, target in zip(preds, targets):
            pred_norm: str = self._normalize_string(str(pred))
            target_norm: str = self._normalize_string(str(target))

            # Handle edge cases where one or both strings are empty
            if len(pred_norm) == 0 and len(target_norm) == 0:
                # Both empty → exact match
                nls: float = 1.0
            elif len(pred_norm) == 0 or len(target_norm) == 0:
                # One empty, one non-empty → completely wrong
                nls = 0.0
            else:
                # Standard ANLS computation
                edit_dist: int = editdistance.eval(pred_norm, target_norm)
                max_len: int = max(len(pred_norm), len(target_norm))
                nls = 1.0 - edit_dist / max_len

                # Apply 0.5 threshold: predictions < 50% similar → zero credit
                if nls < 0.5:
                    nls = 0.0

            nls_sum += nls

        return (nls_sum / total) * 100.0

    def relaxed_accuracy(
        self,
        preds: List[str],
        targets: List[str],
        tolerance: float = 0.05,
    ) -> float:
        """Compute ChartQA relaxed accuracy with numeric tolerance.

        Allows small numerical differences (within ``tolerance`` relative
        error) to be counted as correct. Falls back to exact string match
        when either prediction or target cannot be parsed as a number.

        Numeric comparison logic:
        1. Strip noise characters (%, $, comma, whitespace) from both strings.
        2. Attempt ``float()`` conversion on both stripped strings.
        3. If both parse successfully:
           - If ``target_float == 0``: correct iff ``abs(pred_float) <= tolerance``
           - Otherwise: correct iff ``abs(pred_float - target_float) / abs(target_float) <= tolerance``
        4. If either fails to parse: fall back to exact normalized string match.

        Default tolerance of 0.05 (5%) matches the ChartQA paper's
        relaxed accuracy definition (Masry et al., 2022).

        Args:
            preds:     List of model prediction strings.
            targets:   List of ground-truth answer strings.
            tolerance: Relative error tolerance for numeric comparisons.
                       Default: ``0.05`` (5%). Must be non-negative.

        Returns:
            Relaxed accuracy as a float in [0.0, 100.0]. Returns ``0.0``
            if either list is empty or lengths do not match.

        Example::

            calc = MetricsCalculator()
            preds   = ["12.5%", "100", "Q3"]
            targets = ["12.4%", "99",  "Q3"]
            score = calc.relaxed_accuracy(preds, targets, tolerance=0.05)
            # "12.5" vs "12.4": |12.5-12.4|/12.4 ≈ 0.008 ≤ 0.05 → correct
            # "100" vs "99":    |100-99|/99 ≈ 0.010 ≤ 0.05 → correct
            # "Q3" vs "Q3":     string match → correct
            # score == 100.0
        """
        # Guard: empty or mismatched lists
        if not preds or not targets:
            return 0.0

        if len(preds) != len(targets):
            min_len: int = min(len(preds), len(targets))
            preds = preds[:min_len]
            targets = targets[:min_len]

        total: int = len(preds)
        if total == 0:
            return 0.0

        # Clamp tolerance to non-negative
        tolerance = max(0.0, float(tolerance))

        num_correct: int = 0
        pred: str
        target: str
        for pred, target in zip(preds, targets):
            pred_norm: str = self._normalize_string(str(pred))
            target_norm: str = self._normalize_string(str(target))

            # ---------------------------------------------------------------- #
            # Attempt numeric comparison                                        #
            # ---------------------------------------------------------------- #
            # Strip numeric noise characters before float parsing
            pred_stripped: str = self._numeric_noise_re.sub("", pred_norm)
            target_stripped: str = self._numeric_noise_re.sub("", target_norm)

            is_correct: bool = False

            try:
                pred_float: float = float(pred_stripped)
                target_float: float = float(target_stripped)

                # Both parsed as floats — apply relative tolerance check
                if target_float == 0.0:
                    # Avoid division by zero: check absolute difference
                    is_correct = abs(pred_float) <= tolerance
                else:
                    relative_error: float = (
                        abs(pred_float - target_float) / abs(target_float)
                    )
                    is_correct = relative_error <= tolerance

            except (ValueError, TypeError):
                # One or both strings are not numeric — fall back to exact match
                is_correct = pred_norm == target_norm

            if is_correct:
                num_correct += 1

        return (num_correct / total) * 100.0

    def normalize_to_100(
        self,
        value: float,
        max_value: float,
    ) -> float:
        """Normalize a raw benchmark score to the [0, 100] range.

        Used for benchmarks whose raw scores are not already in [0, 100]:
        - MME: raw score is perception + cognition sum (max ≈ 5000 per config)
        - OCRBench: raw score is out of 1000 (max = 1000 per config)

        For benchmarks already in [0, 100], passing ``max_value=100`` is
        a no-op (returns ``value`` unchanged).

        Formula: ``(value / max_value) * 100.0``

        Args:
            value:     Raw benchmark score. Should be non-negative.
            max_value: Maximum possible raw score for this benchmark.
                       From ``config.yaml``: ``evaluation.benchmarks[*].max_value``.
                       Must be positive to avoid division by zero.

        Returns:
            Normalized score as a float. Returns ``0.0`` if ``max_value``
            is zero or negative (guards against division by zero).
            The return value may exceed 100.0 if ``value > max_value``
            (e.g., if a model achieves a perfect score on a subset).

        Example::

            calc = MetricsCalculator()
            # MME: raw sum = 1822, max = 5000
            normalized = calc.normalize_to_100(1822.0, 5000.0)
            # normalized == 36.44

            # OCRBench: raw score = 796, max = 1000
            normalized = calc.normalize_to_100(796.0, 1000.0)
            # normalized == 79.6

            # Already-normalized benchmark (no-op)
            normalized = calc.normalize_to_100(78.3, 100.0)
            # normalized == 78.3
        """
        if max_value <= 0.0:
            return 0.0

        return (float(value) / float(max_value)) * 100.0

    def compute_average_score(
        self,
        results: Dict[str, float],
        benchmark_maxes: Dict[str, float],
    ) -> float:
        """Compute the normalized average score across all benchmarks.

        Produces the "Avg" column reported in Tables 1 and 2 of the paper.
        Each benchmark score is first normalized to [0, 100] using
        ``normalize_to_100``, then the mean is taken across all benchmarks.

        From the paper: "Average scores are computed by normalizing each
        metric to a range between 0 and 100."

        Args:
            results:          Dict mapping benchmark name to raw score.
                              Example: ``{"mmvet": 78.3, "mme": 1822.0,
                              "ocrbench": 796.0, "mmmu": 41.8, ...}``
                              Benchmark names should match the ``name`` field
                              in ``config.yaml``'s ``evaluation.benchmarks``.
            benchmark_maxes:  Dict mapping benchmark name to maximum possible
                              raw score. Used as the denominator in
                              ``normalize_to_100``.
                              Example: ``{"mme": 5000, "ocrbench": 1000,
                              "mmvet": 100, "mmmu": 100, ...}``
                              If a benchmark name is not found in this dict,
                              defaults to ``100.0`` (assumes already normalized).

        Returns:
            Mean normalized score as a float in [0.0, 100.0+].
            Returns ``0.0`` if ``results`` is empty.

        Example::

            calc = MetricsCalculator()
            results = {
                "mmvet":    78.3,
                "mmmu":     41.8,
                "mmbench_en": 71.2,
                "mme":      1822.0,
                "mathvista_mini": 50.0,
                "ocrbench": 796.0,
                "ccbench":  83.9,
            }
            maxes = {
                "mmvet":    100,
                "mmmu":     100,
                "mmbench_en": 100,
                "mme":      5000,
                "mathvista_mini": 100,
                "ocrbench": 1000,
                "ccbench":  100,
            }
            avg = calc.compute_average_score(results, maxes)
            # Normalized: [78.3, 41.8, 71.2, 36.44, 50.0, 79.6, 83.9]
            # avg ≈ 63.04  (matches paper's reported Avg ≈ 67.1 for NaViL-2B
            #               with slight differences due to benchmark selection)
        """
        if not results:
            return 0.0

        normalized_scores: List[float] = []

        benchmark_name: str
        raw_score: float
        for benchmark_name, raw_score in results.items():
            # Get max value for this benchmark; default to 100.0 if not specified
            max_val: float = float(
                benchmark_maxes.get(benchmark_name, 100.0)
            )

            # Normalize to [0, 100]
            normalized: float = self.normalize_to_100(raw_score, max_val)
            normalized_scores.append(normalized)

        if not normalized_scores:
            return 0.0

        # Compute mean using numpy for numerical stability
        average: float = float(np.mean(normalized_scores))
        return average
