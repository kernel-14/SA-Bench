## evaluation/metrics.py
"""Evaluation metrics for SCoRe: Self-Correction via Reinforcement Learning.

This module implements the five core self-correction evaluation metrics
defined in Section 3 of the paper, plus the MBPP-R offline repair accuracy
from Table 3. It is a pure computation module with no dependencies on other
project files.

Paper definition (Section 3, "Metrics"):
    (1) Accuracy@t1: model accuracy at the first attempt
    (2) Accuracy@t2: model accuracy at the second attempt
    (3) Δ(t1, t2): net improvement = Accuracy@t2 - Accuracy@t1
        — the primary self-correction metric
    (4) Δ^{i→c}(t1, t2): fraction of problems incorrect at t1 that become
        correct at t2 — measures how many new problems self-correction solves
    (5) Δ^{c→i}(t1, t2): fraction of problems correct at t1 that become
        incorrect at t2 — measures erroneous revision rate

Mathematical identity (verified in compute_all_metrics):
    Δ(t1, t2) = Δ^{i→c}(t1, t2) - Δ^{c→i}(t1, t2)

This identity holds because every problem falls into exactly one of four
transition categories (i→i, i→c, c→i, c→c), and only i→c and c→i
contribute to the accuracy difference.

Validation against Table 2 (base model on MATH):
    accuracy_t1 = 52.6%, accuracy_t2 = 41.4%, delta = -11.2%
    i2c_rate = 4.6%, c2i_rate = 15.8%
    Consistency: 4.6% - 15.8% = -11.2% ✓

Validation against Table 2 (SCoRe on MATH):
    accuracy_t1 = 60.0%, accuracy_t2 = 64.4%, delta = 4.4%
    i2c_rate = 5.8%, c2i_rate = 1.4%
    Consistency: 5.8% - 1.4% = 4.4% ✓

Input data contract:
    Each element in results: List[dict] must contain:
        'reward_t1': float in {0.0, 1.0} — binary correctness of first attempt
        'reward_t2': float in {0.0, 1.0} — binary correctness of second attempt
    For compute_mbpp_r_accuracy, additionally:
        'mbpp_r_reward': float in {0.0, 1.0} — offline repair correctness

Typical usage:
    from evaluation.metrics import Metrics

    results = [
        {'reward_t1': 1.0, 'reward_t2': 1.0},
        {'reward_t1': 0.0, 'reward_t2': 1.0},
        {'reward_t1': 1.0, 'reward_t2': 0.0},
        {'reward_t1': 0.0, 'reward_t2': 0.0},
    ]
    all_metrics = Metrics.compute_all_metrics(results)
    # {'accuracy_t1': 0.5, 'accuracy_t2': 0.5, 'delta_t1_t2': 0.0,
    #  'i2c_rate': 0.25, 'c2i_rate': 0.25, ...}
"""

import logging
import math
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold for classifying binary rewards.
# Rewards from RewardFunction.compute_reward() are always exactly 0.0 or 1.0,
# but we use a threshold for robustness against any upstream floating-point
# imprecision (e.g., 0.9999999 or 1.0000001 from numerical operations).
# ---------------------------------------------------------------------------
_CORRECT_THRESHOLD: float = 0.5

# Floating-point tolerance for the consistency check:
# |Δ(t1,t2) - (Δ^{i→c} - Δ^{c→i})| must be below this value.
_CONSISTENCY_TOLERANCE: float = 1e-9


class Metrics:
    """Stateless utility class for computing SCoRe self-correction metrics.

    All methods are static — no instance state is required. The class
    exists purely as a namespace for the metric functions, consistent with
    the design specification.

    All methods operate on a List[dict] where each dict contains
    'reward_t1' and 'reward_t2' float values (binary: 0.0 or 1.0).
    This is the output format produced by Evaluator._run_two_turn_inference().

    The five metrics correspond exactly to the paper's Section 3 definitions
    and are reported in Tables 1, 2, 3, and 4.
    """

    # -------------------------------------------------------------------------
    # Individual metric methods
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_accuracy_t1(results: List[Dict[str, Any]]) -> float:
        """Compute Accuracy@t1: model accuracy on the first attempt.

        From Section 3: "Accuracy@1: the model's accuracy at the first attempt."

        Args:
            results: List of result dicts, each containing 'reward_t1'
                (float in {0.0, 1.0}). The binary reward convention is
                established in Section 6: "we use binary rewards during
                training, indicating whether the model's answer matches
                the ground truth one."

        Returns:
            Float in [0.0, 1.0] representing the fraction of problems
            where the first attempt was correct. Returns 0.0 for an
            empty results list.
        """
        if not results:
            logger.debug(
                "compute_accuracy_t1: Empty results list. Returning 0.0."
            )
            return 0.0

        num_correct: int = sum(
            1
            for r in results
            if Metrics._is_correct(float(r.get("reward_t1", 0.0)))
        )
        accuracy: float = num_correct / len(results)

        logger.debug(
            "compute_accuracy_t1: %d/%d correct = %.4f.",
            num_correct,
            len(results),
            accuracy,
        )
        return accuracy

    @staticmethod
    def compute_accuracy_t2(results: List[Dict[str, Any]]) -> float:
        """Compute Accuracy@t2: model accuracy on the second attempt.

        From Section 3: "Accuracy@12: the model's accuracy at the second
        attempt." (The paper uses @t2 notation in the tables.)

        Args:
            results: List of result dicts, each containing 'reward_t2'
                (float in {0.0, 1.0}).

        Returns:
            Float in [0.0, 1.0] representing the fraction of problems
            where the second attempt was correct. Returns 0.0 for an
            empty results list.
        """
        if not results:
            logger.debug(
                "compute_accuracy_t2: Empty results list. Returning 0.0."
            )
            return 0.0

        num_correct: int = sum(
            1
            for r in results
            if Metrics._is_correct(float(r.get("reward_t2", 0.0)))
        )
        accuracy: float = num_correct / len(results)

        logger.debug(
            "compute_accuracy_t2: %d/%d correct = %.4f.",
            num_correct,
            len(results),
            accuracy,
        )
        return accuracy

    @staticmethod
    def compute_delta(results: List[Dict[str, Any]]) -> float:
        """Compute Δ(t1, t2): net self-correction improvement.

        From Section 3: "Δ(t1, t2): the net improvement in model accuracy
        between the first and second attempts, which measures the efficacy
        of self-correction."

        This is the PRIMARY self-correction metric reported throughout the
        paper. The paper's key result is that SCoRe achieves Δ = +4.4% on
        MATH (Table 2), the first significantly positive delta, compared to
        -11.2% for the base model.

        Computed as: Δ = Accuracy@t2 - Accuracy@t1

        This is equivalent to: Δ = Δ^{i→c} - Δ^{c→i}
        (verified by the mathematical identity in compute_all_metrics).

        Args:
            results: List of result dicts containing 'reward_t1' and
                'reward_t2' (floats in {0.0, 1.0}).

        Returns:
            Float in [-1.0, 1.0] representing the net accuracy change.
            Positive values indicate genuine self-correction improvement.
            Negative values indicate the model degrades its answers.
            Returns 0.0 for an empty results list.
        """
        if not results:
            logger.debug(
                "compute_delta: Empty results list. Returning 0.0."
            )
            return 0.0

        accuracy_t1: float = Metrics.compute_accuracy_t1(results)
        accuracy_t2: float = Metrics.compute_accuracy_t2(results)
        delta: float = accuracy_t2 - accuracy_t1

        logger.debug(
            "compute_delta: accuracy_t1=%.4f, accuracy_t2=%.4f, "
            "delta=%.4f.",
            accuracy_t1,
            accuracy_t2,
            delta,
        )
        return delta

    @staticmethod
    def compute_i2c(results: List[Dict[str, Any]]) -> float:
        """Compute Δ^{i→c}(t1, t2): incorrect-to-correct transition rate.

        From Section 3: "Δ^{i→c}(t1, t2): the fraction of problems that
        are incorrect in the first attempt but become correct at the second
        attempt, which measures how many new problems can self-correction solve."

        IMPORTANT: The denominator is the TOTAL number of problems, not
        just the incorrect ones. This is confirmed by the mathematical
        identity Δ = Δ^{i→c} - Δ^{c→i}, which only holds with total
        problem count as denominator.

        Verification from Table 2 (base model):
            i2c_rate = 4.6%, c2i_rate = 15.8%
            delta = 4.6% - 15.8% = -11.2% ✓

        Args:
            results: List of result dicts containing 'reward_t1' and
                'reward_t2' (floats in {0.0, 1.0}).

        Returns:
            Float in [0.0, 1.0] representing the fraction of ALL problems
            where the model successfully corrected an incorrect first attempt.
            Returns 0.0 for an empty results list.
        """
        if not results:
            logger.debug(
                "compute_i2c: Empty results list. Returning 0.0."
            )
            return 0.0

        num_i2c: int = sum(
            1
            for r in results
            if (
                Metrics._is_incorrect(float(r.get("reward_t1", 0.0)))
                and Metrics._is_correct(float(r.get("reward_t2", 0.0)))
            )
        )
        i2c_rate: float = num_i2c / len(results)

        logger.debug(
            "compute_i2c: %d/%d i→c transitions = %.4f.",
            num_i2c,
            len(results),
            i2c_rate,
        )
        return i2c_rate

    @staticmethod
    def compute_c2i(results: List[Dict[str, Any]]) -> float:
        """Compute Δ^{c→i}(t1, t2): correct-to-incorrect transition rate.

        From Section 3: "Δ^{c→i}(t1, t2): the fraction of problems that
        are correct in the first attempt but become incorrect at the second
        attempt, which measures how well the model understands what makes
        a response correct."

        A high Δ^{c→i} indicates behavior collapse — the model is making
        unnecessary edits to correct answers. The paper shows that SCoRe
        reduces this from 15.8% (base model) to 1.4% (Table 2), which is
        a key indicator of successful self-correction training.

        IMPORTANT: The denominator is the TOTAL number of problems (same
        as compute_i2c), not just the correct ones. This ensures the
        mathematical identity Δ = Δ^{i→c} - Δ^{c→i} holds.

        Args:
            results: List of result dicts containing 'reward_t1' and
                'reward_t2' (floats in {0.0, 1.0}).

        Returns:
            Float in [0.0, 1.0] representing the fraction of ALL problems
            where the model erroneously changed a correct first attempt to
            incorrect. Returns 0.0 for an empty results list.
        """
        if not results:
            logger.debug(
                "compute_c2i: Empty results list. Returning 0.0."
            )
            return 0.0

        num_c2i: int = sum(
            1
            for r in results
            if (
                Metrics._is_correct(float(r.get("reward_t1", 0.0)))
                and Metrics._is_incorrect(float(r.get("reward_t2", 0.0)))
            )
        )
        c2i_rate: float = num_c2i / len(results)

        logger.debug(
            "compute_c2i: %d/%d c→i transitions = %.4f.",
            num_c2i,
            len(results),
            c2i_rate,
        )
        return c2i_rate

    @staticmethod
    def compute_all_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute all five self-correction metrics plus raw counts.

        Calls all five individual metric methods and packages results into
        a single dict. Also computes raw transition counts for interpretability
        and verifies the mathematical consistency identity:
            Δ(t1, t2) = Δ^{i→c}(t1, t2) - Δ^{c→i}(t1, t2)

        The returned dict keys match the config.yaml evaluation.metrics list:
            'accuracy_t1', 'accuracy_t2', 'delta_t1_t2', 'i2c_rate', 'c2i_rate'

        Args:
            results: List of result dicts, each containing 'reward_t1' and
                'reward_t2' (floats in {0.0, 1.0}).

        Returns:
            Dict with the following keys:
                'accuracy_t1' (float): Accuracy@t1 — first attempt accuracy.
                'accuracy_t2' (float): Accuracy@t2 — second attempt accuracy.
                'delta_t1_t2' (float): Δ(t1,t2) — net self-correction gain.
                'i2c_rate' (float): Δ^{i→c} — incorrect-to-correct rate.
                'c2i_rate' (float): Δ^{c→i} — correct-to-incorrect rate.
                'num_problems' (int): Total number of problems evaluated.
                'num_i2c' (int): Raw count of i→c transitions.
                'num_c2i' (int): Raw count of c→i transitions.
                'num_correct_both' (int): Count where both t1 and t2 correct.
                'num_incorrect_both' (int): Count where both t1 and t2 incorrect.
                'consistency_check_passed' (bool): Whether
                    |delta - (i2c_rate - c2i_rate)| < tolerance.

            All float values are 0.0 and all int values are 0 for an empty
            results list.
        """
        if not results:
            logger.debug(
                "compute_all_metrics: Empty results list. "
                "Returning zero-valued metrics dict."
            )
            return {
                "accuracy_t1": 0.0,
                "accuracy_t2": 0.0,
                "delta_t1_t2": 0.0,
                "i2c_rate": 0.0,
                "c2i_rate": 0.0,
                "num_problems": 0,
                "num_i2c": 0,
                "num_c2i": 0,
                "num_correct_both": 0,
                "num_incorrect_both": 0,
                "consistency_check_passed": True,
            }

        # ------------------------------------------------------------------
        # Compute the five core metrics
        # ------------------------------------------------------------------
        accuracy_t1: float = Metrics.compute_accuracy_t1(results)
        accuracy_t2: float = Metrics.compute_accuracy_t2(results)
        delta_t1_t2: float = Metrics.compute_delta(results)
        i2c_rate: float = Metrics.compute_i2c(results)
        c2i_rate: float = Metrics.compute_c2i(results)

        # ------------------------------------------------------------------
        # Compute raw transition counts for interpretability
        # ------------------------------------------------------------------
        num_problems: int = len(results)
        num_i2c: int = 0
        num_c2i: int = 0
        num_correct_both: int = 0
        num_incorrect_both: int = 0

        for r in results:
            r1_correct: bool = Metrics._is_correct(
                float(r.get("reward_t1", 0.0))
            )
            r2_correct: bool = Metrics._is_correct(
                float(r.get("reward_t2", 0.0))
            )

            if not r1_correct and r2_correct:
                num_i2c += 1
            elif r1_correct and not r2_correct:
                num_c2i += 1
            elif r1_correct and r2_correct:
                num_correct_both += 1
            else:
                num_incorrect_both += 1

        # ------------------------------------------------------------------
        # Verify the mathematical consistency identity:
        # Δ(t1, t2) = Δ^{i→c}(t1, t2) - Δ^{c→i}(t1, t2)
        #
        # Proof: Every problem is in exactly one of {i→i, i→c, c→i, c→c}.
        # Accuracy@t2 - Accuracy@t1
        #   = (num_i2c + num_cc)/N - (num_ci + num_cc)/N
        #   = (num_i2c - num_ci) / N
        #   = Δ^{i→c} - Δ^{c→i}
        # ------------------------------------------------------------------
        expected_delta: float = i2c_rate - c2i_rate
        consistency_error: float = abs(delta_t1_t2 - expected_delta)
        consistency_check_passed: bool = (
            consistency_error < _CONSISTENCY_TOLERANCE
        )

        if not consistency_check_passed:
            logger.warning(
                "compute_all_metrics: Consistency check FAILED. "
                "delta_t1_t2=%.8f, i2c_rate - c2i_rate=%.8f, "
                "error=%.2e (tolerance=%.2e). "
                "This indicates a bug in metric computation.",
                delta_t1_t2,
                expected_delta,
                consistency_error,
                _CONSISTENCY_TOLERANCE,
            )
        else:
            logger.debug(
                "compute_all_metrics: Consistency check passed. "
                "delta=%.6f, i2c-c2i=%.6f, error=%.2e.",
                delta_t1_t2,
                expected_delta,
                consistency_error,
            )

        # ------------------------------------------------------------------
        # Log summary for monitoring
        # ------------------------------------------------------------------
        logger.info(
            "compute_all_metrics: n=%d, "
            "acc@t1=%.3f, acc@t2=%.3f, delta=%.3f, "
            "i2c=%.3f, c2i=%.3f | "
            "i→c=%d, c→i=%d, c→c=%d, i→i=%d.",
            num_problems,
            accuracy_t1,
            accuracy_t2,
            delta_t1_t2,
            i2c_rate,
            c2i_rate,
            num_i2c,
            num_c2i,
            num_correct_both,
            num_incorrect_both,
        )

        return {
            # Five core metrics (keys match config.yaml evaluation.metrics)
            "accuracy_t1": accuracy_t1,
            "accuracy_t2": accuracy_t2,
            "delta_t1_t2": delta_t1_t2,
            "i2c_rate": i2c_rate,
            "c2i_rate": c2i_rate,
            # Raw counts for interpretability
            "num_problems": num_problems,
            "num_i2c": num_i2c,
            "num_c2i": num_c2i,
            "num_correct_both": num_correct_both,
            "num_incorrect_both": num_incorrect_both,
            # Consistency check result
            "consistency_check_passed": consistency_check_passed,
        }

    @staticmethod
    def compute_mbpp_r_accuracy(results: List[Dict[str, Any]]) -> float:
        """Compute accuracy on the MBPP-R offline repair task.

        From Table 3 of the paper: MBPP-R is an offline repair task that
        requires correcting incorrect first-attempt programs generated from
        PaLM 2 (Ni et al., 2024). Unlike the two-turn self-correction
        metrics, MBPP-R uses a fixed set of pre-generated incorrect programs
        (not self-generated), so it has a single repair accuracy metric.

        Paper results (Table 3):
            Base model: 47.3%
            Self-Refine: 30.7%
            Pair-SFT: 59.8%
            SCoRe: 60.6%

        The paper notes: "Pair-SFT works nearly as well on the static repair
        task MBPP-R, but actually degrades the base model when evaluated in
        the self-correction setting, thus underscoring the importance of
        on-policy sampling for self-correction."

        Args:
            results: List of result dicts, each containing 'mbpp_r_reward'
                (float in {0.0, 1.0}). This key is populated by
                Evaluator.evaluate_mbpp_r() when running offline repair
                evaluation.

        Returns:
            Float in [0.0, 1.0] representing the fraction of MBPP-R
            problems where the model successfully repaired the incorrect
            program. Returns 0.0 for an empty results list.

        Raises:
            KeyError: If any result dict is missing the 'mbpp_r_reward' key.
                This is a hard error (not silently defaulted to 0.0) because
                missing keys indicate a data pipeline bug that would silently
                corrupt the evaluation results.
        """
        if not results:
            logger.debug(
                "compute_mbpp_r_accuracy: Empty results list. Returning 0.0."
            )
            return 0.0

        # Validate that all result dicts have the required key before
        # computing the mean. Fail fast on the first missing key.
        for i, r in enumerate(results):
            if "mbpp_r_reward" not in r:
                raise KeyError(
                    f"compute_mbpp_r_accuracy: Result dict at index {i} is "
                    "missing the 'mbpp_r_reward' key. "
                    "This key is populated by Evaluator.evaluate_mbpp_r() "
                    "and must be present in all MBPP-R result dicts. "
                    f"Available keys: {list(r.keys())}. "
                    "Check that the results list was produced by "
                    "Evaluator.evaluate_mbpp_r() and not by "
                    "Evaluator.evaluate() (which produces 'reward_t1'/'reward_t2')."
                )

        num_correct: int = sum(
            1
            for r in results
            if Metrics._is_correct(float(r["mbpp_r_reward"]))
        )
        accuracy: float = num_correct / len(results)

        logger.info(
            "compute_mbpp_r_accuracy: %d/%d correct = %.4f (%.1f%%).",
            num_correct,
            len(results),
            accuracy,
            accuracy * 100.0,
        )
        return accuracy

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_correct(reward: float) -> bool:
        """Classify a binary reward value as correct.

        Uses a threshold of 0.5 rather than exact equality with 1.0 for
        robustness against floating-point imprecision. Since rewards from
        RewardFunction.compute_reward() are always exactly 0.0 or 1.0,
        this threshold correctly classifies all valid reward values.

        Args:
            reward: A float reward value, expected to be in {0.0, 1.0}.

        Returns:
            True if reward >= 0.5 (classified as correct), False otherwise.
        """
        return reward >= _CORRECT_THRESHOLD

    @staticmethod
    def _is_incorrect(reward: float) -> bool:
        """Classify a binary reward value as incorrect.

        The logical complement of _is_correct. Uses the same threshold
        for consistency.

        Args:
            reward: A float reward value, expected to be in {0.0, 1.0}.

        Returns:
            True if reward < 0.5 (classified as incorrect), False otherwise.
        """
        return reward < _CORRECT_THRESHOLD
