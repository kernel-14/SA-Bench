## rewards/__init__.py
"""Unified reward function interface for SCoRe: Self-Correction via Reinforcement Learning.

This module implements RewardFunction, the single entry point for all reward
computation in the SCoRe pipeline. It abstracts task-specific reward logic
(MATH answer matching vs. code execution) behind a uniform interface consumed
by RolloutBuffer, SCoReStage1Trainer, SCoReStage2Trainer, SFTBaselineTrainer,
and Evaluator.

The binary reward convention ({0.0, 1.0}) is a system-wide invariant. Every
downstream consumer — REINFORCE loss, KL penalty computation, reward shaping
(shaped_r2 = r2 + alpha * (r2 - r1)), and all five evaluation metrics —
depends on this contract being upheld.

Config values used (from config.yaml):
    task: "math" or "code"
    reward.code.execution_timeout: 5  (passed to CodeReward)
    reward.code.max_memory_mb: 512    (passed to CodeReward)

Typical usage:
    from rewards import RewardFunction

    # For MATH task
    reward_fn = RewardFunction(task="math")
    score = reward_fn.compute_reward(model_response, ground_truth_answer)

    # For code task
    reward_fn = RewardFunction(task="code")
    score = reward_fn.compute_reward(
        prediction=generated_code,
        ground_truth="",          # unused for code; test_cases carries verification
        test_cases=["assert func(1) == 2", "assert func(2) == 4"],
    )

    # Batch computation (parallel for code, sequential for math)
    rewards = reward_fn.batch_compute(predictions, ground_truths, test_cases)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional

from rewards.math_reward import MathReward
from rewards.code_reward import CodeReward

logger = logging.getLogger(__name__)

# Maximum number of parallel workers for code reward batch computation.
# Each worker spawns a subprocess with a 5-second timeout. Bounded to
# avoid resource exhaustion on machines with limited file descriptors.
_MAX_CODE_WORKERS: int = 32


class RewardFunction:
    """Unified reward function interface for MATH and code generation tasks.

    Dispatches to MathReward (sympy-based answer matching) or CodeReward
    (sandboxed subprocess execution) based on the task specified at
    construction time. Provides both single-sample and batch interfaces.

    The binary reward convention is strictly enforced: all return values
    are floats in {0.0, 1.0}. This is critical for downstream loss
    computations that multiply rewards by log-probability tensors.

    Attributes:
        task: Task identifier, either "math" or "code". Immutable after
            construction.
        _reward: The underlying task-specific reward object. Either a
            MathReward or CodeReward instance. Private — external callers
            use compute_reward() and batch_compute().
    """

    def __init__(self, task: str) -> None:
        """Initialize RewardFunction for the specified task.

        Instantiates the appropriate underlying reward class:
            - "math" → MathReward() with pre-compiled regex patterns and
              sympy-based equivalence checking.
            - "code" → CodeReward() with default timeout=5 seconds and
              max_memory_mb=512, matching config.yaml defaults.

        Args:
            task: Task identifier. Must be "math" or "code". This value
                corresponds to config.task in config.yaml.

        Raises:
            ValueError: If task is not "math" or "code". Fails fast to
                prevent silent dispatch to None.
        """
        if task not in ("math", "code"):
            raise ValueError(
                f"Invalid task '{task}'. RewardFunction only supports "
                "'math' and 'code'. This value must match config.task "
                "in config.yaml."
            )

        self.task: str = task

        if task == "math":
            # MathReward uses sympy for symbolic equivalence checking and
            # regex-based answer extraction from the canonical format:
            # "Final Answer: The final answer is $answer$. I hope it is correct."
            self._reward: object = MathReward()
            logger.info(
                "RewardFunction initialized for task='math' using MathReward "
                "(sympy-based answer matching)."
            )
        else:
            # CodeReward spawns isolated subprocesses with a hard timeout.
            # Default values match config.yaml:
            #   reward.code.execution_timeout: 5
            #   reward.code.max_memory_mb: 512
            self._reward = CodeReward(timeout=5, max_memory_mb=512)
            logger.info(
                "RewardFunction initialized for task='code' using CodeReward "
                "(sandboxed subprocess execution, timeout=5s, max_memory=512MB)."
            )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def compute_reward(
        self,
        prediction: str,
        ground_truth: str,
        test_cases: Optional[List[str]] = None,
    ) -> float:
        """Compute the binary correctness reward for a single prediction.

        Dispatches to the appropriate task-specific reward function. The
        return value is always a float in {0.0, 1.0} — never int, never bool.

        For MATH:
            Calls MathReward.compute(prediction, ground_truth). The
            test_cases argument is ignored. Returns 1.0 if the extracted
            answer is mathematically equivalent to ground_truth, else 0.0.

        For code:
            Calls CodeReward.compute(code=prediction, test_cases=test_cases).
            The ground_truth argument is semantically unused for code tasks —
            test_cases carries the verification logic (MBPP assertions or
            HumanEval test function). Returns 1.0 if all test cases pass,
            else 0.0.

        Args:
            prediction: The full model response string. For MATH, this is
                the complete response containing "Final Answer: ...". For
                code, this is the raw model response containing the generated
                Python code (possibly with markdown fences).
            ground_truth: The ground truth answer string. For MATH, this is
                the expected answer (e.g., "\\frac{3}{7}" or "42"). For code,
                this parameter is unused — pass an empty string or None.
            test_cases: For code tasks, a list of test assertion strings
                (MBPP) or a list containing the HumanEval test function
                string. For MATH tasks, this parameter is ignored. If None
                or empty for a code task, returns 0.0 defensively (a code
                submission with no test cases cannot be verified).

        Returns:
            1.0 if the prediction is correct, 0.0 otherwise. Always a float.
        """
        # Guard: empty prediction → immediate failure without calling
        # the underlying reward (avoids unnecessary sympy/subprocess overhead)
        if not prediction or not prediction.strip():
            logger.debug(
                "compute_reward: empty prediction string. Returning 0.0."
            )
            return 0.0

        if self.task == "math":
            math_reward: MathReward = self._reward  # type: ignore[assignment]
            result: float = math_reward.compute(prediction, ground_truth)
            # Enforce float type — MathReward.compute() already returns float,
            # but be explicit for type safety
            return float(result)

        else:
            # Code task: test_cases is required for verification
            if not test_cases:
                logger.debug(
                    "compute_reward (code): test_cases is None or empty. "
                    "Cannot verify code correctness. Returning 0.0."
                )
                return 0.0

            code_reward: CodeReward = self._reward  # type: ignore[assignment]
            # CodeReward.extract_code() handles markdown fences, [BEGIN]/[DONE]
            # delimiters, and raw code — called internally by compute() via
            # the prediction string. We pass prediction directly as the code
            # argument; CodeReward.compute() expects the extracted code string.
            # Extract code first to match CodeReward's interface.
            extracted_code: str = code_reward.extract_code(prediction)
            result = code_reward.compute(
                code=extracted_code,
                test_cases=test_cases,
            )
            return float(result)

    def batch_compute(
        self,
        predictions: List[str],
        ground_truths: List[str],
        test_cases: Optional[List[Optional[List[str]]]] = None,
    ) -> List[float]:
        """Compute rewards for an entire batch of predictions.

        Returns a List[float] of the same length as predictions, in the
        same order. The return order guarantee is non-negotiable — callers
        (RolloutBuffer.sample_trajectories) assign reward_t1 and reward_t2
        by index.

        Parallelism strategy:
            - MATH: Sequential execution via list comprehension. Math reward
              involves sympy parsing (CPU-bound, milliseconds per sample).
              The GIL is not a bottleneck for the typical batch sizes used
              in MATH training (batch_size=512 from Table 5, but individual
              reward calls are fast).
            - Code: Parallel execution via ThreadPoolExecutor. Each code
              evaluation spawns a subprocess with a 5-second timeout. With
              batch_size=128 (MBPP config from Table 5), sequential execution
              would take up to 640 seconds per batch — completely infeasible.
              Parallel execution reduces this to ~5 seconds (one timeout
              cycle) in the worst case.

        Args:
            predictions: List of model response strings. Length N.
            ground_truths: List of ground truth answer strings. Length N.
                For code tasks, these are semantically unused — pass a list
                of empty strings if ground truth is not available.
            test_cases: For code tasks, a list of N test case lists, where
                test_cases[i] is the list of test assertions for predictions[i].
                For MATH tasks, pass None (ignored). If None for code tasks,
                all rewards default to 0.0.

        Returns:
            List of float rewards in {0.0, 1.0}, length N, in the same
            order as predictions.

        Raises:
            ValueError: If len(predictions) != len(ground_truths), or if
                test_cases is provided and len(test_cases) != len(predictions).
                These assertions catch misaligned batch construction bugs
                before they silently corrupt reward signals.
        """
        # ------------------------------------------------------------------
        # Input validation
        # ------------------------------------------------------------------
        n: int = len(predictions)

        if len(ground_truths) != n:
            raise ValueError(
                f"batch_compute: len(predictions)={n} != "
                f"len(ground_truths)={len(ground_truths)}. "
                "Batch sizes must match."
            )

        if test_cases is not None and len(test_cases) != n:
            raise ValueError(
                f"batch_compute: len(predictions)={n} != "
                f"len(test_cases)={len(test_cases)}. "
                "test_cases must have the same length as predictions."
            )

        # Empty batch — return immediately
        if n == 0:
            return []

        # ------------------------------------------------------------------
        # MATH: sequential execution
        # ------------------------------------------------------------------
        if self.task == "math":
            rewards: List[float] = [
                self.compute_reward(
                    prediction=predictions[i],
                    ground_truth=ground_truths[i],
                    test_cases=None,
                )
                for i in range(n)
            ]
            logger.debug(
                "batch_compute (math): computed %d rewards sequentially. "
                "Mean reward: %.3f.",
                n,
                sum(rewards) / n if n > 0 else 0.0,
            )
            return rewards

        # ------------------------------------------------------------------
        # Code: parallel execution via ThreadPoolExecutor
        # ------------------------------------------------------------------
        # Determine the number of workers: bounded by _MAX_CODE_WORKERS
        # and the actual batch size to avoid spawning unnecessary threads.
        num_workers: int = min(n, _MAX_CODE_WORKERS)

        # Prepare per-sample test_cases (None → list of None values)
        effective_test_cases: List[Optional[List[str]]] = (
            test_cases if test_cases is not None
            else [None] * n
        )

        # Submit all tasks to the thread pool
        # Using a list to preserve submission order for result collection
        futures: List[Future] = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i in range(n):
                future: Future = executor.submit(
                    self.compute_reward,
                    predictions[i],
                    ground_truths[i],
                    effective_test_cases[i],
                )
                futures.append(future)

            # Collect results in submission order (preserves batch ordering)
            # Future.result() blocks until the task completes or raises.
            # Since compute_reward() never raises (all exceptions are caught
            # internally), this is safe.
            code_rewards: List[float] = []
            for i, future in enumerate(futures):
                try:
                    reward_val: float = future.result()
                    code_rewards.append(float(reward_val))
                except Exception as exc:
                    # Belt-and-suspenders: compute_reward should never raise,
                    # but catch anything that slips through.
                    logger.warning(
                        "batch_compute (code): future[%d] raised unexpected "
                        "exception: %s. Defaulting to 0.0.",
                        i,
                        exc,
                    )
                    code_rewards.append(0.0)

        logger.debug(
            "batch_compute (code): computed %d rewards with %d workers. "
            "Mean reward: %.3f.",
            n,
            num_workers,
            sum(code_rewards) / n if n > 0 else 0.0,
        )
        return code_rewards
