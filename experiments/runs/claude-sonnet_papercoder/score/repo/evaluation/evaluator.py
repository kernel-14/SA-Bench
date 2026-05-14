```python
## evaluation/evaluator.py
"""Evaluator for SCoRe: Self-Correction via Reinforcement Learning.

This module implements the Evaluator class, the central orchestrator for all
evaluation scenarios described in the SCoRe paper. It runs two-turn (and
multi-turn) inference, computes the five paper metrics, and supports the
inference-compute scaling experiments from Section 6.2.

Paper evaluation protocol (Section 6):
    - Greedy decoding (temperature=0.0) for all main results.
    - Temperature=0.7 for inference-compute scaling experiments (Section 6.2).
    - Two sequential attempts per problem (l=1, i.e., one round of self-correction).
    - Metrics: Accuracy@t1, Accuracy@t2, Δ(t1,t2), Δ^{i→c}, Δ^{c→i}.

The Evaluator is stateless between calls — each evaluate*() call is
independent and reproducible. No model weights are modified.

Typical usage:
    from evaluation.evaluator import Evaluator

    evaluator = Evaluator(model, reward_fn, prompt_templates, config)

    # Main evaluation (Table 2 / Table 3 format)
    results = evaluator.evaluate(test_data, temperature=0.0)
    print(f"Accuracy@t2: {results['accuracy_t2']:.3f}")
    print(f"Delta: {results['delta_t1_t2']:.3f}")

    # MBPP-R offline repair (Table 3)
    mbpp_r_acc = evaluator.evaluate_mbpp_r(mbpp_r_data)

    # Inference-compute scaling (Figure 1 right)
    scaling_results = evaluator.evaluate_inference_scaling(test_data, k=16)
"""

import collections
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from config import Config
from data.prompt_templates import PromptTemplates
from evaluation.edit_distance_analysis import EditDistanceAnalysis
from evaluation.metrics import Metrics
from models.model_wrapper import ModelWrapper
from rewards import RewardFunction
from training.rollout_buffer import Trajectory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Maximum number of problems to process in a single batched generation call.
# Balances GPU memory usage against throughput. With max_new_tokens=1024 and
# bfloat16, a batch of 8 MATH problems fits comfortably on a 40GB GPU.
_EVAL_BATCH_SIZE: int = 8

# Sentinel value for turns that could not be evaluated (context overflow).
_OVERFLOW_REWARD: float = 0.0

# Maximum context length safety margin (tokens). We stop multi-attempt
# evaluation if the prompt would exceed model_max_length - max_new_tokens - margin.
_CONTEXT_SAFETY_MARGIN: int = 128


class Evaluator:
    """Orchestrates all evaluation scenarios for the SCoRe pipeline.

    Runs two-turn (and multi-turn) inference on test datasets and computes
    the five self-correction metrics defined in Section 3 of the paper.
    Also supports MBPP-R offline repair evaluation and inference-compute
    scaling experiments.

    The Evaluator does NOT modify model weights. It is purely inference
    and measurement.

    Attributes:
        model: The ModelWrapper instance to evaluate. Can be the base model,
            a SCoRe-trained model, or any SFT baseline model.
        reward_fn: Unified reward function (MathReward or CodeReward).
        prompt_templates: Prompt builder for both MATH and code tasks.
        metrics: Metrics computation utility (stateless).
        edit_analysis: Edit distance analysis utility.
        config: Global configuration instance.
        task: Task identifier ('math' or 'code'), read from config.task.
        eval_temperature: Evaluation temperature (0.0 = greedy), from
            config.eval_temperature.
        scaling_temperature: Temperature for inference-compute scaling
            experiments (0.7), from config.scaling_temperature.
        max_new_tokens: Maximum new tokens per generation call (1024),
            from config.max_new_tokens.
        edit_distance_enabled: Whether to run edit distance analysis during
            evaluate(), from config.edit_distance_analysis_enabled.
        no_edit_threshold: Threshold for "no edit" classification, from
            config.edit_distance_no_edit_threshold.
        large_edit_threshold: Threshold for "large edit" classification,
            from config.edit_distance_large_edit_threshold.
        multi_attempt_num_attempts: Number of attempts for multi-attempt
            evaluation, from config.multi_attempt_num_attempts.
    """

    def __init__(
        self,
        model: ModelWrapper,
        reward_fn: RewardFunction,
        prompt_templates: PromptTemplates,
        config: Config,
    ) -> None:
        """Initialize the Evaluator.

        Args:
            model: The ModelWrapper instance to evaluate. Must be initialized
                and ready for inference. Weights are not modified.
            reward_fn: Unified reward function. Must be initialized for the
                correct task (config.task). Dispatches to MathReward or
                CodeReward internally.
            prompt_templates: Prompt builder. Provides build_turn1_prompt()
                and build_turn2_prompt() for both MATH and code tasks.
            config: Global Config instance. Reads evaluation-relevant
                hyperparameters: eval_temperature, scaling_temperature,
                max_new_tokens, edit distance thresholds, multi-attempt
                settings.

        Raises:
            ValueError: If config.task is not 'math' or 'code'.
        """
        if config.task not in ("math", "code"):
            raise ValueError(
                f"Evaluator: Invalid task '{config.task}'. "
                "Must be 'math' or 'code'."
            )

        self.model: ModelWrapper = model
        self.reward_fn: RewardFunction = reward_fn
        self.prompt_templates: PromptTemplates = prompt_templates
        self.config: Config = config

        # Instantiate stateless utility classes
        self.metrics: Metrics = Metrics()
        self.edit_analysis: EditDistanceAnalysis = EditDistanceAnalysis(
            no_edit_threshold=config.edit_distance_no_edit_threshold,
            large_edit_threshold=config.edit_distance_large_edit_threshold,
        )

        # Resolve evaluation-relevant config values
        self.task: str = config.task
        # Section 6: "greedy decoding (i.e. temperature 0)"
        self.eval_temperature: float = config.eval_temperature
        # Section 6.2: "we set temperature to be 0.7"
        self.scaling_temperature: float = config.scaling_temperature
        # config.max_new_tokens = 1024 (default)
        self.max_new_tokens: int = config.max_new_tokens
        # Edit distance analysis settings
        self.edit_distance_enabled: bool = config.edit_distance_analysis_enabled
        self.no_edit_threshold: float = config.edit_distance_no_edit_threshold
        self.large_edit_threshold: float = config.edit_distance_large_edit_threshold
        # Multi-attempt evaluation settings
        self.multi_attempt_num_attempts: int = config.multi_attempt_num_attempts

        logging.basicConfig(
            level=getattr(logging, config.log_level, logging.INFO)
        )
        logger.info(
            "Evaluator initialized: task='%s', eval_temperature=%.1f, "
            "scaling_temperature=%.1f, max_new_tokens=%d, "
            "edit_distance_enabled=%s.",
            self.task,
            self.eval_temperature,
            self.scaling_temperature,
            self.max_new_tokens,
            self.edit_distance_enabled,
        )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        test_data: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Run two-turn evaluation on the full test set.

        Primary evaluation method. Runs two-turn self-correction inference
        on all problems in test_data with the specified temperature (default
        greedy, temperature=0.0 per Section 6 of the paper) and computes
        all five paper metrics.

        Implements the evaluation protocol from Section 6:
            "We report the self-correction accuracy on a number of tasks
            with two sequential attempts at the problem, i.e., one round
            of self-correction."

        Batches turn-1 and turn-2 generation across problems for GPU
        efficiency. Turn-2 prompts depend on turn-1 responses, so the
        two turns cannot be batched together, but all turn-1 calls can
        be batched, then all turn-2 calls can be batched.

        Args:
            test_data: List of problem dicts from DatasetLoader.load_test_data().
                For MATH: each dict has 'problem' (str) and 'answer' (str).
                For code (HumanEval): each dict has 'prompt' (str),
                'canonical_solution' (str), 'test' (str), 'entry_point' (str).
            temperature: Sampling temperature. Default 0.0 = greedy decoding
                (Section 6: "greedy decoding (i.e. temperature 0)"). Set to
                0.7 for inference-compute scaling experiments (Section 6.2).

        Returns:
            Dict containing:
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
                'consistency_check_passed' (bool): Metric consistency check.
                'edit_distance_mean' (float): Mean edit distance ratio (if enabled).
                'edit_distance_std' (float): Std of edit distance ratios (if enabled).
                'fraction_no_edit' (float): Fraction with near-zero edit (if enabled).
                'fraction_large_edit' (float): Fraction with large edit (if enabled).
                'per_problem_results' (List[dict]): Per-problem result dicts.
                'total_problems' (int): Alias for num_problems.
        """
        if not test_data:
            logger.warning(
                "evaluate: test_data is empty. Returning zero-valued metrics."
            )
            empty_metrics: Dict[str, Any] = Metrics.compute_all_metrics([])
            empty_metrics.update({
                "edit_distance_mean": 0.0,
                "edit_distance_std": 0.0,
                "fraction_no_edit": 0.0,
                "fraction_large_edit": 0.0,
                "per_problem_results": [],
                "total_problems": 0,
            })
            return empty_metrics

        logger.info(
            "evaluate: Running two-turn evaluation on %d problems "
            "(temperature=%.2f, task='%s').",
            len(test_data),
            temperature,
            self.task,
        )

        # ------------------------------------------------------------------
        # Run two-turn inference for all problems.
        # We use batched generation for efficiency: collect all turn-1
        # prompts, generate all turn-1 responses in batches, then collect
        # all turn-2 prompts, generate all turn-2 responses in batches.
        # ------------------------------------------------------------------
        per_problem_results: List[Dict[str, Any]] = (
            self._run_batched_two_turn_inference(test_data, temperature)
        )

        # ------------------------------------------------------------------
        # Compute the five paper metrics from the per-problem results.
        # ------------------------------------------------------------------
        all_metrics: Dict[str, Any] = Metrics.compute_all_metrics(
            per_problem_results
        )

        # ------------------------------------------------------------------
        # Edit distance analysis (if enabled via config).
        # Constructs lightweight Trajectory-like objects for analyze_batch().
        # ------------------------------------------------------------------
        edit_stats: Dict[str, Any] = {
            "edit_distance_mean": 0.0,
            "edit_distance_std": 0.0,
            "fraction_no_edit": 0.0,
            "fraction_large_edit": 0.0,
        }

        if self.edit_distance_enabled:
            # Build minimal Trajectory objects for analyze_batch().
            # Only turn1_response and turn2_response are needed.
            pseudo_trajectories: List[Trajectory] = [
                Trajectory(
                    problem=r.get("problem", ""),
                    ground_truth=r.get("ground_truth", ""),
                    turn1_response=r.get("turn1_response", ""),
                    turn2_response=r.get("turn2_response", ""),
                    reward_t1=float(r.get("reward_t1", 0.0)),
                    reward_t2=float(r.get("reward_t2", 0.0)),
                )
                for r in per_problem_results
            ]

            batch_stats: Dict[str, Any] = self.edit_analysis.analyze_batch(
                pseudo_trajectories
            )
            edit_stats = {
                "edit_distance_mean": batch_stats.get("mean", 0.0),
                "edit_distance_std": batch_stats.get("std", 0.0),
                "fraction_no_edit": batch_stats.get("fraction_no_edit", 0.0),
                "fraction_large_edit": batch_stats.get("fraction_large_edit", 0.0),
            }

            logger.info(
                "evaluate: Edit distance analysis: mean=%.4f, std=%.4f, "
                "fraction_no_edit=%.3f, fraction_large_edit=%.3f.",
                edit_stats["edit_distance_mean"],
                edit_stats["edit_distance_std"],
                edit_stats["fraction_no_edit"],
                edit_stats["fraction_large_edit"],
            )

        # ------------------------------------------------------------------
        # Assemble the final results dict.
        # ------------------------------------------------------------------
        all_metrics.update(edit_stats)
        all_metrics["per_problem_results"] = per_problem_results
        all_metrics["total_problems"] = len(per_problem_results)

        logger.info(
            "evaluate: Results — acc@t1=%.3f, acc@t2=%.3f, "
            "delta=%.3f, i2c=%.3f, c2i=%.3f.",
            all_metrics.get("accuracy_t1", 0.0),
            all_metrics.get("accuracy_t2", 0.0),
            all_metrics.get("delta_t1_t2", 0.0),
            all_metrics.get("i2c_rate", 0.0),
            all_metrics.get("c2i_rate", 0.0),
        )

        return all_metrics

    def evaluate_mbpp_r(
        self, mbpp_r_data: List[Dict[str, Any]]
    ) -> float:
        """Evaluate offline repair performance on the MBPP-R dataset.

        MBPP-R is an offline repair task where the model must correct
        pre-generated incorrect programs (from PaLM 2 in the paper, or
        equivalent). Unlike the two-turn self-correction evaluation, the
        first-turn response is already provided — we skip turn-1 generation
        and go directly to turn-2 correction.

        From Table 3 of the paper:
            Base model: 47.3%, Pair-SFT: 59.8%, SCoRe: 60.6%

        The paper notes: "Pair-SFT works nearly as well on the static repair
        task MBPP-R, but actually degrades the base model when evaluated in
        the self-correction setting, thus underscoring the importance of
        on-policy sampling for self-correction."

        Args:
            mbpp_r_data: List of offline repair dicts from
                DatasetLoader.load_mbpp_r_data(). Each dict must contain:
                    'text' (str): Task description.
                    'incorrect_code' (str): Pre-generated incorrect program.
                    'test_list' (List[str]): Test assertion strings.
                    'test_setup_code' (str): Optional setup code.
                Returns 0.0 immediately if the list is empty (MBPP-R
                dataset unavailable — graceful degradation).

        Returns:
            Float in [0.0, 1.0] representing the fraction of MBPP-R
            problems where the model successfully repaired the incorrect
            program. Returns 0.0 if mbpp_r_data is empty.
        """
        if not mbpp_r_data:
            logger.warning(
                "evaluate_mbpp_r: mbpp_r_data is empty. "
                "MBPP-R evaluation skipped. Returning 0.0. "
                "To enable MBPP-R evaluation, provide the dataset via "
                "config.mbpp_r_dataset_name."
            )
            return 0.0

        logger.info(
            "evaluate_mbpp_r: Evaluating offline repair on %d MBPP-R problems.",
            len(mbpp_r_data),
        )

        rewards: List[float] = []

        # Process in batches for GPU efficiency
        for batch_start in range(0, len(mbpp_r_data), _EVAL_BATCH_SIZE):
            batch: List[Dict[str, Any]] = mbpp_r_data[
                batch_start: batch_start + _EVAL_BATCH_SIZE
            ]

            # ------------------------------------------------------------------
            # Build turn-2 prompts using the pre-generated incorrect code
            # as the "turn-1 response". The model must correct it.
            # ------------------------------------------------------------------
            turn2_prompts: List[str] = []
            test_cases_batch: List[List[str]] = []

            for item in batch:
                problem_text: str = str(item.get("text", ""))
                incorrect_code: str = str(item.get("incorrect_code", ""))

                # Build turn-2 prompt: problem context + incorrect code +
                # self-correction instruction (same as standard turn-2)
                prompt_t2: str = self.prompt_templates.build_turn2_prompt(
                    problem=problem_text,
                    turn1_response=incorrect_code,
                    task="code",  # MBPP-R is always a code task
                )
                turn2_prompts.append(prompt_t2)

                # Build test cases for reward computation
                raw_test_list: Any = item.get("test_list", [])
                if isinstance(raw_test_list, str):
                    test_cases: List[str] = [raw_test_list]
                else:
                    test_cases = [str(t) for t in raw_test_list]

                setup_code: str = str(item.get("test_setup_code", "")).strip()
                if setup_code:
                    test_cases = [setup_code] + test_cases

                test_cases_batch.append(test_cases)

            # ------------------------------------------------------------------
            # Generate turn-2 responses (corrections) with greedy decoding.
            # ------------------------------------------------------------------
            try:
                turn2_responses: List[str] = self.model.generate(
                    prompts=turn2_prompts,
                    temperature=0.0,  # Greedy decoding for evaluation
                    max_new_tokens=self.max_new_tokens,
                )
            except Exception as exc:
                logger.error(
                    "evaluate_mbpp_r: Generation failed for batch starting "
                    "at index %d: %s. Assigning 0.0 reward to all items "
                    "in this batch.",
                    batch_start,
                    exc,
                )
                rewards.extend([0.0] * len(batch))
                continue

            # ------------------------------------------------------------------
            # Compute rewards for each corrected response.
            # ------------------------------------------------------------------
            batch_rewards: List[float] = self.reward_fn.batch_compute(
                predictions=turn2_responses,
                ground_truths=[""] * len(batch),  # Unused for code tasks
                test_cases=test_cases_batch,
            )
            rewards.extend(batch_rewards)

        # Compute mean accuracy
        if not rewards:
            logger.warning(
                "evaluate_mbpp_r: No rewards computed. Returning 0.0."
            )
            return 0.0

        mbpp_r_accuracy: float = sum(rewards) / len(rewards)

        logger.info(
            "evaluate_mbpp_r: MBPP-R accuracy = %.4f (%.1f%%) "
            "on %d problems.",
            mbpp_r_accuracy,
            mbpp_r_accuracy * 100.0,
            len(rewards),
        )

        return mbpp_r_accuracy

    def evaluate_inference_scaling(
        self,
        test_data: List[Dict[str, Any]],
        k: int = 16,
    ) -> Dict[str, Any]:
        """Evaluate inference-compute scaling strategies (Figure 1 right, Section 6.2).

        Compares two strategies with a fixed total sample budget of 2K:

        Strategy A (Parallel): Sample 2K solutions independently, majority vote.
            "the default strategy is to sample all solutions in parallel to
            perform majority voting."

        Strategy B (Sequential + Parallel): Sample K solutions in parallel,
            apply one round of self-correction to each (yielding 2K total
            responses), majority vote over all 2K.
            "instead of sampling 2K solutions in parallel, it is more
            compute-efficient to sample K solutions in parallel, then perform
            one round of self-correction on each solution."

        Paper result (Section 6.2): "With 32 solution budget per problem,
        parallel sampling shows a 7.4% accuracy gain, while combining it
        with sequential sampling using self-correction yields a 10.5%
        improvement."

        Uses temperature=0.7 (config.scaling_temperature) for sampling,
        as specified in Section 6.2: "we set temperature to be 0.7."

        Args:
            test_data: List of problem dicts from DatasetLoader.load_test_data().
            k: Number of parallel samples per problem. Total budget = 2K.
                The paper tests k values from config.evaluation.inference_scaling.k_values
                = [1, 2, 4, 8, 16, 32]. The caller (main.py) iterates over
                k_values and calls this method for each k.

        Returns:
            Dict containing:
                'parallel_accuracy' (float): Strategy A accuracy (2K parallel).
                'sequential_accuracy' (float): Strategy B accuracy (K parallel + K sequential).
                'k' (int): The k value used.
                'total_budget' (int): Total samples per problem = 2K.
                'per_problem_results' (List[dict]): Per-problem results with
                    both strategy outcomes.
        """
        if not test_data:
            logger.warning(
                "evaluate_inference_scaling: test_data is empty. "
                "Returning zero-valued results."
            )
            return {
                "parallel_accuracy": 0.0,
                "sequential_accuracy": 0.0,
                "k": k,
                "total_budget": 2 * k,
                "per_problem_results": [],
            }

        if k <= 0:
            raise ValueError(
                f"evaluate_inference_scaling: k must be positive (got {k})."
            )

        temperature: float = self.scaling_temperature  # 0.7 per Section 6.2
        total_budget: int = 2 * k

        logger.info(
            "evaluate_inference_scaling: k=%d, total_budget=%d, "
            "temperature=%.2f, num_problems=%d.",
            k,
            total_budget,
            temperature,
            len(test_data),
        )

        per_problem_results: List[Dict[str, Any]] = []
        parallel_correct: int = 0
        sequential_correct: int = 0

        for problem_idx, problem in enumerate(test_data):
            problem_text, ground_truth, test_cases = (
                self._extract_problem_fields(problem)
            )

            # Build the turn-1 prompt for this problem
            prompt_t1: str = self._build_turn1_prompt(problem)

            # ------------------------------------------------------------------
            # Strategy A: Sample 2K responses in parallel, majority vote.
            # ------------------------------------------------------------------
            try:
                # Generate 2K independent responses at temperature 0.7
                parallel_2k_responses: List[str] = self.model.generate(
                    prompts=[prompt_t1] * total_budget,
                    temperature=temperature,
                    max_new_tokens=self.max_new_tokens,
                )
            except Exception as exc:
                logger.error(
                    "evaluate_inference_scaling: Strategy A generation failed "
                    "for problem %d: %s. Assigning 0.0 reward.",
                    problem_idx,
                    exc,
                )
                parallel_2k_responses = [""] * total_budget

            # Majority vote over 2K responses
            parallel_voted_answer: str = self._majority_vote(
                parallel_2k_responses
            )
            # Compute reward for the majority-voted answer
            parallel_reward: float = self.reward_fn.compute_reward(
                prediction=parallel_voted_answer,
                ground_truth=ground_truth,
                test_cases=test_cases if test_cases else None,
            )
            if parallel_reward >= 0.5:
                parallel_correct += 1

            # ------------------------------------------------------------------
            # Strategy B: Sample K responses in parallel, then one round of
            # self-correction on each, majority vote over all 2K.
            # ------------------------------------------------------------------
            try:
                # Step 1: Generate K independent turn-1 responses
                turn1_k_responses: List[str] = self.model.generate(
                    prompts=[prompt_t1] * k,
                    temperature=temperature,
                    max_new_tokens=self.max_new_tokens,
                )
            except Exception as exc:
                logger.error(
                    "evaluate_inference_scaling: Strategy B turn-1 generation "
                    "failed for problem %d: %s. Assigning 0.0 reward.",
                    problem_idx,
                    exc,
                )
                turn1_k_responses = [""] * k

            # Step 2: Build K turn-2 prompts (one per turn-1 response)
            turn2_prompts: List[str] = [
                self.prompt_templates.build_turn2_prompt(
                    problem=problem_text,
                    turn1_response=t1_resp,
                    task=self.task,
                )
                for t1_resp in turn1_k_responses
            ]

            # Step 3: Generate K turn-2 responses
            try:
                turn2_k_responses: List[str] = self.model.generate(
                    prompts=turn2_prompts,
                    temperature=temperature,
                    max_new_tokens=self.max_new_tokens,
                )
            except Exception as exc:
                logger.error(
                    "evaluate_inference_scaling: Strategy B turn-2 generation "
                    "failed for problem %d: %s. Using empty turn-2 responses.",
                    problem_idx,
                    exc,
                )
                turn2_k_responses = [""] * k

            # Step 4: Pool all 2K responses (K turn-1 + K turn-2)
            all_2k_responses: List[str] = turn1_k_responses + turn2_k_responses

            # Step 5: Majority vote over all 2K responses
            sequential_voted_answer: str = self._majority_vote(
                all_2k_responses
            )
            sequential_reward: float = self.reward_fn.compute_reward(
                prediction=sequential_voted_answer,
                ground_truth=ground_truth,
                test_cases=test_cases if test_cases else None,
            )
            if sequential_reward >= 0.5:
                sequential_correct += 1

            # Record per-problem results
            per_problem_results.append({
                "problem": problem_text,
                "ground_truth": ground_truth,
                "parallel_reward": parallel_reward,
                "sequential_reward": sequential_reward,
                "parallel_voted_answer": parallel_voted_answer,
                "sequential_voted_answer": sequential_voted_answer,
            })

            if (problem_idx + 1) % 50 == 0:
                logger.info(
                    "evaluate_inference_scaling: Processed %d/%d problems. "
                    "Parallel acc=%.3f, Sequential acc=%.3f.",
                    problem_idx + 1,
                    len(test_data),
                    parallel_correct / (problem_idx + 1),
                    sequential_correct / (problem_idx + 1),
                )

        n: int = len(test_data)
        parallel_accuracy: float = parallel_correct / n if n > 0 else 0.0
        sequential_accuracy: float = sequential_correct / n if n > 0 else 0.0

        logger.info(
            "evaluate_inference_scaling: k=%d, total_budget=%d. "
            "Parallel accuracy=%.4f, Sequential accuracy=%.4f. "
            "Sequential improvement=%.4f.",
            k,
            total_budget,
            parallel_accuracy,
            sequential_accuracy,
            sequential_accuracy - parallel_accuracy,
        )

        return {
            "parallel_accuracy": parallel_accuracy,
            "sequential_accuracy": sequential_accuracy,
            "k": k,
            "total_budget": total_budget,
            "per_problem_results": per_problem_results,
        }

    def evaluate_multi_attempt(
        self,
        test_data: List[Dict[str, Any]],
        num_attempts: int = 10,
    ) -> Dict[str, Any]:
        """Evaluate sequential self-correction over multiple attempts (Figure 8).

        Implements the multi-attempt scaling experiment from Appendix A.1:
            "We investigate the performance of various models when asked to
            iteratively self-correct over multiple attempts, despite only
            being trained over two attempts (or not at all, in the case of
            the base model)."

        The model was only trained for 2 turns, so this tests generalization
        to more turns. The paper finds that SCoRe's performance increases
        slightly past two turns but plateaus.

        Uses greedy decoding (temperature=0.0) for reproducibility.

        Args:
            test_data: List of problem dicts from DatasetLoader.load_test_data().
            num_attempts: Number of sequential self-correction attempts.
                From config.multi_attempt_num_attempts = 10 (Appendix A.1,
                Figure 8 shows 10 attempts).

        Returns:
            Dict containing:
                'per_turn_accuracy' (List[float]): Accuracy at each turn,
                    length = num_attempts. per_turn_accuracy[0] = Accuracy@t1,
                    per_turn_accuracy[1] = Accuracy@t2, etc.
                'per_problem_results' (List[dict]): Per-problem results with
                    rewards for all turns.
        """
        if not test_data:
            logger.warning(
                "evaluate_multi_attempt: test_data is empty. "
                "Returning zero-valued results."
            )
            return {
                "per_turn_accuracy": [0.0] * num_attempts,
                "per_problem_results": [],
            }

        if num_attempts <= 0:
            raise ValueError(
                f"evaluate_multi_attempt: num_attempts must be positive