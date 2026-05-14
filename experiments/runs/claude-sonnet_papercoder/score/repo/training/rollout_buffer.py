```python
## training/rollout_buffer.py
"""Rollout buffer for SCoRe: Self-Correction via Reinforcement Learning.

This module implements the two-turn trajectory sampling engine for SCoRe's
multi-turn RL training pipeline. It is the central data collection component
consumed by SCoReStage1Trainer, SCoReStage2Trainer, and REINFORCETrainer.

The Trajectory dataclass captures the complete state of a two-turn rollout:
both prompts, both responses, both rewards, and log-probabilities from both
the policy and reference models. This is the τ = {x₁, ŷ₁, r̂₁, x₂, ŷ₂, r̂₂}
notation from the paper.

Key design invariants:
    - All log-probability tensors stored in Trajectory are .detach()ed to
      prevent memory accumulation across training steps.
    - Reference model computations are wrapped in torch.no_grad() defensively,
      even though ModelWrapper(freeze=True) already sets eval mode.
    - Batch generation is used throughout for GPU efficiency.
    - Offline data mixing (Section 5.3) is handled transparently: batch items
      with a pre-populated 'turn1_response' key skip turn 1 generation.

Alignment with paper equations:
    ŷ₁ ~ π_θ(·|x)                    → policy_model.generate(turn1_prompts)
    ŷ₂ ~ π_θ(·|[x, ŷ₁, p₁])         → policy_model.generate(turn2_prompts)
    r̂(y₁, y*)                         → reward_fn.batch_compute(turn1_responses)
    r̂(y₂, y*)                         → reward_fn.batch_compute(turn2_responses)
    b̂(y₂|y₁,y*) = α·(r̂₂ - r̂₁)      → compute_shaped_rewards(trajectories, alpha)
    D_KL(π_θ||π_ref)                  → compute_kl_divergence(logprobs, ref_logprobs)

Typical usage:
    from training.rollout_buffer import RolloutBuffer, Trajectory

    buffer = RolloutBuffer(policy_model, ref_model, reward_fn, prompt_templates, config)
    trajectories = buffer.sample_trajectories(batch)
    trajectories = buffer.compute_shaped_rewards(trajectories, alpha=config.alpha)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from config import Config
from data.prompt_templates import PromptTemplates
from models.model_wrapper import ModelWrapper
from rewards import RewardFunction

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """Complete state of a two-turn self-correction rollout.

    This is the fundamental data unit flowing through the entire SCoRe
    training pipeline. It corresponds to the trajectory notation
    τ = {x₁, ŷ₁, r̂(y₁,y*), x₂, ŷ₂, r̂(y₂,y*)} from Section 5.2 of the paper.

    All Tensor fields are scalar tensors (sum of per-token log-probs over
    the response, not the prompt). They are stored detached from the
    computation graph to prevent memory accumulation across training steps.
    Trainers recompute log-probs with gradients enabled during loss computation.

    Attributes:
        problem: The raw problem text (math problem statement or code task
            description). Used to rebuild prompts if needed.
        ground_truth: The ground truth answer string (for MATH) or empty
            string (for code, where test_cases carries verification logic).
        test_cases: For code tasks, list of assertion strings used to
            evaluate correctness. For MATH tasks, empty list [].
        turn1_prompt: The full prompt fed to the model at turn 1. Built by
            PromptTemplates.build_turn1_prompt(). Includes system prompt
            and problem for MATH; 3-shot examples and problem for code.
        turn1_response: The model's generated response at turn 1 (newly
            generated tokens only, not the prompt). Sampled at training
            temperature (1.0 per Table 5).
        turn2_prompt: The full prompt fed to the model at turn 2. This is
            turn1_prompt + turn1_response + self-correction instruction.
            Built by PromptTemplates.build_turn2_prompt(). Critically does
            NOT reveal whether turn 1 was correct (intrinsic self-correction).
        turn2_response: The model's generated response at turn 2 (newly
            generated tokens only).
        reward_t1: Binary reward {0.0, 1.0} for turn 1 correctness.
        reward_t2: Binary reward {0.0, 1.0} for turn 2 correctness.
        logprob_t1: Scalar tensor: sum of per-token log P(token | context)
            under the POLICY model for turn1_response tokens only. Detached.
        logprob_t2: Scalar tensor: sum of per-token log P(token | context)
            under the POLICY model for turn2_response tokens only. Detached.
        ref_logprob_t1: Scalar tensor: sum of per-token log P(token | context)
            under the REFERENCE (frozen base) model for turn1_response tokens.
            Used for β₂ KL penalty in Stage I (Equation 3).
        ref_logprob_t2: Scalar tensor: sum of per-token log P(token | context)
            under the REFERENCE model for turn2_response tokens.
            Used for β₁ KL penalty in both stages (Equations 3 and 4).
        shaped_reward_t2: The reward-shaped second-attempt reward used in
            Stage II. Initially 0.0; populated by compute_shaped_rewards().
            Value: r2 + alpha * (r2 - r1). See Section 5.2, Equation 5.
    """

    # Problem context
    problem: str = ""
    ground_truth: str = ""
    test_cases: List[str] = field(default_factory=list)

    # Turn 1
    turn1_prompt: str = ""
    turn1_response: str = ""

    # Turn 2
    turn2_prompt: str = ""
    turn2_response: str = ""

    # Rewards (binary: 0.0 or 1.0)
    reward_t1: float = 0.0
    reward_t2: float = 0.0

    # Policy model log-probabilities (summed over response tokens, detached)
    logprob_t1: Optional[torch.Tensor] = field(default=None)
    logprob_t2: Optional[torch.Tensor] = field(default=None)

    # Reference model log-probabilities (summed over response tokens, detached)
    ref_logprob_t1: Optional[torch.Tensor] = field(default=None)
    ref_logprob_t2: Optional[torch.Tensor] = field(default=None)

    # Stage II reward shaping: r2 + alpha * (r2 - r1)
    # Populated by RolloutBuffer.compute_shaped_rewards()
    shaped_reward_t2: float = 0.0


class RolloutBuffer:
    """Two-turn trajectory sampling engine for SCoRe multi-turn RL.

    Handles the complete rollout collection process:
        1. Generate turn 1 responses from the policy model.
        2. Compute turn 1 rewards.
        3. Build turn 2 prompts (problem + turn1_response + correction instruction).
        4. Generate turn 2 responses from the policy model.
        5. Compute turn 2 rewards.
        6. Compute log-probabilities from both policy and reference models.
        7. Assemble Trajectory objects.

    Also provides utilities for reward shaping (Stage II) and KL divergence
    computation (used by trainers during loss computation).

    The policy_model reference is mutable — callers update model weights
    externally between training steps, and the buffer always uses the
    current state of policy_model at the time of each sample_trajectories call.

    Attributes:
        policy_model: The trainable policy model (updated each step).
        ref_model: The frozen reference model (never updated).
        reward_fn: Unified reward function (dispatches to MATH or code).
        prompt_templates: Prompt builder for both tasks.
        config: Global configuration instance.
    """

    def __init__(
        self,
        policy_model: ModelWrapper,
        ref_model: ModelWrapper,
        reward_fn: RewardFunction,
        prompt_templates: PromptTemplates,
        config: Config,
    ) -> None:
        """Initialize RolloutBuffer.

        Args:
            policy_model: The trainable policy model (π_θ). Updated externally
                between training steps. The buffer always uses the current
                state at the time of each sample_trajectories() call.
            ref_model: The frozen reference model (π_ref). Must have been
                initialized with freeze=True in ModelWrapper. Never updated.
            reward_fn: Unified reward function. Dispatches to MathReward
                (sympy-based) or CodeReward (subprocess execution) based on
                the task set at construction time.
            prompt_templates: Prompt builder. Provides build_turn1_prompt()
                and build_turn2_prompt() for both MATH and code tasks.
            config: Global Config instance. Reads:
                config.task (for task-specific reward extraction),
                config.sampling_temperature (1.0 per Table 5),
                config.max_new_tokens (1024 default).
        """
        self.policy_model: ModelWrapper = policy_model
        self.ref_model: ModelWrapper = ref_model
        self.reward_fn: RewardFunction = reward_fn
        self.prompt_templates: PromptTemplates = prompt_templates
        self.config: Config = config

        logger.info(
            "RolloutBuffer initialized for task='%s', "
            "sampling_temperature=%.1f, max_new_tokens=%d.",
            config.task,
            config.sampling_temperature,
            config.max_new_tokens,
        )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def sample_trajectories(
        self, batch: List[Dict[str, Any]]
    ) -> List[Trajectory]:
        """Sample two-turn trajectories for a batch of problems.

        This is the primary method called by all trainers at each training
        step. Processes the entire batch in vectorized calls for GPU efficiency.

        Handles offline data mixing transparently: if a batch item contains
        a pre-populated 'turn1_response' key (from base model offline samples,
        Section 5.3), that response is used directly without generating from
        the policy model. Log-probabilities are still computed for the offline
        response under both the current policy and reference model.

        Args:
            batch: List of problem dicts. Each dict must contain:
                For MATH:
                    'problem' (str): The problem text.
                    'answer' (str): The ground truth answer.
                    Optional 'turn1_response' (str): Pre-generated turn 1
                        response for offline data mixing (Stage II).
                For code (MBPP training):
                    'text' (str): The task description.
                    'code' (str): The canonical solution (unused for reward).
                    'test_list' (List[str]): Test assertion strings.
                    'test_setup_code' (str): Setup code for tests.
                    Optional 'turn1_response' (str): Pre-generated turn 1
                        response for offline data mixing.
                For code (HumanEval test):
                    'prompt' (str): Function signature + docstring.
                    'canonical_solution' (str): Reference solution.
                    'test' (str): Test function string.
                    'entry_point' (str): Function name to test.
                    Optional 'turn1_response' (str): Pre-generated turn 1.

        Returns:
            List of Trajectory objects, one per batch item, in the same
            order as the input batch. All Tensor fields are detached.

        Raises:
            ValueError: If batch is empty.
        """
        if not batch:
            raise ValueError(
                "sample_trajectories: batch must be non-empty. "
                "Received an empty list."
            )

        batch_size: int = len(batch)
        logger.debug(
            "sample_trajectories: Processing batch of %d problems.", batch_size
        )

        # ------------------------------------------------------------------
        # Step 1: Extract problem fields and build turn 1 prompts
        # ------------------------------------------------------------------
        problems: List[str] = []
        ground_truths: List[str] = []
        test_cases_batch: List[List[str]] = []
        turn1_prompts: List[str] = []

        for item in batch:
            problem_text, ground_truth, test_cases = self._extract_problem_fields(item)
            problems.append(problem_text)
            ground_truths.append(ground_truth)
            test_cases_batch.append(test_cases)

            # Build turn 1 prompt using the appropriate template
            t1_prompt: str = self.prompt_templates.build_turn1_prompt(
                problem_text, self.config.task
            )
            turn1_prompts.append(t1_prompt)

        # ------------------------------------------------------------------
        # Step 2: Generate turn 1 responses (or use pre-generated offline ones)
        # ------------------------------------------------------------------
        # Identify which items need generation vs. which have offline responses
        needs_generation_mask: List[bool] = [
            "turn1_response" not in item or not item["turn1_response"]
            for item in batch
        ]
        num_needs_generation: int = sum(needs_generation_mask)

        turn1_responses: List[str] = [""] * batch_size

        # Fill in pre-generated offline responses first
        for i, item in enumerate(batch):
            if not needs_generation_mask[i]:
                turn1_responses[i] = str(item["turn1_response"])
                logger.debug(
                    "sample_trajectories: Using offline turn1_response for "
                    "item %d (offline data mixing, Section 5.3).",
                    i,
                )

        # Generate responses for items that need it (batched for efficiency)
        if num_needs_generation > 0:
            generation_indices: List[int] = [
                i for i, needs in enumerate(needs_generation_mask) if needs
            ]
            prompts_to_generate: List[str] = [
                turn1_prompts[i] for i in generation_indices
            ]

            generated_t1: List[str] = self.policy_model.generate(
                prompts=prompts_to_generate,
                temperature=self.config.sampling_temperature,
                max_new_tokens=self.config.max_new_tokens,
            )

            # Place generated responses back in the correct positions
            for idx, gen_response in zip(generation_indices, generated_t1):
                turn1_responses[idx] = gen_response

        # ------------------------------------------------------------------
        # Step 3: Compute turn 1 rewards
        # ------------------------------------------------------------------
        rewards_t1: List[float] = self._compute_rewards_batch(
            responses=turn1_responses,
            ground_truths=ground_truths,
            test_cases_batch=test_cases_batch,
        )

        # ------------------------------------------------------------------
        # Step 4: Build turn 2 prompts
        # ------------------------------------------------------------------
        turn2_prompts: List[str] = []
        for i, item in enumerate(batch):
            t2_prompt: str = self.prompt_templates.build_turn2_prompt(
                problem=problems[i],
                turn1_response=turn1_responses[i],
                task=self.config.task,
            )
            turn2_prompts.append(t2_prompt)

        # ------------------------------------------------------------------
        # Step 5: Generate turn 2 responses (always from current policy)
        # ------------------------------------------------------------------
        turn2_responses: List[str] = self.policy_model.generate(
            prompts=turn2_prompts,
            temperature=self.config.sampling_temperature,
            max_new_tokens=self.config.max_new_tokens,
        )

        # ------------------------------------------------------------------
        # Step 6: Compute turn 2 rewards
        # ------------------------------------------------------------------
        rewards_t2: List[float] = self._compute_rewards_batch(
            responses=turn2_responses,
            ground_truths=ground_truths,
            test_cases_batch=test_cases_batch,
        )

        # ------------------------------------------------------------------
        # Step 7: Compute policy model log-probabilities
        # These are stored detached — trainers recompute with gradients
        # enabled during loss computation.
        # ------------------------------------------------------------------
        logprobs_t1: torch.Tensor = self.policy_model.compute_log_probs(
            prompts=turn1_prompts,
            responses=turn1_responses,
        ).detach()
        # Shape: (batch_size,)

        logprobs_t2: torch.Tensor = self.policy_model.compute_log_probs(
            prompts=turn2_prompts,
            responses=turn2_responses,
        ).detach()
        # Shape: (batch_size,)

        # ------------------------------------------------------------------
        # Step 8: Compute reference model log-probabilities
        # The reference model is frozen (freeze=True in ModelWrapper), but
        # we wrap in no_grad defensively to ensure no gradient accumulation.
        # ------------------------------------------------------------------
        with torch.no_grad():
            ref_logprobs_t1: torch.Tensor = self.ref_model.compute_log_probs(
                prompts=turn1_prompts,
                responses=turn1_responses,
            ).detach()
            # Shape: (batch_size,)

            ref_logprobs_t2: torch.Tensor = self.ref_model.compute_log_probs(
                prompts=turn2_prompts,
                responses=turn2_responses,
            ).detach()
            # Shape: (batch_size,)

        # ------------------------------------------------------------------
        # Step 9: Assemble Trajectory objects
        # ------------------------------------------------------------------
        trajectories: List[Trajectory] = []
        for i in range(batch_size):
            trajectory: Trajectory = Trajectory(
                problem=problems[i],
                ground_truth=ground_truths[i],
                test_cases=test_cases_batch[i],
                turn1_prompt=turn1_prompts[i],
                turn1_response=turn1_responses[i],
                turn2_prompt=turn2_prompts[i],
                turn2_response=turn2_responses[i],
                reward_t1=float(rewards_t1[i]),
                reward_t2=float(rewards_t2[i]),
                logprob_t1=logprobs_t1[i].detach(),
                logprob_t2=logprobs_t2[i].detach(),
                ref_logprob_t1=ref_logprobs_t1[i].detach(),
                ref_logprob_t2=ref_logprobs_t2[i].detach(),
                shaped_reward_t2=0.0,  # Populated by compute_shaped_rewards()
            )
            trajectories.append(trajectory)

        # Log summary statistics for monitoring
        mean_r1: float = sum(rewards_t1) / batch_size if batch_size > 0 else 0.0
        mean_r2: float = sum(rewards_t2) / batch_size if batch_size > 0 else 0.0
        delta: float = mean_r2 - mean_r1
        i2c: float = sum(
            1.0 for t in trajectories if t.reward_t1 == 0.0 and t.reward_t2 == 1.0
        ) / batch_size
        c2i: float = sum(
            1.0 for t in trajectories if t.reward_t1 == 1.0 and t.reward_t2 == 0.0
        ) / batch_size

        logger.debug(
            "sample_trajectories: batch_size=%d, "
            "mean_r1=%.3f, mean_r2=%.3f, delta=%.3f, "
            "i2c=%.3f, c2i=%.3f.",
            batch_size,
            mean_r1,
            mean_r2,
            delta,
            i2c,
            c2i,
        )

        return trajectories

    def compute_shaped_rewards(
        self,
        trajectories: List[Trajectory],
        alpha: float,
    ) -> List[Trajectory]:
        """Apply the Stage II reward shaping bonus to a list of trajectories.

        Implements the reward shaping formula from Section 5.2, Equation 5:
            b̂(y₂ | y₁, y*) = α · (r̂(y₂, y*) - r̂(y₁, y*))
            shaped_r2 = r̂(y₂, y*) + b̂(y₂ | y₁, y*) = r2 + alpha * (r2 - r1)

        Behavioral analysis of the bonus (with alpha=10 from Table 5):
            i→c (r1=0, r2=1): shaped_r2 = 1 + 10*(1-0) = 11  (strong positive)
            c→c (r1=1, r2=1): shaped_r2 = 1 + 10*(1-1) = 1   (normal reward)
            i→i (r1=0, r2=0): shaped_r2 = 0 + 10*(0-0) = 0   (no reward)
            c→i (r1=1, r2=0): shaped_r2 = 0 + 10*(0-1) = -10 (heavy penalty)

        This asymmetry prevents behavior collapse: the model is heavily
        penalized for breaking correct answers and strongly rewarded for
        fixing incorrect ones. From Section 5.2: "assigns a heavy negative
        penalty to transitions that change a correct response to incorrect."

        Modifies trajectories in-place and returns the same list for
        convenience (allows chaining: trajectories = buffer.compute_shaped_rewards(...)).

        Args:
            trajectories: List of Trajectory objects from sample_trajectories().
                Each trajectory's shaped_reward_t2 field is updated in-place.
            alpha: Reward shaping multiplier. From Table 5: α = 10 for both
                MATH and code tasks. Paper states: "α is a positive constant
                multiplier, ideally larger than 1.0."

        Returns:
            The same list of trajectories with shaped_reward_t2 populated.
            Returned for convenience (same object, modified in-place).

        Raises:
            ValueError: If alpha <= 0.0 (paper requires positive alpha).
        """
        if alpha <= 0.0:
            raise ValueError(
                f"alpha must be positive (got {alpha}). "
                "Paper Section 5.2: 'α is a positive constant multiplier, "
                "ideally larger than 1.0'."
            )

        num_i2c: int = 0
        num_c2i: int = 0
        num_cc: int = 0
        num_ii: int = 0

        for trajectory in trajectories:
            r1: float = trajectory.reward_t1
            r2: float = trajectory.reward_t2

            # shaped_r2 = r2 + alpha * (r2 - r1)
            trajectory.shaped_reward_t2 = r2 + alpha * (r2 - r1)

            # Track transition statistics for logging
            if r1 == 0.0 and r2 == 1.0:
                num_i2c += 1
            elif r1 == 1.0 and r2 == 0.0:
                num_c2i += 1
            elif r1 == 1.0 and r2 == 1.0:
                num_cc += 1
            else:
                num_ii += 1

        n: int = len(trajectories)
        if n > 0:
            mean_shaped: float = sum(t.shaped_reward_t2 for t in trajectories) / n
            logger.debug(
                "compute_shaped_rewards: alpha=%.1f, n=%d, "
                "mean_shaped_r2=%.3f, "
                "i2c=%d (%.1f%%), c2i=%d (%.1f%%), "
                "cc=%d (%.1f%%), ii=%d (%.1f%%).",
                alpha,
                n,
                mean_shaped,
                num_i2c,
                100.0 * num_i2c / n,
                num_c2i,
                100.0 * num_c2i / n,
                num_cc,
                100.0 * num_cc / n,
                num_ii,
                100.0 * num_ii / n,
            )

        return trajectories

    def compute_kl_divergence(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample KL divergence approximation.

        Implements the sample-based KL estimate used in the REINFORCE
        objective (Ahmadian et al., 2024 — "Back to Basics"):

            D_KL(π_θ || π_ref) ≈ Σ_t [log π_θ(a_t|s_t) - log π_ref(a_t|s_t)]

        This is the standard sample-based KL estimate, not the exact KL.
        Both logprobs and ref_logprobs are already summed over response tokens
        (computed by ModelWrapper.compute_log_probs()), so this reduces to
        simple element-wise subtraction.

        Using sum (not mean) over tokens is consistent with the REINFORCE
        policy gradient where the log probability is also summed over tokens.
        Using mean would create an inconsistency between the reward signal
        (per-sequence) and the KL penalty (per-token).

        This method is a utility called by trainers during loss computation:
            - SCoReStage1Trainer._compute_kl_penalty_t1(): uses trajectory
              logprob_t1 and ref_logprob_t1 (strong β₂ penalty, Equation 3)
            - SCoReStage1Trainer._compute_kl_penalty_t2(): uses trajectory
              logprob_t2 and ref_logprob_t2 (weak β₁ penalty)
            - SCoReStage2Trainer: uses both turns with β₁ (Equation 4)

        Args:
            logprobs: Per-sample summed log-probabilities under the policy
                model π_θ. Shape: (batch_size,). These are the values from
                trajectory.logprob_t1 or trajectory.logprob_t2 (or freshly
                recomputed with gradients for the loss).
            ref_logprobs: Per-sample summed log-probabilities under the
                reference model π_ref. Shape: (batch_size,). These are the
                stored trajectory.ref_logprob_t1 or trajectory.ref_logprob_t2
                values (always detached, reference model never changes).

        Returns:
            Per-sample KL divergence estimates of shape (batch_size,).
            Values are typically positive (policy has drifted from reference)
            but can be negative for individual samples due to the sample-based
            approximation. The mean over the batch is used in the loss.

        Raises:
            ValueError: If logprobs and ref_logprobs have different shapes.
        """
        if logprobs.shape != ref_logprobs.shape:
            raise ValueError(
                f"compute_kl_divergence: logprobs.shape={logprobs.shape} != "
                f"ref_logprobs.shape={ref_logprobs.shape}. "
                "Both tensors must have the same shape (batch_size,)."
            )

        # KL(π_θ || π_ref) ≈ log π_θ - log π_ref (per sample, summed over tokens)
        kl_per_sample: torch.Tensor = logprobs - ref_logprobs

        logger.debug(
            "compute_kl_divergence: batch_size=%d, "
            "mean_kl=%.4f, min_kl=%.4f, max_kl=%.4f.",
            logprobs.shape[0],
            kl_per_sample.mean().item(),
            kl_per_sample.min().item(),
            kl_per_sample.max().item(),
        )

        return kl_per_sample

    def _sample_single_trajectory(
        self, problem: Dict[str, Any]
    ) -> Trajectory:
        """Sample a single two-turn trajectory for one problem.

        Convenience wrapper around sample_trajectories() for single-problem
        use cases: debugging, qualitative analysis, and multi-attempt
        evaluation (Evaluator.evaluate_multi_attempt).

        Args:
            problem: A single problem dict with the same schema as items
                in the batch passed to sample_trajectories().

        Returns:
            A single Trajectory object for the given problem.
        """
        trajectories: List[Trajectory] = self.sample_trajectories([problem])
        return trajectories[0]

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _extract_problem_fields(
        self, item: Dict[str, Any]
    ) -> tuple:
        """Extract problem text, ground truth, and test cases from a dataset item.

        Normalizes the different schemas used by MATH, MBPP, and HumanEval
        datasets into a uniform (problem_text, ground_truth, test_cases) tuple.

        MATH schema:
            'problem' → problem_text
            'answer'  → ground_truth
            []        → test_cases (empty for MATH)

        MBPP schema:
            'text'      → problem_text
            'code'      → ground_truth (canonical solution, unused for reward)
            'test_list' → test_cases (assertion strings)
            'test_setup_code' is prepended to test_cases if non-empty

        HumanEval schema:
            'prompt'             → problem_text
            'canonical_solution' → ground_truth (unused for reward)
            'test'               → test_cases (single test function string)
            'entry_point'        → appended to test_cases as check() call

        Args:
            item: A dataset item dict from DatasetLoader.

        Returns:
            Tuple of (problem_text: str, ground_truth: str, test_cases: List[str]).
        """
        if self.config.task == "math":
            problem_text: str = str(item.get("problem", ""))
            ground_truth: str = str(item.get("answer", ""))
            test_cases: List[str] = []
            return problem_text, ground_truth, test_cases

        else:
            # Code task — distinguish MBPP from HumanEval by key presence
            if "text" in item:
                # MBPP schema
                problem_text = str(item.get("text", ""))
                ground_truth = str(item.get("code", ""))

                # Build test cases from test_list
                raw_test_list: Any = item.get("test_list", [])
                if isinstance(raw_test_list, str):
                    test_cases = [raw_test_list]
                else:
                    test_cases = [str(t) for t in raw_test_list]

                # Prepend test_setup_code if present and non-empty
                setup_code: str = str(item.get("test_setup_code", "")).strip()
                if setup_code:
                    test_cases = [setup_code] + test_cases

            elif "prompt" in item:
                # HumanEval schema
                problem_text = str(item.get("prompt", ""))
                ground_truth = str(item.get("canonical_solution", ""))

                # HumanEval test is a single string containing the test function
                # The test function is called as check(entry_point)
                test_function: str = str(item.get("