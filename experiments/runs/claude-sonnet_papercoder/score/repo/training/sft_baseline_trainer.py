```python
## training/sft_baseline_trainer.py
"""SFT baseline trainers for SCoRe: Self-Correction via Reinforcement Learning.

This module implements the STaR and Pair-SFT supervised fine-tuning baselines
described in Section 4 of the paper. These baselines demonstrate the
insufficiency of SFT-based approaches for self-correction, serving as
comparison points against SCoRe in Table 1 and Table 2.

Paper context (Section 4):
    "A natural approach for training self-correction is to utilize some form
    of supervised fine-tuning on data collected from a base model. [...] we
    assess the empirical efficacy of two such approaches for self-correction:
    STaR (Zelikman et al., 2022), and a version of Welleck et al. (2023) that
    trains only one model."

Key failure modes demonstrated by these baselines:
    1. Distribution shift: SFT on offline data cannot generalize to the
       model's own self-generated errors at test time (Figure 5).
    2. Behavior collapse: Models learn to make no edits (Figure 4a), or
       learn to produce a good first attempt and copy it.

Config values used (from config.yaml):
    task: "math" or "code"
    sft_baselines.star.num_iterations: 3
    sft_baselines.star.include_correct_pairs: false
    sft_baselines.pair_sft.include_correct_pairs: false
    training.math.learning_rate: 5e-6
    training.code.learning_rate: 1e-5
    training.math.batch_size: 512
    training.code.batch_size: 128
    model.train_temperature: 1.0
    model.max_new_tokens: 1024

Typical usage:
    from training.sft_baseline_trainer import SFTBaselineTrainer

    trainer = SFTBaselineTrainer(policy_model, config, method="star")
    dataset = trainer.build_star_dataset(base_model, train_data, reward_fn)
    trained_model = trainer.train_sft(dataset)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from config import Config
from data.prompt_templates import PromptTemplates
from models.model_wrapper import ModelWrapper
from rewards import RewardFunction
from training.rollout_buffer import Trajectory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Maximum number of re-sampling attempts in Pair-SFT to find a correct
# response for an incorrect first attempt. After this many attempts, the
# problem is skipped (no valid repair pair can be constructed).
# ---------------------------------------------------------------------------
_MAX_RESAMPLE_ATTEMPTS: int = 10

# Maximum sequence length for SFT examples (in tokens). Sequences longer
# than this are truncated to prevent OOM. Logged as a warning when triggered.
_MAX_SFT_SEQ_LEN: int = 2048

# Ignore index for label masking (PyTorch cross-entropy convention)
_IGNORE_INDEX: int = -100


# ---------------------------------------------------------------------------
# Internal dataset class for HuggingFace Trainer
# ---------------------------------------------------------------------------


class _SFTDataset(TorchDataset):
    """Minimal torch Dataset wrapping a list of SFT example dicts.

    Each example dict must contain:
        'input_ids' (List[int]): Token ids for the full two-turn context.
        'labels' (List[int]): Token ids with prompt positions masked as -100.
        'attention_mask' (List[int]): Binary mask (1 for real tokens).

    The 'metadata' key is preserved but not returned by __getitem__ to
    avoid issues with HuggingFace Trainer's data collation.

    Attributes:
        examples: List of SFT example dicts.
    """

    def __init__(self, examples: List[Dict[str, Any]]) -> None:
        """Initialize the dataset.

        Args:
            examples: List of SFT example dicts from _format_as_sft_example().
        """
        self.examples: List[Dict[str, Any]] = examples

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single example as a dict of tensors.

        Args:
            idx: Index into the examples list.

        Returns:
            Dict with keys 'input_ids', 'labels', 'attention_mask', each
            as a 1D torch.long tensor. The 'metadata' key is excluded to
            prevent DataCollator issues.
        """
        example: Dict[str, Any] = self.examples[idx]
        return {
            "input_ids": torch.tensor(
                example["input_ids"], dtype=torch.long
            ),
            "labels": torch.tensor(
                example["labels"], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                example["attention_mask"], dtype=torch.long
            ),
        }


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------


class SFTBaselineTrainer:
    """SFT baseline trainer implementing STaR and Pair-SFT for self-correction.

    Implements the two SFT-based baselines from Section 4 of the paper:

    STaR (Zelikman et al., 2022):
        - Samples two-turn traces from the current model.
        - Filters to keep only i→c transitions (incorrect t1, correct t2).
        - Runs SFT on the filtered dataset.
        - Iterates for 3 rounds (config.sft_baselines.star.num_iterations).

    Pair-SFT (Welleck et al., 2023 variant):
        - Samples first attempts from the base model.
        - For incorrect first attempts, pairs with a correct response
          (re-sampled from the base model).
        - Runs SFT on the paired dataset (single iteration).

    Both methods demonstrate failure modes:
        - STaR: behavior collapse (low Δ^{i→c}, model learns not to edit)
        - Pair-SFT: distribution shift (good on fixed first attempts,
          poor on self-generated first attempts, Figure 5)

    Attributes:
        policy_model: The trainable policy model. Updated in-place by
            train_sft(). For STaR, this is updated after each iteration.
        config: Global configuration instance.
        method: Either 'star' or 'pair_sft'. Determines dataset construction.
        prompt_templates: Prompt builder for both MATH and code tasks.
        learning_rate: Task-specific learning rate from config.
        batch_size: Task-specific batch size from config.
        max_grad_norm: Maximum gradient norm for clipping.
        adam_beta1: Adam optimizer beta1.
        adam_beta2: Adam optimizer beta2.
        adam_epsilon: Adam optimizer epsilon.
        weight_decay: L2 weight decay.
        train_temperature: Sampling temperature for dataset construction.
        max_new_tokens: Maximum new tokens per generation call.
    """

    def __init__(
        self,
        policy_model: ModelWrapper,
        config: Config,
        method: str = "star",
    ) -> None:
        """Initialize SFTBaselineTrainer.

        Args:
            policy_model: The trainable policy model (π_θ). Must have been
                initialized with freeze=False in ModelWrapper. Its weights
                are updated in-place by train_sft(). For STaR, this model
                is used for sampling in each iteration.
            config: Global Config instance. Reads task-specific training
                hyperparameters and SFT baseline configuration.
            method: Baseline method identifier. Must be 'star' or 'pair_sft'.
                - 'star': STaR (Zelikman et al., 2022) — filters i→c traces.
                - 'pair_sft': Pair-SFT (Welleck et al., 2023 variant) —
                  synthetic pairing of incorrect t1 with correct t2.

        Raises:
            ValueError: If method is not 'star' or 'pair_sft'.
            ValueError: If config.task is not 'math' or 'code'.
        """
        if method not in ("star", "pair_sft"):
            raise ValueError(
                f"SFTBaselineTrainer: Invalid method '{method}'. "
                "Must be 'star' or 'pair_sft'. "
                "These correspond to the two SFT baselines in Section 4 "
                "of the paper (Table 1)."
            )

        if config.task not in ("math", "code"):
            raise ValueError(
                f"SFTBaselineTrainer: Invalid task '{config.task}'. "
                "Must be 'math' or 'code'."
            )

        self.policy_model: ModelWrapper = policy_model
        self.config: Config = config
        self.method: str = method
        self.prompt_templates: PromptTemplates = PromptTemplates()

        # ------------------------------------------------------------------
        # Resolve task-specific hyperparameters from Config.
        # Config.from_dict() flattens the nested YAML into flat fields.
        # ------------------------------------------------------------------
        self.learning_rate: float = config.learning_rate
        self.batch_size: int = config.batch_size
        self.max_grad_norm: float = config.max_grad_norm
        self.adam_beta1: float = config.adam_beta1
        self.adam_beta2: float = config.adam_beta2
        self.adam_epsilon: float = config.adam_epsilon
        self.weight_decay: float = config.weight_decay

        # Sampling parameters for dataset construction
        # Table 5: "Sampling temperature: 1.0"
        self.train_temperature: float = config.sampling_temperature
        self.max_new_tokens: int = config.max_new_tokens

        logging.basicConfig(
            level=getattr(logging, config.log_level, logging.INFO)
        )
        logger.info(
            "SFTBaselineTrainer initialized: method='%s', task='%s', "
            "lr=%.2e, batch_size=%d, train_temperature=%.1f.",
            method,
            config.task,
            self.learning_rate,
            self.batch_size,
            self.train_temperature,
        )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def build_star_dataset(
        self,
        base_model: ModelWrapper,
        train_data: List[Dict[str, Any]],
        reward_fn: RewardFunction,
        num_iterations: int = 3,
    ) -> List[Dict[str, Any]]:
        """Build the STaR training dataset D_STaR via iterative self-training.

        Implements the STaR approach from Section 4 of the paper:
            "The STaR approach filters these trajectories to retain only
            those that successfully revise incorrect responses and runs SFT
            on the resulting dataset."
            "We run 3 iterations for STaR following the protocol in
            Singh et al. (2024)."

        Each iteration:
            1. Sample two-turn traces from the current model.
            2. Filter to keep only i→c transitions (reward_t1=0, reward_t2=1).
            3. Optionally include c→c pairs (if include_correct_pairs=True).
            4. Accumulate filtered examples.
            5. Fine-tune the current model on all accumulated examples.
            6. Use the fine-tuned model for the next iteration.

        Args:
            base_model: The initial base model for iteration 0. Subsequent
                iterations use the model fine-tuned in the previous iteration.
                This is the same object as self.policy_model — passed
                explicitly to make the iteration logic clear.
            train_data: List of problem dicts from DatasetLoader.load_train_data().
                Each dict has task-specific fields (see RolloutBuffer for schema).
            reward_fn: Unified reward function for computing binary rewards.
                Dispatches to MathReward or CodeReward based on config.task.
            num_iterations: Number of STaR iterations. From config.yaml:
                sft_baselines.star.num_iterations = 3.

        Returns:
            List of SFT example dicts accumulated across all iterations.
            Each dict has keys: 'input_ids', 'labels', 'attention_mask',
            'metadata'. The list may be empty if no i→c transitions are
            found (early training with a weak base model).
        """
        include_correct_pairs: bool = self.config.star_include_correct_pairs
        all_examples: List[Dict[str, Any]] = []
        current_model: ModelWrapper = base_model

        logger.info(
            "build_star_dataset: Starting %d STaR iterations on %d problems. "
            "include_correct_pairs=%s.",
            num_iterations,
            len(train_data),
            include_correct_pairs,
        )

        for iteration in range(num_iterations):
            logger.info(
                "build_star_dataset: Iteration %d/%d.",
                iteration + 1,
                num_iterations,
            )

            # ------------------------------------------------------------------
            # Step 1: Sample two-turn traces from the current model.
            # Process in mini-batches to avoid OOM on large train_data.
            # ------------------------------------------------------------------
            trajectories: List[Trajectory] = self._sample_two_turn_traces(
                model=current_model,
                train_data=train_data,
                reward_fn=reward_fn,
                iteration=iteration,
            )

            # ------------------------------------------------------------------
            # Step 2: Filter to keep i→c transitions (and optionally c→c).
            # ------------------------------------------------------------------
            iteration_examples: List[Dict[str, Any]] = (
                self._filter_successful_corrections(trajectories)
            )

            if include_correct_pairs:
                # D_STaR+: also include correct-to-correct pairs
                cc_examples: List[Dict[str, Any]] = (
                    self._extract_correct_to_correct(trajectories)
                )
                iteration_examples.extend(cc_examples)
                logger.info(
                    "build_star_dataset: Iteration %d: %d i→c examples + "
                    "%d c→c examples = %d total.",
                    iteration + 1,
                    len(iteration_examples) - len(cc_examples),
                    len(cc_examples),
                    len(iteration_examples),
                )
            else:
                logger.info(
                    "build_star_dataset: Iteration %d: %d i→c examples "
                    "(out of %d trajectories).",
                    iteration + 1,
                    len(iteration_examples),
                    len(trajectories),
                )

            if not iteration_examples:
                logger.warning(
                    "build_star_dataset: Iteration %d produced 0 training "
                    "examples. No i→c transitions found. Skipping fine-tuning "
                    "for this iteration. This is expected early in training "
                    "when the base model rarely self-corrects.",
                    iteration + 1,
                )
                # Still continue to next iteration — the model may improve
                # with more sampling attempts in subsequent iterations
                continue

            # ------------------------------------------------------------------
            # Step 3: Accumulate examples across iterations.
            # ------------------------------------------------------------------
            all_examples.extend(iteration_examples)
            logger.info(
                "build_star_dataset: Accumulated %d total examples after "
                "iteration %d.",
                len(all_examples),
                iteration + 1,
            )

            # ------------------------------------------------------------------
            # Step 4: Fine-tune the current model on all accumulated examples.
            # The fine-tuned model is used for sampling in the next iteration.
            # ------------------------------------------------------------------
            if iteration < num_iterations - 1:
                # Only fine-tune if there are more iterations to run
                # (the final iteration's fine-tuning is done by the caller
                # via train_sft() on the returned dataset)
                logger.info(
                    "build_star_dataset: Fine-tuning model for iteration %d "
                    "(to use in iteration %d).",
                    iteration + 1,
                    iteration + 2,
                )
                current_model = self.train_sft(all_examples)

        logger.info(
            "build_star_dataset: Completed %d iterations. "
            "Total examples: %d.",
            num_iterations,
            len(all_examples),
        )
        return all_examples

    def build_pair_sft_dataset(
        self,
        base_model: ModelWrapper,
        train_data: List[Dict[str, Any]],
        reward_fn: RewardFunction,
        include_correct_pairs: bool = False,
    ) -> List[Dict[str, Any]]:
        """Build the Pair-SFT training dataset D_SFT via synthetic pairing.

        Implements the Pair-SFT approach from Section 4 of the paper:
            "Another approach is to use base model data from above to
            construct 'synthetic' repair traces by pairing incorrect
            responses with correct ones (Welleck et al., 2023). We study
            a variant of this method that we call Pair-SFT, which does not
            train a separate corrector model and does not augment this
            initial dataset with multi-turn traces."
            "We run... only one iteration for Pair-SFT, following the
            protocol in Welleck et al. (2023)."

        Single iteration:
            1. Sample first attempts from the base model.
            2. For incorrect first attempts, find a correct response by
               re-sampling (up to _MAX_RESAMPLE_ATTEMPTS times).
            3. Pair each incorrect t1 with a correct t2 (synthetic trace).
            4. Optionally include correct-to-correct pairs (D_SFT+).

        Key distinction from STaR: The turn2 response is independently
        sampled — it does not need to be a genuine correction of the
        specific errors in turn1. This is the "synthetic pairing" aspect.

        Args:
            base_model: The base model for sampling. Pair-SFT uses only
                the base model (single iteration, no iterative update).
            train_data: List of problem dicts from DatasetLoader.
            reward_fn: Unified reward function for binary reward computation.
            include_correct_pairs: Whether to include c→c pairs (D_SFT+).
                From config.yaml: sft_baselines.pair_sft.include_correct_pairs.
                Default False (D_SFT). Set True for D_SFT+ ablation.

        Returns:
            List of SFT example dicts. Each dict has keys:
            'input_ids', 'labels', 'attention_mask', 'metadata'.
        """
        logger.info(
            "build_pair_sft_dataset: Building Pair-SFT dataset from %d "
            "problems. include_correct_pairs=%s.",
            len(train_data),
            include_correct_pairs,
        )

        # ------------------------------------------------------------------
        # Step 1: Sample first attempts from the base model.
        # ------------------------------------------------------------------
        trajectories: List[Trajectory] = self._sample_two_turn_traces(
            model=base_model,
            train_data=train_data,
            reward_fn=reward_fn,
            iteration=0,
        )

        # ------------------------------------------------------------------
        # Step 2: Pair incorrect first attempts with correct responses.
        # For trajectories where reward_t2 is already 1.0, use the sampled
        # turn2 response directly. For those where reward_t2 is 0.0,
        # re-sample up to _MAX_RESAMPLE_ATTEMPTS times.
        # ------------------------------------------------------------------
        examples: List[Dict[str, Any]] = self._pair_incorrect_with_correct(
            trajectories=trajectories,
            base_model=base_model,
            reward_fn=reward_fn,
        )

        # ------------------------------------------------------------------
        # Step 3: Optionally include correct-to-correct pairs (D_SFT+).
        # ------------------------------------------------------------------
        if include_correct_pairs:
            cc_examples: List[Dict[str, Any]] = (
                self._extract_correct_to_correct(trajectories)
            )
            examples.extend(cc_examples)
            logger.info(
                "build_pair_sft_dataset: %d repair pairs + %d c→c pairs = "
                "%d total examples.",
                len(examples) - len(cc_examples),
                len(cc_examples),
                len(examples),
            )
        else:
            logger.info(
                "build_pair_sft_dataset: %d repair pairs (out of %d "
                "trajectories with incorrect t1).",
                len(examples),
                sum(1 for t in trajectories if t.reward_t1 == 0.0),
            )

        return examples

    def train_sft(
        self, dataset: List[Dict[str, Any]]
    ) -> ModelWrapper:
        """Fine-tune the policy model on an SFT dataset using HuggingFace Trainer.

        Runs standard supervised fine-tuning with cross-entropy loss on
        response tokens only (prompt tokens masked with -100 in labels).
        Uses the HuggingFace Trainer for compatibility with DeepSpeed and
        gradient accumulation.

        Args:
            dataset: List of SFT example dicts from build_star_dataset() or
                build_pair_sft_dataset(). Each dict must have keys:
                'input_ids' (List[int]), 'labels' (List[int]),
                'attention_mask' (List[int]).

        Returns:
            The updated self.policy_model with fine-tuned weights. The model
            is updated in-place by the HuggingFace Trainer, so the returned
            object is the same as self.policy_model.

        Raises:
            ValueError: If dataset is empty (cannot train on empty data).
        """
        if not dataset:
            raise ValueError(
                "train_sft: dataset is empty. Cannot run SFT on an empty "
                "dataset. Check that build_star_dataset() or "
                "build_pair_sft_dataset() produced valid examples."
            )

        logger.info(
            "train_sft: Starting SFT on %d examples. method='%s', "
            "lr=%.2e, batch_size=%d.",
            len(dataset),
            self.method,
            self.learning_rate,
            self.batch_size,
        )

        # ------------------------------------------------------------------
        # Step 1: Wrap the list of dicts in a torch Dataset.
        # ------------------------------------------------------------------
        sft_dataset: _SFTDataset = _SFTDataset(dataset)

        # ------------------------------------------------------------------
        # Step 2: Configure TrainingArguments from config values.
        # ------------------------------------------------------------------
        output_dir: str = os.path.join(
            self.config.output_dir, f"sft_{self.method}"
        )
        os.makedirs(output_dir, exist_ok=True)

        # Determine whether to use bf16 based on config.torch_dtype
        use_bf16: bool = self.config.torch_dtype == "bfloat16"
        use_fp16: bool = self.config.torch_dtype == "float16"

        # Check if DeepSpeed config exists before passing it
        deepspeed_config: Optional[str] = None
        if (
            self.config.deepspeed_config
            and os.path.isfile(self.config.deepspeed_config)
        ):
            deepspeed_config = self.config.deepspeed_config
            logger.info(
                "train_sft: Using DeepSpeed config: '%s'.", deepspeed_config
            )
        else:
            logger.debug(
                "train_sft: DeepSpeed config '%s' not found. "
                "Running without DeepSpeed.",
                self.config.deepspeed_config,
            )

        training_args: TrainingArguments = TrainingArguments(
            output_dir=output_dir,
            # Single pass over the dataset per call to train_sft().
            # STaR handles iteration externally in build_star_dataset().
            num_train_epochs=1,
            per_device_train_batch_size=max(1, self.batch_size),
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            weight_decay=self.weight_decay,
            # Logging
            logging_steps=10,
            logging_dir=os.path.join(output_dir, "logs"),
            # Checkpointing: disabled here — CheckpointUtils handles this
            save_strategy="no",
            # Mixed precision
            bf16=use_bf16,
            fp16=use_fp16,
            # Gradient accumulation
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            # Disable automatic column removal — we handle this in _SFTDataset
            remove_unused_columns=False,
            # Disable evaluation during SFT (evaluation is done by Evaluator)
            do_eval=False,
            # Report to none — logging handled by LoggingUtils
            report_to="none",
            # DeepSpeed (optional)
            deepspeed=deepspeed_config,
            # Disable dataloader pin_memory if running on CPU
            dataloader_pin_memory=torch.cuda.is_available(),
        )

        # ------------------------------------------------------------------
        # Step 3: Configure DataCollatorForSeq2Seq.
        # Critical: label_pad_token_id=-100 ensures padded label positions
        # do not contribute to the cross-entropy loss.
        # ------------------------------------------------------------------
        data_collator: DataCollatorForSeq2Seq = DataCollatorForSeq2Seq(
            tokenizer=self.policy_model.tokenizer,
            model=self.policy_model.model,
            padding=True,
            label_pad_token_id=_IGNORE_INDEX,
            pad_to_multiple_of=8 if (use_bf16 or use_fp16) else None,
        )

        # ------------------------------------------------------------------
        # Step 4: Instantiate and run the HuggingFace Trainer.
        # ------------------------------------------------------------------
        trainer: Trainer = Trainer(
            model=self.policy_model.model,
            args=training_args,
            train_dataset=sft_dataset,
            data_collator=data_collator,
        )

        logger.info("train_sft: Starting Trainer.train()...")
        trainer.train()
        logger.info("train_sft: Training complete.")

        # The Trainer updates self.policy_model.model in-place.
        # Set back to training mode (Trainer may have set eval mode).
        self.policy_model.model.train()

        return self.policy_model

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _sample_two_turn_traces(
        self,
        model: ModelWrapper,
        train_data: List[Dict[str, Any]],
        reward_fn: RewardFunction,
        iteration: int = 0,
    ) -> List[Trajectory]:
        """Sample two-turn self-correction traces from a model.

        Processes train_data in mini-batches for GPU efficiency. Each
        problem gets one two-turn trace: turn1 response sampled from the
        model, then turn2 response sampled given the turn1 context.

        This is a simplified version of RolloutBuffer.sample_trajectories()
        that does not compute log-probabilities (not needed for SFT dataset
        construction — only rewards matter for filtering).

        Args:
            model: The model to sample from. For STaR iteration 0, this is
                the base model. For subsequent iterations, this is the
                fine-tuned model from the previous iteration.
            train_data: List of problem dicts.
            reward_fn: Reward function for computing binary rewards.
            iteration: Current STaR iteration index (for metadata logging).

        Returns:
            List of Trajectory objects with populated turn1/turn2 responses
            and rewards. Log-probability fields are None (not computed here).
        """
        trajectories: List[Trajectory] = []

        # Process in mini-batches to avoid OOM
        # Use a reasonable mini-batch size for generation
        gen_batch_size: int = min(8, len(train_data))

        logger.info(
            "_sample_two_turn_traces: Sampling %d traces (iteration=%d, "
            "gen_batch_size=%d).",
            len(train_data),
            iteration,
            gen_batch_size,
        )

        for batch_start in tqdm(
            range(0, len(train_data), gen_batch_size),
            desc=f"Sampling traces (iter {iteration + 1})",
            unit="batch",
        ):
            batch: List[Dict[str, Any]] = train_data[
                batch_start: batch_start + gen_batch_size
            ]

            # ------------------------------------------------------------------
            # Extract problem fields for this mini-batch
            # ------------------------------------------------------------------
            problems: List[str] = []
            ground_truths: List[str] = []
            test_cases_batch: List[List[str]] = []
            turn1_prompts: List[str] = []

            for item in batch:
                problem_text, ground_truth, test_cases = (
                    self._extract_problem_fields(item)
                )
                problems.append(problem_text)
                ground_truths.append(ground_truth)
                test_cases_batch.append(test_cases)

                t1_prompt: str = self.prompt_templates.build_turn1_prompt(
                    problem_text, self.config.task
                )
                turn1_prompts.append(t1_prompt)

            # ------------------------------------------------------------------
            # Generate turn 1 responses
            # ------------------------------------------------------------------
            turn1_responses: List[str] = model.generate(
                prompts=turn1_prompts,
                temperature=self.train_temperature,
                max_new_tokens=self.max_new_tokens,
            )

            # ------------------------------------------------------------------
            # Compute turn 1 rewards
            # ------------------------------------------------------------------
            rewards_t1: List[float] = reward_fn.batch_compute(
                predictions=turn1_responses,
                ground_truths=ground_truths,
                test_cases=test_cases_batch,
            )

            # ------------------------------------------------------------------
            # Build turn 2 prompts and generate turn 2 responses
            # ------------------------------------------------------------------
            turn2_prompts: List[str] = [
                self.prompt_templates.build_turn2_prompt(
                    problem=problems[i],
                    turn1_response=turn1_responses[i],
                    task=self.config.task,
                )
                for i in range(len(batch))
            ]

            turn2_responses: List[str] = model.generate(
                prompts=turn2_prompts,
                temperature=self.train_temperature,
                max_new_tokens=self.max_new_tokens,
            )

            # ------------------------------------------------------------------
            # Compute turn 2 rewards
            # ------------------------------------------------------------------
            rewards_t2: List[float] = reward_fn.batch_compute(
                predictions=turn2_responses,
                ground_truths=ground_truths,
                test_cases=test_cases_batch,
            )

            # ------------------------------------------------------------------
            # Assemble Trajectory objects (no log-probs needed for SFT)
            # ------------------------------------------------------------------
            for i in range(len(batch)):
                traj: Trajectory = Trajectory(
                    problem=problems[i],
                    ground_truth=ground_truths[i],
                    test_cases=test_cases_batch[i],
                    turn1_prompt=turn1_prompts[i],
                    turn1_response=turn1_responses[i],
                    turn2_prompt=turn2_prompts[i],
                    turn2_response=turn2_responses[i],
                    reward_t1=float(rewards_t1[i]),
                    reward_t2=float(rewards_t2[i]),
                    # Log-probs not needed for SFT dataset construction
                    logprob_t1=None,
                    log