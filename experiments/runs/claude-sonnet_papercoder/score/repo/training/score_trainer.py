```python
## training/score_trainer.py
"""SCoRe two-stage multi-turn RL trainer for Self-Correction via Reinforcement Learning.

This module implements the three core training classes that execute the SCoRe
pipeline described in the paper:

    SCoReStage1Trainer: Stage I — decoupled RL that trains the model to produce
        high-reward second-attempt responses while keeping first-attempt
        distribution close to the base model via a strong KL penalty β₂.
        Implements Equation 3 from the paper.

    SCoReStage2Trainer: Stage II — joint multi-turn RL with reward shaping that
        incentivizes genuine self-correction (i→c transitions) and penalizes
        regression (c→i transitions). Implements Equation 4 with the reward
        shaping bonus from Section 5.2.

    SCoReTrainer: Orchestrator that initializes all components, builds the base
        model buffer for offline augmentation, runs Stage I then Stage II, and
        returns the final trained model.

Paper equations implemented:
    Stage I (Equation 3):
        loss = -mean(r2 * logprob_t2 - β₂ * KL_t1 - β₁ * KL_t2)

    Stage II (Equation 4 + reward shaping from Section 5.2):
        shaped_r2 = r2 + α * (r2 - r1)
        loss = -mean(r1 * logprob_t1 + shaped_r2 * logprob_t2 - β₁ * (KL_t1 + KL_t2))

Hyperparameters (Table 5, Appendix B):
    MATH:  α=10, β₁=0.01, β₂=0.1,  lr=5e-6, batch_size=512, total_steps=3000
    Code:  α=10, β₁=0.01, β₂=0.25, lr=1e-5, batch_size=128, total_steps=1500

Typical usage:
    from training.score_trainer import SCoReTrainer

    trainer = SCoReTrainer(config)
    trained_model = trainer.train(train_data)
"""

import itertools
import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.utils as nn_utils
import torch.optim as optim
from tqdm import tqdm

from config import Config
from data.prompt_templates import PromptTemplates
from models.model_wrapper import ModelWrapper
from rewards import RewardFunction
from training.rollout_buffer import RolloutBuffer, Trajectory
from utils.checkpoint_utils import CheckpointUtils
from utils.logging_utils import LoggingUtils

logger = logging.getLogger(__name__)

# Ignore index for label masking (PyTorch convention)
_IGNORE_INDEX: int = -100

# Minimum std for reward normalization denominator (prevents division by zero)
_REWARD_NORM_EPS_FLOOR: float = 1e-8


class SCoReStage1Trainer:
    """Stage I trainer: decoupled RL with strong first-turn KL constraint.

    Trains the model to produce high-reward second-attempt responses while
    keeping the first-attempt distribution close to the base model. This
    decouples the two turns and prevents the behavior collapse that occurs
    in standard multi-turn RL (Figure 6 of the paper).

    Implements Equation 3:
        max_θ E[r̂(y₂, y*) - β₂ · D_KL(π_θ(·|x₁) || π_ref(·|x₁))]

    The β₂ KL penalty on the first turn (e.g., 0.1 for MATH, 0.25 for code)
    is significantly larger than the standard β₁ (0.01), enforcing a strict
    constraint that prevents the first-turn distribution from drifting.

    Attributes:
        policy_model: The trainable policy model (π_θ). Updated each step.
        ref_model: The frozen reference model (π_ref). Never updated.
        rollout_buffer: Two-turn trajectory sampling engine.
        config: Global configuration instance.
        logger: Experiment tracking logger (wandb + Python logging).
        optimizer: Adam optimizer for policy_model parameters.
        beta1: Standard KL penalty weight (β₁ = 0.01 from Table 5).
        beta2: Stage I first-turn KL penalty weight (β₂ = 0.1/0.25 from Table 5).
        alpha: Reward shaping multiplier (not used in Stage I, stored for reference).
        normalize_rewards: Whether to apply per-batch reward normalization.
        reward_norm_eps: Epsilon for reward normalization denominator.
        max_grad_norm: Maximum gradient norm for clipping.
        batch_size: Training batch size (512 MATH / 128 code from Table 5).
        checkpoint_utils: Checkpoint save/load utility.
    """

    def __init__(
        self,
        policy_model: ModelWrapper,
        ref_model: ModelWrapper,
        rollout_buffer: RolloutBuffer,
        config: Config,
        logger_util: LoggingUtils,
    ) -> None:
        """Initialize SCoReStage1Trainer.

        Args:
            policy_model: The trainable policy model (π_θ). Must have been
                initialized with freeze=False in ModelWrapper.
            ref_model: The frozen reference model (π_ref). Must have been
                initialized with freeze=True in ModelWrapper. Never updated.
            rollout_buffer: Two-turn trajectory sampling engine providing
                sample_trajectories() for on-policy rollout collection.
            config: Global Config instance. Reads task-specific hyperparameters
                from the flattened config fields (Config.from_dict() handles
                the nested YAML → flat dataclass mapping).
            logger_util: LoggingUtils instance for wandb and Python logging.

        Raises:
            ValueError: If config.task is not 'math' or 'code'.
        """
        if config.task not in ("math", "code"):
            raise ValueError(
                f"SCoReStage1Trainer: Invalid task '{config.task}'. "
                "Must be 'math' or 'code'."
            )

        self.policy_model: ModelWrapper = policy_model
        self.ref_model: ModelWrapper = ref_model
        self.rollout_buffer: RolloutBuffer = rollout_buffer
        self.config: Config = config
        self.logger_util: LoggingUtils = logger_util

        # ------------------------------------------------------------------
        # Resolve task-specific hyperparameters from Config.
        # Config.from_dict() flattens the nested YAML structure into flat
        # fields, so we read directly from config attributes.
        # ------------------------------------------------------------------

        # SCoRe hyperparameters (Table 5, Appendix B)
        # β₁: standard KL penalty weight (0.01 for both tasks)
        self.beta1: float = config.beta1
        # β₂: Stage I first-turn KL penalty weight (0.1 MATH / 0.25 code)
        self.beta2: float = config.beta2
        # α: reward shaping multiplier (stored for reference, not used in Stage I)
        self.alpha: float = config.alpha

        # Reward normalization settings
        self.normalize_rewards: bool = config.normalize_rewards
        self.reward_norm_eps: float = max(
            config.reward_norm_eps, _REWARD_NORM_EPS_FLOOR
        )

        # Training hyperparameters
        self.max_grad_norm: float = config.max_grad_norm
        self.batch_size: int = config.batch_size
        self.eval_every_n_steps: int = config.eval_every_n_steps
        self.save_every_n_steps: int = config.save_every_n_steps

        # ------------------------------------------------------------------
        # Verify reference model is frozen (belt-and-suspenders check)
        # ------------------------------------------------------------------
        for param in self.ref_model.model.parameters():
            if param.requires_grad:
                logger.warning(
                    "SCoReStage1Trainer: Reference model parameter has "
                    "requires_grad=True. Forcing to False. The reference "
                    "model (π_ref) must never be updated during training."
                )
                param.requires_grad = False

        # ------------------------------------------------------------------
        # Initialize Adam optimizer on policy model parameters only.
        # The reference model parameters are NOT included.
        # ------------------------------------------------------------------
        self.optimizer: optim.Adam = optim.Adam(
            self.policy_model.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )

        # ------------------------------------------------------------------
        # Checkpoint utility for saving Stage I checkpoints
        # ------------------------------------------------------------------
        self.checkpoint_utils: CheckpointUtils = CheckpointUtils()

        logger.info(
            "SCoReStage1Trainer initialized: task='%s', "
            "lr=%.2e, beta1=%.4f, beta2=%.4f, alpha=%.1f, "
            "batch_size=%d, max_grad_norm=%.1f, normalize_rewards=%s.",
            config.task,
            config.learning_rate,
            self.beta1,
            self.beta2,
            self.alpha,
            self.batch_size,
            self.max_grad_norm,
            self.normalize_rewards,
        )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def train(
        self,
        train_data: List[Dict[str, Any]],
        num_steps: int,
    ) -> ModelWrapper:
        """Execute Stage I training for the specified number of gradient steps.

        Cycles through train_data indefinitely (shuffling at each epoch
        boundary) until num_steps gradient updates have been performed.
        Logs metrics every eval_every_n_steps and saves checkpoints every
        save_every_n_steps.

        Args:
            train_data: List of problem dicts from DatasetLoader.load_train_data().
                Cycled indefinitely to fill num_steps batches.
            num_steps: Number of gradient update steps to perform. From
                config.stage1_steps (e.g., 1500 for MATH, 750 for code).

        Returns:
            self.policy_model with Stage I weights. This is the initialization
            for Stage II training. The model is set to eval mode after training
            completes (callers should set back to train() if needed).
        """
        if not train_data:
            raise ValueError(
                "SCoReStage1Trainer.train(): train_data is empty. "
                "Cannot train on an empty dataset."
            )

        logger.info(
            "SCoReStage1Trainer.train(): Starting Stage I for %d steps. "
            "train_data size=%d, batch_size=%d.",
            num_steps,
            len(train_data),
            self.batch_size,
        )

        self.policy_model.model.train()

        # ------------------------------------------------------------------
        # Create an infinite cycling iterator over shuffled train_data.
        # We shuffle a copy of the indices at each epoch boundary to ensure
        # different orderings across epochs without modifying train_data.
        # ------------------------------------------------------------------
        data_iterator = self._make_cycling_iterator(train_data)

        best_reward_t2: float = -float("inf")

        for step in tqdm(
            range(1, num_steps + 1),
            desc="Stage I Training",
            unit="step",
        ):
            # ------------------------------------------------------------------
            # Sample a batch of problems from the cycling iterator
            # ------------------------------------------------------------------
            batch: List[Dict[str, Any]] = [
                next(data_iterator) for _ in range(self.batch_size)
            ]

            # ------------------------------------------------------------------
            # Execute one training step
            # ------------------------------------------------------------------
            metrics: Dict[str, float] = self.train_step(batch)
            metrics["stage"] = 1.0
            metrics["step"] = float(step)

            # ------------------------------------------------------------------
            # Log metrics at specified frequency
            # ------------------------------------------------------------------
            if step % self.eval_every_n_steps == 0:
                self.logger_util.log_metrics(metrics, step=step)
                logger.info(
                    "Stage I step %d/%d: loss=%.4f, mean_r1=%.3f, "
                    "mean_r2=%.3f, delta=%.3f, kl_t1=%.4f, kl_t2=%.4f, "
                    "frac_changed=%.3f.",
                    step,
                    num_steps,
                    metrics.get("loss", 0.0),
                    metrics.get("mean_reward_t1", 0.0),
                    metrics.get("mean_reward_t2", 0.0),
                    metrics.get("delta_t1_t2", 0.0),
                    metrics.get("mean_kl_t1", 0.0),
                    metrics.get("mean_kl_t2", 0.0),
                    metrics.get("fraction_answer_changed", 0.0),
                )

            # ------------------------------------------------------------------
            # Save checkpoint at specified frequency
            # ------------------------------------------------------------------
            if step % self.save_every_n_steps == 0:
                checkpoint_path: str = f"{self.config.output_dir}/stage1_step_{step}"
                self.checkpoint_utils.save(
                    model=self.policy_model,
                    path=checkpoint_path,
                    step=step,
                    metrics=metrics,
                )
                # Track best checkpoint by mean_reward_t2
                current_r2: float = metrics.get("mean_reward_t2", 0.0)
                if current_r2 > best_reward_t2:
                    best_reward_t2 = current_r2
                    best_path: str = f"{self.config.output_dir}/stage1_best"
                    self.checkpoint_utils.save(
                        model=self.policy_model,
                        path=best_path,
                        step=step,
                        metrics=metrics,
                    )
                    logger.info(
                        "Stage I: New best checkpoint at step %d "
                        "(mean_reward_t2=%.3f) saved to '%s'.",
                        step,
                        best_reward_t2,
                        best_path,
                    )

        # ------------------------------------------------------------------
        # Save final Stage I checkpoint
        # ------------------------------------------------------------------
        final_path: str = f"{self.config.output_dir}/stage1_final"
        final_metrics: Dict[str, float] = {
            "stage": 1.0,
            "step": float(num_steps),
            "best_reward_t2": best_reward_t2,
        }
        self.checkpoint_utils.save(
            model=self.policy_model,
            path=final_path,
            step=num_steps,
            metrics=final_metrics,
        )
        logger.info(
            "SCoReStage1Trainer.train(): Stage I complete. "
            "Final checkpoint saved to '%s'. Best mean_reward_t2=%.3f.",
            final_path,
            best_reward_t2,
        )

        return self.policy_model

    def train_step(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Execute one Stage I gradient update step.

        Samples trajectories, computes Stage I loss (Equation 3), performs
        backward pass with gradient clipping, and returns metrics.

        Args:
            batch: List of problem dicts of length batch_size.

        Returns:
            Dict with keys: 'loss', 'mean_reward_t1', 'mean_reward_t2',
            'delta_t1_t2', 'mean_kl_t1', 'mean_kl_t2',
            'fraction_answer_changed'.
        """
        # ------------------------------------------------------------------
        # Step 1: Sample two-turn trajectories from the current policy.
        # Log probs stored in trajectories are detached (no gradient).
        # ------------------------------------------------------------------
        trajectories: List[Trajectory] = (
            self.rollout_buffer.sample_trajectories(batch)
        )

        # ------------------------------------------------------------------
        # Step 2: Zero gradients before computing the loss.
        # ------------------------------------------------------------------
        self.optimizer.zero_grad()

        # ------------------------------------------------------------------
        # Step 3: Compute Stage I loss (Equation 3).
        # ------------------------------------------------------------------
        loss: torch.Tensor = self.compute_stage1_loss(trajectories)

        # ------------------------------------------------------------------
        # Step 4: Guard against NaN/Inf loss (can occur with extreme log probs)
        # ------------------------------------------------------------------
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                "train_step (Stage I): Loss is NaN or Inf (%.6f). "
                "Skipping gradient update for this step.",
                loss.item() if not torch.isnan(loss) else float("nan"),
            )
            return self._compute_metrics_from_trajectories(
                trajectories=trajectories,
                loss_value=float("nan"),
            )

        # ------------------------------------------------------------------
        # Step 5: Backward pass.
        # ------------------------------------------------------------------
        loss.backward()

        # ------------------------------------------------------------------
        # Step 6: Gradient clipping.
        # ------------------------------------------------------------------
        nn_utils.clip_grad_norm_(
            self.policy_model.model.parameters(),
            max_norm=self.max_grad_norm,
        )

        # ------------------------------------------------------------------
        # Step 7: Optimizer step.
        # ------------------------------------------------------------------
        self.optimizer.step()

        # ------------------------------------------------------------------
        # Step 8: Compute and return metrics.
        # ------------------------------------------------------------------
        return self._compute_metrics_from_trajectories(
            trajectories=trajectories,
            loss_value=loss.item(),
        )

    def compute_stage1_loss(
        self, trajectories: List[Trajectory]
    ) -> torch.Tensor:
        """Compute Stage I REINFORCE loss with asymmetric KL penalties.

        Implements Equation 3 from the paper:
            max_θ E[r̂(y₂, y*) - β₂ · D_KL(π_θ(·|x₁) || π_ref(·|x₁))]

        As a minimization loss (negated):
            loss = -mean(normalized_r2 * logprob_t2 - β₂ * KL_t1 - β₁ * KL_t2)

        The asymmetry β₂ >> β₁ (e.g., 0.1 vs 0.01) enforces a strict
        constraint on the first-turn distribution while allowing the second
        turn to improve freely.

        CRITICAL: logprob_t2 values are taken from the stored trajectory
        tensors. These tensors retain gradient tracking from the forward
        pass during rollout sampling (they are NOT detached in the rollout
        buffer for the policy model). This ensures correct gradient flow.

        Args:
            trajectories: List of Trajectory objects from sample_trajectories().
                Must have non-None logprob_t1, logprob_t2, ref_logprob_t1,
                ref_logprob_t2 fields.

        Returns:
            Scalar loss tensor with gradient tracking enabled.

        Raises:
            ValueError: If trajectories is empty.
        """
        batch_size: int = len(trajectories)
        if batch_size == 0:
            raise ValueError(
                "compute_stage1_loss: trajectories list is empty."
            )

        # Determine device from policy model
        device: torch.device = self.policy_model.device

        # ------------------------------------------------------------------
        # Step 1: Extract turn-2 rewards and normalize.
        # ------------------------------------------------------------------
        rewards_t2: List[float] = [t.reward_t2 for t in trajectories]
        normalized_r2: torch.Tensor = self._normalize_reward_list(
            rewards_t2, device=device
        )
        # Shape: (batch_size,)

        # ------------------------------------------------------------------
        # Step 2: Compute KL penalties.
        # KL_t1: strong β₂ penalty on first turn (prevents first-turn drift)
        # KL_t2: weak β₁ penalty on second turn (standard regularization)
        # ------------------------------------------------------------------
        kl_t1: torch.Tensor = self._compute_kl_penalty_t1(
            trajectories, device=device
        )
        # Shape: (batch_size,)

        kl_t2: torch.Tensor = self._compute_kl_penalty_t2(
            trajectories, device=device
        )
        # Shape: (batch_size,)

        # ------------------------------------------------------------------
        # Step 3: Extract turn-2 log-probabilities.
        # These are stored in the trajectory as tensors from the rollout.
        # We stack them to get a (batch_size,) tensor.
        # ------------------------------------------------------------------
        logprob_t2: torch.Tensor = torch.stack(
            [t.logprob_t2 for t in trajectories]
        ).to(device)
        # Shape: (batch_size,)

        # ------------------------------------------------------------------
        # Step 4: Compute per-sample Stage I objective.
        # Equation 3: r2 * logprob_t2 - β₂ * KL_t1 - β₁ * KL_t2
        # ------------------------------------------------------------------
        per_sample_obj: torch.Tensor = (
            normalized_r2 * logprob_t2
            - self.beta2 * kl_t1
            - self.beta1 * kl_t2
        )
        # Shape: (batch_size,)

        # ------------------------------------------------------------------
        # Step 5: Negate and average for minimization loss.
        # ------------------------------------------------------------------
        loss: torch.Tensor = -per_sample_obj.mean()

        logger.debug(
            "compute_stage1_loss: batch_size=%d, "
            "mean_norm_r2=%.4f, mean_logprob_t2=%.4f, "
            "mean_kl_t1=%.4f, mean_kl_t2=%.4f, loss=%.4f.",
            batch_size,
            normalized_r2.mean().item(),
            logprob_t2.mean().item(),
            kl_t1.mean().item(),
            kl_t2.mean().item(),
            loss.item(),
        )

        return loss

    # -------------------------------------------------------------------------
    # Protected helpers (also used by SCoReStage2Trainer via composition)
    # -------------------------------------------------------------------------

    def _compute_kl_penalty_t1(
        self,
        trajectories: List[Trajectory],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute per-sample KL divergence approximation for turn 1.

        KL_t1[i] = log π_θ(y₁[i] | x₁[i]) - log π_ref(y₁[i] | x₁[i])

        Both logprob_t1 and ref_logprob_t1 are scalar tensors (sum of
        per-token log-probs over the response, per the shared knowledge
        convention). The KL approximation is their difference.

        Args:
            trajectories: List of Trajectory objects with non-None
                logprob_t1 and ref_logprob_t1 fields.
            device: Target device for the output tensor.

        Returns:
            Tensor of shape (batch_size,) with per-sample KL estimates.
        """
        kl_values: List[torch.Tensor] = []
        for t in trajectories:
            if t.logprob_t1 is None or t.ref_logprob_t1 is None:
                # Defensive: if log probs are missing, use zero KL
                logger.warning(
                    "_compute_kl_penalty_t1: logprob_t1 or ref_logprob_t1 "
                    "is None for a trajectory. Using KL=0.0 for this sample."
                )
                kl_values.append(torch.tensor(0.0, device=device))
            else:
                # KL approximation: log π_θ - log π_ref (per sample, summed over tokens)
                kl: torch.Tensor = (
                    t.logprob_t1.to(device) - t.ref_logprob_t1.to(device)
                )
                kl_values.append(kl)

        return torch.stack(kl_values)
        # Shape: (batch_size,)

    def _compute_kl_penalty_t2(
        self,
        trajectories: List[Trajectory],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute per-sample KL divergence approximation for turn 2.

        KL_t2[i] = log π_θ(y₂[i] | x₂[i]) - log π_ref(y₂[i] | x₂[i])

        Identical structure to _compute_kl_penalty_t1 but uses turn-2
        log-probabilities.

        Args:
            trajectories: List of Trajectory objects with non-None
                logprob_t2 and ref_logprob_t2 fields.
            device: Target device for the output tensor.

        Returns:
            Tensor of shape (batch_size,) with per-sample KL estimates.
        """
        kl_values: List[torch.Tensor] = []
        for t in trajectories:
            if t.logprob_t2 is None or t.ref_logprob_t2 is None:
                logger.warning(
                    "_compute_kl_penalty_t2: logprob_t2 or ref_logprob_t2 "
                    "is None for a trajectory. Using KL=0.0 for this sample."
                )
                kl_values.append(torch.tensor(0.0, device=device))
            else:
                kl: torch.Tensor = (
                    t.logprob_t2.to(device) - t.ref_logprob_t2.to(device)
                )
                kl_values.append(kl)

        return torch.stack(kl_values)
        # Shape: (batch_size,)

    def _normalize_reward_list(
        self,
        rewards: List[float],
        device: torch.device,
    ) -> torch.Tensor:
        """Apply per-batch reward normalization for REINFORCE variance reduction.

        Implements: normalized_r = (r - mean(r)) / (std(r) + eps)

        When all rewards are identical (e.g., all 0.0 or all 1.0), std=0
        and normalized rewards are all 0.0 (no learning signal — correct
        behavior since identical rewards provide no gradient direction).

        Args:
            rewards: List of raw binary reward values {0.0, 1.0}.
            device: Target device for the output tensor.

        Returns:
            1D float32 tensor of shape (len(rewards),) on the specified device.
            If normalize_rewards=False, returns raw rewards as tensor.
        """
        rewards_tensor: torch.Tensor = torch.tensor(
            rewards, dtype=torch.float32, device=device
        )

        if not self.normalize_rewards:
            return rewards_tensor

        mean: torch.Tensor = rewards_tensor.mean()
        std: torch.Tensor = rewards_tensor.std()

        # Handle single-element batch or zero-variance batch
        if rewards_tensor.numel() <= 1 or torch.isnan(std) or std.item() < _REWARD_NORM_EPS_FLOOR:
            # Return zero-centered rewards without std normalization
            return rewards_tensor - mean

        normalized: torch.Tensor = (rewards_tensor - mean) / (
            std + self.reward_norm_eps
        )
        return normalized

    def _compute_metrics_from_trajectories(
        self,
        trajectories: List[Trajectory],
        loss_value: float,
    ) -> Dict[str, float]:
        """Compute training metrics from trajectory data.

        All metrics use raw (unnormalized) rewards for interpretability and
        direct correspondence with the paper's reported metrics.

        Args:
            trajectories: List of Trajectory objects.
            loss_value: Scalar loss value from compute_stage1_loss().item().

        Returns:
            Dict with keys: 'loss', 'mean_reward_t1', 'mean_reward_t2',
            'delta_t1_t2', 'mean_kl_t1', 'mean_kl_t2',
            'fraction_answer_changed'.
        """
        n: int = len(trajectories)
        if n == 0:
            return {
                "loss": loss_value,
                "mean_reward_t1": 0.0,
                "mean_reward_t2": 0.0,
                "delta_t1_t2": 0.0,
                "mean_kl_t1": 0.0,
                "mean_kl_t2": 0.0,
                "fraction_answer_changed": 0.0,
            }

        rewards_t1: List[float] = [t.reward_t1 for t in trajectories]
        rewards_t2: List[float] = [t.reward_t2 for t in trajectories]

        mean_r1: float = sum(rewards_t1) / n
        mean_r2: float = sum(rewards_t2) / n
        delta: float = mean_r2 - mean_r1

        # KL divergence from stored (detached) log probs
        kl_t1_vals: List[float] = []
        kl_t2_vals: List[float] = []
        for t in trajectories:
            if t.logprob_t1 is not None and t.ref_logprob_t1 is not None:
                kl_t1_vals.append(
                    (t.logprob_t1 - t.ref_logprob_t1).item()
                )
            if t.logprob_t2 is not None and t.ref_logprob_t2 is not None:
                kl_t2_vals.append(
                    (t.logprob_t2 - t.ref_logprob_t2).item()
                )

        mean_kl_t1: float = (
            sum(kl_t1_vals) / len(kl_t1_vals) if kl_t1_vals else 0.0
        )
        mean_kl_t2: float = (
            sum(kl_t2_vals) / len(kl_t2_vals) if kl_t2_vals else 0.0
        )

        # Fraction of trajectories where the model changed its answer
        # (monitors behavior collapse — low fraction = collapse)
        num_changed: int = sum(
            1
            for t in trajectories
            if t.turn1_response.strip() != t.turn2_response.strip()
        )
        fraction_changed: float = num_changed / n

        return {
            "loss": loss_value,
            "mean_reward_t1": mean_r1,
            "mean_reward_t2": mean_r2,
            "delta_t1_t2": delta,
            "mean_kl_t1": mean_kl_t1,
            "mean_kl_t2": mean_kl_t2,
            "fraction_answer_changed": fraction_changed,
        }

    def _make_cycling_iterator(
        self, data: List[Dict[str, Any]]
    ):
        """Create an infinite cycling iterator over shuffled data.

        Shuffles the data at each epoch boundary to ensure different
        orderings across epochs. Uses a local copy of indices to avoid
        modifying the original data list.

        Args:
            data: The dataset to cycle over.

        Yields:
            Individual data items, cycling indefinitely with reshuffling
            at each epoch boundary.
        """
        indices: List[int] = list(range(len(data)))
        while True:
            random.shuffle(indices)
            for idx in indices:
                yield data[idx]


class SCoReStage2Trainer:
    """Stage II trainer: joint multi-turn 