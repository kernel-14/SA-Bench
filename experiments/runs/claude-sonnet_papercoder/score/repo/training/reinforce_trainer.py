## training/reinforce_trainer.py
"""REINFORCE trainer for SCoRe: Self-Correction via Reinforcement Learning.

This module implements REINFORCETrainer, which serves two roles:
    1. A standalone single-turn RL baseline that demonstrates behavior
       collapse without the two-stage design (ablation: "w/o multi-turn
       training" from Table 4).
    2. A base class whose optimizer setup, reward normalization, and
       gradient update patterns are inherited by SCoReStage1Trainer and
       SCoReStage2Trainer.

The base REINFORCE objective (Equation 2 from the paper):
    max_θ E_{x,y ~ π_θ(·|x)} [ r̂(y, y*) - β₁ · D_KL(π_θ(·|x) || π_ref(·|x)) ]

As a minimization loss (negated):
    loss = -mean( normalized_reward_t2 * logprob_t2 - β₁ * KL_t2 )

Key design decisions:
    - Log-probabilities are RECOMPUTED inside compute_loss() with gradient
      tracking, rather than using the stored (detached) values from the
      trajectory. This ensures correct gradient flow for the policy update.
    - Per-batch reward normalization (subtract mean, divide by std) is
      applied for variance reduction, controlled by config.score.normalize_rewards.
    - KL divergence is approximated as sum(log π_θ - log π_ref) per sample,
      consistent with the REINFORCE-style approximation in Ahmadian et al. (2024).
    - Metrics are computed from raw (unnormalized) rewards for interpretability
      and alignment with the paper's reported metrics.

Hyperparameters (from Table 5, Appendix B):
    MATH:  β₁ = 0.01, learning_rate = 5e-6, batch_size = 512
    Code:  β₁ = 0.01, learning_rate = 1e-5, batch_size = 128

Typical usage:
    from training.reinforce_trainer import REINFORCETrainer

    trainer = REINFORCETrainer(policy_model, ref_model, rollout_buffer, config)
    for batch in dataloader:
        metrics = trainer.train_step(batch)
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.utils as nn_utils
import torch.optim as optim

from config import Config
from models.model_wrapper import ModelWrapper
from training.rollout_buffer import RolloutBuffer, Trajectory

logger = logging.getLogger(__name__)


class REINFORCETrainer:
    """REINFORCE with KL-divergence penalty for single-turn RL fine-tuning.

    Implements the base RL training loop used in SCoRe. Serves as both a
    standalone single-turn baseline and the base class for Stage I and
    Stage II trainers.

    The single-turn variant (this class) optimizes only the second-attempt
    reward, demonstrating that without the two-stage design, the model
    collapses to non-correcting behavior (Figure 6 of the paper).

    Attributes:
        policy_model: The trainable policy model (π_θ). Updated each step.
        ref_model: The frozen reference model (π_ref). Never updated.
        rollout_buffer: Two-turn trajectory sampling engine.
        config: Global configuration instance.
        optimizer: Adam optimizer for policy_model parameters.
        beta1: Standard KL penalty weight (β₁). From Table 5: 0.01 for
            both MATH and code tasks.
        normalize_rewards: Whether to apply per-batch reward normalization.
            From config.score.normalize_rewards (default True).
        reward_norm_eps: Epsilon for reward normalization denominator.
            From config.score.reward_norm_eps (default 1e-8).
        max_grad_norm: Maximum gradient norm for clipping.
            From config.training.{task}.max_grad_norm (default 1.0).
    """

    def __init__(
        self,
        policy_model: ModelWrapper,
        ref_model: ModelWrapper,
        rollout_buffer: RolloutBuffer,
        config: Config,
    ) -> None:
        """Initialize REINFORCETrainer.

        Resolves task-specific hyperparameters from the nested config
        structure and instantiates the Adam optimizer.

        Args:
            policy_model: The trainable policy model (π_θ). Must have been
                initialized with freeze=False in ModelWrapper. Its parameters
                are passed to the Adam optimizer.
            ref_model: The frozen reference model (π_ref). Must have been
                initialized with freeze=True in ModelWrapper. Never updated
                by this trainer.
            rollout_buffer: Two-turn trajectory sampling engine. Provides
                sample_trajectories() for collecting on-policy rollouts.
            config: Global Config instance. Reads task-specific training
                hyperparameters from config.training.{task} and SCoRe
                hyperparameters from config.score.{task}.

        Raises:
            ValueError: If config.task is not "math" or "code".
        """
        if config.task not in ("math", "code"):
            raise ValueError(
                f"REINFORCETrainer: Invalid task '{config.task}'. "
                "Must be 'math' or 'code'."
            )

        self.policy_model: ModelWrapper = policy_model
        self.ref_model: ModelWrapper = ref_model
        self.rollout_buffer: RolloutBuffer = rollout_buffer
        self.config: Config = config

        # ------------------------------------------------------------------
        # Resolve task-specific hyperparameters from Config.
        # Config.from_dict() already flattens the nested YAML structure into
        # flat fields, so we read directly from config attributes.
        # ------------------------------------------------------------------

        # Training hyperparameters (from config.training.{task} in YAML,
        # flattened into Config fields by Config.from_dict())
        learning_rate: float = config.learning_rate
        adam_beta1: float = config.adam_beta1
        adam_beta2: float = config.adam_beta2
        adam_epsilon: float = config.adam_epsilon
        weight_decay: float = config.weight_decay
        self.max_grad_norm: float = config.max_grad_norm

        # SCoRe hyperparameters (from config.score.{task} in YAML,
        # flattened into Config fields by Config.from_dict())
        # β₁: standard KL penalty weight (Table 5: 0.01 for both tasks)
        self.beta1: float = config.beta1
        # Per-batch reward normalization for REINFORCE variance reduction
        self.normalize_rewards: bool = config.normalize_rewards
        self.reward_norm_eps: float = config.reward_norm_eps

        # ------------------------------------------------------------------
        # Instantiate Adam optimizer on policy model parameters.
        # The reference model parameters are NOT included — it is frozen.
        # ------------------------------------------------------------------
        self.optimizer: optim.Adam = optim.Adam(
            self.policy_model.model.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
            weight_decay=weight_decay,
        )

        logger.info(
            "REINFORCETrainer initialized: task='%s', "
            "lr=%.2e, beta1=%.4f, max_grad_norm=%.1f, "
            "normalize_rewards=%s, reward_norm_eps=%.2e.",
            config.task,
            learning_rate,
            self.beta1,
            self.max_grad_norm,
            self.normalize_rewards,
            self.reward_norm_eps,
        )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def train_step(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Execute one REINFORCE training step on a batch of problems.

        Performs the complete training iteration:
            1. Sample two-turn trajectories from the current policy.
            2. Zero gradients.
            3. Compute REINFORCE loss (with KL penalty).
            4. Backward pass.
            5. Gradient clipping.
            6. Optimizer step.
            7. Compute and return metrics.

        Args:
            batch: List of problem dicts from DatasetLoader. Each dict
                contains the problem text, ground truth, and test cases
                (schema varies by task — see RolloutBuffer._extract_problem_fields).

        Returns:
            Dict with keys:
                'loss' (float): Scalar loss value for this step.
                'mean_reward_t1' (float): Mean turn-1 reward (= Accuracy@t1).
                'mean_reward_t2' (float): Mean turn-2 reward (= Accuracy@t2).
                'delta_t1_t2' (float): mean_reward_t2 - mean_reward_t1
                    (= Δ(t1,t2), the primary self-correction metric).
                'mean_kl_t1' (float): Mean per-sample KL divergence at turn 1.
                'mean_kl_t2' (float): Mean per-sample KL divergence at turn 2.

        Raises:
            ValueError: If batch is empty (propagated from RolloutBuffer).
        """
        # ------------------------------------------------------------------
        # Step 1: Sample two-turn trajectories from the current policy.
        # This is an on-policy rollout — the policy model generates both
        # turn 1 and turn 2 responses at the current parameter values.
        # Log probs stored in trajectories are detached (no gradient).
        # ------------------------------------------------------------------
        trajectories: List[Trajectory] = self.rollout_buffer.sample_trajectories(
            batch
        )

        # ------------------------------------------------------------------
        # Step 2: Zero gradients before computing the loss.
        # Must be done AFTER sampling (sampling uses no_grad internally)
        # and BEFORE compute_loss (which recomputes log probs with gradients).
        # ------------------------------------------------------------------
        self.optimizer.zero_grad()

        # ------------------------------------------------------------------
        # Step 3: Compute REINFORCE loss with KL penalty.
        # compute_loss() recomputes log probs with gradient tracking.
        # ------------------------------------------------------------------
        loss: torch.Tensor = self.compute_loss(trajectories)

        # ------------------------------------------------------------------
        # Step 4: Backward pass — compute gradients.
        # ------------------------------------------------------------------
        loss.backward()

        # ------------------------------------------------------------------
        # Step 5: Gradient clipping to prevent exploding gradients.
        # max_grad_norm = 1.0 from config.training.{task}.max_grad_norm.
        # ------------------------------------------------------------------
        nn_utils.clip_grad_norm_(
            self.policy_model.model.parameters(),
            max_norm=self.max_grad_norm,
        )

        # ------------------------------------------------------------------
        # Step 6: Optimizer step — update policy model parameters.
        # ------------------------------------------------------------------
        self.optimizer.step()

        # ------------------------------------------------------------------
        # Step 7: Compute metrics from raw (unnormalized) rewards.
        # These correspond directly to the paper's reported metrics:
        #   mean_reward_t1 = Accuracy@t1
        #   mean_reward_t2 = Accuracy@t2
        #   delta_t1_t2    = Δ(t1, t2)
        # ------------------------------------------------------------------
        metrics: Dict[str, float] = self._compute_metrics(
            trajectories=trajectories,
            loss_value=loss.item(),
        )

        logger.debug(
            "train_step: loss=%.4f, mean_r1=%.3f, mean_r2=%.3f, "
            "delta=%.3f, mean_kl_t1=%.4f, mean_kl_t2=%.4f.",
            metrics["loss"],
            metrics["mean_reward_t1"],
            metrics["mean_reward_t2"],
            metrics["delta_t1_t2"],
            metrics["mean_kl_t1"],
            metrics["mean_kl_t2"],
        )

        return metrics

    def compute_loss(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute the REINFORCE loss with KL penalty (Equation 2).

        Delegates to _compute_single_turn_loss() in the base class.
        Subclasses (SCoReStage1Trainer, SCoReStage2Trainer) override this
        method to implement their specific objectives (Equations 3 and 4).

        Args:
            trajectories: List of Trajectory objects from sample_trajectories().
                Must contain non-empty turn2_prompt, turn2_response, reward_t2,
                and ref_logprob_t2 fields.

        Returns:
            Scalar loss tensor with gradient tracking enabled (for backward()).
        """
        return self._compute_single_turn_loss(trajectories)

    # -------------------------------------------------------------------------
    # Protected methods (intended for use by subclasses)
    # -------------------------------------------------------------------------

    def _compute_single_turn_loss(
        self, trajectories: List[Trajectory]
    ) -> torch.Tensor:
        """Compute single-turn REINFORCE loss using only turn-2 data.

        Implements the base REINFORCE objective (Equation 2):
            loss = -mean( normalized_reward_t2 * logprob_t2 - β₁ * KL_t2 )

        where:
            logprob_t2 = log π_θ(y₂ | x₂)  [recomputed with gradients]
            KL_t2 = log π_θ(y₂ | x₂) - log π_ref(y₂ | x₂)  [per sample]
            normalized_reward_t2 = per-batch normalized binary reward

        This is the "single-turn" variant that only optimizes the second
        attempt. It serves as the ablation baseline ("w/o multi-turn training"
        in Table 4) and as a building block for Stage I and Stage II losses.

        CRITICAL: Log-probabilities are RECOMPUTED here with gradient tracking,
        not taken from the stored (detached) trajectory values. This ensures
        correct gradient flow: the policy gradient ∇_θ log π_θ(y|x) * R
        requires the log-prob to be a function of the current θ.

        Args:
            trajectories: List of Trajectory objects. Must have non-empty
                turn2_prompt, turn2_response, reward_t2, ref_logprob_t2.

        Returns:
            Scalar loss tensor. Negative because we minimize the negated
            REINFORCE objective (gradient ascent → gradient descent).
        """
        batch_size: int = len(trajectories)
        if batch_size == 0:
            raise ValueError(
                "_compute_single_turn_loss: trajectories list is empty."
            )

        # ------------------------------------------------------------------
        # Step 1: Extract turn-2 rewards (raw binary values for normalization)
        # ------------------------------------------------------------------
        rewards_t2: List[float] = [t.reward_t2 for t in trajectories]

        # ------------------------------------------------------------------
        # Step 2: Normalize rewards for variance reduction.
        # Returns a tensor on the model's device.
        # ------------------------------------------------------------------
        normalized_rewards_t2: torch.Tensor = self._normalize_rewards(
            rewards_t2
        )
        # Shape: (batch_size,), on model device

        # ------------------------------------------------------------------
        # Step 3: Recompute turn-2 log-probabilities WITH gradient tracking.
        # We use the stored prompt/response strings from the trajectory,
        # not the pre-computed (detached) logprob_t2 scalars.
        # This is the key step that enables the policy gradient to flow.
        # ------------------------------------------------------------------
        prompts_t2: List[str] = [t.turn2_prompt for t in trajectories]
        responses_t2: List[str] = [t.turn2_response for t in trajectories]

        # compute_log_probs() returns shape (batch_size,) with gradients
        logprobs_t2: torch.Tensor = self.policy_model.compute_log_probs(
            prompts=prompts_t2,
            responses=responses_t2,
        )
        # Shape: (batch_size,), gradients enabled

        # ------------------------------------------------------------------
        # Step 4: Compute KL divergence for turn 2.
        # KL_t2 = log π_θ(y₂|x₂) - log π_ref(y₂|x₂)
        # ref_logprob_t2 is a detached scalar tensor stored in the trajectory.
        # We convert to a plain tensor on the same device as logprobs_t2.
        # ------------------------------------------------------------------
        ref_logprobs_t2: torch.Tensor = torch.tensor(
            [t.ref_logprob_t2.item() for t in trajectories],
            dtype=torch.float32,
            device=logprobs_t2.device,
        )
        # Shape: (batch_size,), no gradient (reference model is frozen)

        kl_t2: torch.Tensor = logprobs_t2 - ref_logprobs_t2
        # Shape: (batch_size,)
        # Positive values: policy has drifted from reference (penalized)
        # Negative values: policy is closer to reference than at init

        # ------------------------------------------------------------------
        # Step 5: Compute per-sample REINFORCE objective terms.
        # policy_gradient_term = normalized_reward * log π_θ(y₂|x₂)
        # kl_penalty_term = β₁ * KL_t2
        # per_sample_objective = policy_gradient_term - kl_penalty_term
        # ------------------------------------------------------------------
        # Ensure normalized_rewards_t2 is on the same device as logprobs_t2
        normalized_rewards_t2 = normalized_rewards_t2.to(logprobs_t2.device)

        policy_gradient_term: torch.Tensor = normalized_rewards_t2 * logprobs_t2
        kl_penalty_term: torch.Tensor = self.beta1 * kl_t2
        per_sample_objective: torch.Tensor = (
            policy_gradient_term - kl_penalty_term
        )
        # Shape: (batch_size,)

        # ------------------------------------------------------------------
        # Step 6: Negate and average to get the minimization loss.
        # We minimize -E[objective] = gradient ascent on the objective.
        # ------------------------------------------------------------------
        loss: torch.Tensor = -per_sample_objective.mean()

        logger.debug(
            "_compute_single_turn_loss: batch_size=%d, "
            "mean_normalized_r2=%.4f, mean_logprob_t2=%.4f, "
            "mean_kl_t2=%.4f, loss=%.4f.",
            batch_size,
            normalized_rewards_t2.mean().item(),
            logprobs_t2.mean().item(),
            kl_t2.mean().item(),
            loss.item(),
        )

        return loss

    def _normalize_rewards(
        self, rewards: List[float]
    ) -> torch.Tensor:
        """Apply per-batch reward normalization for REINFORCE variance reduction.

        Implements the standard REINFORCE variance reduction technique:
            normalized_reward = (reward - mean(rewards)) / (std(rewards) + eps)

        This is controlled by config.score.normalize_rewards (default True).
        When all rewards in the batch are identical (e.g., all 0.0 or all 1.0),
        std = 0 and the normalized rewards are all ~0 (near-zero gradient).
        This is acceptable — identical rewards provide no learning signal.

        The returned tensor is placed on the policy model's device to ensure
        compatibility with logprob tensors in loss computation.

        Args:
            rewards: List of raw binary reward values {0.0, 1.0} for a batch.
                Length must equal the batch size.

        Returns:
            1D float32 tensor of shape (len(rewards),) on the policy model's
            device. If normalize_rewards=False, returns the raw rewards as a
            tensor without normalization.
        """
        # Determine the target device from the policy model
        device: torch.device = self.policy_model.device

        # Convert to float32 tensor on the target device
        rewards_tensor: torch.Tensor = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=device,
        )
        # Shape: (batch_size,)

        if not self.normalize_rewards:
            # No normalization — return raw rewards as tensor
            logger.debug(
                "_normalize_rewards: normalize_rewards=False, "
                "returning raw rewards. mean=%.3f.",
                rewards_tensor.mean().item(),
            )
            return rewards_tensor

        # ------------------------------------------------------------------
        # Per-batch normalization: subtract mean, divide by std + eps
        # ------------------------------------------------------------------
        mean: torch.Tensor = rewards_tensor.mean()
        std: torch.Tensor = rewards_tensor.std()

        # std() returns NaN for single-element tensors; handle gracefully
        if rewards_tensor.numel() <= 1 or torch.isnan(std):
            logger.debug(
                "_normalize_rewards: batch_size=%d, std is NaN or undefined. "
                "Returning zero-centered rewards without std normalization.",
                rewards_tensor.numel(),
            )
            return rewards_tensor - mean

        normalized: torch.Tensor = (rewards_tensor - mean) / (
            std + self.reward_norm_eps
        )

        logger.debug(
            "_normalize_rewards: batch_size=%d, "
            "raw_mean=%.3f, raw_std=%.3f, "
            "normalized_mean=%.6f, normalized_std=%.6f.",
            len(rewards),
            mean.item(),
            std.item(),
            normalized.mean().item(),
            normalized.std().item() if normalized.numel() > 1 else 0.0,
        )

        return normalized

    def _compute_metrics(
        self,
        trajectories: List[Trajectory],
        loss_value: float,
    ) -> Dict[str, float]:
        """Compute training metrics from raw trajectory data.

        All metrics use raw (unnormalized) rewards for interpretability and
        direct correspondence with the paper's reported metrics:
            mean_reward_t1 = Accuracy@t1
            mean_reward_t2 = Accuracy@t2
            delta_t1_t2    = Δ(t1, t2) — the primary self-correction metric

        KL divergence values are computed from the stored (detached) log probs
        in the trajectory, not the recomputed ones used in the loss. This is
        fine for monitoring purposes — the values are numerically identical
        since the policy parameters haven't changed between sampling and
        metrics computation within the same train_step call.

        Args:
            trajectories: List of Trajectory objects from sample_trajectories().
            loss_value: The scalar loss value from compute_loss().item().

        Returns:
            Dict with keys: 'loss', 'mean_reward_t1', 'mean_reward_t2',
            'delta_t1_t2', 'mean_kl_t1', 'mean_kl_t2'.
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
            }

        # Raw rewards (binary {0.0, 1.0})
        rewards_t1: List[float] = [t.reward_t1 for t in trajectories]
        rewards_t2: List[float] = [t.reward_t2 for t in trajectories]

        mean_reward_t1: float = sum(rewards_t1) / n
        mean_reward_t2: float = sum(rewards_t2) / n
        delta_t1_t2: float = mean_reward_t2 - mean_reward_t1

        # KL divergence from stored (detached) log probs
        # KL_t = log π_θ(y_t|x_t) - log π_ref(y_t|x_t) per sample
        kl_t1_vals: List[float] = [
            (t.logprob_t1 - t.ref_logprob_t1).item()
            for t in trajectories
            if t.logprob_t1 is not None and t.ref_logprob_t1 is not None
        ]
        kl_t2_vals: List[float] = [
            (t.logprob_t2 - t.ref_logprob_t2).item()
            for t in trajectories
            if t.logprob_t2 is not None and t.ref_logprob_t2 is not None
        ]

        mean_kl_t1: float = (
            sum(kl_t1_vals) / len(kl_t1_vals) if kl_t1_vals else 0.0
        )
        mean_kl_t2: float = (
            sum(kl_t2_vals) / len(kl_t2_vals) if kl_t2_vals else 0.0
        )

        return {
            "loss": loss_value,
            "mean_reward_t1": mean_reward_t1,
            "mean_reward_t2": mean_reward_t2,
            "delta_t1_t2": delta_t1_t2,
            "mean_kl_t1": mean_kl_t1,
            "mean_kl_t2": mean_kl_t2,
        }
