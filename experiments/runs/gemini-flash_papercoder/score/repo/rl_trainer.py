from dataclasses import dataclass, field
import logging
import abc
import random
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset_utils import Problem
from model_utils import LLMForSelfCorrection
from prompt_manager import PromptManager
from reward_utils import BaseRewardFunction

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    """
    Represents a single multi-turn interaction (rollout) with the language model.
    Contains problem details, prompts, model responses, log-probabilities,
    rewards, and KL divergences for both turns.
    """

    problem: Problem
    # Turn 1 data
    prompt_t1: str
    response_t1: str
    log_probs_t1: List[float]  # Log-probabilities of policy for response_t1
    ref_log_probs_t1: List[float]  # Log-probabilities of ref model for response_t1
    reward_t1: float
    kl_t1: float

    # Turn 2 data
    prompt_t2: str
    response_t2: str
    log_probs_t2: List[float]  # Log-probabilities of policy for response_t2
    ref_log_probs_t2: List[float]  # Log-probabilities of ref model for response_t2
    reward_t2: float
    kl_t2: float


class BaseRLTrainer(abc.ABC):
    """
    Abstract base class for Reinforcement Learning trainers in the SCoRe framework.
    Provides common functionalities for two-stage RL training, including initialization,
    multi-turn rollout generation, KL divergence calculation, and the main training loop.
    """

    def __init__(
        self,
        model_wrapper: LLMForSelfCorrection,
        ref_model_wrapper: LLMForSelfCorrection,
        dataloader: DataLoader,
        prompt_manager: PromptManager,
        reward_function: BaseRewardFunction,
        config: Config,
    ) -> None:
        """
        Initializes the BaseRLTrainer.

        Args:
            model_wrapper: An instance of LLMForSelfCorrection representing the trainable policy model.
            ref_model_wrapper: An instance of LLMForSelfCorrection representing the fixed reference model.
            dataloader: DataLoader for fetching training problems.
            prompt_manager: Manager for constructing prompts.
            reward_function: Function for calculating rewards and bonuses.
            config: Global configuration object.
        """
        self.model_wrapper = model_wrapper
        self.ref_model_wrapper = ref_model_wrapper
        self.dataloader = dataloader
        self.prompt_manager = prompt_manager
        self.reward_function = reward_function
        self.config = config

        self.model = self.model_wrapper.get_current_model()
        # The ref_model in LLMForSelfCorrection is already the underlying HF model.
        # This ensures we get the actual model, not the wrapper.
        self.ref_model = self.ref_model_wrapper.get_current_model()

        # Determine task context for prompts based on config.task_type
        self.task_context: Literal["math", "mbpp_train"]
        if self.config.task_type == "math":
            self.task_context = "math"
        elif self.config.task_type == "code":
            # For MBPP training, use 'mbpp_train' context for 3-shot prompting
            self.task_context = "mbpp_train"
        else:
            raise ValueError(
                f"Unsupported task_type: {self.config.task_type} for RLTrainer."
            )

        # Load stage-specific hyperparameters
        stage_name = self.__class__.__name__  # e.g., 'Stage1RLTrainer', 'Stage2RLTrainer'
        if "Stage1" in stage_name:
            stage_hparams_key = (
                "training_stage1_math"
                if self.config.task_type == "math"
                else "training_stage1_mbpp"
            )
        elif "Stage2" in stage_name:
            stage_hparams_key = (
                "training_stage2_math"
                if self.config.task_type == "math"
                else "training_stage2_mbpp"
            )
        else:
            raise ValueError(f"Unknown RLTrainer stage: {stage_name}")

        stage_hparams = getattr(self.config, stage_hparams_key)
        if stage_hparams is None:
            raise ValueError(
                f"Hyperparameters for {stage_hparams_key} not found in config."
            )

        self.learning_rate: float = stage_hparams["learning_rate"]
        self.training_steps: int = stage_hparams["training_steps"]
        self.batch_size: int = stage_hparams[
            "batch_size"
        ]  # Number of rollouts to collect for each gradient update
        self.sampling_temperature: float = stage_hparams["sampling_temperature"]
        self.alpha_reward_shaping: float = stage_hparams["alpha_reward_shaping"]
        self.beta1_kl_penalty: float = stage_hparams["beta1_kl_penalty"]

        # beta2 is only defined for Stage 1. For Stage 2, it will be 0.0 by default.
        self.beta2_kl_penalty_stage1: float = stage_hparams.get(
            "beta2_kl_penalty_stage1", 0.0
        )

        # Max tokens for generation during rollouts (from evaluation settings in config)
        self.max_new_tokens: int = self.config.evaluation.get("max_new_tokens", 1024)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )

        # Initialize dataloader iterator to handle exhaustion
        self._dataloader_iterator = iter(self.dataloader)
        self._current_dataloader_batch: Optional[List[Problem]] = None
        self._current_dataloader_batch_idx: int = 0

    def _get_next_problem(self) -> Problem:
        """
        Retrieves the next single problem from the dataloader.
        Handles batching from the DataLoader and resetting the iterator if exhausted.

        Returns:
            A single Problem object.
        """
        if (
            self._current_dataloader_batch is None
            or self._current_dataloader_batch_idx >= len(self._current_dataloader_batch)
        ):
            # Fetch a new batch from the dataloader
            try:
                self._current_dataloader_batch = next(self._dataloader_iterator)
                self._current_dataloader_batch_idx = 0
            except StopIteration:
                logger.info("Dataloader exhausted. Resetting iterator for next epoch.")
                self._dataloader_iterator = iter(self.dataloader)
                self._current_dataloader_batch = next(self._dataloader_iterator)
                self._current_dataloader_batch_idx = 0
        
        problem = self._current_dataloader_batch[self._current_dataloader_batch_idx]
        self._current_dataloader_batch_idx += 1
        return problem

    def _run_rollout(self, problem: Problem) -> Rollout:
        """
        Executes a two-turn interaction (rollout) for a single problem.

        Args:
            problem: The Problem object for which to run the rollout.

        Returns:
            A Rollout object containing all generated data for both turns.
        """
        # Turn 1
        prompt_t1 = self.prompt_manager.get_first_turn_prompt(
            problem_text=problem.text, task_context=self.task_context
        )
        response_t1, log_probs_t1 = self.model_wrapper.generate(
            prompt=prompt_t1,
            temperature=self.sampling_temperature,
            max_new_tokens=self.max_new_tokens,
        )
        ref_log_probs_t1 = self.ref_model_wrapper.get_log_probs(
            prompt_t1, response_t1
        )
        reward_t1 = self.reward_function.calculate_reward(
            response=response_t1, ground_truth=problem.ground_truth
        )
        kl_t1 = self._calculate_kl_divergence(log_probs_t1, ref_log_probs_t1)

        # Turn 2
        prompt_t2 = self.prompt_manager.get_second_turn_prompt(
            problem_text=problem.text,
            first_response=response_t1,
            task_context=self.task_context,
        )
        response_t2, log_probs_t2 = self.model_wrapper.generate(
            prompt=prompt_t2,
            temperature=self.sampling_temperature,
            max_new_tokens=self.max_new_tokens,
        )
        ref_log_probs_t2 = self.ref_model_wrapper.get_log_probs(
            prompt_t2, response_t2
        )
        reward_t2 = self.reward_function.calculate_reward(
            response=response_t2, ground_truth=problem.ground_truth
        )
        kl_t2 = self._calculate_kl_divergence(log_probs_t2, ref_log_probs_t2)

        return Rollout(
            problem=problem,
            prompt_t1=prompt_t1,
            response_t1=response_t1,
            log_probs_t1=log_probs_t1,
            ref_log_probs_t1=ref_log_probs_t1,
            reward_t1=reward_t1,
            kl_t1=kl_t1,
            prompt_t2=prompt_t2,
            response_t2=response_t2,
            log_probs_t2=log_probs_t2,
            ref_log_probs_t2=ref_log_probs_t2,
            reward_t2=reward_t2,
            kl_t2=kl_t2,
        )

    def _calculate_kl_divergence(
        self, log_probs_policy: List[float], log_probs_ref: List[float]
    ) -> float:
        """
        Calculates the KL divergence between policy and reference model for a generated sequence.
        This is an approximation using the difference in log-probabilities of the sampled sequence.

        Args:
            log_probs_policy: List of log-probabilities for the generated tokens under the policy.
            log_probs_ref: List of log-probabilities for the generated tokens under the reference model.

        Returns:
            The sum of log_prob_policy - log_prob_ref for each token. Returns 0.0 if lists are empty.
        """
        if not log_probs_policy or not log_probs_ref:
            return 0.0

        # Ensure lengths match, though they should for the same generated sequence.
        # We take the minimum length to prevent errors if there's a tokenization mismatch.
        min_len = min(len(log_probs_policy), len(log_probs_ref))
        log_probs_policy_tensor = torch.tensor(
            log_probs_policy[:min_len], dtype=torch.float32, device=self.model.device
        )
        log_probs_ref_tensor = torch.tensor(
            log_probs_ref[:min_len], dtype=torch.float32, device=self.ref_model.device
        )

        # Sum of log(P(token)) - log(Q(token)) over the sequence
        return torch.sum(log_probs_policy_tensor - log_probs_ref_tensor).item()

    @abc.abstractmethod
    def _compute_loss(self, rollout: Rollout) -> torch.Tensor:
        """
        Abstract method to compute the loss for a single rollout based on the
        specific objective of the training stage.

        Args:
            rollout: The Rollout object containing interaction data.

        Returns:
            A scalar torch.Tensor representing the loss for this rollout.
        """
        raise NotImplementedError

    def train(self) -> None:
        """
        Executes the main training loop for the RL stage.
        """
        self.model.train()  # Set policy model to training mode
        self.ref_model.eval()  # Set reference model to evaluation mode (no gradients)

        # Ensure optimizer is correctly tied to the current model parameters
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )

        progress_bar = tqdm(
            range(self.training_steps),
            desc=f"{self.__class__.__name__} Training",
            unit="step",
        )

        for step in progress_bar:
            rollouts: List[Rollout] = []
            # Collect self.batch_size rollouts
            for _ in range(self.batch_size):
                problem = self._get_next_problem()
                rollouts.append(self._run_rollout(problem))

            if not rollouts:
                logger.warning("No rollouts collected in current step. Skipping optimization.")
                continue

            # Compute and accumulate loss for the batch
            total_loss = torch.tensor(0.0, device=self.model.device)
            for rollout in rollouts:
                loss_for_rollout = self._compute_loss(rollout)
                total_loss += loss_for_rollout

            total_loss /= len(rollouts)  # Average loss over the batch of rollouts

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")
            logger.debug(f"Step {step+1}/{self.training_steps}, Loss: {total_loss.item():.4f}")

        progress_bar.close()
        logger.info(f"{self.__class__.__name__} training completed.")


class Stage1RLTrainer(BaseRLTrainer):
    """
    Implements Stage I of SCoRe training.
    Objective: Optimize second-turn correctness while heavily regularizing
    the first turn to stay close to the reference model.
    """

    def __init__(
        self,
        model_wrapper: LLMForSelfCorrection,
        ref_model_wrapper: LLMForSelfCorrection,
        dataloader: DataLoader,
        prompt_manager: PromptManager,
        reward_function: BaseRewardFunction,
        config: Config,
    ) -> None:
        """
        Initializes the Stage1RLTrainer.
        """
        super().__init__(
            model_wrapper,
            ref_model_wrapper,
            dataloader,
            prompt_manager,
            reward_function,
            config,
        )

    def _compute_loss(self, rollout: Rollout) -> torch.Tensor:
        """
        Computes the loss for a single rollout in Stage I.
        Loss = - (sum_log_probs_t2 * reward_t2)
               + (beta1_kl_penalty + beta2_kl_penalty_stage1) * kl_t1
               + beta1_kl_penalty * kl_t2

        Args:
            rollout: The Rollout object for which to compute the loss.

        Returns:
            A scalar torch.Tensor representing the loss.
        """
        # Policy gradient term for Turn 2 (only optimize second turn's log-probs for reward)
        # Sum of log_probs_t2 is multiplied by reward_t2.
        # Ensure tensor is on correct device.
        log_probs_t2_tensor = torch.tensor(
            rollout.log_probs_t2, dtype=torch.float32, device=self.model.device
        )
        policy_gradient_loss_t2 = (
            -(torch.sum(log_probs_t2_tensor) * rollout.reward_t2)
            if len(log_probs_t2_tensor) > 0
            else torch.tensor(0.0, device=self.model.device)
        )

        # KL penalty terms
        # Ensure tensors are on correct device.
        kl_t1_tensor = torch.tensor(
            rollout.kl_t1, dtype=torch.float32, device=self.model.device
        )
        kl_t2_tensor = torch.tensor(
            rollout.kl_t2, dtype=torch.float32, device=self.model.device
        )

        kl_penalty_loss = (
            (self.beta1_kl_penalty + self.beta2_kl_penalty_stage1) * kl_t1_tensor
            + self.beta1_kl_penalty * kl_t2_tensor
        )

        total_loss = policy_gradient_loss_t2 + kl_penalty_loss
        return total_loss


class Stage2RLTrainer(BaseRLTrainer):
    """
    Implements Stage II of SCoRe training.
    Objective: Jointly optimize both attempts with reward shaping to incentivize
    self-correction progress.
    """

    def __init__(
        self,
        model_wrapper: LLMForSelfCorrection,
        ref_model_wrapper: LLMForSelfCorrection,
        dataloader: DataLoader,
        prompt_manager: PromptManager,
        reward_function: BaseRewardFunction,
        config: Config,
    ) -> None:
        """
        Initializes the Stage2RLTrainer.
        """
        super().__init__(
            model_wrapper,
            ref_model_wrapper,
            dataloader,
            prompt_manager,
            reward_function,
            config,
        )

    def _compute_loss(self, rollout: Rollout) -> torch.Tensor:
        """
        Computes the loss for a single rollout in Stage II with reward shaping.
        Loss = - (sum_log_probs_t1 * shaped_reward_t1 + sum_log_probs_t2 * shaped_reward_t2)
               + beta1_kl_penalty * kl_t1
               + beta1_kl_penalty * kl_t2

        Args:
            rollout: The Rollout object for which to compute the loss.

        Returns:
            A scalar torch.Tensor representing the loss.
        """
        # Calculate reward shaping bonus
        bonus_t2 = self.reward_function.calculate_bonus(
            reward_t1=rollout.reward_t1,
            reward_t2=rollout.reward_t2,
            alpha=self.alpha_reward_shaping,
        )

        shaped_reward_t1 = rollout.reward_t1
        shaped_reward_t2 = rollout.reward_t2 + bonus_t2

        # Policy gradient terms for both turns, weighted by shaped rewards
        # Ensure tensors are on correct device.
        log_probs_t1_tensor = torch.tensor(
            rollout.log_probs_t1, dtype=torch.float32, device=self.model.device
        )
        log_probs_t2_tensor = torch.tensor(
            rollout.log_probs_t2, dtype=torch.float32, device=self.model.device
        )

        policy_gradient_loss_t1 = (
            -(torch.sum(log_probs_t1_tensor) * shaped_reward_t1)
            if len(log_probs_t1_tensor) > 0
            else torch.tensor(0.0, device=self.model.device)
        )
        policy_gradient_loss_t2 = (
            -(torch.sum(log_probs_t2_tensor) * shaped_reward_t2)
            if len(log_probs_t2_tensor) > 0
            else torch.tensor(0.0, device=self.model.device)
        )

        policy_gradient_loss = policy_gradient_loss_t1 + policy_gradient_loss_t2

        # KL penalty terms (using beta1 for both)
        # Ensure tensors are on correct device.
        kl_t1_tensor = torch.tensor(
            rollout.kl_t1, dtype=torch.float32, device=self.model.device
        )
        kl_t2_tensor = torch.tensor(
            rollout.kl_t2, dtype=torch.float32, device=self.model.device
        )

        kl_penalty_loss = (
            self.beta1_kl_penalty * kl_t1_tensor
        ) + (
            self.beta1_kl_penalty * kl_t2_tensor
        )

        total_loss = policy_gradient_loss + kl_penalty_loss
        return total_loss

