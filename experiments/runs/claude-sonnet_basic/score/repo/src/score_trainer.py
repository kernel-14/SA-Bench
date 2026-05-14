"""
SCoRe: Self-Correction via Reinforcement Learning
Core training algorithm implementation.

Paper: "Training Language Models to Self-Correct via Reinforcement Learning"
Kumar et al., 2024

This implements the two-stage multi-turn RL approach:
- Stage I: Train second-attempt while constraining first-attempt to base model
- Stage II: Joint optimization with reward shaping bonus
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class SCoReConfig:
    """
    Configuration for SCoRe training.
    Hyperparameters from Appendix B of the paper.
    """
    # Task type
    task_type: str = "math"  # "math" or "code"
    
    # Model
    model_name: str = "google/gemma-2b"
    
    # Optimizer
    learning_rate: float = 5e-6  # 5e-6 for MATH, 1e-5 for MBPP
    optimizer: str = "adam"
    
    # Training
    training_steps: int = 3000  # 3000 for MATH, 1500 for MBPP
    batch_size: int = 512  # 512 for MATH, 128 for MBPP
    sampling_temperature: float = 1.0
    
    # SCoRe-specific hyperparameters
    alpha: float = 10.0  # Reward shaping multiplier (same for both tasks)
    beta1: float = 0.01  # KL penalty weight for standard RL (both turns in Stage II)
    beta2: float = 0.1   # KL penalty weight for first-turn in Stage I (0.1 for MATH, 0.25 for MBPP)
    
    # Stage I / Stage II split
    stage1_steps: int = 1000  # Approximate split; paper doesn't specify exact ratio
    
    # Evaluation
    eval_every: int = 100
    eval_temperature: float = 0.0  # Greedy decoding for evaluation
    
    # Data
    max_seq_len: int = 2048
    
    # Offline data augmentation (Section 5.3)
    use_offline_first_attempts: bool = True  # Augment with base model samples


@dataclass
class RolloutBatch:
    """A batch of two-turn rollouts for training."""
    # Problem inputs
    problems: List[str]
    ground_truths: List[Any]
    
    # First attempt
    first_prompts: List[str]
    first_responses: List[str]
    first_rewards: List[float]
    
    # Second attempt
    second_prompts: List[str]
    second_responses: List[str]
    second_rewards: List[float]
    
    # Token-level data for policy gradient
    first_input_ids: Optional[List[torch.Tensor]] = None
    first_response_ids: Optional[List[torch.Tensor]] = None
    second_input_ids: Optional[List[torch.Tensor]] = None
    second_response_ids: Optional[List[torch.Tensor]] = None


class REINFORCEWithKL:
    """
    REINFORCE policy gradient with KL-divergence penalty.
    
    Implements the base RL fine-tuning approach from Ahmadian et al. (2024),
    as described in Section 3 of the paper (Equation 2):
    
    max_theta E[r(y, y*) - beta1 * KL(pi_theta || pi_ref)]
    
    This is extended to multi-turn in SCoRe.
    """
    
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        config: SCoReConfig,
    ):
        """
        Args:
            model: The policy model being trained
            ref_model: The frozen reference model (base model)
            tokenizer: Tokenizer for the model
            config: SCoRe configuration
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        
        self.optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate
        )
    
    def compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities of response tokens given input.
        
        Args:
            model: Language model
            input_ids: Input token IDs [seq_len] (1D)
            response_ids: Response token IDs [resp_len] (1D)
            
        Returns:
            Log probability (scalar, sum over response tokens)
        """
        # Concatenate input and response, add batch dim
        full_ids = torch.cat([input_ids, response_ids]).unsqueeze(0)  # [1, total_len]
        
        outputs = model(input_ids=full_ids)
        logits = outputs.logits[0]  # [total_len, vocab_size]
        
        # Get logits for response tokens (shifted by 1 for next-token prediction)
        input_len = input_ids.shape[0]
        resp_logits = logits[input_len - 1:-1]  # [resp_len, vocab_size]
        
        # Compute log probs for actual response tokens
        log_probs = F.log_softmax(resp_logits, dim=-1)
        token_log_probs = log_probs[
            torch.arange(len(response_ids), device=response_ids.device),
            response_ids
        ]  # [resp_len]
        
        return token_log_probs.sum()  # scalar
    
    def compute_kl_divergence(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between policy and reference model.
        KL(pi_theta || pi_ref) approximated as log(pi_theta) - log(pi_ref).
        
        Args:
            input_ids: Input token IDs [seq_len]
            response_ids: Response token IDs [resp_len]
            
        Returns:
            KL divergence (scalar)
        """
        log_prob_policy = self.compute_log_probs(self.model, input_ids, response_ids)
        
        with torch.no_grad():
            log_prob_ref = self.compute_log_probs(self.ref_model, input_ids, response_ids)
        
        return log_prob_policy - log_prob_ref


class SCoReTrainer:
    """
    Main SCoRe trainer implementing the two-stage multi-turn RL approach.
    
    Stage I (Section 5.1):
    - Optimize second-attempt reward
    - Constrain first-attempt to be close to base model via KL penalty (beta2)
    - Objective: max E[r(y2, y*) - beta2 * KL(pi_theta(.|x1) || pi_ref(.|x1))]
    
    Stage II (Section 5.2):
    - Jointly optimize both attempts
    - Use reward shaping bonus: b(y2|y1,y*) = alpha * (r(y2,y*) - r(y1,y*))
    - Objective: max E[sum_i r(yi, y*) - beta1 * KL(pi_theta(.|xi) || pi_ref(.|xi))]
      with shaped reward for second attempt
    """
    
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        config: SCoReConfig,
        reward_fn,
    ):
        """
        Args:
            model: Policy model to train
            ref_model: Frozen reference (base) model
            tokenizer: Tokenizer
            config: SCoRe configuration
            reward_fn: Function(response, ground_truth) -> float
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        
        self.rl = REINFORCEWithKL(model, ref_model, tokenizer, config)
        
        self.global_step = 0
        self.stage = 1  # Start with Stage I
        
        # Metrics tracking
        self.metrics_history = []
    
    def generate_response(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate a response from the current policy.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0 = greedy)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (response_text, input_ids, response_ids)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_len,
            padding=False,
        ).to(self.model.device)
        
        with torch.no_grad():
            if temperature == 0.0:
                # Greedy decoding
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        
        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        response_ids = output_ids[0][input_len:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return response, inputs["input_ids"][0], response_ids
    
    def collect_rollouts(
        self,
        problems: List[str],
        ground_truths: List[Any],
        first_prompts: List[str],
        correction_instruction: str,
        temperature: float = 1.0,
    ) -> RolloutBatch:
        """
        Collect two-turn rollouts from the current policy.
        
        Args:
            problems: List of problem strings
            ground_truths: List of ground truth answers
            first_prompts: Formatted prompts for first attempt
            correction_instruction: Self-correction instruction for second attempt
            temperature: Sampling temperature
            
        Returns:
            RolloutBatch with all rollout data
        """
        self.model.eval()
        
        first_responses = []
        first_rewards = []
        first_input_ids = []
        first_response_ids = []
        second_prompts = []
        second_responses = []
        second_rewards = []
        second_input_ids = []
        second_response_ids = []
        
        for problem, gt, first_prompt in zip(problems, ground_truths, first_prompts):
            # Generate first attempt
            y1, x1_ids, y1_ids = self.generate_response(first_prompt, temperature=temperature)
            r1 = self.reward_fn(y1, gt)
            
            first_responses.append(y1)
            first_rewards.append(r1)
            first_input_ids.append(x1_ids)
            first_response_ids.append(y1_ids)
            
            # Build second-attempt prompt
            second_prompt = f"{first_prompt}\n\n{y1}\n\n{correction_instruction}"
            second_prompts.append(second_prompt)
            
            # Generate second attempt
            y2, x2_ids, y2_ids = self.generate_response(second_prompt, temperature=temperature)
            r2 = self.reward_fn(y2, gt)
            
            second_responses.append(y2)
            second_rewards.append(r2)
            second_input_ids.append(x2_ids)
            second_response_ids.append(y2_ids)
        
        self.model.train()
        
        return RolloutBatch(
            problems=problems,
            ground_truths=ground_truths,
            first_prompts=first_prompts,
            first_responses=first_responses,
            first_rewards=first_rewards,
            second_prompts=second_prompts,
            second_responses=second_responses,
            second_rewards=second_rewards,
            first_input_ids=first_input_ids,
            first_response_ids=first_response_ids,
            second_input_ids=second_input_ids,
            second_response_ids=second_response_ids,
        )
    
    def stage1_loss(self, batch: RolloutBatch) -> torch.Tensor:
        """
        Compute Stage I loss.
        
        Objective (Equation 3 from paper):
        max E[r(y2, y*) - beta2 * KL(pi_theta(.|x1) || pi_ref(.|x1))]
        
        - Optimize second-attempt reward
        - Apply strong KL penalty on first-attempt to keep it close to base model
        - Standard (small) KL penalty also applied to second attempt
        
        Args:
            batch: Rollout batch
            
        Returns:
            Scalar loss
        """
        total_loss = torch.tensor(0.0, device=self.model.device)
        
        for i in range(len(batch.problems)):
            x1_ids = batch.first_input_ids[i]
            y1_ids = batch.first_response_ids[i]
            x2_ids = batch.second_input_ids[i]
            y2_ids = batch.second_response_ids[i]
            
            r2 = batch.second_rewards[i]
            
            # Log prob of second attempt (for REINFORCE)
            log_prob_y2 = self.rl.compute_log_probs(self.model, x2_ids, y2_ids)
            
            # REINFORCE loss for second attempt
            reinforce_loss = -r2 * log_prob_y2
            
            # Strong KL penalty on first attempt (beta2)
            kl_first = self.rl.compute_kl_divergence(x1_ids, y1_ids)
            
            # Standard KL penalty on second attempt (beta1)
            kl_second = self.rl.compute_kl_divergence(x2_ids, y2_ids)
            
            # Stage I objective: maximize r(y2) - beta2 * KL(first) - beta1 * KL(second)
            sample_loss = (
                reinforce_loss
                + self.config.beta2 * kl_first
                + self.config.beta1 * kl_second
            )
            
            total_loss = total_loss + sample_loss
        
        return total_loss / len(batch.problems)
    
    def stage2_loss(self, batch: RolloutBatch) -> torch.Tensor:
        """
        Compute Stage II loss with reward shaping.
        
        Objective (Equation 4 + reward shaping from Section 5.2):
        max E[sum_i r(yi, y*) - beta1 * KL(pi_theta(.|xi) || pi_ref(.|xi))]
        
        With shaped reward for second attempt:
        r2_shaped = r2 + alpha * (r2 - r1)
        
        This bonus:
        - Rewards i->c transitions (incorrect->correct): +alpha
        - Penalizes c->i transitions (correct->incorrect): -alpha
        - Zero for same-correctness transitions
        
        Args:
            batch: Rollout batch
            
        Returns:
            Scalar loss
        """
        total_loss = torch.tensor(0.0, device=self.model.device)
        
        for i in range(len(batch.problems)):
            x1_ids = batch.first_input_ids[i]
            y1_ids = batch.first_response_ids[i]
            x2_ids = batch.second_input_ids[i]
            y2_ids = batch.second_response_ids[i]
            
            r1 = batch.first_rewards[i]
            r2 = batch.second_rewards[i]
            
            # Log probs for both attempts
            log_prob_y1 = self.rl.compute_log_probs(self.model, x1_ids, y1_ids)
            log_prob_y2 = self.rl.compute_log_probs(self.model, x2_ids, y2_ids)
            
            # Shaped reward for second attempt
            # b(y2|y1,y*) = alpha * (r(y2,y*) - r(y1,y*))
            bonus = self.config.alpha * (r2 - r1)
            r2_shaped = r2 + bonus
            
            # REINFORCE losses
            reinforce_loss_y1 = -r1 * log_prob_y1
            reinforce_loss_y2 = -r2_shaped * log_prob_y2
            
            # KL penalties for both attempts (beta1)
            kl_first = self.rl.compute_kl_divergence(x1_ids, y1_ids)
            kl_second = self.rl.compute_kl_divergence(x2_ids, y2_ids)
            
            # Stage II objective
            sample_loss = (
                reinforce_loss_y1
                + reinforce_loss_y2
                + self.config.beta1 * kl_first
                + self.config.beta1 * kl_second
            )
            
            total_loss = total_loss + sample_loss
        
        return total_loss / len(batch.problems)
    
    def train_step(
        self,
        batch: RolloutBatch,
        stage: int,
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Rollout batch
            stage: 1 for Stage I, 2 for Stage II
            
        Returns:
            Dictionary of metrics
        """
        self.rl.optimizer.zero_grad()
        
        if stage == 1:
            loss = self.stage1_loss(batch)
        else:
            loss = self.stage2_loss(batch)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.rl.optimizer.step()
        
        # Compute metrics
        r1_mean = sum(batch.first_rewards) / len(batch.first_rewards)
        r2_mean = sum(batch.second_rewards) / len(batch.second_rewards)
        delta = r2_mean - r1_mean
        
        n_incorrect_t1 = sum(1 for r in batch.first_rewards if r == 0.0)
        n_correct_t1 = sum(1 for r in batch.first_rewards if r == 1.0)
        
        # i->c: fraction of incorrect t1 that become correct t2
        n_i_to_c = sum(
            1 for r1, r2 in zip(batch.first_rewards, batch.second_rewards)
            if r1 == 0.0 and r2 == 1.0
        )
        i_to_c = n_i_to_c / max(1, n_incorrect_t1)
        
        # c->i: fraction of correct t1 that become incorrect t2
        n_c_to_i = sum(
            1 for r1, r2 in zip(batch.first_rewards, batch.second_rewards)
            if r1 == 1.0 and r2 == 0.0
        )
        c_to_i = n_c_to_i / max(1, n_correct_t1)
        
        # Fraction of problems where answer changed
        n_changed = sum(
            1 for y1, y2 in zip(batch.first_responses, batch.second_responses)
            if y1 != y2
        )
        
        return {
            "loss": loss.item(),
            "accuracy_t1": r1_mean,
            "accuracy_t2": r2_mean,
            "delta_t1_t2": delta,
            "i_to_c": i_to_c,
            "c_to_i": c_to_i,
            "fraction_changed": n_changed / len(batch.problems),
            "stage": stage,
            "step": self.global_step,
        }
    
    def evaluate(
        self,
        eval_problems: List[str],
        eval_ground_truths: List[Any],
        eval_first_prompts: List[str],
        correction_instruction: str,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a held-out set.
        Uses greedy decoding (temperature=0) as per paper.
        
        Args:
            eval_problems: Evaluation problems
            eval_ground_truths: Ground truth answers
            eval_first_prompts: Formatted prompts for first attempt
            correction_instruction: Self-correction instruction
            
        Returns:
            Dictionary of evaluation metrics
        """
        batch = self.collect_rollouts(
            eval_problems,
            eval_ground_truths,
            eval_first_prompts,
            correction_instruction,
            temperature=self.config.eval_temperature,
        )
        
        r1_mean = sum(batch.first_rewards) / len(batch.first_rewards)
        r2_mean = sum(batch.second_rewards) / len(batch.second_rewards)
        delta = r2_mean - r1_mean
        
        n_incorrect_t1 = sum(1 for r in batch.first_rewards if r == 0.0)
        n_correct_t1 = sum(1 for r in batch.first_rewards if r == 1.0)
        
        i_to_c = sum(
            1 for r1, r2 in zip(batch.first_rewards, batch.second_rewards)
            if r1 == 0.0 and r2 == 1.0
        ) / max(1, n_incorrect_t1)
        
        c_to_i = sum(
            1 for r1, r2 in zip(batch.first_rewards, batch.second_rewards)
            if r1 == 1.0 and r2 == 0.0
        ) / max(1, n_correct_t1)
        
        return {
            "eval/accuracy_t1": r1_mean,
            "eval/accuracy_t2": r2_mean,
            "eval/delta_t1_t2": delta,
            "eval/i_to_c": i_to_c,
            "eval/c_to_i": c_to_i,
        }
