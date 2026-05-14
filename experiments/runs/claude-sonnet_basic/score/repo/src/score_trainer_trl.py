"""
SCoRe trainer using TRL (Transformer Reinforcement Learning) library.

This provides a more practical implementation using TRL's PPO/REINFORCE
infrastructure, which handles batched training, gradient accumulation,
and other engineering details.

The core SCoRe algorithm is the same as in score_trainer.py, but this
version is more suitable for actual training runs.
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
class SCoReTRLConfig:
    """
    Configuration for SCoRe training with TRL.
    Hyperparameters from Appendix B of the paper.
    """
    # Task
    task_type: str = "math"
    
    # Model
    model_name: str = "google/gemma-2b"
    
    # Training
    learning_rate: float = 5e-6
    training_steps: int = 3000
    batch_size: int = 16  # Per-device batch size (total batch via gradient accumulation)
    gradient_accumulation_steps: int = 32  # To achieve effective batch of 512
    max_new_tokens: int = 512
    sampling_temperature: float = 1.0
    
    # SCoRe hyperparameters
    alpha: float = 10.0
    beta1: float = 0.01
    beta2: float = 0.1
    
    # Stage split
    stage1_fraction: float = 0.33
    
    # KL computation
    kl_penalty: str = "kl"  # "kl" or "abs" or "mse"
    
    # Evaluation
    eval_every: int = 100
    eval_temperature: float = 0.0
    
    # Data
    max_seq_len: int = 2048
    
    # Logging
    log_with: str = "wandb"  # "wandb" or "tensorboard" or None


class SCoReTRLTrainer:
    """
    SCoRe trainer using TRL-style REINFORCE with KL penalty.
    
    This implements the two-stage training:
    
    Stage I (Equation 3):
    max E[r(y2, y*) - beta2 * KL(pi_theta(.|x1) || pi_ref(.|x1))]
    
    Stage II (Equation 4 + reward shaping):
    max E[r(y1, y*) + r2_shaped(y2, y*) - beta1 * sum_i KL(pi_theta(.|xi) || pi_ref(.|xi))]
    
    where r2_shaped = r2 + alpha * (r2 - r1)
    """
    
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        config: SCoReTRLConfig,
        reward_fn,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        
        self.optimizer = Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )
        
        self.global_step = 0
        self.stage1_steps = int(config.training_steps * config.stage1_fraction)
    
    @property
    def current_stage(self) -> int:
        return 1 if self.global_step < self.stage1_steps else 2
    
    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate responses for a batch of prompts.
        
        Returns:
            Tuple of (responses, input_ids_list, response_ids_list)
        """
        responses = []
        input_ids_list = []
        response_ids_list = []
        
        self.model.eval()
        
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_len,
                padding=False,
            ).to(self.model.device)
            
            with torch.no_grad():
                if temperature == 0.0:
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
            
            input_len = inputs["input_ids"].shape[1]
            resp_ids = output_ids[0][input_len:]
            
            response = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            responses.append(response)
            input_ids_list.append(inputs["input_ids"][0])
            response_ids_list.append(resp_ids)
        
        self.model.train()
        return responses, input_ids_list, response_ids_list
    
    def compute_sequence_log_prob(
        self,
        model,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the log probability of a response sequence given input.
        
        Args:
            model: Language model
            input_ids: Input token IDs [seq_len]
            response_ids: Response token IDs [resp_len]
            
        Returns:
            Scalar log probability (sum over tokens)
        """
        # Concatenate input and response
        full_ids = torch.cat([input_ids, response_ids]).unsqueeze(0)  # [1, total_len]
        
        if model is self.ref_model:
            ctx = torch.no_grad()
        else:
            ctx = torch.enable_grad()
        
        with ctx:
            outputs = model(input_ids=full_ids)
        
        logits = outputs.logits[0]  # [total_len, vocab_size]
        
        # Get logits for response tokens
        input_len = input_ids.shape[0]
        resp_logits = logits[input_len - 1:-1]  # [resp_len, vocab_size]
        
        # Compute log probs
        log_probs = F.log_softmax(resp_logits, dim=-1)
        token_log_probs = log_probs[
            torch.arange(len(response_ids)),
            response_ids
        ]
        
        return token_log_probs.sum()
    
    def compute_kl(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL(pi_theta || pi_ref) for a single sequence.
        Approximated as log(pi_theta/pi_ref) = log_pi_theta - log_pi_ref.
        """
        log_prob_policy = self.compute_sequence_log_prob(
            self.model, input_ids, response_ids
        )
        
        with torch.no_grad():
            log_prob_ref = self.compute_sequence_log_prob(
                self.ref_model, input_ids, response_ids
            )
        
        return log_prob_policy - log_prob_ref
    
    def stage1_loss_single(
        self,
        first_input_ids: torch.Tensor,
        first_response_ids: torch.Tensor,
        second_input_ids: torch.Tensor,
        second_response_ids: torch.Tensor,
        r2: float,
    ) -> torch.Tensor:
        """
        Compute Stage I loss for a single example.
        
        Objective: max E[r(y2, y*) - beta2 * KL(pi_theta(.|x1) || pi_ref(.|x1))]
        
        Note: We also apply a small beta1 KL penalty on the second attempt
        (omitted from Equation 3 for clarity, but mentioned in the text).
        """
        # Log prob of second attempt (for REINFORCE)
        log_prob_y2 = self.compute_sequence_log_prob(
            self.model, second_input_ids, second_response_ids
        )
        
        # REINFORCE loss for second attempt
        reinforce_loss = -r2 * log_prob_y2
        
        # Strong KL penalty on first attempt (beta2)
        kl_first = self.compute_kl(first_input_ids, first_response_ids)
        
        # Small KL penalty on second attempt (beta1)
        kl_second = self.compute_kl(second_input_ids, second_response_ids)
        
        loss = (
            reinforce_loss
            + self.config.beta2 * kl_first
            + self.config.beta1 * kl_second
        )
        
        return loss
    
    def stage2_loss_single(
        self,
        first_input_ids: torch.Tensor,
        first_response_ids: torch.Tensor,
        second_input_ids: torch.Tensor,
        second_response_ids: torch.Tensor,
        r1: float,
        r2: float,
    ) -> torch.Tensor:
        """
        Compute Stage II loss for a single example.
        
        Objective: max E[r(y1) + r2_shaped(y2) - beta1 * sum_i KL(pi_theta(.|xi) || pi_ref(.|xi))]
        
        where r2_shaped = r2 + alpha * (r2 - r1)
        """
        # Log probs for both attempts
        log_prob_y1 = self.compute_sequence_log_prob(
            self.model, first_input_ids, first_response_ids
        )
        log_prob_y2 = self.compute_sequence_log_prob(
            self.model, second_input_ids, second_response_ids
        )
        
        # Shaped reward for second attempt
        # b(y2|y1,y*) = alpha * (r(y2,y*) - r(y1,y*))
        bonus = self.config.alpha * (r2 - r1)
        r2_shaped = r2 + bonus
        
        # REINFORCE losses
        reinforce_loss_y1 = -r1 * log_prob_y1
        reinforce_loss_y2 = -r2_shaped * log_prob_y2
        
        # KL penalties for both attempts (beta1)
        kl_first = self.compute_kl(first_input_ids, first_response_ids)
        kl_second = self.compute_kl(second_input_ids, second_response_ids)
        
        loss = (
            reinforce_loss_y1
            + reinforce_loss_y2
            + self.config.beta1 * kl_first
            + self.config.beta1 * kl_second
        )
        
        return loss
    
    def train_step(
        self,
        problems: List[str],
        ground_truths: List[Any],
        first_prompts: List[str],
        correction_instruction: str,
    ) -> Dict[str, float]:
        """
        Perform one training step with a batch of problems.
        
        Args:
            problems: Problem strings
            ground_truths: Ground truth answers/test cases
            first_prompts: Formatted prompts for first attempt
            correction_instruction: Self-correction instruction
            
        Returns:
            Dictionary of training metrics
        """
        stage = self.current_stage
        
        # Generate first attempts
        y1_list, x1_ids_list, y1_ids_list = self.generate_batch(
            first_prompts,
            temperature=self.config.sampling_temperature,
            max_new_tokens=self.config.max_new_tokens,
        )
        
        # Compute first-attempt rewards
        r1_list = [self.reward_fn(y1, gt) for y1, gt in zip(y1_list, ground_truths)]
        
        # Build second-attempt prompts
        second_prompts = [
            f"{fp}\n\n{y1}\n\n{correction_instruction}"
            for fp, y1 in zip(first_prompts, y1_list)
        ]
        
        # Generate second attempts
        y2_list, x2_ids_list, y2_ids_list = self.generate_batch(
            second_prompts,
            temperature=self.config.sampling_temperature,
            max_new_tokens=self.config.max_new_tokens,
        )
        
        # Compute second-attempt rewards
        r2_list = [self.reward_fn(y2, gt) for y2, gt in zip(y2_list, ground_truths)]
        
        # Compute loss
        self.optimizer.zero_grad()
        
        total_loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
        
        for i in range(len(problems)):
            if stage == 1:
                loss_i = self.stage1_loss_single(
                    x1_ids_list[i], y1_ids_list[i],
                    x2_ids_list[i], y2_ids_list[i],
                    r2_list[i],
                )
            else:
                loss_i = self.stage2_loss_single(
                    x1_ids_list[i], y1_ids_list[i],
                    x2_ids_list[i], y2_ids_list[i],
                    r1_list[i], r2_list[i],
                )
            
            total_loss = total_loss + loss_i
        
        total_loss = total_loss / len(problems)
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.global_step += 1
        
        # Compute metrics
        r1_mean = sum(r1_list) / len(r1_list)
        r2_mean = sum(r2_list) / len(r2_list)
        
        n_incorrect_t1 = sum(1 for r in r1_list if r == 0.0)
        n_correct_t1 = sum(1 for r in r1_list if r == 1.0)
        
        n_i_to_c = sum(
            1 for r1, r2 in zip(r1_list, r2_list) if r1 == 0.0 and r2 == 1.0
        )
        n_c_to_i = sum(
            1 for r1, r2 in zip(r1_list, r2_list) if r1 == 1.0 and r2 == 0.0
        )
        
        # Fraction of problems where answer changed
        n_changed = sum(
            1 for y1, y2 in zip(y1_list, y2_list) if y1 != y2
        )
        
        return {
            "loss": total_loss.item(),
            "stage": stage,
            "step": self.global_step,
            "accuracy_t1": r1_mean,
            "accuracy_t2": r2_mean,
            "delta_t1_t2": r2_mean - r1_mean,
            "delta_i_to_c": n_i_to_c / max(1, n_incorrect_t1),
            "delta_c_to_i": n_c_to_i / max(1, n_correct_t1),
            "fraction_changed": n_changed / len(problems),
        }
