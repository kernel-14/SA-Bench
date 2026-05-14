"""
MA-RLHF Trainer: Main training loop for Macro-Action RLHF.

This module orchestrates the complete MA-RLHF training pipeline:
1. SFT (Supervised Fine-Tuning)
2. Reward Model Training
3. MA-PPO Training (the core contribution)

Integrates with the HuggingFace transformers ecosystem and supports
DeepSpeed integration as described in the paper (Appendix B.2).
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Any, Literal
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Local imports
from ma_rlhf.termination import get_macro_action_positions
from ma_rlhf.value_estimation import get_macro_action_values
from ma_rlhf.ma_ppo import (
    policy_loss_macro_action,
    policy_loss_macro_action_joint,
    critic_loss_macro_action,
    compute_macro_action_returns_and_advantages,
    compute_macro_rewards,
)
from ma_rlhf.rlhf_utils import (
    compute_shaped_reward,
    compute_reward_model_loss,
    compute_program_synthesis_reward,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class MAConfig:
    """Configuration for Macro-Action RLHF training."""
    # Model
    model_name_or_path: str = "google/gemma-2b"
    model_size: str = "2B"
    
    # Data
    dataset_name: str = "tldr"  # tldr, hhrlhf, webgpt, apps
    max_prompt_length: int = 512
    max_response_length: int = 512
    
    # SFT
    sft_batch_size: int = 512
    sft_epochs: int = 3
    sft_learning_rate: float = 5e-5
    sft_warmup_ratio: float = 0.1
    
    # RM
    rm_batch_size: int = 64
    rm_epochs: int = 1
    rm_learning_rate: float = 1e-5
    
    # PPO / MA-PPO
    ppo_batch_size: int = 256
    policy_learning_rate: float = 1.5e-5
    critic_learning_rate: float = 1.5e-5
    ppo_epochs: int = 1
    rollout: int = 1
    clip_ratio: float = 0.2
    gamma: float = 1.0       # γ in GAE
    lam: float = 0.95         # λ in GAE
    kl_coefficient: float = 0.05  # β
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = 50
    warmup_steps: int = 200
    
    # Macro Action Settings
    use_macro_actions: bool = True
    macro_termination: str = "ngram"  # ngram, randomized_ngram, ppl, parser
    n_gram: int = 5  # Default n-gram length
    value_assignment: str = "equal"  # equal, unit, position_decayed
    cutoff: int = 5  # For parsing-based termination
    
    # Training
    total_steps: int = 4600
    save_interval: int = 200
    eval_interval: int = 200
    seed: int = 42
    
    # Output
    output_dir: str = "./output"
    use_deepspeed: bool = False


# ---------------------------------------------------------------------------
# Macro Action Scheduler
# ---------------------------------------------------------------------------

class MacroActionScheduler:
    """
    Manages macro action computation during MA-PPO training.
    
    This encapsulates the full pipeline:
    1. Get macro action boundaries (termination)
    2. Compute macro action values (value estimation)
    3. Compute macro action rewards
    4. Compute advantages and returns via GAE
    5. Compute policy and critic losses
    """
    
    def __init__(self, config: MAConfig):
        self.config = config
        self.termination = config.macro_termination
        self.n_gram = config.n_gram
        self.value_assignment = config.value_assignment
        self.cutoff = config.cutoff
        self.gamma = config.gamma
        self.lam = config.lam
        self.clip_ratio = config.clip_ratio
    
    def get_macro_boundaries(
        self,
        start: int,
        mask: torch.Tensor,
        ppl: Optional[torch.Tensor] = None,
        parse_tree = None,
    ) -> List[int]:
        """Determine macro action boundary positions."""
        if not self.config.use_macro_actions:
            # Vanilla PPO: each token is its own macro action
            seq_len = mask.size(1)
            return list(range(start, seq_len))
        
        return get_macro_action_positions(
            start=start,
            mask=mask,
            termination=self.termination,
            n_gram=self.n_gram if self.termination == 'ngram' else None,
            ppl=ppl,
            cutoff=self.cutoff,
            parse_tree=parse_tree,
        )
    
    def compute_macro_values(
        self,
        token_values: torch.Tensor,
        mask: torch.Tensor,
        start: int,
        sequence: List[int],
    ) -> torch.Tensor:
        """Aggregate token values into macro action values."""
        if not self.config.use_macro_actions:
            # With n=1 (vanilla PPO), each token is its own macro action
            return token_values[:, start:]
        
        return get_macro_action_values(
            values=token_values,
            mask=mask,
            start=start,
            sequence=sequence,
            value_assignment=self.value_assignment,
        )
    
    def compute_macro_rewards(
        self,
        token_rewards: torch.Tensor,
        mask: torch.Tensor,
        start: int,
        sequence: List[int],
    ) -> torch.Tensor:
        """Aggregate token rewards into macro action rewards."""
        if not self.config.use_macro_actions:
            return token_rewards[:, start:]
        
        return compute_macro_rewards(
            token_rewards=token_rewards,
            mask=mask,
            start=start,
            sequence=sequence,
            rho=1.0,  # ρ = 1 as per paper
        )
    
    def compute_advantages_returns(
        self,
        macro_values: torch.Tensor,
        macro_rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns using GAE at macro level."""
        return compute_macro_action_returns_and_advantages(
            macro_values=macro_values,
            macro_rewards=macro_rewards,
            gamma=self.gamma,
            lam=self.lam,
        )
    
    def compute_policy_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        sequence: List[int],
    ) -> torch.Tensor:
        """Compute MA-PPO policy loss."""
        return policy_loss_macro_action(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask,
            sequence=sequence,
            cliprange=self.clip_ratio,
        )
    
    def compute_critic_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor,
        sequence: List[int],
    ) -> torch.Tensor:
        """Compute MA-PPO critic loss."""
        return critic_loss_macro_action(
            values=values,
            old_values=old_values,
            returns=returns,
            mask=mask,
            sequence=sequence,
        )


# ---------------------------------------------------------------------------
# Data Utilities
# ---------------------------------------------------------------------------

class RLHFDataset(Dataset):
    """
    Dataset for RLHF training stages.
    
    Handles:
    - SFT: prompt + chosen response pairs
    - RM: prompt + chosen + rejected triples
    - PPO: prompts for generation
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        stage: str = "sft",  # sft, rm, ppo
        max_prompt_length: int = 512,
        max_response_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.stage = stage
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        
        if self.stage == "sft":
            response = item["chosen"]
            full_text = prompt + response
            encoded = self.tokenizer(
                full_text,
                max_length=self.max_prompt_length + self.max_response_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": encoded["input_ids"].squeeze(0),
            }
        
        elif self.stage == "rm":
            chosen = item["chosen"]
            rejected = item["rejected"]
            
            encoded_prompt = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=self.max_prompt_length
            )
            
            encoded_chosen = self.tokenizer(
                prompt + chosen, return_tensors="pt", truncation=True,
                max_length=self.max_prompt_length + self.max_response_length
            )
            
            encoded_rejected = self.tokenizer(
                prompt + rejected, return_tensors="pt", truncation=True,
                max_length=self.max_prompt_length + self.max_response_length
            )
            
            return {
                "input_ids_chosen": encoded_chosen["input_ids"].squeeze(0),
                "attention_mask_chosen": encoded_chosen["attention_mask"].squeeze(0),
                "input_ids_rejected": encoded_rejected["input_ids"].squeeze(0),
                "attention_mask_rejected": encoded_rejected["attention_mask"].squeeze(0),
            }
        
        elif self.stage == "ppo":
            encoded = self.tokenizer(
                prompt,
                max_length=self.max_prompt_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }


# ---------------------------------------------------------------------------
# MA-RLHF Trainer
# ---------------------------------------------------------------------------

class MARLHFTrainer:
    """
    Main trainer for the MA-RLHF pipeline.
    
    Manages the three-stage training process:
    1. SFT training
    2. Reward model training  
    3. MA-PPO training
    """
    
    def __init__(
        self,
        config: MAConfig,
        policy_model,
        reference_model,
        reward_model,
        critic_model,
        tokenizer,
    ):
        self.config = config
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.critic_model = critic_model
        self.tokenizer = tokenizer
        
        # Initialize macro action scheduler
        self.ma_scheduler = MacroActionScheduler(config)
        
        # Training state
        self.global_step = 0
        
    def generate_response(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a response using the policy model.
        
        Returns the full sequence (prompt + response) with log probabilities.
        """
        self.policy_model.eval()
        
        with torch.no_grad():
            outputs = self.policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_response_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        return {
            "sequences": outputs.sequences,
            "scores": outputs.scores,
        }
    
    def get_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get token-level log probabilities from a model.
        
        Returns:
            log_probs: Log probabilities for each token position.
            values: Value predictions (if model has a value head).
        """
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs of the actual tokens
        shifted_input_ids = input_ids[:, 1:]
        token_log_probs = torch.gather(
            log_probs[:, :-1, :], dim=-1, index=shifted_input_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs, None  # values from separate critic
    
    def get_rm_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get reward model score for a sequence."""
        self.reward_model.eval()
        with torch.no_grad():
            outputs = self.reward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        # RM score is the output at the last token position
        return outputs.logits[:, -1].squeeze(-1)
    
    def compute_ppo_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Execute one MA-PPO training step.
        
        Pipeline:
        1. Generate responses from policy model
        2. Get RM scores
        3. Get log probs from policy and reference models
        4. Compute shaped rewards with KL penalty
        5. Get macro action boundaries
        6. Compute macro values and rewards
        7. Compute advantages and returns via GAE
        8. Compute policy and critic losses
        9. Backward pass and optimization
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # Step 1: Generate responses
        gen_outputs = self.generate_response(input_ids, attention_mask)
        full_sequences = gen_outputs["sequences"]
        
        # Create full attention mask
        prompt_len = input_ids.size(1)
        response_len = full_sequences.size(1) - prompt_len
        response_mask = torch.ones(
            input_ids.size(0), response_len,
            dtype=attention_mask.dtype, device=attention_mask.device
        )
        full_attention_mask = torch.cat([attention_mask, response_mask], dim=1)
        
        # Step 2: Get RM scores
        rm_scores = self.get_rm_score(full_sequences, full_attention_mask)
        
        # Step 3: Get log probs
        policy_log_probs, _ = self.get_log_probs(
            self.policy_model, full_sequences, full_attention_mask
        )
        ref_log_probs, _ = self.get_log_probs(
            self.reference_model, full_sequences, full_attention_mask
        )
        
        # Step 4: Compute shaped rewards
        response_log_probs = policy_log_probs[:, prompt_len-1:]
        ref_response_log_probs = ref_log_probs[:, prompt_len-1:]
        shaped_rewards = compute_shaped_reward(
            rm_scores=rm_scores,
            log_probs=response_log_probs,
            ref_log_probs=ref_response_log_probs,
            mask=response_mask,
            beta=self.config.kl_coefficient,
        )
        
        # Step 5: Get critic values
        old_values = self.critic_model(
            input_ids=full_sequences,
            attention_mask=full_attention_mask,
        ).logits
        old_response_values = old_values[:, prompt_len-1:]
        
        # Step 6: Get macro action boundaries
        start = prompt_len - 1
        sequence = self.ma_scheduler.get_macro_boundaries(
            start=start,
            mask=torch.cat([
                torch.ones(input_ids.size(0), prompt_len, device=attention_mask.device),
                response_mask
            ], dim=1),
        )
        
        # Step 7: Compute macro values and rewards
        macro_old_values = self.ma_scheduler.compute_macro_values(
            token_values=old_response_values,
            mask=response_mask,
            start=0,  # relative to response
            sequence=[s - start for s in sequence],
        )
        
        macro_old_rewards = self.ma_scheduler.compute_macro_rewards(
            token_rewards=shaped_rewards,
            mask=response_mask,
            start=0,
            sequence=[s - start for s in sequence],
        )
        
        # Step 8: Compute advantages and returns
        advantages, returns = self.ma_scheduler.compute_advantages_returns(
            macro_values=macro_old_values,
            macro_rewards=macro_old_rewards,
        )
        
        # Step 9: Compute losses and optimize
        # Get current log probs and values
        current_log_probs, _ = self.get_log_probs(
            self.policy_model, full_sequences, full_attention_mask
        )
        current_values = self.critic_model(
            input_ids=full_sequences,
            attention_mask=full_attention_mask,
        ).logits
        
        current_response_log_probs = current_log_probs[:, prompt_len-1:]
        current_response_values = current_values[:, prompt_len-1:]
        
        policy_loss = self.ma_scheduler.compute_policy_loss(
            logprobs=current_response_log_probs,
            old_logprobs=response_log_probs,
            advantages=advantages,
            mask=response_mask,
            sequence=[s - start for s in sequence],
        )
        
        value_loss = self.ma_scheduler.compute_critic_loss(
            values=current_response_values,
            old_values=old_response_values,
            returns=returns,
            mask=response_mask,
            sequence=[s - start for s in sequence],
        )
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "rm_score": rm_scores.mean().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_l2": torch.norm(advantages).item(),
        }


# ---------------------------------------------------------------------------
# Entry Points
# ---------------------------------------------------------------------------

def create_config_from_args(args: Optional[List[str]] = None) -> MAConfig:
    """Create MAConfig from command-line arguments or defaults."""
    parser = argparse.ArgumentParser(description="MA-RLHF Training")
    
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-2b")
    parser.add_argument("--dataset", type=str, default="tldr",
                        choices=["tldr", "hhrlhf", "webgpt", "apps"])
    parser.add_argument("--output_dir", type=str, default="./output")
    
    # Macro action specific
    parser.add_argument("--use_macro_actions", action="store_true", default=True)
    parser.add_argument("--macro_termination", type=str, default="ngram",
                        choices=["ngram", "randomized_ngram", "ppl", "parser"])
    parser.add_argument("--n_gram", type=int, default=5)
    parser.add_argument("--value_assignment", type=str, default="equal",
                        choices=["equal", "unit", "position_decayed"])
    
    # Training
    parser.add_argument("--ppo_batch_size", type=int, default=256)
    parser.add_argument("--policy_learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--critic_learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--kl_coefficient", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--total_steps", type=int, default=4600)
    
    if args:
        parsed = parser.parse_args(args)
    else:
        parsed = parser.parse_args([])
    
    config = MAConfig()
    for key, value in vars(parsed).items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
