"""
REINFORCE-style policy gradient training with KL-divergence penalty.

Implements the base RL approach described in Section 3 (Preliminaries), 
based on Ahmadian et al. (2024) "Back to Basics: Revisiting REINFORCE 
Style Optimization for Learning from Human Feedback in LLMs".

Equation 2 from the paper:
    max_θ E_{x_t, y_t ∼ π_θ(·|x_t)} [ r̂(y_t, y*) - β₁ D_KL(π_θ(·|x_t) || π_ref(·|x_t)) ]

This is the core RL training method used in both Stage I and Stage II of SCoRe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class REINFORCEConfig:
    """Configuration for REINFORCE training."""
    beta1: float = 0.01  # KL penalty coefficient for standard RL (Eq. 2)
    beta2: float = 0.1   # KL penalty for first-turn only (Stage I, Eq. 3)
    alpha: float = 10.0  # Reward shaping progress bonus multiplier (Stage II)
    sampling_temperature: float = 1.0
    learning_rate: float = 5e-6
    max_grad_norm: float = 1.0
    discount_gamma: float = 0.0  # Discount factor (gamma=0 means instantaneous rewards)


def compute_kl_divergence(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute KL divergence between policy and reference model.
    
    D_KL(π_θ(·|x) || π_ref(·|x)) = E_{y∼π_θ}[log π_θ(y|x) - log π_ref(y|x)]
    
    Uses the approximation: D_KL ≈ log π_θ(y|x) - log π_ref(y|x)
    which is an unbiased estimator of the KL.
    """
    kl_per_token = log_probs - ref_log_probs
    
    if mask is not None:
        kl_per_token = kl_per_token * mask
        if reduction == "mean":
            return (kl_per_token.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)).mean()
        elif reduction == "sum":
            return kl_per_token.sum(dim=-1).mean()
        else:
            return kl_per_token
    else:
        if reduction == "mean":
            return kl_per_token.mean(dim=-1).mean()
        elif reduction == "sum":
            return kl_per_token.sum(dim=-1).mean()
        else:
            return kl_per_token


class REINFORCEPolicyGradient:
    """
    REINFORCE policy gradient trainer for LLMs.
    
    Implements Ahmadian et al. (2024) style on-policy RL optimization:
        - On-policy sampling from the current model
        - REINFORCE gradient estimation
        - KL-divergence penalty against a reference (frozen) policy
    
    Supports both single-turn and multi-turn RL training as used in SCoRe.
    """
    
    def __init__(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        tokenizer: object,
        config: REINFORCEConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
            )
        else:
            self.optimizer = optimizer
    
    def _get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities of response tokens under the model.
        
        Runs the full sequence (input + response) through the model and
        extracts log probabilities for only the response tokens.
        """
        full_ids = torch.cat([input_ids, response_ids], dim=-1)
        full_mask = torch.cat([
            attention_mask,
            torch.ones_like(response_ids)
        ], dim=-1)
        
        is_current = (model == self.model)
        with torch.set_grad_enabled(is_current):
            outputs = model(input_ids=full_ids, attention_mask=full_mask)
            logits = outputs.logits
        
        # logits at position t predict token at t+1
        # We need logits for the response portion
        shift_logits = logits[:, input_ids.shape[1] - 1:-1, :]
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1,
            index=response_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs
    
    def single_turn_step(
        self,
        batch: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute a single-turn REINFORCE update (Equation 2).
        
        Args:
            batch: {'input_ids', 'attention_mask', 'response_ids'}
            rewards: Binary rewards for each sequence (B,)
        
        Returns:
            loss, metrics_dict
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        response_ids = batch['response_ids']
        
        log_probs = self._get_log_probs(
            self.model, input_ids, attention_mask, response_ids
        )
        
        with torch.no_grad():
            ref_log_probs = self._get_log_probs(
                self.reference_model, input_ids, attention_mask, response_ids
            )
        
        kl_div = compute_kl_divergence(log_probs, ref_log_probs)
        
        # PG loss: -E[reward * log π]
        seq_log_probs = log_probs.sum(dim=-1)
        pg_loss = -(rewards * seq_log_probs).mean()
        kl_loss = self.config.beta1 * kl_div
        
        loss = pg_loss + kl_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        
        self.optimizer.step()
        
        metrics = {
            "pg_loss": pg_loss.item(),
            "kl_loss": kl_loss.item(),
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_kl": kl_div.item(),
        }
        return loss, metrics
    
    def two_turn_step(
        self,
        batch: Dict[str, torch.Tensor],
        rewards_t1: torch.Tensor,
        rewards_t2: torch.Tensor,
        is_stage1: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute a two-turn REINFORCE update for SCoRe.
        
        Implements both Stage I (Equation 3) and Stage II (Equation 4).
        
        Stage I (Eq. 3):
            max_θ E[r̂(y₂, y*) - β₂·D_KL(π_θ(·|x₁) || π_ref(·|x₁))]
        KL is applied only to the first turn, plus default KL on second.
        
        Stage II (Eq. 4) with reward shaping:
            max_θ E[Σ r̂(yᵢ, y*) + α·(r̂(y₂)-r̂(y₁))] 
                  - β₁·(KL_t1 + KL_t2)
        
        The progress bonus b(y₂|y₁) = α·(r̂(y₂) - r̂(y₁)) rewards transitions
        that flip correctness and penalizes changing correct→incorrect.
        """
        # Get log probs for both turns
        log_probs_t1 = self._get_log_probs(
            self.model,
            batch['input_ids_t1'], batch['attention_mask_t1'],
            batch['response_ids_t1'],
        )
        log_probs_t2 = self._get_log_probs(
            self.model,
            batch['input_ids_t2'], batch['attention_mask_t2'],
            batch['response_ids_t2'],
        )
        
        with torch.no_grad():
            ref_log_probs_t1 = self._get_log_probs(
                self.reference_model,
                batch['input_ids_t1'], batch['attention_mask_t1'],
                batch['response_ids_t1'],
            )
            ref_log_probs_t2 = self._get_log_probs(
                self.reference_model,
                batch['input_ids_t2'], batch['attention_mask_t2'],
                batch['response_ids_t2'],
            )
        
        # KL divergences
        kl_t1 = compute_kl_divergence(log_probs_t1, ref_log_probs_t1)
        kl_t2 = compute_kl_divergence(log_probs_t2, ref_log_probs_t2)
        
        seq_log_probs_t1 = log_probs_t1.sum(dim=-1)
        seq_log_probs_t2 = log_probs_t2.sum(dim=-1)
        
        if is_stage1:
            # Stage I: Only optimize second turn, KL constrain first turn
            pg_loss = -(rewards_t2 * seq_log_probs_t2).mean()
            kl_loss = self.config.beta2 * kl_t1 + self.config.beta1 * kl_t2
            loss = pg_loss + kl_loss
            
            metrics = {
                "pg_loss": pg_loss.item(),
                "kl_loss": kl_loss.item(),
                "kl_t1": kl_t1.item(),
                "kl_t2": kl_t2.item(),
                "loss": loss.item(),
                "mean_reward_t1": rewards_t1.mean().item(),
                "mean_reward_t2": rewards_t2.mean().item(),
                "stage": "stage1",
            }
        else:
            # Stage II: Joint optimization with reward shaping
            # Progress bonus: α·(r̂(y₂) - r̂(y₁))
            progress_bonus = self.config.alpha * (rewards_t2 - rewards_t1)
            shaped_reward_t2 = rewards_t2 + progress_bonus
            
            pg_loss_t1 = -(rewards_t1 * seq_log_probs_t1).mean()
            pg_loss_t2 = -(shaped_reward_t2 * seq_log_probs_t2).mean()
            pg_loss = pg_loss_t1 + pg_loss_t2
            
            # Standard KL penalty on both turns
            kl_loss = self.config.beta1 * (kl_t1 + kl_t2)
            loss = pg_loss + kl_loss
            
            metrics = {
                "pg_loss": pg_loss.item(),
                "pg_loss_t1": pg_loss_t1.item(),
                "pg_loss_t2": pg_loss_t2.item(),
                "kl_loss": kl_loss.item(),
                "kl_t1": kl_t1.item(),
                "kl_t2": kl_t2.item(),
                "loss": loss.item(),
                "mean_reward_t1": rewards_t1.mean().item(),
                "mean_reward_t2": rewards_t2.mean().item(),
                "mean_shaped_reward_t2": shaped_reward_t2.mean().item(),
                "mean_progress_bonus": progress_bonus.mean().item(),
                "stage": "stage2",
            }
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        
        self.optimizer.step()
        
        return loss, metrics


class RewardCalculator:
    """
    Compute binary rewards for model responses.
    
    For MATH: checks if the model's final answer matches ground truth.
    For coding: checks if the model's code passes all test cases.
    
    Implements the oracle reward r̂(y, y*) from Section 3.
    """
    
    @staticmethod
    def extract_final_answer_math(text: str) -> Optional[str]:
        """Extract final answer from MATH-style response."""
        import re
        pattern = r"Final Answer:\s*The final answer is\s*(.+?)(?:\.\s*I hope it is correct\.)?"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    @staticmethod
    def extract_final_answer_code(text: str) -> Optional[str]:
        """Extract Python code from a code generation response."""
        import re
        pattern = r"```python\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        pattern = r"```\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
    
    @staticmethod
    def check_math_answer(predicted: str, ground_truth: str) -> float:
        """Compare predicted answer with ground truth. Returns 1.0 or 0.0."""
        import re
        
        pred = predicted.strip().lower()
        gt = ground_truth.strip().lower()
        
        if pred == gt:
            return 1.0
        
        # Numeric comparison
        try:
            pred_nums = re.findall(r'-?\d+\.?\d*', pred)
            gt_nums = re.findall(r'-?\d+\.?\d*', gt)
            if pred_nums and gt_nums:
                pred_val = float(pred_nums[-1])
                gt_val = float(gt_nums[-1])
                if abs(pred_val - gt_val) < 1e-6:
                    return 1.0
        except (ValueError, IndexError):
            pass
        
        # Whitespace-insensitive matching
        pred_clean = re.sub(r'\s+', '', pred)
        gt_clean = re.sub(r'\s+', '', gt)
        if pred_clean == gt_clean:
            return 1.0
        
        return 0.0
    
    @staticmethod
    def check_code_correctness(code: str, test_cases: List[Dict]) -> float:
        """
        Check if generated code passes all test cases.
        Returns 1.0 if all pass, 0.0 otherwise.
        """
        import subprocess
        import tempfile
        import sys
        import os
        
        if not test_cases:
            return 0.0
        
        test_script = code + "\n\n"
        for i, test in enumerate(test_cases):
            inp = test.get('input', '')
            exp = test.get('expected', '')
            if inp:
                test_script += f"assert {inp} == {exp}, f'Test {i} failed'\n"
        
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False
            ) as f:
                f.write(test_script)
                temp_path = f.name
            
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True, text=True, timeout=10,
            )
            os.unlink(temp_path)
            return 1.0 if result.returncode == 0 else 0.0
        except Exception:
            return 0.0


def compute_reinforce_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    kl_div: torch.Tensor,
    beta: float,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the REINFORCE loss with KL penalty.
    
    Loss = -E[advantage * log π] + β * KL
    """
    if mask is not None:
        seq_log_probs = (log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    else:
        seq_log_probs = log_probs.sum(dim=-1)
    
    pg_loss = -(advantages * seq_log_probs).mean()
    
    if isinstance(kl_div, torch.Tensor):
        kl_loss = beta * kl_div
    else:
        kl_loss = beta * kl_div
    
    loss = pg_loss + kl_loss
    
    metrics = {
        "pg_loss": pg_loss.item(),
        "kl_div": kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div,
        "loss": loss.item(),
        "mean_advantage": advantages.mean().item(),
    }
    return loss, metrics


__all__ = [
    "REINFORCEConfig",
    "REINFORCEPolicyGradient",
    "RewardCalculator",
    "compute_kl_divergence",
    "compute_reinforce_loss",
]
