"""
SFT-based baselines for self-correction training.

Implements the STaR and Pair-SFT approaches analyzed in Section 4:
"SFT on Self-Generated Data is Insufficient for Self-Correction"

STaR (Zelikman et al., 2022): 
    Filter self-correction traces to retain only successful corrections,
    then run SFT on the filtered data. Iterative: re-sample with updated model.

Pair-SFT (based on Welleck et al., 2023):
    Construct synthetic repair traces by pairing incorrect responses 
    with correct ones. Does NOT train a separate corrector model.
    Runs only one iteration.

These baselines are analyzed in Section 4 to demonstrate the failure modes
of SFT-based approaches: distribution shift and behavior collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import logging

from .reinforce import RewardCalculator

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT-based self-correction training."""
    learning_rate: float = 5e-6
    max_grad_norm: float = 1.0
    batch_size: int = 512
    num_epochs: int = 3
    max_prompt_length: int = 2048
    max_response_length: int = 2048
    task: str = "math"
    temperature: float = 1.0


class STaRTrainer:
    """
    STaR: Self-Taught Reasoner approach applied to self-correction.
    
    Algorithm (Section 4):
    1. Sample self-correction traces from the model
    2. Filter to retain only traces where:
       - Turn 1 is incorrect
       - Turn 2 is correct (successful self-correction)
    3. Optionally add "correct-to-correct" traces (D_STaR^+)
    4. Run SFT on the filtered data
    5. Repeat for multiple iterations
    
    The paper uses 3 iterations following Singh et al. (2024).
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: object,
        config: SFTConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_calc = RewardCalculator()
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=config.learning_rate
            )
        else:
            self.optimizer = optimizer
    
    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 1.0,
    ) -> str:
        """Generate a response from the model."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.config.max_prompt_length,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=(temperature > 0),
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        return self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
    
    def collect_traces(
        self,
        problems: List[Dict],
        prompts_t1: List[str],
        prompts_t2: List[str],
        include_correct_to_correct: bool = False,
    ) -> List[Dict]:
        """
        Collect self-correction traces and filter for successful corrections.
        
        Args:
            problems: List of {'problem': str, 'answer': Any}
            prompts_t1: First-turn prompts
            prompts_t2: Second-turn prompts (with correction instruction)
            include_correct_to_correct: Whether to include "correct→correct" traces
        
        Returns:
            List of training traces with prompt, response pairs
        """
        traces = []
        
        for prob, p1, p2 in zip(problems, prompts_t1, prompts_t2):
            # Generate turn 1
            resp1 = self.generate_response(p1, temperature=self.config.temperature)
            reward1 = self._compute_reward(resp1, prob['answer'])
            
            # Generate turn 2 (self-correction)
            # Build full turn 2 prompt with turn 1 response
            full_p2 = p2 + "\n\n" + resp1
            
            resp2 = self.generate_response(full_p2, temperature=self.config.temperature)
            reward2 = self._compute_reward(resp2, prob['answer'])
            
            # D_STaR: retain only successful corrections (incorrect → correct)
            if reward1 <= 0.5 and reward2 > 0.5:
                traces.append({
                    "prompt_t1": p1,
                    "response_t1": resp1,
                    "prompt_t2": full_p2,
                    "response_t2": resp2,
                    "is_correction": True,
                })
            # D_STaR^+: also include correct → correct
            elif include_correct_to_correct and reward1 > 0.5 and reward2 > 0.5:
                traces.append({
                    "prompt_t1": p1,
                    "response_t1": resp1,
                    "prompt_t2": full_p2,
                    "response_t2": resp2,
                    "is_correction": False,
                })
        
        return traces
    
    def _compute_reward(self, response: str, ground_truth: Any) -> float:
        """Compute reward for a response."""
        if self.config.task == "math":
            answer = self.reward_calc.extract_final_answer_math(response)
            if answer is None:
                return 0.0
            return self.reward_calc.check_math_answer(answer, str(ground_truth))
        else:
            code = self.reward_calc.extract_final_answer_code(response)
            if isinstance(ground_truth, dict):
                test_cases = [ground_truth]
            else:
                test_cases = ground_truth
            return self.reward_calc.check_code_correctness(code, test_cases)
    
    def sft_step(
        self,
        prompt: str,
        target_response: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single SFT step: maximize log-likelihood of target response given prompt.
        """
        # Tokenize
        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        target_tokens = self.tokenizer(
            target_response,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_response_length,
        )
        
        device = next(self.model.parameters()).device
        input_ids = prompt_tokens['input_ids'].to(device)
        attention_mask = prompt_tokens['attention_mask'].to(device)
        target_ids = target_tokens['input_ids'].to(device)
        
        # Concatenate
        full_ids = torch.cat([input_ids, target_ids], dim=-1)
        full_mask = torch.cat([
            attention_mask,
            torch.ones_like(target_ids)
        ], dim=-1)
        
        # Labels: -100 for prompt, target_ids for response
        labels = torch.cat([
            torch.full_like(input_ids, -100),
            target_ids,
        ], dim=-1)
        
        outputs = self.model(
            input_ids=full_ids,
            attention_mask=full_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        
        self.optimizer.step()
        
        return loss, {"sft_loss": loss.item()}
    
    def run_iteration(
        self,
        problems: List[Dict],
        prompts_t1: List[str],
        prompts_t2: List[str],
        iteration: int = 0,
        include_correct_to_correct: bool = False,
    ) -> Dict[str, Any]:
        """
        Run one iteration of STaR:
        1. Collect traces
        2. Train on filtered traces
        """
        logger.info(f"STaR iteration {iteration}: collecting traces")
        
        traces = self.collect_traces(
            problems, prompts_t1, prompts_t2,
            include_correct_to_correct=include_correct_to_correct,
        )
        
        logger.info(f"Collected {len(traces)} traces")
        
        if len(traces) == 0:
            logger.warning("No traces collected, skipping iteration")
            return {"num_traces": 0}
        
        # Train on traces
        self.model.train()
        losses = []
        
        # Train on turn 2 (correction) and optionally turn 1
        num_batches = (len(traces) + self.config.batch_size - 1) // self.config.batch_size
        
        for epoch in range(self.config.num_epochs):
            indices = np.random.permutation(len(traces))
            for b in range(num_batches):
                start = b * self.config.batch_size
                end = min(start + self.config.batch_size, len(traces))
                batch_indices = indices[start:end]
                
                for idx in batch_indices:
                    trace = traces[idx]
                    # Train on both turns
                    loss1, _ = self.sft_step(
                        trace['prompt_t1'], trace['response_t1']
                    )
                    loss2, _ = self.sft_step(
                        trace['prompt_t2'], trace['response_t2']
                    )
                    losses.append((loss1.item() + loss2.item()) / 2)
        
        self.model.eval()
        
        return {
            "num_traces": len(traces),
            "mean_loss": np.mean(losses),
        }
    
    def train(
        self,
        problems: List[Dict],
        prompts_t1: List[str],
        prompts_t2: List[str],
        num_iterations: int = 3,
        include_correct_to_correct: bool = False,
    ) -> Dict[str, Any]:
        """Run multiple STaR iterations."""
        history = []
        for it in range(num_iterations):
            result = self.run_iteration(
                problems, prompts_t1, prompts_t2,
                iteration=it,
                include_correct_to_correct=include_correct_to_correct,
            )
            history.append(result)
            logger.info(f"Iteration {it}: {result}")
        return {"history": history}


class PairSFTTrainer:
    """
    Pair-SFT: Train on synthetically paired repair traces.
    
    Based on Welleck et al. (2023) but without a separate corrector model.
    Described in Section 4:
    
    1. Generate first-attempt responses from base model
    2. If incorrect, pair with a correct response (from the dataset or model)
    3. Run SFT on these (incorrect → correct) pairs
    4. Optionally add (correct → correct) pairs (D_SFT^+)
    
    Only one iteration is run, following the protocol in Welleck et al. (2023).
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: object,
        config: SFTConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_calc = RewardCalculator()
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=config.learning_rate
            )
        else:
            self.optimizer = optimizer
    
    def generate_response(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate a response from the model."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.config.max_prompt_length,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        return self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
    
    def construct_dataset(
        self,
        problems: List[Dict],
        prompts_t1: List[str],
        prompts_t2: List[str],
        ground_truth_solutions: Optional[List[str]] = None,
        include_correct_to_correct: bool = False,
    ) -> List[Dict]:
        """
        Construct the Pair-SFT dataset by pairing incorrect first attempts 
        with correct target responses.
        
        For D_SFT: pairs incorrect→correct
        For D_SFT^+: also includes correct→correct pairs
        """
        dataset = []
        
        for i, (prob, p1, p2) in enumerate(zip(problems, prompts_t1, prompts_t2)):
            resp1 = self.generate_response(p1)
            reward1 = self._compute_reward(resp1, prob['answer'])
            
            if reward1 <= 0.5:
                # Incorrect first attempt → pair with correct
                if ground_truth_solutions and i < len(ground_truth_solutions):
                    target = ground_truth_solutions[i]
                else:
                    # Generate a correct solution
                    target = self._generate_correct_solution(p1, prob['answer'])
                
                dataset.append({
                    "prompt_t1": p1,
                    "response_t1": resp1,
                    "prompt_t2": p2 + "\n\n" + resp1,
                    "response_t2": target,
                })
            elif include_correct_to_correct:
                # Correct first attempt → pair with itself
                dataset.append({
                    "prompt_t1": p1,
                    "response_t1": resp1,
                    "prompt_t2": p2 + "\n\n" + resp1,
                    "response_t2": resp1,  # Keep correct
                })
        
        return dataset
    
    def _compute_reward(self, response: str, ground_truth: Any) -> float:
        if self.config.task == "math":
            answer = self.reward_calc.extract_final_answer_math(response)
            if answer is None:
                return 0.0
            return self.reward_calc.check_math_answer(answer, str(ground_truth))
        else:
            code = self.reward_calc.extract_final_answer_code(response)
            if isinstance(ground_truth, dict):
                test_cases = [ground_truth]
            else:
                test_cases = ground_truth
            return self.reward_calc.check_code_correctness(code, test_cases)
    
    def _generate_correct_solution(
        self, prompt: str, ground_truth: Any, max_attempts: int = 10
    ) -> str:
        """Try to generate a correct solution by repeated sampling."""
        for _ in range(max_attempts):
            response = self.generate_response(prompt)
            reward = self._compute_reward(response, ground_truth)
            if reward > 0.5:
                return response
        # Fallback: return ground truth formatted
        return f"The answer is {ground_truth}."
    
    def train(
        self,
        dataset: List[Dict],
        num_epochs: int = 3,
    ) -> Dict[str, Any]:
        """Train on the constructed Pair-SFT dataset."""
        self.model.train()
        losses = []
        
        num_batches = (len(dataset) + self.config.batch_size - 1) // self.config.batch_size
        
        for epoch in range(num_epochs):
            indices = np.random.permutation(len(dataset))
            pbar = tqdm(range(num_batches), desc=f"Pair-SFT epoch {epoch}")
            
            for b in pbar:
                start = b * self.config.batch_size
                end = min(start + self.config.batch_size, len(dataset))
                batch_indices = indices[start:end]
                
                batch_losses = []
                for idx in batch_indices:
                    trace = dataset[idx]
                    
                    # Tokenize turn 1
                    t1_in = self.tokenizer(
                        trace['prompt_t1'], return_tensors="pt",
                        truncation=True, max_length=self.config.max_prompt_length
                    )
                    t1_tgt = self.tokenizer(
                        trace['response_t1'], return_tensors="pt",
                        truncation=True, max_length=self.config.max_response_length
                    )
                    
                    # Tokenize turn 2
                    t2_in = self.tokenizer(
                        trace['prompt_t2'], return_tensors="pt",
                        truncation=True, max_length=self.config.max_prompt_length
                    )
                    t2_tgt = self.tokenizer(
                        trace['response_t2'], return_tensors="pt",
                        truncation=True, max_length=self.config.max_response_length
                    )
                    
                    device = next(self.model.parameters()).device
                    
                    # Turn 1 loss
                    full1 = torch.cat([t1_in['input_ids'].to(device), t1_tgt['input_ids'].to(device)], dim=-1)
                    mask1 = torch.cat([t1_in['attention_mask'].to(device), torch.ones_like(t1_tgt['input_ids']).to(device)], dim=-1)
                    labels1 = torch.cat([torch.full_like(t1_in['input_ids'], -100), t1_tgt['input_ids'].to(device)], dim=-1)
                    
                    out1 = self.model(input_ids=full1, attention_mask=mask1, labels=labels1)
                    
                    # Turn 2 loss
                    full2 = torch.cat([t2_in['input_ids'].to(device), t2_tgt['input_ids'].to(device)], dim=-1)
                    mask2 = torch.cat([t2_in['attention_mask'].to(device), torch.ones_like(t2_tgt['input_ids']).to(device)], dim=-1)
                    labels2 = torch.cat([torch.full_like(t2_in['input_ids'], -100), t2_tgt['input_ids'].to(device)], dim=-1)
                    
                    out2 = self.model(input_ids=full2, attention_mask=mask2, labels=labels2)
                    
                    loss = out1.loss + out2.loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    batch_losses.append(loss.item())
                
                pbar.set_postfix({"loss": np.mean(batch_losses)})
                losses.extend(batch_losses)
        
        self.model.eval()
        
        return {
            "mean_loss": np.mean(losses),
            "num_traces": len(dataset),
        }


__all__ = ["SFTConfig", "STaRTrainer", "PairSFTTrainer"]
