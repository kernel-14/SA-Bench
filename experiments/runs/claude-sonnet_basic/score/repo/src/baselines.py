"""
Baseline methods for comparison with SCoRe.

Implements:
1. STaR (Zelikman et al., 2022) - SFT on successful correction traces
2. Pair-SFT (Welleck et al., 2023 variant) - SFT on paired incorrect/correct responses
3. Self-Refine (Madaan et al., 2023) - Prompting-based self-correction

From Section 4 of the paper.
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    """Configuration for baseline methods."""
    learning_rate: float = 5e-6
    batch_size: int = 32
    num_epochs: int = 3
    max_seq_len: int = 2048
    task_type: str = "math"  # "math" or "code"
    
    # STaR specific
    star_iterations: int = 3  # Paper uses 3 iterations for STaR
    
    # Whether to include correct->correct pairs (D+ variants)
    include_correct_to_correct: bool = False


class STaRTrainer:
    """
    STaR (Self-Taught Reasoner) baseline for self-correction.
    
    From Zelikman et al. (2022), adapted for self-correction as in Section 4.
    
    Algorithm:
    1. Sample two-turn traces from the model
    2. Filter to keep only traces where:
       - First attempt is incorrect
       - Second attempt is correct (successful correction)
    3. Run SFT on these filtered traces
    4. Repeat for multiple iterations
    
    Dataset D_STaR: successful correction traces (incorrect -> correct)
    Dataset D_STaR+: also includes correct -> correct traces
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: BaselineConfig,
        reward_fn,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)
    
    def collect_star_data(
        self,
        problems: List[str],
        ground_truths: List[str],
        first_prompts: List[str],
        correction_instruction: str,
        temperature: float = 1.0,
    ) -> List[Dict]:
        """
        Collect and filter self-correction traces for STaR.
        
        Keeps traces where first attempt is wrong and second is correct.
        If include_correct_to_correct=True, also keeps correct->correct traces.
        
        Returns:
            List of dicts with 'prompt', 'response', 'type' keys
        """
        training_data = []
        
        for problem, gt, first_prompt in zip(problems, ground_truths, first_prompts):
            # Generate first attempt
            inputs = self.tokenizer(
                first_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_len,
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            input_len = inputs["input_ids"].shape[1]
            y1 = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            r1 = self.reward_fn(y1, gt)
            
            # Build second-attempt prompt
            second_prompt = f"{first_prompt}\n\n{y1}\n\n{correction_instruction}"
            
            # Generate second attempt
            inputs2 = self.tokenizer(
                second_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_len,
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs2 = self.model.generate(
                    **inputs2,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            input_len2 = inputs2["input_ids"].shape[1]
            y2 = self.tokenizer.decode(outputs2[0][input_len2:], skip_special_tokens=True)
            r2 = self.reward_fn(y2, gt)
            
            # STaR filtering: keep incorrect->correct traces
            if r1 == 0.0 and r2 == 1.0:
                # Add both turns as training data
                training_data.append({
                    "prompt": first_prompt,
                    "response": y1,
                    "type": "first_attempt_incorrect",
                })
                training_data.append({
                    "prompt": second_prompt,
                    "response": y2,
                    "type": "second_attempt_correct",
                })
            
            # D_STaR+ variant: also include correct->correct
            if self.config.include_correct_to_correct and r1 == 1.0 and r2 == 1.0:
                training_data.append({
                    "prompt": first_prompt,
                    "response": y1,
                    "type": "first_attempt_correct",
                })
                training_data.append({
                    "prompt": second_prompt,
                    "response": y2,
                    "type": "second_attempt_correct",
                })
        
        return training_data
    
    def sft_step(self, prompt: str, response: str) -> torch.Tensor:
        """
        Compute SFT loss (negative log likelihood) for a single example.
        
        Args:
            prompt: Input prompt
            response: Target response
            
        Returns:
            Scalar loss
        """
        # Tokenize
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_len,
        ).to(self.model.device)
        
        response_ids = self.tokenizer(
            response,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)
        
        # Concatenate
        full_ids = torch.cat(
            [prompt_ids["input_ids"], response_ids["input_ids"]], dim=1
        )
        
        # Forward pass
        outputs = self.model(input_ids=full_ids)
        logits = outputs.logits
        
        # Compute loss only on response tokens
        input_len = prompt_ids["input_ids"].shape[1]
        resp_logits = logits[:, input_len - 1:-1, :]
        
        loss = F.cross_entropy(
            resp_logits.reshape(-1, resp_logits.shape[-1]),
            response_ids["input_ids"].reshape(-1),
        )
        
        return loss
    
    def train_iteration(
        self,
        training_data: List[Dict],
    ) -> Dict[str, float]:
        """
        Run one SFT training iteration on collected data.
        
        Args:
            training_data: List of (prompt, response) pairs
            
        Returns:
            Training metrics
        """
        total_loss = 0.0
        n_steps = 0
        
        # Shuffle data
        import random
        random.shuffle(training_data)
        
        for item in training_data:
            self.optimizer.zero_grad()
            loss = self.sft_step(item["prompt"], item["response"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_steps += 1
        
        return {
            "loss": total_loss / max(1, n_steps),
            "n_training_examples": len(training_data),
        }


class PairSFTTrainer:
    """
    Pair-SFT baseline: SFT on synthetically paired incorrect/correct responses.
    
    Based on Welleck et al. (2023), adapted as a single-model approach.
    
    Algorithm:
    1. Sample responses from the base model
    2. For each incorrect response, pair it with a correct response to the same problem
    3. Train on these (incorrect, correct) pairs as self-correction traces
    
    Dataset D_SFT: pairs of (incorrect first attempt, correct second attempt)
    Dataset D_SFT+: also includes (correct first attempt, correct second attempt)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: BaselineConfig,
        reward_fn,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)
    
    def build_pair_sft_dataset(
        self,
        problems: List[str],
        ground_truths: List[str],
        first_prompts: List[str],
        correction_instruction: str,
        n_samples: int = 5,
        temperature: float = 1.0,
    ) -> List[Dict]:
        """
        Build the Pair-SFT dataset by sampling multiple responses and pairing them.
        
        For each problem:
        1. Sample n_samples responses
        2. Find incorrect responses (r=0) and correct responses (r=1)
        3. Pair each incorrect response with a correct response
        
        Args:
            problems: List of problems
            ground_truths: Ground truth answers
            first_prompts: Formatted prompts
            correction_instruction: Self-correction instruction
            n_samples: Number of samples per problem
            temperature: Sampling temperature
            
        Returns:
            List of training examples
        """
        training_data = []
        
        for problem, gt, first_prompt in zip(problems, ground_truths, first_prompts):
            # Sample multiple responses
            responses = []
            rewards = []
            
            for _ in range(n_samples):
                inputs = self.tokenizer(
                    first_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_seq_len,
                ).to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                input_len = inputs["input_ids"].shape[1]
                y = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                r = self.reward_fn(y, gt)
                
                responses.append(y)
                rewards.append(r)
            
            # Find incorrect and correct responses
            incorrect = [(y, r) for y, r in zip(responses, rewards) if r == 0.0]
            correct = [(y, r) for y, r in zip(responses, rewards) if r == 1.0]
            
            if not correct:
                continue  # Skip if no correct response found
            
            # Pair each incorrect response with a correct one
            for y_incorrect, _ in incorrect:
                # Use the first correct response as the target
                y_correct = correct[0][0]
                
                # Build self-correction trace
                second_prompt = f"{first_prompt}\n\n{y_incorrect}\n\n{correction_instruction}"
                
                training_data.append({
                    "prompt": first_prompt,
                    "response": y_incorrect,
                    "type": "first_attempt_incorrect",
                })
                training_data.append({
                    "prompt": second_prompt,
                    "response": y_correct,
                    "type": "second_attempt_correct",
                })
            
            # D_SFT+ variant: also include correct->correct pairs
            if self.config.include_correct_to_correct:
                for y_correct_first, _ in correct:
                    y_correct_second = correct[0][0]
                    second_prompt = f"{first_prompt}\n\n{y_correct_first}\n\n{correction_instruction}"
                    
                    training_data.append({
                        "prompt": first_prompt,
                        "response": y_correct_first,
                        "type": "first_attempt_correct",
                    })
                    training_data.append({
                        "prompt": second_prompt,
                        "response": y_correct_second,
                        "type": "second_attempt_correct",
                    })
        
        return training_data
    
    def sft_step(self, prompt: str, response: str) -> torch.Tensor:
        """Compute SFT loss for a single example."""
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_len,
        ).to(self.model.device)
        
        response_ids = self.tokenizer(
            response,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)
        
        full_ids = torch.cat(
            [prompt_ids["input_ids"], response_ids["input_ids"]], dim=1
        )
        
        outputs = self.model(input_ids=full_ids)
        logits = outputs.logits
        
        input_len = prompt_ids["input_ids"].shape[1]
        resp_logits = logits[:, input_len - 1:-1, :]
        
        loss = F.cross_entropy(
            resp_logits.reshape(-1, resp_logits.shape[-1]),
            response_ids["input_ids"].reshape(-1),
        )
        
        return loss
    
    def train(self, training_data: List[Dict]) -> Dict[str, float]:
        """
        Train on the Pair-SFT dataset.
        
        Args:
            training_data: List of (prompt, response) pairs
            
        Returns:
            Training metrics
        """
        import random
        
        total_loss = 0.0
        n_steps = 0
        
        for epoch in range(self.config.num_epochs):
            random.shuffle(training_data)
            
            for item in training_data:
                self.optimizer.zero_grad()
                loss = self.sft_step(item["prompt"], item["response"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                n_steps += 1
        
        return {
            "loss": total_loss / max(1, n_steps),
            "n_training_examples": len(training_data),
        }


class SelfRefine:
    """
    Self-Refine baseline (Madaan et al., 2023).
    
    Prompting-based approach: simply prompt the model to self-correct
    without any fine-tuning. Uses the same self-correction instruction
    as SCoRe but without training.
    """
    
    def __init__(self, model, tokenizer, max_seq_len: int = 2048):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate a response using greedy decoding."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        ).to(self.model.device)
        
        with torch.no_grad():
            if temperature == 0.0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        
        input_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    def self_correct(
        self,
        first_prompt: str,
        correction_instruction: str,
        temperature: float = 0.0,
    ) -> Tuple[str, str]:
        """
        Perform one round of self-correction via prompting.
        
        Args:
            first_prompt: Prompt for first attempt
            correction_instruction: Self-correction instruction
            temperature: Sampling temperature
            
        Returns:
            Tuple of (first_attempt, second_attempt)
        """
        # First attempt
        y1 = self.generate(first_prompt, temperature=temperature)
        
        # Build self-correction prompt
        second_prompt = f"{first_prompt}\n\n{y1}\n\n{correction_instruction}"
        
        # Second attempt
        y2 = self.generate(second_prompt, temperature=temperature)
        
        return y1, y2
