"""
SCoRe: Self-Correction via Multi-Turn Reinforcement Learning.

Implements the full SCoRe training algorithm as described in Section 5 of the paper:
"Training Language Models to Self-Correct via Reinforcement Learning."

The algorithm consists of two stages:

Stage I (Section 5.1): Train an initialization that decouples the two attempts.
    - Maximize second-attempt reward while constraining first-turn distribution
      to be close to the base model via KL divergence (Eq. 3).
    - Objective: max_θ E[r̂(y₂, y*) - β₂·D_KL(π_θ(·|x₁) || π_ref(·|x₁))]

Stage II (Section 5.2): Multi-turn RL with reward shaping.
    - Jointly optimize both attempts with a progress bonus (Eq. 4).
    - Reward shaping: b(y₂|y₁) = α·(r̂(y₂) - r̂(y₁))
    - Objective: max_θ E[Σ r̂(yᵢ) + α·(r̂(y₂)-r̂(y₁)) - β₁·Σ D_KL(π_θ || π_ref)]
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import logging
import os
import json

from .reinforce import (
    REINFORCEConfig,
    REINFORCEPolicyGradient,
    RewardCalculator,
)

logger = logging.getLogger(__name__)


@dataclass
class SCoReConfig:
    """Complete configuration for SCoRe training."""
    # Model settings
    model_name: str = ""
    base_model_path: str = ""
    
    # Stage I settings
    stage1_steps: int = 1500
    stage1_beta2: float = 0.1   # KL on first turn
    stage1_beta1: float = 0.01  # KL on second turn
    
    # Stage II settings
    stage2_steps: int = 1500
    stage2_beta1: float = 0.01
    stage2_alpha: float = 10.0  # Progress bonus multiplier
    
    # Training settings
    batch_size: int = 512
    learning_rate: float = 5e-6
    max_grad_norm: float = 1.0
    sampling_temperature: float = 1.0  # For training
    eval_temperature: float = 0.0  # Greedy eval
    
    # Data settings
    max_prompt_length: int = 2048
    max_response_length: int = 2048
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    save_every: int = 500
    eval_every: int = 500
    
    # Offline data mixing (optional)
    mix_offline_first_attempts: bool = False
    offline_mix_ratio: float = 0.0
    
    # Task type
    task: str = "math"  # "math" or "code"
    
    # For MATH: 3000 steps, batch 512, lr 5e-6 (Table 5 left)
    # For MBPP: 1500 steps, batch 128, lr 1e-5 (Table 5 right)


class SCoReTrainer:
    """
    Main trainer for the SCoRe algorithm.
    
    Implements the two-stage training process:
    
    Stage I: Decouple attempts by training second attempt while
             constraining first attempt to reference model.
    
    Stage II: Joint optimization with reward shaping to prevent
              behavior collapse.
    
    The paper reports using Gemini 1.0 Pro for code and Gemini 1.5 Flash 
    for MATH. We abstract the model loading to support any HuggingFace model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        tokenizer: object,
        config: SCoReConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Ensure reference is frozen
        for p in self.reference_model.parameters():
            p.requires_grad = False
        self.reference_model.eval()
        
        self.reward_calc = RewardCalculator()
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=config.learning_rate
            )
        else:
            self.optimizer = optimizer
        
        # Create REINFORCE trainer (shares optimizer)
        self.rl_config = REINFORCEConfig(
            beta1=config.stage1_beta1,
            beta2=config.stage1_beta2,
            alpha=config.stage2_alpha,
            sampling_temperature=config.sampling_temperature,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
        )
        self.rl_trainer = REINFORCEPolicyGradient(
            model=model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            config=self.rl_config,
            optimizer=self.optimizer,
        )
        
        self.global_step = 0
        self.metrics_history = defaultdict(list)
    
    def build_turn_prompt(
        self,
        problem: str,
        previous_response: Optional[str] = None,
        correction_instruction: Optional[str] = None,
    ) -> str:
        """
        Build the prompt for a given turn.
        
        Turn 1: problem + instruction to solve
        Turn 2: problem + previous solution + correction instruction
        
        Uses the prompts from Appendix C of the paper.
        """
        if previous_response is None:
            # First turn
            if self.config.task == "math":
                return f"""You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."

Problem: {problem}"""
            else:
                return f"""You are an expert Python programmer. Write a solution to the following problem. Only output the final correct Python program!

Problem: {problem}"""
        else:
            # Second turn - with correction instruction
            if self.config.task == "math":
                return f"""You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."

Problem: {problem}

Your previous solution:
{previous_response}

There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct." """
            else:
                return f"""You are an expert Python programmer. Write a solution to the following problem. Only output the final correct Python program!

Problem: {problem}

Your previous solution:
{previous_response}

There might be an error in the code above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final correct Python program!"""
    
    def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the current policy.
        """
        if temperature is None:
            temperature = self.config.sampling_temperature
        if max_tokens is None:
            max_tokens = self.config.max_response_length
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
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
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        return response
    
    def compute_reward(
        self, 
        response: str, 
        ground_truth: Any,
    ) -> float:
        """
        Compute binary reward for a response.
        
        For MATH: extracts final answer and checks against ground truth.
        For coding: extracts code and runs against test cases.
        """
        if self.config.task == "math":
            answer = self.reward_calc.extract_final_answer_math(response)
            if answer is None:
                return 0.0
            return self.reward_calc.check_math_answer(answer, str(ground_truth))
        else:
            code = self.reward_calc.extract_final_answer_code(response)
            test_cases = ground_truth  # List of {'input': ..., 'expected': ...}
            if isinstance(test_cases, dict):
                test_cases = [test_cases]
            return self.reward_calc.check_code_correctness(code, test_cases)
    
    def sample_two_turn_trajectory(
        self,
        problem: str,
        ground_truth: Any,
    ) -> Dict[str, Any]:
        """
        Sample a complete two-turn self-correction trajectory on-policy.
        
        Returns:
            Dictionary with prompts, responses, and rewards for both turns.
        """
        # Turn 1: Generate initial solution
        prompt_t1 = self.build_turn_prompt(problem)
        response_t1 = self.generate_response(prompt_t1)
        reward_t1 = self.compute_reward(response_t1, ground_truth)
        
        # Turn 2: Generate self-correction
        prompt_t2 = self.build_turn_prompt(problem, response_t1)
        response_t2 = self.generate_response(prompt_t2)
        reward_t2 = self.compute_reward(response_t2, ground_truth)
        
        return {
            "problem": problem,
            "ground_truth": ground_truth,
            "prompt_t1": prompt_t1,
            "response_t1": response_t1,
            "reward_t1": reward_t1,
            "prompt_t2": prompt_t2,
            "response_t2": response_t2,
            "reward_t2": reward_t2,
        }
    
    def tokenize_trajectory(
        self, 
        trajectory: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a two-turn trajectory for REINFORCE training.
        
        Returns tensors needed for two_turn_step:
        - input_ids_t1, attention_mask_t1, response_ids_t1
        - input_ids_t2, attention_mask_t2, response_ids_t2
        """
        # Turn 1
        t1_inputs = self.tokenizer(
            trajectory["prompt_t1"],
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        t1_resp = self.tokenizer(
            trajectory["response_t1"],
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_response_length,
        )
        
        # Turn 2
        t2_inputs = self.tokenizer(
            trajectory["prompt_t2"],
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        t2_resp = self.tokenizer(
            trajectory["response_t2"],
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_response_length,
        )
        
        return {
            "input_ids_t1": t1_inputs["input_ids"],
            "attention_mask_t1": t1_inputs["attention_mask"],
            "response_ids_t1": t1_resp["input_ids"],
            "input_ids_t2": t2_inputs["input_ids"],
            "attention_mask_t2": t2_inputs["attention_mask"],
            "response_ids_t2": t2_resp["input_ids"],
        }
    
    def run_stage1(
        self,
        train_problems: List[Dict],
        val_problems: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Run Stage I of SCoRe: Training an initialization that decouples attempts.
        
        Equation 3:
            max_θ E[r̂(y₂, y*) - β₂ D_KL(π_θ(·|x₁) || π_ref(·|x₁))]
        
        Optimizes second-attempt accuracy while constraining the first-turn
        distribution to stay close to the reference (base) model.
        
        Args:
            train_problems: List of {'problem': ..., 'answer': ...}
            val_problems: Optional validation problems
        
        Returns:
            Dictionary of training statistics
        """
        logger.info(f"Starting SCoRe Stage I for {self.config.stage1_steps} steps")
        
        # Set Stage I config
        self.rl_trainer.config.beta1 = self.config.stage1_beta1
        self.rl_trainer.config.beta2 = self.config.stage1_beta2
        
        stage1_metrics = []
        
        pbar = tqdm(range(self.config.stage1_steps), desc="Stage I")
        for step in pbar:
            # Sample batch of problems
            batch_indices = np.random.choice(
                len(train_problems),
                size=min(self.config.batch_size, len(train_problems)),
                replace=False,
            )
            batch_problems = [train_problems[i] for i in batch_indices]
            
            # Sample two-turn trajectories on-policy
            trajectories = []
            rewards_t1_list = []
            rewards_t2_list = []
            
            for prob in batch_problems:
                traj = self.sample_two_turn_trajectory(
                    prob['problem'], prob['answer']
                )
                trajectories.append(traj)
                rewards_t1_list.append(traj['reward_t1'])
                rewards_t2_list.append(traj['reward_t2'])
            
            # Tokenize trajectories
            tokenized_batch = self._collate_trajectories(trajectories)
            
            # Convert rewards
            rewards_t1 = torch.tensor(rewards_t1_list, dtype=torch.float32)
            rewards_t2 = torch.tensor(rewards_t2_list, dtype=torch.float32)
            
            device = next(self.model.parameters()).device
            rewards_t1 = rewards_t1.to(device)
            rewards_t2 = rewards_t2.to(device)
            
            # Stage I step
            loss, metrics = self.rl_trainer.two_turn_step(
                batch=tokenized_batch,
                rewards_t1=rewards_t1,
                rewards_t2=rewards_t2,
                is_stage1=True,
            )
            
            self.global_step += 1
            stage1_metrics.append(metrics)
            
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "r_t1": f"{metrics['mean_reward_t1']:.2f}",
                "r_t2": f"{metrics['mean_reward_t2']:.2f}",
                "kl_t1": f"{metrics['kl_t1']:.4f}",
            })
            
            # Evaluation
            if val_problems and (step + 1) % self.config.eval_every == 0:
                eval_metrics = self.evaluate(val_problems)
                logger.info(f"Stage I eval @ step {step+1}: {eval_metrics}")
            
            # Save checkpoint
            if (step + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"stage1_step{step+1}")
        
        logger.info("Stage I complete")
        return {
            "stage1_metrics": stage1_metrics,
            "final_reward_t1": np.mean([m['mean_reward_t1'] for m in stage1_metrics[-100:]]),
            "final_reward_t2": np.mean([m['mean_reward_t2'] for m in stage1_metrics[-100:]]),
        }
    
    def run_stage2(
        self,
        train_problems: List[Dict],
        val_problems: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Run Stage II of SCoRe: Multi-turn RL with reward shaping.
        
        Equation 4 with progress bonus:
            max_θ E[Σ r̂(yᵢ) + α·(r̂(y₂)-r̂(y₁)) - β₁·Σ D_KL(π_θ || π_ref)]
        
        The reward shaping term b(y₂|y₁) = α·(r̂(y₂)-r̂(y₁)) rewards 
        progress towards self-correction and penalizes degrading correct 
        answers.
        
        Args:
            train_problems: List of {'problem': ..., 'answer': ...}
            val_problems: Optional validation problems
        
        Returns:
            Dictionary of training statistics
        """
        logger.info(f"Starting SCoRe Stage II for {self.config.stage2_steps} steps")
        
        # Set Stage II config
        self.rl_trainer.config.beta1 = self.config.stage2_beta1
        self.rl_trainer.config.alpha = self.config.stage2_alpha
        
        stage2_metrics = []
        
        pbar = tqdm(range(self.config.stage2_steps), desc="Stage II")
        for step in pbar:
            # Sample batch of problems
            batch_indices = np.random.choice(
                len(train_problems),
                size=min(self.config.batch_size, len(train_problems)),
                replace=False,
            )
            batch_problems = [train_problems[i] for i in batch_indices]
            
            # Optionally mix in offline first attempts from base model
            trajectories = []
            rewards_t1_list = []
            rewards_t2_list = []
            
            for prob in batch_problems:
                traj = self.sample_two_turn_trajectory(
                    prob['problem'], prob['answer']
                )
                trajectories.append(traj)
                rewards_t1_list.append(traj['reward_t1'])
                rewards_t2_list.append(traj['reward_t2'])
            
            # Tokenize trajectories
            tokenized_batch = self._collate_trajectories(trajectories)
            
            rewards_t1 = torch.tensor(rewards_t1_list, dtype=torch.float32)
            rewards_t2 = torch.tensor(rewards_t2_list, dtype=torch.float32)
            
            device = next(self.model.parameters()).device
            rewards_t1 = rewards_t1.to(device)
            rewards_t2 = rewards_t2.to(device)
            
            # Stage II step
            loss, metrics = self.rl_trainer.two_turn_step(
                batch=tokenized_batch,
                rewards_t1=rewards_t1,
                rewards_t2=rewards_t2,
                is_stage1=False,
            )
            
            self.global_step += 1
            stage2_metrics.append(metrics)
            
            # Compute progress statistics
            correct_t1 = (rewards_t1 > 0.5).float().mean().item()
            correct_t2 = (rewards_t2 > 0.5).float().mean().item()
            delta = correct_t2 - correct_t1
            
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "r_t1": f"{correct_t1:.2f}",
                "r_t2": f"{correct_t2:.2f}",
                "Δ": f"{delta:.3f}",
                "bonus": f"{metrics['mean_progress_bonus']:.2f}",
            })
            
            if val_problems and (step + 1) % self.config.eval_every == 0:
                eval_metrics = self.evaluate(val_problems)
                logger.info(f"Stage II eval @ step {step+1}: {eval_metrics}")
            
            if (step + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"stage2_step{step+1}")
        
        logger.info("Stage II complete")
        return {
            "stage2_metrics": stage2_metrics,
            "final_correct_t1": np.mean([(m['mean_reward_t1']) for m in stage2_metrics[-100:]]),
            "final_correct_t2": np.mean([(m['mean_reward_t2']) for m in stage2_metrics[-100:]]),
        }
    
    def _collate_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Collate multiple tokenized trajectories into a batch."""
        # Tokenize all trajectories
        all_tokenized = [self.tokenize_trajectory(t) for t in trajectories]
        
        device = next(self.model.parameters()).device
        
        # Pad and stack turn 1 inputs
        t1_input_ids = self._pad_sequences(
            [t['input_ids_t1'] for t in all_tokenized]
        ).to(device)
        t1_attention_mask = self._pad_sequences(
            [t['attention_mask_t1'] for t in all_tokenized]
        ).to(device)
        t1_response_ids = self._pad_sequences(
            [t['response_ids_t1'] for t in all_tokenized]
        ).to(device)
        
        # Pad and stack turn 2 inputs
        t2_input_ids = self._pad_sequences(
            [t['input_ids_t2'] for t in all_tokenized]
        ).to(device)
        t2_attention_mask = self._pad_sequences(
            [t['attention_mask_t2'] for t in all_tokenized]
        ).to(device)
        t2_response_ids = self._pad_sequences(
            [t['response_ids_t2'] for t in all_tokenized]
        ).to(device)
        
        return {
            "input_ids_t1": t1_input_ids,
            "attention_mask_t1": t1_attention_mask,
            "response_ids_t1": t1_response_ids,
            "input_ids_t2": t2_input_ids,
            "attention_mask_t2": t2_attention_mask,
            "response_ids_t2": t2_response_ids,
        }
    
    def _pad_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad a list of tensors to equal length."""
        max_len = max(s.shape[-1] for s in sequences)
        padded = []
        for s in sequences:
            if s.shape[-1] < max_len:
                pad = torch.zeros(
                    *s.shape[:-1], max_len - s.shape[-1],
                    dtype=s.dtype,
                )
                s = torch.cat([s, pad], dim=-1)
            padded.append(s)
        return torch.cat(padded, dim=0)
    
    def evaluate(
        self,
        eval_problems: List[Dict],
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate self-correction performance on a set of problems.
        
        Computes the metrics from Section 3:
        - Accuracy@t1: Accuracy at first attempt
        - Accuracy@t2: Accuracy at second attempt
        - Δ(t1, t2): Net improvement (t2 - t1)
        - Δ(i→c): Fraction incorrect→correct (correction rate)
        - Δ(c→i): Fraction correct→incorrect (degradation rate)
        """
        self.model.eval()
        
        if num_samples is not None:
            indices = np.random.choice(len(eval_problems), num_samples, replace=False)
            eval_problems = [eval_problems[i] for i in indices]
        
        correct_t1 = 0
        correct_t2 = 0
        i_to_c = 0  # incorrect -> correct
        c_to_i = 0  # correct -> incorrect
        total = 0
        
        for prob in tqdm(eval_problems, desc="Evaluating", leave=False):
            traj = self.sample_two_turn_trajectory(
                prob['problem'], prob['answer']
            )
            
            r1 = traj['reward_t1']
            r2 = traj['reward_t2']
            
            if r1 > 0.5:
                correct_t1 += 1
                if r2 <= 0.5:
                    c_to_i += 1
            else:
                if r2 > 0.5:
                    i_to_c += 1
            
            if r2 > 0.5:
                correct_t2 += 1
            
            total += 1
        
        self.model.train()
        
        results = {
            "accuracy_t1": correct_t1 / total,
            "accuracy_t2": correct_t2 / total,
            "delta_t1_t2": (correct_t2 - correct_t1) / total,
            "i_to_c": i_to_c / total,
            "c_to_i": c_to_i / total,
            "num_samples": total,
        }
        return results
    
    def evaluate_self_consistency(
        self,
        eval_problems: List[Dict],
        num_parallel: int = 16,
        num_sequential: int = 2,
        temperature: float = 0.7,
    ) -> Dict[str, float]:
        """
        Evaluate with self-consistency decoding (Section 6.2).
        
        Instead of sampling 2K solutions in parallel, samples K solutions 
        in parallel, then performs one round of self-correction on each.
        """
        self.model.eval()
        
        correct_parallel = 0
        correct_sequential = 0
        total = 0
        
        for prob in tqdm(eval_problems, desc="Self-Consistency", leave=False):
            # Parallel sampling: K independent solutions
            parallel_solutions = []
            for _ in range(num_parallel):
                prompt = self.build_turn_prompt(prob['problem'])
                response = self.generate_response(prompt, temperature=temperature)
                parallel_solutions.append(response)
            
            # Majority vote on parallel
            answers = []
            for s in parallel_solutions:
                ans = self.reward_calc.extract_final_answer_math(s)
                answers.append(ans)
            
            # Simple majority
            from collections import Counter
            answer_counts = Counter(answers)
            most_common = answer_counts.most_common(1)[0][0]
            if most_common is not None:
                correct_parallel += self.reward_calc.check_math_answer(
                    most_common, str(prob['answer'])
                )
            
            # Sequential: K parallel, then self-correct each
            sequential_correct = []
            for s in parallel_solutions:
                prompt_t2 = self.build_turn_prompt(
                    prob['problem'], s
                )
                corrected = self.generate_response(prompt_t2, temperature=temperature)
                r = self.compute_reward(corrected, prob['answer'])
                sequential_correct.append(r)
            
            # Majority vote on corrected
            if sum(sequential_correct) > len(sequential_correct) / 2:
                correct_sequential += 1
            
            total += 1
        
        self.model.train()
        
        return {
            "parallel_majority": correct_parallel / total,
            "sequential_correction": correct_sequential / total,
            "num_samples": total,
        }
    
    def save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, f"{name}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        logger.info(f"Checkpoint loaded: {path}")


__all__ = ["SCoReConfig", "SCoReTrainer"]
