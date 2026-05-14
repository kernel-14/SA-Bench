"""
Inference-compute scaling with self-correction (Section 6.2).

Implements the comparison between:
1. Parallel sampling: sample K solutions independently, majority vote
2. Sequential + parallel: sample K/2 solutions, self-correct each, majority vote

From the paper: "instead of sampling 2K solutions in parallel, it is more
compute-efficient to sample K solutions in parallel, then perform one round
of self-correction on each solution."

Results from paper (32 solution budget):
- Parallel sampling: +7.4% accuracy gain
- Sequential (self-correction) + parallel: +10.5% improvement
"""

import torch
from typing import List, Dict, Optional, Callable
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def majority_vote(answers: List[str]) -> str:
    """
    Select the most common answer from a list.
    
    Args:
        answers: List of answer strings
        
    Returns:
        Most common answer
    """
    if not answers:
        return ""
    
    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def extract_answer_for_voting(response: str, task_type: str = "math") -> str:
    """
    Extract the answer from a response for majority voting.
    
    Args:
        response: Model response
        task_type: "math" or "code"
        
    Returns:
        Extracted answer string
    """
    if task_type == "math":
        from src.reward import extract_math_answer, normalize_math_answer
        answer = extract_math_answer(response)
        if answer:
            return normalize_math_answer(answer)
        return response.strip()[-50:]  # Fallback: use last 50 chars
    else:
        # For code, use the full response
        return response.strip()


def parallel_sampling(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    task_type: str = "math",
) -> str:
    """
    Sample n_samples solutions in parallel and return majority vote.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        n_samples: Number of samples
        temperature: Sampling temperature
        max_new_tokens: Max tokens to generate
        task_type: Task type for answer extraction
        
    Returns:
        Majority vote answer
    """
    responses = []
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    for _ in range(n_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        input_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        responses.append(response)
    
    # Extract answers and majority vote
    answers = [extract_answer_for_voting(r, task_type) for r in responses]
    return majority_vote(answers)


def sequential_then_parallel(
    model,
    tokenizer,
    prompt: str,
    correction_instruction: str,
    n_samples: int,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    task_type: str = "math",
) -> str:
    """
    Sample n_samples solutions, self-correct each, then majority vote.
    
    This is the "sequential + parallel" strategy from Section 6.2.
    Total compute: 2 * n_samples (one for first attempt, one for correction).
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        correction_instruction: Self-correction instruction
        n_samples: Number of initial samples
        temperature: Sampling temperature
        max_new_tokens: Max tokens to generate
        task_type: Task type for answer extraction
        
    Returns:
        Majority vote answer after self-correction
    """
    second_responses = []
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    for _ in range(n_samples):
        # First attempt
        with torch.no_grad():
            outputs1 = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        input_len = inputs["input_ids"].shape[1]
        y1 = tokenizer.decode(outputs1[0][input_len:], skip_special_tokens=True)
        
        # Build self-correction prompt
        second_prompt = f"{prompt}\n\n{y1}\n\n{correction_instruction}"
        
        inputs2 = tokenizer(
            second_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)
        
        # Second attempt (self-correction)
        with torch.no_grad():
            outputs2 = model.generate(
                **inputs2,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        input_len2 = inputs2["input_ids"].shape[1]
        y2 = tokenizer.decode(outputs2[0][input_len2:], skip_special_tokens=True)
        second_responses.append(y2)
    
    # Extract answers and majority vote
    answers = [extract_answer_for_voting(r, task_type) for r in second_responses]
    return majority_vote(answers)


def evaluate_inference_scaling(
    model,
    tokenizer,
    problems: List[str],
    ground_truths: List[str],
    first_prompts: List[str],
    correction_instruction: str,
    reward_fn: Callable,
    sample_budgets: List[int],
    temperature: float = 0.7,
    task_type: str = "math",
) -> Dict[str, List[float]]:
    """
    Evaluate inference-compute scaling for different sample budgets.
    
    Compares:
    1. Parallel sampling with N samples
    2. Sequential (self-correction) + parallel with N/2 samples
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        problems: Problem strings
        ground_truths: Ground truth answers
        first_prompts: Formatted prompts
        correction_instruction: Self-correction instruction
        reward_fn: Reward function
        sample_budgets: List of total sample budgets to evaluate
        temperature: Sampling temperature (paper uses 0.7)
        task_type: Task type
        
    Returns:
        Dict with 'parallel' and 'sequential' accuracy lists
    """
    results = {
        "parallel": [],
        "sequential": [],
        "sample_budgets": sample_budgets,
    }
    
    for budget in sample_budgets:
        logger.info(f"Evaluating with sample budget {budget}...")
        
        # Parallel: use all budget for parallel sampling
        parallel_correct = 0
        for prob, gt, prompt in zip(problems, ground_truths, first_prompts):
            answer = parallel_sampling(
                model, tokenizer, prompt,
                n_samples=budget,
                temperature=temperature,
                task_type=task_type,
            )
            # For majority vote, we need to check if the voted answer is correct
            # We use the reward function on a synthetic response containing the answer
            r = reward_fn(f"Final Answer: The final answer is ${answer}$. I hope it is correct.", gt)
            parallel_correct += r
        
        parallel_acc = parallel_correct / len(problems)
        results["parallel"].append(parallel_acc)
        
        # Sequential: use budget/2 for initial samples, then self-correct each
        seq_n_samples = max(1, budget // 2)
        sequential_correct = 0
        
        for prob, gt, prompt in zip(problems, ground_truths, first_prompts):
            answer = sequential_then_parallel(
                model, tokenizer, prompt,
                correction_instruction=correction_instruction,
                n_samples=seq_n_samples,
                temperature=temperature,
                task_type=task_type,
            )
            r = reward_fn(f"Final Answer: The final answer is ${answer}$. I hope it is correct.", gt)
            sequential_correct += r
        
        sequential_acc = sequential_correct / len(problems)
        results["sequential"].append(sequential_acc)
        
        logger.info(
            f"Budget {budget}: Parallel={parallel_acc:.3f}, "
            f"Sequential={sequential_acc:.3f}"
        )
    
    return results
