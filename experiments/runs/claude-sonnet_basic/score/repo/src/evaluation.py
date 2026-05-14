"""
Evaluation utilities for SCoRe.

Computes the metrics described in Section 3 of the paper:
- Accuracy@t1: First-attempt accuracy
- Accuracy@t2: Second-attempt accuracy  
- Delta(t1, t2): Net improvement (t2 - t1)
- Delta^(i->c)(t1, t2): Fraction of incorrect t1 that become correct t2
- Delta^(c->i)(t1, t2): Fraction of correct t1 that become incorrect t2
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import json
import os


def compute_self_correction_metrics(
    first_rewards: List[float],
    second_rewards: List[float],
) -> Dict[str, float]:
    """
    Compute all self-correction metrics from the paper.
    
    Args:
        first_rewards: Binary rewards for first attempts (0 or 1)
        second_rewards: Binary rewards for second attempts (0 or 1)
        
    Returns:
        Dictionary with all metrics:
        - accuracy_t1: Accuracy at first attempt
        - accuracy_t2: Accuracy at second attempt
        - delta_t1_t2: Net improvement (t2 - t1)
        - delta_i_to_c: Fraction of incorrect t1 -> correct t2
        - delta_c_to_i: Fraction of correct t1 -> incorrect t2
    """
    assert len(first_rewards) == len(second_rewards), \
        "Must have same number of first and second rewards"
    
    n = len(first_rewards)
    if n == 0:
        return {
            "accuracy_t1": 0.0,
            "accuracy_t2": 0.0,
            "delta_t1_t2": 0.0,
            "delta_i_to_c": 0.0,
            "delta_c_to_i": 0.0,
        }
    
    # Accuracy@t1 and Accuracy@t2
    accuracy_t1 = sum(first_rewards) / n
    accuracy_t2 = sum(second_rewards) / n
    
    # Delta(t1, t2): net improvement
    delta_t1_t2 = accuracy_t2 - accuracy_t1
    
    # Count transitions
    n_incorrect_t1 = sum(1 for r in first_rewards if r == 0.0)
    n_correct_t1 = sum(1 for r in first_rewards if r == 1.0)
    
    # Delta^(i->c): fraction of incorrect t1 that become correct t2
    n_i_to_c = sum(
        1 for r1, r2 in zip(first_rewards, second_rewards)
        if r1 == 0.0 and r2 == 1.0
    )
    delta_i_to_c = n_i_to_c / max(1, n_incorrect_t1)
    
    # Delta^(c->i): fraction of correct t1 that become incorrect t2
    n_c_to_i = sum(
        1 for r1, r2 in zip(first_rewards, second_rewards)
        if r1 == 1.0 and r2 == 0.0
    )
    delta_c_to_i = n_c_to_i / max(1, n_correct_t1)
    
    return {
        "accuracy_t1": accuracy_t1,
        "accuracy_t2": accuracy_t2,
        "delta_t1_t2": delta_t1_t2,
        "delta_i_to_c": delta_i_to_c,
        "delta_c_to_i": delta_c_to_i,
        # Also report raw counts for debugging
        "n_total": n,
        "n_incorrect_t1": n_incorrect_t1,
        "n_correct_t1": n_correct_t1,
        "n_i_to_c": n_i_to_c,
        "n_c_to_i": n_c_to_i,
    }


def compute_edit_distance_ratio(response1: str, response2: str) -> float:
    """
    Compute the edit distance ratio between two responses.
    
    Defined as: edit_distance(r1, r2) / (len(r1) + len(r2))
    
    This is used in Figure 4 of the paper to analyze how much
    models change their responses between attempts.
    
    Args:
        response1: First response string
        response2: Second response string
        
    Returns:
        Edit distance ratio in [0, 1]
    """
    # Use character-level edit distance
    m, n = len(response1), len(response2)
    
    if m + n == 0:
        return 0.0
    
    # Dynamic programming for edit distance
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if response1[i - 1] == response2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    edit_dist = dp[m][n]
    return edit_dist / (m + n)


def compute_edit_distance_ratios(
    first_responses: List[str],
    second_responses: List[str],
) -> List[float]:
    """
    Compute edit distance ratios for a batch of response pairs.
    
    Args:
        first_responses: List of first attempt responses
        second_responses: List of second attempt responses
        
    Returns:
        List of edit distance ratios
    """
    return [
        compute_edit_distance_ratio(r1, r2)
        for r1, r2 in zip(first_responses, second_responses)
    ]


def format_results_table(
    method_results: Dict[str, Dict[str, float]],
    task: str = "MATH",
) -> str:
    """
    Format results as a table similar to Tables 2 and 3 in the paper.
    
    Args:
        method_results: Dict mapping method name to metrics dict
        task: Task name for display
        
    Returns:
        Formatted table string
    """
    header = f"\n{'='*80}\n"
    header += f"Self-Correction Performance on {task}\n"
    header += f"{'='*80}\n"
    
    # Column headers
    cols = ["Method", "Acc.@t1", "Acc.@t2", "Δ(t1,t2)", "Δ^(i→c)", "Δ^(c→i)"]
    col_widths = [30, 10, 10, 12, 12, 12]
    
    header_row = "".join(f"{col:<{w}}" for col, w in zip(cols, col_widths))
    separator = "-" * sum(col_widths)
    
    rows = [header, header_row, separator]
    
    for method, metrics in method_results.items():
        acc_t1 = metrics.get("accuracy_t1", 0.0)
        acc_t2 = metrics.get("accuracy_t2", 0.0)
        delta = metrics.get("delta_t1_t2", 0.0)
        i_to_c = metrics.get("delta_i_to_c", 0.0)
        c_to_i = metrics.get("delta_c_to_i", 0.0)
        
        row = (
            f"{method:<30}"
            f"{acc_t1*100:.1f}%{'':<4}"
            f"{acc_t2*100:.1f}%{'':<4}"
            f"{delta*100:+.1f}%{'':<5}"
            f"{i_to_c*100:.1f}%{'':<5}"
            f"{c_to_i*100:.1f}%{'':<5}"
        )
        rows.append(row)
    
    rows.append("=" * sum(col_widths))
    return "\n".join(rows)


def evaluate_model_on_dataset(
    model,
    tokenizer,
    problems: List[str],
    ground_truths: List[str],
    first_prompts: List[str],
    correction_instruction: str,
    reward_fn,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset and compute all self-correction metrics.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        problems: List of problem strings
        ground_truths: Ground truth answers
        first_prompts: Formatted prompts for first attempt
        correction_instruction: Self-correction instruction
        reward_fn: Function(response, ground_truth) -> float
        temperature: Sampling temperature (0 = greedy)
        max_new_tokens: Maximum tokens to generate
        batch_size: Evaluation batch size
        
    Returns:
        Dictionary of metrics
    """
    import torch
    
    first_rewards = []
    second_rewards = []
    first_responses = []
    second_responses = []
    
    model.eval()
    
    for i in range(0, len(problems), batch_size):
        batch_problems = problems[i:i + batch_size]
        batch_gts = ground_truths[i:i + batch_size]
        batch_prompts = first_prompts[i:i + batch_size]
        
        for problem, gt, first_prompt in zip(batch_problems, batch_gts, batch_prompts):
            # Generate first attempt
            inputs = tokenizer(
                first_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(model.device)
            
            with torch.no_grad():
                if temperature == 0.0:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                else:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            
            input_len = inputs["input_ids"].shape[1]
            y1 = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            r1 = reward_fn(y1, gt)
            
            first_responses.append(y1)
            first_rewards.append(r1)
            
            # Build second-attempt prompt
            second_prompt = f"{first_prompt}\n\n{y1}\n\n{correction_instruction}"
            
            # Generate second attempt
            inputs2 = tokenizer(
                second_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(model.device)
            
            with torch.no_grad():
                if temperature == 0.0:
                    outputs2 = model.generate(
                        **inputs2,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                else:
                    outputs2 = model.generate(
                        **inputs2,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            
            input_len2 = inputs2["input_ids"].shape[1]
            y2 = tokenizer.decode(outputs2[0][input_len2:], skip_special_tokens=True)
            r2 = reward_fn(y2, gt)
            
            second_responses.append(y2)
            second_rewards.append(r2)
    
    model.train()
    
    metrics = compute_self_correction_metrics(first_rewards, second_rewards)
    
    # Add edit distance analysis
    edit_ratios = compute_edit_distance_ratios(first_responses, second_responses)
    metrics["mean_edit_distance_ratio"] = np.mean(edit_ratios)
    metrics["fraction_no_edit"] = sum(1 for r in edit_ratios if r < 0.01) / len(edit_ratios)
    
    return metrics


def save_results(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    task: str = "MATH",
) -> None:
    """
    Save evaluation results to a JSON file and print a formatted table.
    
    Args:
        results: Dict mapping method name to metrics
        output_path: Path to save JSON results
        task: Task name
    """
    # Save JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print table
    print(format_results_table(results, task))
    print(f"\nResults saved to {output_path}")
