"""Utility functions for SCoRe training.

- Checkpoint management
- Logging
- Inference-compute scaling (Section 6.2)
- Majority voting
"""

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch


def save_checkpoint(
    policy: "LLMPolicy",  # type: ignore
    output_dir: str,
    step: int,
    prefix: str = "checkpoint",
) -> str:
    """Save a model checkpoint."""
    ckpt_dir = os.path.join(output_dir, f"{prefix}_step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    policy.model.save_pretrained(ckpt_dir)
    policy.tokenizer.save_pretrained(ckpt_dir)
    return ckpt_dir


def load_checkpoint(
    policy: "LLMPolicy",  # type: ignore
    ckpt_dir: str,
) -> None:
    """Load a model checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    policy.model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=policy.dtype if hasattr(policy, "dtype") else None,
        device_map="auto" if policy.device == "cuda" else None,
    )
    policy.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    if policy.tokenizer.pad_token is None:
        policy.tokenizer.pad_token = policy.tokenizer.eos_token


def compute_metrics_from_results(
    results: List[Dict],
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """Compute and optionally save metrics from evaluation results."""
    from metrics import compute_self_correction_metrics

    metrics = compute_self_correction_metrics(results)

    if save_path:
        with open(save_path, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)

    return metrics


def majority_vote(
    answers: List[str],
) -> str:
    """Select the most common answer from a list.

    Used for inference-compute scaling (Section 6.2).
    """
    if not answers:
        return ""
    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def extract_answer_from_response(
    response: str,
    is_code: bool = False,
) -> str:
    """Extract the answer string from a model response.

    For math: extracts the boxed answer or final answer line.
    For code: returns the full response (extraction done by test execution).
    """
    if is_code:
        return response

    from rewards import extract_boxed_answer, extract_final_answer_line

    boxed = extract_boxed_answer(response)
    if boxed is not None:
        return boxed

    final = extract_final_answer_line(response)
    if final is not None:
        return final

    return response.strip()


def compute_inference_scaling(
    policy: "LLMPolicy",  # type: ignore
    problems: List[Dict],
    is_code: bool,
    max_new_tokens: int,
    num_samples: int = 32,
    temperature: float = 0.7,
) -> Dict[str, float]:
    """Evaluate inference-compute scaling (Section 6.2).

    Compares:
    - Parallel: 2K samples, majority voting
    - Sequential: K samples, each with one round of self-correction

    With K such that total sample budget = 2K for parallel vs K for sequential
    (since sequential uses 2 samples per problem: initial + correction).
    """
    from prompts import (
        build_math_first_turn_prompt,
        build_math_second_turn_prompt,
        build_mbpp_first_turn_prompt,
        build_mbpp_second_turn_prompt,
    )
    from rewards import compute_reward

    K = num_samples // 2  # Sequential: K initial + K corrections = num_samples
    parallel_samples = num_samples  # Parallel: num_samples independent samples

    parallel_correct = 0
    sequential_correct = 0
    total = len(problems)

    for prob in problems:
        if is_code:
            prompt = build_mbpp_first_turn_prompt(
                prob["task_description"], prob["test_cases"]
            )
        else:
            prompt = build_math_first_turn_prompt(prob["problem_text"])

        # Parallel: sample 2K solutions, majority vote
        parallel_answers = []
        for _ in range(parallel_samples):
            responses = policy.generate(
                [prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
            answer = extract_answer_from_response(responses[0], is_code)
            parallel_answers.append(answer)
        voted = majority_vote(parallel_answers)

        # Check if majority vote is correct
        if is_code:
            p_correct = compute_reward(
                voted, prob["code_solution"],
                is_code=True, test_cases=prob["test_cases"],
            ) > 0.5
        else:
            p_correct = compute_reward(voted, prob["answer"]) > 0.5
        if p_correct:
            parallel_correct += 1

        # Sequential: K samples, each corrected once
        sequential_answers = []
        for _ in range(K):
            responses_t1 = policy.generate(
                [prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )

            if is_code:
                prompt_t2 = build_mbpp_second_turn_prompt(
                    prob["task_description"], prob["test_cases"], responses_t1[0]
                )
            else:
                prompt_t2 = build_math_second_turn_prompt(
                    prob["problem_text"], responses_t1[0]
                )

            responses_t2 = policy.generate(
                [prompt_t2],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
            answer = extract_answer_from_response(responses_t2[0], is_code)
            sequential_answers.append(answer)

        voted_seq = majority_vote(sequential_answers)

        if is_code:
            s_correct = compute_reward(
                voted_seq, prob["code_solution"],
                is_code=True, test_cases=prob["test_cases"],
            ) > 0.5
        else:
            s_correct = compute_reward(voted_seq, prob["answer"]) > 0.5
        if s_correct:
            sequential_correct += 1

    parallel_acc = parallel_correct / total if total > 0 else 0
    sequential_acc = sequential_correct / total if total > 0 else 0

    return {
        "parallel_accuracy": parallel_acc,
        "sequential_accuracy": sequential_acc,
        "improvement": sequential_acc - parallel_acc,
        "num_problems": total,
        "sample_budget": num_samples,
    }
