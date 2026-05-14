"""
Evaluation for SCoRe.

Implements all metrics from Section 3:
- Accuracy@t1: first-attempt accuracy
- Accuracy@t2: second-attempt accuracy
- Δ(t1, t2): net improvement (Acc@t2 - Acc@t1)
- Δ^(i→c)(t1, t2): fraction incorrect at t1 that become correct at t2
- Δ^(c→i)(t1, t2): fraction correct at t1 that become incorrect at t2

Also implements:
- Edit distance ratio analysis (Figure 4)
- Inference-compute scaling with self-consistency (Section 6.2)
- MBPP-R offline repair evaluation (Table 3)
"""

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import Levenshtein
import torch
from tqdm import tqdm

from data import (
    Problem,
    load_humaneval_dataset,
    load_math_dataset,
    load_mbpp_dataset,
    load_mbpp_repair_dataset,
)
from model import LLMPolicy, load_policy_and_ref
from rewards import compute_reward
from train import collect_rollout


# ---------------------------------------------------------------------------
# Core metrics (Section 3)
# ---------------------------------------------------------------------------

def compute_self_correction_metrics(
    correct_t1: List[bool],
    correct_t2: List[bool],
) -> Dict[str, float]:
    """
    Compute all self-correction metrics from Section 3.

    Args:
        correct_t1: list of booleans, True if first attempt is correct
        correct_t2: list of booleans, True if second attempt is correct

    Returns:
        dict with acc_t1, acc_t2, delta_t1_t2, delta_i2c, delta_c2i
    """
    n = len(correct_t1)
    assert n == len(correct_t2), "Lists must have the same length"

    acc_t1 = sum(correct_t1) / n
    acc_t2 = sum(correct_t2) / n
    delta_t1_t2 = acc_t2 - acc_t1

    # Δ^(i→c): incorrect at t1, correct at t2
    delta_i2c = sum(
        (not c1) and c2 for c1, c2 in zip(correct_t1, correct_t2)
    ) / n

    # Δ^(c→i): correct at t1, incorrect at t2
    delta_c2i = sum(
        c1 and (not c2) for c1, c2 in zip(correct_t1, correct_t2)
    ) / n

    return {
        "acc_t1": acc_t1,
        "acc_t2": acc_t2,
        "delta_t1_t2": delta_t1_t2,
        "delta_i2c": delta_i2c,
        "delta_c2i": delta_c2i,
    }


# ---------------------------------------------------------------------------
# Edit distance ratio (Figure 4)
# ---------------------------------------------------------------------------

def edit_distance_ratio(response_t1: str, response_t2: str) -> float:
    """
    Compute the edit distance ratio between two responses.

    Defined as: edit_distance(t1, t2) / (len(t1) + len(t2))

    This is the metric used in Figure 4 to analyze how much the model
    modifies its response between turns.
    """
    if not response_t1 and not response_t2:
        return 0.0
    dist = Levenshtein.distance(response_t1, response_t2)
    total_len = len(response_t1) + len(response_t2)
    return dist / total_len if total_len > 0 else 0.0


def compute_edit_distance_stats(
    responses_t1: List[str],
    responses_t2: List[str],
) -> Dict[str, float]:
    """Compute summary statistics of edit distance ratios."""
    ratios = [
        edit_distance_ratio(r1, r2)
        for r1, r2 in zip(responses_t1, responses_t2)
    ]
    import numpy as np
    arr = np.array(ratios)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "frac_zero": float((arr == 0.0).mean()),  # fraction making no edits
    }


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_on_problems(
    policy: LLMPolicy,
    problems: List[Problem],
    reward_fn,
    temperature: float = 0.0,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Run full two-turn evaluation on a list of problems.

    Returns metrics dict and optionally saves detailed results to output_path.
    """
    policy.model.eval()
    results = []
    correct_t1 = []
    correct_t2 = []
    responses_t1 = []
    responses_t2 = []

    with torch.no_grad():
        for problem in tqdm(problems, desc="Evaluating"):
            rollout = collect_rollout(
                policy, problem, temperature=temperature, reward_fn=reward_fn
            )
            c1 = rollout["reward_t1"] > 0.5
            c2 = rollout["reward_t2"] > 0.5
            correct_t1.append(c1)
            correct_t2.append(c2)
            responses_t1.append(rollout["response_t1"])
            responses_t2.append(rollout["response_t2"])
            results.append({
                "problem_id": problem.problem_id,
                "response_t1": rollout["response_t1"],
                "response_t2": rollout["response_t2"],
                "reward_t1": rollout["reward_t1"],
                "reward_t2": rollout["reward_t2"],
                "correct_t1": c1,
                "correct_t2": c2,
            })

    metrics = compute_self_correction_metrics(correct_t1, correct_t2)
    edit_stats = compute_edit_distance_stats(responses_t1, responses_t2)
    metrics["edit_distance"] = edit_stats

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# MBPP-R offline repair evaluation (Table 3)
# ---------------------------------------------------------------------------

def evaluate_mbpp_repair(
    policy: LLMPolicy,
    repair_problems: List[Dict],
    temperature: float = 0.0,
) -> Dict[str, float]:
    """
    Evaluate on MBPP-R: offline repair task (Ni et al., 2024).

    Each problem provides a fixed incorrect first-attempt program; the model
    must repair it in the second turn.
    """
    policy.model.eval()
    correct = []

    with torch.no_grad():
        for item in tqdm(repair_problems, desc="MBPP-R"):
            response_t2 = policy.generate(item["prompt_t2"], temperature=temperature)
            reward = compute_reward(
                response_t2,
                {"test_list": item["test_list"]},
                "mbpp",
            )
            correct.append(reward > 0.5)

    return {"mbpp_r_accuracy": sum(correct) / len(correct)}


# ---------------------------------------------------------------------------
# Inference-compute scaling (Section 6.2)
# ---------------------------------------------------------------------------

def majority_vote(answers: List[str]) -> str:
    """Select the most common answer from a list (self-consistency)."""
    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def _extract_final_answer(response: str, task: str) -> str:
    """Extract the final answer string for majority voting."""
    import re
    if task == "math":
        match = re.search(
            r"[Ff]inal [Aa]nswer[:\s]+[Tt]he final answer is\s*\$?([^$.\n]+)\$?",
            response,
        )
        if match:
            return match.group(1).strip()
        match = re.search(r"\\boxed\{(.+?)\}", response)
        if match:
            return match.group(1).strip()
    return response.strip()


def evaluate_inference_scaling(
    policy: LLMPolicy,
    problems: List[Problem],
    reward_fn,
    k_values: List[int],
    temperature: float = 0.7,
    use_self_correction: bool = True,
) -> Dict[str, List[float]]:
    """
    Evaluate inference-compute scaling (Section 6.2, Figure 1 right).

    Compares two strategies:
    1. Parallel: sample K solutions independently, majority vote
    2. Sequential + correction: sample K/2 solutions, self-correct each,
       majority vote over 2K responses

    Args:
        k_values: list of total sample budgets to evaluate
        use_self_correction: if True, use sequential self-correction strategy

    Returns:
        dict mapping strategy name to list of accuracies (one per k value)
    """
    policy.model.eval()
    results_parallel = []
    results_sequential = []

    for k in k_values:
        parallel_correct = 0
        sequential_correct = 0

        with torch.no_grad():
            for problem in tqdm(problems, desc=f"Scaling k={k}", leave=False):
                metadata = dict(problem.metadata or {})
                metadata["answer"] = problem.answer

                # Strategy 1: parallel sampling, majority vote over K samples
                parallel_responses = [
                    policy.generate(problem.prompt_t1, temperature=temperature)
                    for _ in range(k)
                ]
                parallel_answers = [
                    _extract_final_answer(r, problem.task) for r in parallel_responses
                ]
                voted_answer = majority_vote(parallel_answers)
                # Check if voted answer is correct
                voted_response = next(
                    r for r, a in zip(parallel_responses, parallel_answers)
                    if a == voted_answer
                )
                if reward_fn(voted_response, metadata, problem.task) > 0.5:
                    parallel_correct += 1

                if use_self_correction:
                    # Strategy 2: K/2 parallel samples, each self-corrected once
                    # majority vote over all 2*(K/2) = K responses
                    k_half = max(1, k // 2)
                    all_responses = []
                    all_answers = []
                    for _ in range(k_half):
                        r1 = policy.generate(problem.prompt_t1, temperature=temperature)
                        prompt_t2 = problem.build_prompt_t2(r1)
                        r2 = policy.generate(prompt_t2, temperature=temperature)
                        all_responses.extend([r1, r2])
                        all_answers.extend([
                            _extract_final_answer(r1, problem.task),
                            _extract_final_answer(r2, problem.task),
                        ])
                    voted_answer_seq = majority_vote(all_answers)
                    voted_response_seq = next(
                        r for r, a in zip(all_responses, all_answers)
                        if a == voted_answer_seq
                    )
                    if reward_fn(voted_response_seq, metadata, problem.task) > 0.5:
                        sequential_correct += 1

        n = len(problems)
        results_parallel.append(parallel_correct / n)
        if use_self_correction:
            results_sequential.append(sequential_correct / n)

    output = {"parallel": results_parallel}
    if use_self_correction:
        output["sequential_with_correction"] = results_sequential
    return output


# ---------------------------------------------------------------------------
# Multi-attempt scaling (Appendix A.1, Figure 8)
# ---------------------------------------------------------------------------

def evaluate_multi_attempt(
    policy: LLMPolicy,
    problems: List[Problem],
    reward_fn,
    num_attempts: int = 10,
    temperature: float = 0.0,
) -> List[float]:
    """
    Evaluate performance over multiple sequential self-correction attempts
    (Appendix A.1, Figure 8).

    Returns a list of accuracies, one per attempt.
    """
    policy.model.eval()
    per_attempt_correct = [[] for _ in range(num_attempts)]

    with torch.no_grad():
        for problem in tqdm(problems, desc="Multi-attempt eval"):
            metadata = dict(problem.metadata or {})
            metadata["answer"] = problem.answer

            # First attempt
            response = policy.generate(problem.prompt_t1, temperature=temperature)
            reward = reward_fn(response, metadata, problem.task)
            per_attempt_correct[0].append(reward > 0.5)

            # Subsequent attempts: each uses the previous response as context
            current_prompt = problem.prompt_t1
            current_response = response
            for attempt in range(1, num_attempts):
                next_prompt = problem.build_prompt_t2(current_response)
                next_response = policy.generate(next_prompt, temperature=temperature)
                reward = reward_fn(next_response, metadata, problem.task)
                per_attempt_correct[attempt].append(reward > 0.5)
                current_prompt = next_prompt
                current_response = next_response

    n = len(problems)
    return [sum(correct) / n for correct in per_attempt_correct]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_reward_fn(task: str):
    def reward_fn(response: str, metadata: dict, task_name: str) -> float:
        return compute_reward(response, metadata, task_name)
    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate SCoRe")
    parser.add_argument("--task", type=str, default="math",
                        choices=["math", "mbpp", "humaneval"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_attempts", type=int, default=2,
                        help="Number of self-correction attempts (default: 2)")
    parser.add_argument("--eval_scaling", action="store_true",
                        help="Run inference-compute scaling evaluation")
    parser.add_argument("--mbpp_repair_path", type=str, default=None,
                        help="Path to MBPP-R dataset for offline repair eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading model from {args.checkpoint}...")
    policy, _ = load_policy_and_ref(args.checkpoint)
    reward_fn = build_reward_fn(args.task)

    print("Loading dataset...")
    if args.task == "math":
        _, eval_problems = load_math_dataset(seed=args.seed)
    elif args.task == "mbpp":
        _, eval_problems = load_mbpp_dataset(seed=args.seed)
    elif args.task == "humaneval":
        eval_problems = load_humaneval_dataset()

    os.makedirs(args.output_dir, exist_ok=True)

    # Main self-correction evaluation
    print("Running self-correction evaluation...")
    output_path = os.path.join(args.output_dir, f"{args.task}_results.json")
    metrics = evaluate_on_problems(
        policy, eval_problems, reward_fn,
        temperature=args.temperature,
        output_path=output_path,
    )

    print("\n=== Self-Correction Results ===")
    print(f"Accuracy@t1:    {metrics['acc_t1']:.4f} ({metrics['acc_t1']*100:.1f}%)")
    print(f"Accuracy@t2:    {metrics['acc_t2']:.4f} ({metrics['acc_t2']*100:.1f}%)")
    print(f"Δ(t1, t2):      {metrics['delta_t1_t2']:.4f} ({metrics['delta_t1_t2']*100:.1f}%)")
    print(f"Δ^(i→c)(t1,t2): {metrics['delta_i2c']:.4f} ({metrics['delta_i2c']*100:.1f}%)")
    print(f"Δ^(c→i)(t1,t2): {metrics['delta_c2i']:.4f} ({metrics['delta_c2i']*100:.1f}%)")
    print(f"Edit dist (mean): {metrics['edit_distance']['mean']:.4f}")
    print(f"Edit dist (frac_zero): {metrics['edit_distance']['frac_zero']:.4f}")

    # MBPP-R offline repair
    if args.mbpp_repair_path:
        print("\nRunning MBPP-R offline repair evaluation...")
        repair_problems = load_mbpp_repair_dataset(args.mbpp_repair_path)
        repair_metrics = evaluate_mbpp_repair(policy, repair_problems, temperature=args.temperature)
        print(f"MBPP-R Accuracy: {repair_metrics['mbpp_r_accuracy']:.4f}")
        with open(os.path.join(args.output_dir, "mbpp_r_results.json"), "w") as f:
            json.dump(repair_metrics, f, indent=2)

    # Multi-attempt scaling (Appendix A.1)
    if args.num_attempts > 2:
        print(f"\nRunning {args.num_attempts}-attempt evaluation...")
        attempt_accs = evaluate_multi_attempt(
            policy, eval_problems, reward_fn,
            num_attempts=args.num_attempts,
            temperature=args.temperature,
        )
        print("Per-attempt accuracies:", [f"{a:.4f}" for a in attempt_accs])
        with open(os.path.join(args.output_dir, "multi_attempt_results.json"), "w") as f:
            json.dump({"attempt_accuracies": attempt_accs}, f, indent=2)

    # Inference-compute scaling (Section 6.2)
    if args.eval_scaling:
        print("\nRunning inference-compute scaling evaluation...")
        k_values = [1, 2, 4, 8, 16, 32]
        scaling_results = evaluate_inference_scaling(
            policy, eval_problems[:100], reward_fn,  # subset for speed
            k_values=k_values,
            temperature=0.7,
        )
        print("Parallel sampling accuracies:", scaling_results["parallel"])
        if "sequential_with_correction" in scaling_results:
            print("Sequential+correction accuracies:", scaling_results["sequential_with_correction"])
        with open(os.path.join(args.output_dir, "scaling_results.json"), "w") as f:
            json.dump({"k_values": k_values, **scaling_results}, f, indent=2)

    # Save all metrics
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
