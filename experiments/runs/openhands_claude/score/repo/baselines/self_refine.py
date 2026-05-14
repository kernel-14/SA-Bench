"""
Self-Refine baseline for self-correction.

Madaan et al. (2023) — prompting-based self-correction without any fine-tuning.

The model is prompted to:
1. Generate an initial response
2. Critique its own response
3. Refine the response based on the critique

In the paper's setting (intrinsic self-correction), the model does not receive
any external feedback — it must deduce whether a mistake exists and correct it.
This is equivalent to the second-turn prompt used in SCoRe.

Note: The paper shows Self-Refine achieves Δ(t1,t2) = -1.0% on MATH and
-1.2% on HumanEval, confirming that prompting alone is insufficient.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from data import (
    Problem,
    load_humaneval_dataset,
    load_math_dataset,
    load_mbpp_dataset,
)
from evaluate import compute_self_correction_metrics, compute_edit_distance_stats
from model import LLMPolicy, load_policy_and_ref
from rewards import compute_reward


class SelfRefineEvaluator:
    """
    Evaluates self-correction performance using the Self-Refine prompting approach.

    Unlike SCoRe, this requires no training — it uses the base model's
    instruction-following ability to self-correct.

    The self-correction instruction is identical to the one used in SCoRe
    (Appendix C), ensuring a fair comparison.
    """

    def __init__(self, policy: LLMPolicy):
        self.policy = policy

    def evaluate(
        self,
        problems: List[Problem],
        temperature: float = 0.0,
        reward_fn=None,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Run Self-Refine evaluation on a list of problems.

        For each problem:
        1. Generate first attempt (turn 1)
        2. Apply self-correction instruction (turn 2)
        3. Evaluate both attempts

        Returns self-correction metrics.
        """
        self.policy.model.eval()
        correct_t1 = []
        correct_t2 = []
        responses_t1 = []
        responses_t2 = []
        results = []

        with torch.no_grad():
            for problem in tqdm(problems, desc="Self-Refine"):
                # Turn 1: initial response
                response_t1 = self.policy.generate(
                    problem.prompt_t1, temperature=temperature
                )

                # Turn 2: self-correction (using the same prompt as SCoRe)
                prompt_t2 = problem.build_prompt_t2(response_t1)
                response_t2 = self.policy.generate(prompt_t2, temperature=temperature)

                metadata = dict(problem.metadata or {})
                metadata["answer"] = problem.answer

                r1 = reward_fn(response_t1, metadata, problem.task) if reward_fn else 0.0
                r2 = reward_fn(response_t2, metadata, problem.task) if reward_fn else 0.0

                c1 = r1 > 0.5
                c2 = r2 > 0.5
                correct_t1.append(c1)
                correct_t2.append(c2)
                responses_t1.append(response_t1)
                responses_t2.append(response_t2)

                results.append({
                    "problem_id": problem.problem_id,
                    "response_t1": response_t1,
                    "response_t2": response_t2,
                    "reward_t1": r1,
                    "reward_t2": r2,
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Self-Refine baseline")
    parser.add_argument("--task", type=str, default="math",
                        choices=["math", "mbpp", "humaneval"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/self_refine")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading model from {args.model}...")
    policy, _ = load_policy_and_ref(args.model)

    print("Loading dataset...")
    if args.task == "math":
        _, eval_problems = load_math_dataset(seed=args.seed)
    elif args.task == "mbpp":
        _, eval_problems = load_mbpp_dataset(seed=args.seed)
    elif args.task == "humaneval":
        eval_problems = load_humaneval_dataset()

    def reward_fn(response, metadata, task):
        return compute_reward(response, metadata, task)

    evaluator = SelfRefineEvaluator(policy)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.task}_results.json")

    print("Running Self-Refine evaluation...")
    metrics = evaluator.evaluate(
        eval_problems,
        temperature=args.temperature,
        reward_fn=reward_fn,
        output_path=output_path,
    )

    print("\n=== Self-Refine Results ===")
    print(f"Accuracy@t1:    {metrics['acc_t1']:.4f} ({metrics['acc_t1']*100:.1f}%)")
    print(f"Accuracy@t2:    {metrics['acc_t2']:.4f} ({metrics['acc_t2']*100:.1f}%)")
    print(f"Δ(t1, t2):      {metrics['delta_t1_t2']:.4f} ({metrics['delta_t1_t2']*100:.1f}%)")
    print(f"Δ^(i→c)(t1,t2): {metrics['delta_i2c']:.4f} ({metrics['delta_i2c']*100:.1f}%)")
    print(f"Δ^(c→i)(t1,t2): {metrics['delta_c2i']:.4f} ({metrics['delta_c2i']*100:.1f}%)")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
