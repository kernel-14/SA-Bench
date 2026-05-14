"""
Evaluation script for MA-PPO trained models.

Evaluates models using:
1. Reward model (RM) scores on validation set
2. Best-of-N (rejection sampling) evaluation
3. Pass@k for code generation (APPS dataset)

Usage:
    python evaluate.py --config configs/tldr_2b.yaml --checkpoint output/checkpoint-4600
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_utils import (
    TLDRDataset,
    HHRLHFDataset,
    WebGPTDataset,
    APPSDataset,
    split_dataset,
    collate_fn_pad,
    compute_code_reward,
)
from reward_model import RewardModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> list:
    data = []
    path = Path(data_path)
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        with open(path) as f:
            data = json.load(f)
    return data


@torch.no_grad()
def evaluate_rm_score(
    policy_model,
    reward_model: RewardModel,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 1.0,
    top_k: int = 50,
    device: str = "cuda",
) -> float:
    """
    Evaluate the average reward model score on a set of prompts.

    Args:
        policy_model: The policy model to generate responses.
        reward_model: The reward model to score responses.
        tokenizer: Tokenizer.
        prompts: List of prompt strings.
        max_new_tokens: Maximum response length.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        top_k: Top-k sampling parameter.
        device: Device string.

    Returns:
        Average RM score across all prompts.
    """
    policy_model.eval()
    reward_model.eval()

    all_scores = []

    for prompt in prompts:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        gen_ids = policy_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        full_mask = (gen_ids != tokenizer.pad_token_id).long()
        score = reward_model(gen_ids, full_mask)
        all_scores.append(score.item())

    return sum(all_scores) / len(all_scores) if all_scores else 0.0


@torch.no_grad()
def evaluate_best_of_n(
    policy_model,
    reward_model: RewardModel,
    tokenizer,
    prompts: List[str],
    n: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    device: str = "cuda",
) -> float:
    """
    Best-of-N (rejection sampling) evaluation.

    Generates N responses per prompt and selects the one with the highest
    reward model score.

    Args:
        n: Number of samples per prompt.
        temperature: Sampling temperature (typically higher for diversity).

    Returns:
        Average best-of-N RM score.
    """
    policy_model.eval()
    reward_model.eval()

    all_best_scores = []

    for prompt in prompts:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        best_score = float("-inf")

        for _ in range(n):
            gen_ids = policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            full_mask = (gen_ids != tokenizer.pad_token_id).long()
            score = reward_model(gen_ids, full_mask).item()
            best_score = max(best_score, score)

        all_best_scores.append(best_score)

    return sum(all_best_scores) / len(all_best_scores) if all_best_scores else 0.0


@torch.no_grad()
def evaluate_pass_at_k(
    policy_model,
    tokenizer,
    test_data: list,
    k: int = 1,
    n_samples: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    device: str = "cuda",
) -> dict:
    """
    Evaluate pass@k metric for code generation on APPS dataset.

    pass@k = E[1 - C(n-c, k) / C(n, k)]
    where n = total samples, c = correct samples.

    Args:
        k: Number of attempts to consider.
        n_samples: Total samples to generate per problem.

    Returns:
        Dict with pass@k scores by difficulty level.
    """
    import math

    def pass_at_k_estimate(n: int, c: int, k: int) -> float:
        """Unbiased estimator for pass@k."""
        if n - c < k:
            return 1.0
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

    policy_model.eval()

    results_by_difficulty = {"introductory": [], "interview": [], "competition": []}
    all_results = []

    for item in test_data:
        question = item.get("question", "")
        difficulty = item.get("difficulty", "interview")
        test_cases_str = item.get("input_output", "{}")
        try:
            test_cases = json.loads(test_cases_str) if isinstance(test_cases_str, str) else test_cases_str
        except json.JSONDecodeError:
            test_cases = {}

        prompt = f"# Problem:\n{question}\n\n# Solution:\n"
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=600,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        n_correct = 0
        for _ in range(n_samples):
            gen_ids = policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_code = tokenizer.decode(
                gen_ids[0, input_ids.size(1):], skip_special_tokens=True
            )
            reward = compute_code_reward(generated_code, test_cases)
            if reward > 0:  # Passed at least some tests
                n_correct += 1

        pass_k = pass_at_k_estimate(n_samples, n_correct, k)
        all_results.append(pass_k)

        if difficulty in results_by_difficulty:
            results_by_difficulty[difficulty].append(pass_k)

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "all": mean(all_results),
        "introductory": mean(results_by_difficulty["introductory"]),
        "interview": mean(results_by_difficulty["interview"]),
        "competition": mean(results_by_difficulty["competition"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to policy checkpoint")
    parser.add_argument("--rm_checkpoint", default=None, help="Path to RM checkpoint")
    parser.add_argument("--eval_type", default="rm_score",
                        choices=["rm_score", "best_of_n", "pass_at_k"])
    parser.add_argument("--n_samples", type=int, default=8, help="N for best-of-N")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_eval_samples", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16
    ).to(device)

    data = load_data(cfg["data_path"])
    task = cfg["task"]

    if args.eval_type in ("rm_score", "best_of_n"):
        # Load reward model
        rm_ckpt = args.rm_checkpoint or cfg.get("rm_checkpoint")
        rm_base = AutoModelForCausalLM.from_pretrained(
            rm_ckpt, torch_dtype=torch.bfloat16
        ).to(device)
        hidden_size = rm_base.config.hidden_size
        reward_model = RewardModel(rm_base, hidden_size).to(device)

        # Load RM weights if available
        rm_weights_path = os.path.join(rm_ckpt, "reward_model.pt")
        if os.path.exists(rm_weights_path):
            reward_model.load_state_dict(torch.load(rm_weights_path, map_location=device))

        # Get validation prompts
        _, _, ppo_data = split_dataset(data)
        val_data = ppo_data[:args.max_eval_samples]

        if task == "tldr":
            dataset = TLDRDataset(val_data, tokenizer, mode="ppo")
        elif task == "hh_rlhf":
            dataset = HHRLHFDataset(val_data, tokenizer, mode="ppo")
        elif task == "webgpt":
            dataset = WebGPTDataset(val_data, tokenizer, mode="ppo")
        else:
            raise ValueError(f"Unknown task: {task}")

        prompts = [dataset[i]["prompt"] for i in range(len(dataset))]

        if args.eval_type == "rm_score":
            score = evaluate_rm_score(
                policy_model, reward_model, tokenizer, prompts,
                temperature=args.temperature, device=str(device)
            )
            logger.info(f"Average RM Score: {score:.4f}")
        else:
            score = evaluate_best_of_n(
                policy_model, reward_model, tokenizer, prompts,
                n=args.n_samples, temperature=args.temperature, device=str(device)
            )
            logger.info(f"Best-of-{args.n_samples} RM Score: {score:.4f}")

    elif args.eval_type == "pass_at_k":
        # APPS evaluation
        test_data = data[:args.max_eval_samples]
        results = evaluate_pass_at_k(
            policy_model, tokenizer, test_data,
            k=1, n_samples=args.n_samples,
            temperature=args.temperature, device=str(device)
        )
        logger.info(f"Pass@1 Results: {json.dumps(results, indent=2)}")

        results_k5 = evaluate_pass_at_k(
            policy_model, tokenizer, test_data,
            k=5, n_samples=max(args.n_samples, 5),
            temperature=args.temperature, device=str(device)
        )
        logger.info(f"Pass@5 Results: {json.dumps(results_k5, indent=2)}")


if __name__ == "__main__":
    main()
