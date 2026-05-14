"""
Evaluation utilities for MA-RLHF.

Implements (§4.1):
  - RM score computation on a validation set
  - Best-of-N (rejection) sampling (§4.4)
  - pass@k metric for code generation (§4.5)
  - GPT-4 pairwise evaluation (§4.1, Appendix F.1)
  - Human evaluation win-rate aggregation (Appendix F.2)
  - L2-norm of advantages and Q-values (§4.5)
"""

import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import PreTrainedTokenizer

from config import EvalConfig, TASK_TLDR, TASK_HH_RLHF, TASK_WEBGPT, TASK_APPS
from model import PolicyModel, RewardModel


# ---------------------------------------------------------------------------
# RM score evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_rm_scores_on_dataset(
    policy: PolicyModel,
    reward_model: RewardModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    device: torch.device,
    num_samples: int = 2000,
    temperature: float = 0.8,
    top_p: float = 1.0,
    top_k: int = 50,
    max_new_tokens: int = 512,
    batch_size: int = 4,
) -> float:
    """Compute mean RM score on a random subset of the dataset (§4.1).

    Randomly samples `num_samples` prompts, generates responses, and scores
    them with the reward model.
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:num_samples]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    policy.eval()
    reward_model.eval()
    all_scores = []

    for batch in loader:
        prompt_ids = batch["input_ids"].to(device)
        prompt_mask = batch["attention_mask"].to(device)

        generated = policy.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        full_mask = torch.ones_like(generated)
        scores = reward_model(generated, full_mask)
        all_scores.extend(scores.cpu().tolist())

    return float(np.mean(all_scores))


# ---------------------------------------------------------------------------
# Best-of-N (rejection) sampling (§4.4)
# ---------------------------------------------------------------------------

@torch.no_grad()
def best_of_n_sampling(
    policy: PolicyModel,
    reward_model: RewardModel,
    tokenizer: PreTrainedTokenizer,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    n: int = 8,
    temperature: float = 0.8,
    top_p: float = 1.0,
    top_k: int = 50,
    max_new_tokens: int = 512,
    device: torch.device = None,
) -> Tuple[torch.Tensor, float]:
    """Generate N responses and return the one with the highest RM score.

    Args:
        prompt_ids: (1, prompt_len)
        prompt_mask: (1, prompt_len)
        n: number of samples.

    Returns:
        (best_response_ids, best_rm_score)
    """
    if device is None:
        device = prompt_ids.device

    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)

    best_score = float("-inf")
    best_ids = None

    for _ in range(n):
        generated = policy.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        full_mask = torch.ones_like(generated)
        score = reward_model(generated, full_mask).item()
        if score > best_score:
            best_score = score
            best_ids = generated

    return best_ids, best_score


@torch.no_grad()
def evaluate_best_of_n(
    policy: PolicyModel,
    reward_model: RewardModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    device: torch.device,
    n_values: List[int] = None,
    temperatures: List[float] = None,
    num_eval_samples: int = 200,
) -> Dict[str, float]:
    """Evaluate Best-of-N across multiple N and temperature values (§4.4, Figure 8).

    Returns a dict mapping "best_of_{n}_temp_{t}" → mean RM score.
    """
    if n_values is None:
        n_values = [4, 8, 16, 32]
    if temperatures is None:
        temperatures = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:num_eval_samples]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    results = {}
    for temp in temperatures:
        for n in n_values:
            scores = []
            for batch in loader:
                prompt_ids = batch["input_ids"].to(device)
                prompt_mask = batch["attention_mask"].to(device)
                _, score = best_of_n_sampling(
                    policy, reward_model, tokenizer,
                    prompt_ids, prompt_mask,
                    n=n, temperature=temp, device=device,
                )
                scores.append(score)
            key = f"best_of_{n}_temp_{temp}"
            results[key] = float(np.mean(scores))
            print(f"Best-of-{n}, T={temp}: {results[key]:.4f}")

    return results


# ---------------------------------------------------------------------------
# pass@k for code generation (§4.5)
# ---------------------------------------------------------------------------

def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al., 2021).

    pass@k = 1 - C(n-c, k) / C(n, k)
    where n = total samples, c = number that pass.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / (n - i) for i in range(c)) / 1.0


def evaluate_code_pass_at_k(
    policy: PolicyModel,
    tokenizer: PreTrainedTokenizer,
    test_dataset,
    device: torch.device,
    k_values: List[int] = None,
    num_samples_per_problem: int = 5,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 5,
    max_new_tokens: int = 512,
) -> Dict[str, float]:
    """Evaluate pass@k on the APPS test set (§4.5, Table 3).

    For each problem, generates `num_samples_per_problem` solutions and
    checks them against the test cases using the compiler signal.

    Returns dict with keys like "pass@1_all", "pass@5_intro", etc.
    """
    if k_values is None:
        k_values = [1, 5]

    from train_ppo import compute_apps_reward
    from config import CodeRewardConfig

    code_reward_cfg = CodeRewardConfig()
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    difficulty_results: Dict[str, List[Tuple[int, int]]] = {
        "introductory": [], "interview": [], "competition": [], "all": []
    }

    policy.eval()
    for batch in loader:
        prompt_ids = batch["input_ids"].to(device)
        prompt_mask = batch["attention_mask"].to(device)
        test_cases = batch["test_cases"][0]
        difficulty = batch["difficulty"][0]

        n_pass = 0
        for _ in range(num_samples_per_problem):
            generated = policy.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            code = tokenizer.decode(
                generated[0, prompt_ids.size(1):], skip_special_tokens=True
            )
            reward = compute_apps_reward(code, test_cases, code_reward_cfg)
            # A solution "passes" if it compiles and gets positive reward
            if reward > code_reward_cfg.partial_pass_base:
                n_pass += 1

        difficulty_results["all"].append((num_samples_per_problem, n_pass))
        diff_key = difficulty.lower() if difficulty.lower() in difficulty_results else "all"
        if diff_key != "all":
            difficulty_results[diff_key].append((num_samples_per_problem, n_pass))

    results = {}
    for diff, nc_list in difficulty_results.items():
        if not nc_list:
            continue
        for k in k_values:
            scores = [_estimate_pass_at_k(n, c, k) * 100 for n, c in nc_list]
            results[f"pass@{k}_{diff}"] = float(np.mean(scores))

    return results


# ---------------------------------------------------------------------------
# GPT-4 pairwise evaluation (§4.1, Appendix F.1)
# ---------------------------------------------------------------------------

TLDR_GPT4_PROMPT = """You will be given two summaries written for an article. Your task is to pick the better one between them, based on the four criteria. Please make sure you read and understand these instructions carefully.
Relevance - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.
Coherence - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."
Consistency - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts.
Fluency - the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
You should output single character to indicate which summary you think is better. 'A' stands for Summary A and 'B' stands for Summary B. If you think both summaries are equally good, output 'E'

Article / Post:{article}

Summary A:{summary_a}
Summary B:{summary_b}

Your Choice (only a single character):"""

HH_RLHF_GPT4_PROMPT = """For the following query to a chatbot assistant, which response is more helpful?
First provide a one-sentence comparison of the two responses and explain which you feel is more helpful. Second, on a new line, state only 'A' or 'B' to indicate which response is more helpful. If they are equally good or bad, state 'E'. Your response should use the json format, with "comparison" and "choice" as keys.

Query: {query}
Response A: {response_a}
Response B: {response_b}
Your Judgment:"""

WEBGPT_GPT4_PROMPT = """You will be given two response written for an question. Your task is to pick the better one between them, based on these criteria.
Factual accuracy - which answer is more factually accurate?
Coherence - which answer is easier to follow?
Usefulness overall - all things considered, which answer would be more helpful to the person who asked this question?
You should output with a json format where the key is the criteria and the value is the choice you made, using 'A' stands for Response A and 'B' stands for Response B. If you think both responses are equally good, output 'E'.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}
Your Judgment (you should also output the reason, note that you are allowed to think both responses are equally good, then output with 'E'):"""


def gpt4_pairwise_eval(
    task: str,
    prompts: List[str],
    responses_a: List[str],
    responses_b: List[str],
    openai_api_key: str,
    model: str = "gpt-4o-2024-05-13",
    randomize_order: bool = True,
) -> Dict[str, float]:
    """GPT-4 pairwise evaluation (§4.1, Appendix F.1).

    Randomly shuffles response order to mitigate position bias (§4.1).

    Returns:
        dict with keys 'win_rate_a', 'win_rate_b', 'tie_rate'.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required for GPT-4 evaluation")

    client = OpenAI(api_key=openai_api_key)

    wins_a = 0
    wins_b = 0
    ties = 0
    n = len(prompts)

    for i in range(n):
        prompt = prompts[i]
        resp_a = responses_a[i]
        resp_b = responses_b[i]

        # Randomize order to mitigate position bias
        swapped = False
        if randomize_order and random.random() < 0.5:
            resp_a, resp_b = resp_b, resp_a
            swapped = True

        if task == TASK_TLDR:
            user_msg = TLDR_GPT4_PROMPT.format(
                article=prompt, summary_a=resp_a, summary_b=resp_b
            )
        elif task == TASK_HH_RLHF:
            user_msg = HH_RLHF_GPT4_PROMPT.format(
                query=prompt, response_a=resp_a, response_b=resp_b
            )
        elif task == TASK_WEBGPT:
            user_msg = WEBGPT_GPT4_PROMPT.format(
                question=prompt, answer_a=resp_a, answer_b=resp_b
            )
        else:
            raise ValueError(f"GPT-4 eval not defined for task: {task}")

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=512,
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()

        # Parse choice
        choice = _parse_gpt4_choice(content, task)
        if swapped:
            if choice == "A":
                choice = "B"
            elif choice == "B":
                choice = "A"

        if choice == "A":
            wins_a += 1
        elif choice == "B":
            wins_b += 1
        else:
            ties += 1

    return {
        "win_rate_a": wins_a / n,
        "win_rate_b": wins_b / n,
        "tie_rate": ties / n,
    }


def _parse_gpt4_choice(content: str, task: str) -> str:
    """Extract A/B/E from GPT-4 response."""
    if task == TASK_TLDR:
        for char in content:
            if char in ("A", "B", "E"):
                return char
        return "E"
    else:
        try:
            data = json.loads(content)
            choice = data.get("choice", "E")
            if isinstance(choice, str) and choice in ("A", "B", "E"):
                return choice
        except Exception:
            pass
        for char in content:
            if char in ("A", "B", "E"):
                return char
        return "E"


# ---------------------------------------------------------------------------
# L2-norm analysis (§4.5, Figure 11)
# ---------------------------------------------------------------------------

def compute_l2_norm(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Compute L2 norm of a tensor, optionally masked."""
    if mask is not None:
        tensor = tensor * mask
    return tensor.norm(p=2).item()


def log_advantage_qvalue_norms(
    advantages: torch.Tensor,
    q_values: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    step: int = 0,
) -> Dict[str, float]:
    """Log L2 norms of advantages and Q-values (§4.5, Figure 11)."""
    return {
        "step": step,
        "l2_norm_advantages": compute_l2_norm(advantages, mask),
        "l2_norm_q_values": compute_l2_norm(q_values, mask),
    }


# ---------------------------------------------------------------------------
# Win-rate aggregation from human annotations
# ---------------------------------------------------------------------------

def compute_win_rate(
    annotations: List[str],
    model_a_label: str = "A",
    model_b_label: str = "B",
    tie_label: str = "tie",
) -> Dict[str, float]:
    """Compute win/tie/loss rates from a list of human annotation labels.

    Args:
        annotations: list of labels, each one of model_a_label, model_b_label, tie_label.

    Returns:
        dict with 'win_rate', 'tie_rate', 'loss_rate' for model A.
    """
    n = len(annotations)
    wins = sum(1 for a in annotations if a == model_a_label)
    ties = sum(1 for a in annotations if a == tie_label)
    losses = sum(1 for a in annotations if a == model_b_label)
    return {
        "win_rate": wins / n,
        "tie_rate": ties / n,
        "loss_rate": losses / n,
    }


# ---------------------------------------------------------------------------
# RM score distribution analysis (§4.4, Figure 10)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_rm_score_distribution(
    policy: PolicyModel,
    reward_model: RewardModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    device: torch.device,
    num_samples: int = 2000,
    temperature: float = 0.8,
    top_p: float = 1.0,
    top_k: int = 50,
    max_new_tokens: int = 512,
) -> List[float]:
    """Collect all RM scores for distribution analysis (Figure 3, 10)."""
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:num_samples]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=4, shuffle=False)

    policy.eval()
    reward_model.eval()
    all_scores = []

    for batch in loader:
        prompt_ids = batch["input_ids"].to(device)
        prompt_mask = batch["attention_mask"].to(device)
        generated = policy.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        full_mask = torch.ones_like(generated)
        scores = reward_model(generated, full_mask)
        all_scores.extend(scores.cpu().tolist())

    return all_scores


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_full_evaluation(
    policy: PolicyModel,
    reward_model: Optional[RewardModel],
    tokenizer: PreTrainedTokenizer,
    task: str,
    val_dataset,
    device: torch.device,
    eval_cfg: EvalConfig = None,
    openai_api_key: Optional[str] = None,
    output_dir: str = "eval_results",
) -> Dict:
    """Run the full evaluation suite for a given task."""
    if eval_cfg is None:
        eval_cfg = EvalConfig()

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # RM score
    if reward_model is not None:
        rm_score = compute_rm_scores_on_dataset(
            policy, reward_model, tokenizer, val_dataset, device,
            num_samples=eval_cfg.rm_eval_samples,
        )
        results["rm_score"] = rm_score
        print(f"RM Score: {rm_score:.4f}")

    # Best-of-N
    if reward_model is not None:
        bon_results = evaluate_best_of_n(
            policy, reward_model, tokenizer, val_dataset, device,
            n_values=eval_cfg.best_of_n_values,
            temperatures=eval_cfg.best_of_n_temperatures,
            num_eval_samples=100,
        )
        results["best_of_n"] = bon_results

    # Save results
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results
