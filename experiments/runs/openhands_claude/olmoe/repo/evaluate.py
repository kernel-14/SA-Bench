"""Evaluation utilities for OLMoE.

Implements the evaluation setup from §3 and Appendix C:

During pretraining (in-loop evaluation):
  - 0-shot completion/cloze formulation (CF)
  - Character-normalized scoring for most tasks
  - MMLU Var: varying few-shots (0-5) for smoother training signal
  - Tasks: ARC-C, ARC-E, BoolQ, COPA, CSQA, HellaSwag, MMLU, OBQA, PIQA, SciQ, SocialIQA, Winogrande

After pretraining (OLMES standard, Gu et al. 2024):
  - max(MCF, CF) formulation
  - 5-shot evaluation
  - PMI normalization for some tasks

After adaptation:
  - MMLU (0-shot EM)
  - GSM8k (8-shot CoT EM)
  - BBH (3-shot EM)
  - HumanEval (0-shot Pass@10)
  - AlpacaEval 1.0 (0-shot %win)
  - XSTest (0-shot F1)
  - IFEval (0-shot Loose Acc)
"""

import json
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import OLMoE


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: OLMoE,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    """Compute perplexity on a dataset.

    Used for validation loss tracking during pretraining (Figure 24 in paper).
    Evaluates on: Books, Reddit, Stack (Dolma 1.7 via Paloma), C4.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, labels=labels)
        ce_loss = out["ce_loss"]

        n_tokens = (labels != -100).sum().item()
        total_loss += ce_loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(1, total_tokens)
    return math.exp(avg_loss)


# ---------------------------------------------------------------------------
# Multiple choice evaluation (completion/cloze formulation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_completion(
    model: OLMoE,
    context: str,
    completion: str,
    tokenizer,
    device: torch.device,
    normalize: str = "none",
) -> float:
    """Score a completion given a context using language model probabilities.

    Args:
        model: OLMoE model
        context: prompt/context string
        completion: candidate completion string
        tokenizer: tokenizer
        device: compute device
        normalize: 'none', 'char' (per character), or 'token' (per token)
    Returns:
        log probability score (higher = better)
    """
    model.eval()
    context_ids = tokenizer.encode(context)
    completion_ids = tokenizer.encode(completion)
    full_ids = context_ids + completion_ids

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    out = model(input_ids=input_ids)
    logits = out["logits"][0]  # (seq_len, vocab)

    # Score only the completion tokens
    log_probs = F.log_softmax(logits, dim=-1)
    completion_start = len(context_ids)
    score = 0.0
    for i, token_id in enumerate(completion_ids):
        pos = completion_start + i - 1  # logit at position predicts next token
        if pos >= 0:
            score += log_probs[pos, token_id].item()

    if normalize == "char":
        score /= max(1, len(completion))
    elif normalize == "token":
        score /= max(1, len(completion_ids))

    return score


@torch.no_grad()
def evaluate_multiple_choice(
    model: OLMoE,
    examples: List[Dict],
    tokenizer,
    device: torch.device,
    formulation: str = "CF",
    normalize: str = "char",
    n_shot: int = 0,
) -> Dict[str, float]:
    """Evaluate on multiple-choice tasks.

    Supports:
    - CF (Completion/Cloze Formulation): rank answer strings by LM probability
    - MCF (Multiple-Choice Formulation): score answer labels A/B/C/D

    Args:
        model: OLMoE model
        examples: list of dicts with 'question', 'choices', 'answer_idx'
        tokenizer: tokenizer
        device: compute device
        formulation: 'CF' or 'MCF'
        normalize: 'none', 'char', or 'pmi'
        n_shot: number of few-shot examples
    Returns:
        dict with 'accuracy' and 'n_correct'
    """
    model.eval()
    n_correct = 0
    n_total = len(examples)

    for ex in examples:
        question = ex["question"]
        choices = ex["choices"]
        answer_idx = ex["answer_idx"]

        if formulation == "CF":
            scores = []
            for choice in choices:
                score = score_completion(
                    model, question, choice, tokenizer, device, normalize=normalize
                )
                scores.append(score)
            pred = scores.index(max(scores))

        elif formulation == "MCF":
            # Score answer labels A, B, C, D
            labels = ["A", "B", "C", "D"][:len(choices)]
            prompt = question + "\nAnswer:"
            scores = []
            for label in labels:
                score = score_completion(
                    model, prompt, f" {label}", tokenizer, device, normalize="none"
                )
                scores.append(score)
            pred = scores.index(max(scores))

        if pred == answer_idx:
            n_correct += 1

    accuracy = n_correct / max(1, n_total)
    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": n_total}


# ---------------------------------------------------------------------------
# MMLU evaluation
# ---------------------------------------------------------------------------

def load_mmlu_examples(data_path: str, split: str = "val") -> List[Dict]:
    """Load MMLU examples from file."""
    examples = []
    import os
    for subject_file in os.listdir(data_path):
        if not subject_file.endswith(".jsonl"):
            continue
        with open(os.path.join(data_path, subject_file)) as f:
            for line in f:
                ex = json.loads(line)
                if ex.get("split", split) == split:
                    examples.append({
                        "question": ex["question"],
                        "choices": ex["choices"],
                        "answer_idx": ex["answer_idx"],
                        "subject": ex.get("subject", ""),
                    })
    return examples


@torch.no_grad()
def evaluate_mmlu(
    model: OLMoE,
    data_path: str,
    tokenizer,
    device: torch.device,
    n_shot: int = 5,
    split: str = "test",
) -> Dict[str, float]:
    """Evaluate on MMLU (Hendrycks et al. 2021).

    Paper uses 5-shot MCF for after-pretraining evaluation (Table 4).
    During pretraining uses MMLU Var (0-5 shot CF) for smoother signal.
    """
    examples = load_mmlu_examples(data_path, split=split)
    results = evaluate_multiple_choice(
        model, examples, tokenizer, device,
        formulation="MCF", normalize="none", n_shot=n_shot,
    )
    return results


# ---------------------------------------------------------------------------
# HellaSwag evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_hellaswag(
    model: OLMoE,
    examples: List[Dict],
    tokenizer,
    device: torch.device,
    n_shot: int = 0,
) -> Dict[str, float]:
    """Evaluate on HellaSwag (Zellers et al. 2019).

    Uses CF with character normalization (Table 11 in paper).
    """
    return evaluate_multiple_choice(
        model, examples, tokenizer, device,
        formulation="CF", normalize="char", n_shot=n_shot,
    )


# ---------------------------------------------------------------------------
# ARC evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_arc(
    model: OLMoE,
    examples: List[Dict],
    tokenizer,
    device: torch.device,
    variant: str = "challenge",
) -> Dict[str, float]:
    """Evaluate on ARC-Challenge or ARC-Easy (Clark et al. 2018).

    ARC-Challenge: CF with PMI normalization (Table 11)
    ARC-Easy: CF with no normalization (Table 11)
    """
    normalize = "pmi" if variant == "challenge" else "none"
    # PMI normalization requires unconditional completion probability
    # For simplicity, use char normalization as approximation
    normalize_approx = "char" if normalize == "pmi" else "none"
    return evaluate_multiple_choice(
        model, examples, tokenizer, device,
        formulation="CF", normalize=normalize_approx,
    )


# ---------------------------------------------------------------------------
# GSM8k evaluation (chain-of-thought)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_gsm8k(
    model: OLMoE,
    examples: List[Dict],
    tokenizer,
    device: torch.device,
    n_shot: int = 8,
    max_new_tokens: int = 512,
) -> Dict[str, float]:
    """Evaluate on GSM8k (Cobbe et al. 2021) with 8-shot chain-of-thought.

    Extracts final numerical answer from generated text.
    """
    import re
    model.eval()
    n_correct = 0

    for ex in examples:
        prompt = ex["question"]
        gold_answer = str(ex["answer"]).strip()

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Extract last number from generated text
        numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", generated_text.replace(",", ""))
        pred = numbers[-1] if numbers else ""

        if pred == gold_answer:
            n_correct += 1

    accuracy = n_correct / max(1, len(examples))
    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": len(examples)}


# ---------------------------------------------------------------------------
# HumanEval evaluation (Pass@k)
# ---------------------------------------------------------------------------

def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Estimate pass@k given n samples and c correct ones.

    Uses the unbiased estimator from Chen et al. 2021.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k)) if n >= k else 0.0


@torch.no_grad()
def evaluate_humaneval(
    model: OLMoE,
    problems: List[Dict],
    tokenizer,
    device: torch.device,
    n_samples: int = 10,
    k: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
) -> Dict[str, float]:
    """Evaluate on HumanEval (Chen et al. 2021) with Pass@k.

    Paper uses Pass@10 (0-shot) for evaluation (Table 5).
    """
    import subprocess
    import tempfile

    model.eval()
    pass_at_k_scores = []

    for problem in problems:
        prompt = problem["prompt"]
        test_code = problem["test"]
        entry_point = problem["entry_point"]

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        n_correct = 0

        for _ in range(n_samples):
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion = tokenizer.decode(
                generated[0][input_ids.shape[1]:], skip_special_tokens=True
            )

            # Execute and test
            full_code = prompt + completion + "\n" + test_code
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(full_code)
                    tmp_path = f.name
                result = subprocess.run(
                    ["python", tmp_path],
                    timeout=10,
                    capture_output=True,
                )
                if result.returncode == 0:
                    n_correct += 1
            except Exception:
                pass

        pass_at_k_scores.append(estimate_pass_at_k(n_samples, n_correct, k))

    avg_pass_at_k = sum(pass_at_k_scores) / max(1, len(pass_at_k_scores))
    return {f"pass@{k}": avg_pass_at_k, "n_problems": len(problems)}


# ---------------------------------------------------------------------------
# Full evaluation suite
# ---------------------------------------------------------------------------

def run_pretraining_eval(
    model: OLMoE,
    eval_data: Dict[str, DataLoader],
    tokenizer,
    device: torch.device,
    step: int,
) -> Dict[str, float]:
    """Run in-loop evaluation during pretraining (Table 11 in paper).

    Evaluates on: ARC-C, ARC-E, BoolQ, COPA, CSQA, HellaSwag, MMLU, OBQA,
    PIQA, SciQ, SocialIQA, Winogrande.
    """
    results = {}

    for task_name, dataloader in eval_data.items():
        if task_name == "perplexity":
            ppl = compute_perplexity(model, dataloader, device, max_batches=100)
            results[f"ppl/{task_name}"] = ppl
        else:
            # Collect examples from dataloader
            examples = []
            for batch in dataloader:
                for i in range(len(batch.get("question", []))):
                    examples.append({
                        "question": batch["question"][i],
                        "choices": [c[i] for c in batch["choices"]],
                        "answer_idx": batch["answer_idx"][i].item(),
                    })

            if examples:
                task_results = evaluate_multiple_choice(
                    model, examples, tokenizer, device,
                    formulation="CF", normalize="char",
                )
                results[f"acc/{task_name}"] = task_results["accuracy"]

    results["step"] = step
    return results
