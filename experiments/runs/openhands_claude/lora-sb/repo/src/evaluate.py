"""Evaluation utilities for all three benchmarks.

Math Reasoning: exact match on final numeric answer (GSM8K, MATH)
Commonsense Reasoning: accuracy on 8 datasets
NLU: task-specific metrics (accuracy, Pearson correlation, Matthews correlation)
"""

from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Math Reasoning Evaluation
# ---------------------------------------------------------------------------

def extract_answer_gsm8k(text: str) -> Optional[str]:
    """Extract the final numeric answer from GSM8K model output."""
    # Look for #### pattern (GSM8K format)
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fallback: last number in the text
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if numbers:
        return numbers[-1]
    return None


def extract_answer_math(text: str) -> Optional[str]:
    """Extract the final answer from MATH model output (boxed format)."""
    # Look for \boxed{...} pattern
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()

    # Fallback: last number
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(answer: str) -> str:
    """Normalize a numeric answer string for comparison."""
    answer = answer.strip().lower()
    answer = answer.replace(",", "")
    # Remove trailing zeros after decimal
    try:
        val = float(answer)
        return str(val)
    except ValueError:
        return answer


def evaluate_math_generation(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    device: torch.device,
    task: str = "gsm8k",
    max_new_tokens: int = 256,
    batch_size: int = 4,
    prompt_template: str = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{question}\n\n### Response:"
    ),
) -> Dict[str, float]:
    """Evaluate model on GSM8K or MATH using exact match on final answer.

    Args:
        model: Fine-tuned causal LM.
        tokenizer: Tokenizer.
        dataset: Evaluation dataset.
        device: Compute device.
        task: 'gsm8k' or 'math'.
        max_new_tokens: Max tokens to generate.
        batch_size: Inference batch size.
        prompt_template: Prompt format string with {question} placeholder.

    Returns:
        Dict with 'accuracy' key.
    """
    model.eval()
    correct = 0
    total = 0

    extract_fn = extract_answer_gsm8k if task == "gsm8k" else extract_answer_math

    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {task}"):
        batch_data = dataset[i: i + batch_size]

        if task == "gsm8k":
            questions = batch_data["question"]
            gold_answers = [extract_answer_gsm8k(ans) for ans in batch_data["answer"]]
        else:
            questions = batch_data["problem"]
            gold_answers = [extract_answer_math(sol) for sol in batch_data["solution"]]

        prompts = [prompt_template.format(question=q) for q in questions]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated part
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for pred_text, gold in zip(decoded, gold_answers):
            pred = extract_fn(pred_text)
            if pred is not None and gold is not None:
                if normalize_answer(pred) == normalize_answer(gold):
                    correct += 1
            total += 1

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return {"accuracy": accuracy}


# ---------------------------------------------------------------------------
# Commonsense Reasoning Evaluation
# ---------------------------------------------------------------------------

COMMONSENSE_ANSWER_MAP = {
    "boolq": {"true": "yes", "false": "no", "yes": "yes", "no": "no"},
    "piqa": {"0": "solution1", "1": "solution2"},
    "social_i_qa": {"1": "answerA", "2": "answerB", "3": "answerC"},
    "hellaswag": {"0": "0", "1": "1", "2": "2", "3": "3"},
    "winogrande": {"1": "option1", "2": "option2"},
    "ARC-Easy": {"A": "A", "B": "B", "C": "C", "D": "D"},
    "ARC-Challenge": {"A": "A", "B": "B", "C": "C", "D": "D"},
    "openbookqa": {"A": "A", "B": "B", "C": "C", "D": "D"},
}


def evaluate_commonsense(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    device: torch.device,
    dataset_name: str,
    max_new_tokens: int = 32,
    batch_size: int = 8,
    prompt_template: str = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
) -> Dict[str, float]:
    """Evaluate model on a commonsense reasoning dataset.

    The model generates the answer and we check if it matches the gold label.
    """
    model.eval()
    correct = 0
    total = 0

    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {dataset_name}"):
        batch_data = dataset[i: i + batch_size]
        instructions = _format_commonsense_instructions(batch_data, dataset_name)
        gold_answers = _get_commonsense_gold(batch_data, dataset_name)

        prompts = [prompt_template.format(instruction=inst) for inst in instructions]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for pred_text, gold in zip(decoded, gold_answers):
            pred = pred_text.strip().lower()
            gold_lower = str(gold).strip().lower()
            if pred.startswith(gold_lower) or gold_lower in pred:
                correct += 1
            total += 1

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return {"accuracy": accuracy}


def _format_commonsense_instructions(batch: Dict, dataset_name: str) -> List[str]:
    """Format commonsense examples as instructions."""
    instructions = []
    n = len(batch[list(batch.keys())[0]])

    for i in range(n):
        if dataset_name == "boolq":
            inst = f"{batch['passage'][i]}\nQuestion: {batch['question'][i]}?\nAnswer:"
        elif dataset_name == "piqa":
            inst = f"Goal: {batch['goal'][i]}\nSolution 1: {batch['sol1'][i]}\nSolution 2: {batch['sol2'][i]}\nWhich solution is better?"
        elif dataset_name == "social_i_qa":
            inst = (f"Context: {batch['context'][i]}\nQuestion: {batch['question'][i]}\n"
                    f"A: {batch['answerA'][i]}\nB: {batch['answerB'][i]}\nC: {batch['answerC'][i]}\nAnswer:")
        elif dataset_name == "hellaswag":
            inst = f"{batch['ctx'][i]}\nComplete the sentence:"
        elif dataset_name == "winogrande":
            inst = f"{batch['sentence'][i]}\nOption 1: {batch['option1'][i]}\nOption 2: {batch['option2'][i]}\nAnswer:"
        elif dataset_name in ("ARC-Easy", "ARC-Challenge"):
            choices = batch["choices"][i]
            choice_text = "\n".join(f"{l}: {t}" for l, t in zip(choices["label"], choices["text"]))
            inst = f"Question: {batch['question'][i]}\n{choice_text}\nAnswer:"
        elif dataset_name == "openbookqa":
            choices = batch["choices"][i]
            choice_text = "\n".join(f"{l}: {t}" for l, t in zip(choices["label"], choices["text"]))
            inst = f"Question: {batch['question_stem'][i]}\n{choice_text}\nAnswer:"
        else:
            inst = str(batch.get("instruction", [""])[i])
        instructions.append(inst)
    return instructions


def _get_commonsense_gold(batch: Dict, dataset_name: str) -> List[str]:
    """Get gold answers for commonsense datasets."""
    n = len(batch[list(batch.keys())[0]])
    gold = []
    for i in range(n):
        if dataset_name == "boolq":
            gold.append("yes" if batch["answer"][i] else "no")
        elif dataset_name == "piqa":
            gold.append(f"solution{batch['label'][i] + 1}")
        elif dataset_name == "social_i_qa":
            label = batch["label"][i]
            gold.append(["A", "B", "C"][int(label) - 1])
        elif dataset_name == "hellaswag":
            gold.append(batch["label"][i])
        elif dataset_name == "winogrande":
            gold.append(f"option{batch['answer'][i]}")
        elif dataset_name in ("ARC-Easy", "ARC-Challenge"):
            gold.append(batch["answerKey"][i])
        elif dataset_name == "openbookqa":
            gold.append(batch["answerKey"][i])
        else:
            gold.append(str(batch.get("output", [""])[i]))
    return gold


# ---------------------------------------------------------------------------
# GLUE Evaluation
# ---------------------------------------------------------------------------

def compute_glue_metrics(
    task_name: str,
    predictions: List[Any],
    labels: List[Any],
) -> Dict[str, float]:
    """Compute task-specific GLUE metrics.

    - CoLA: Matthews correlation coefficient
    - STS-B: Pearson correlation
    - Others: Accuracy
    """
    preds = np.array(predictions)
    labs = np.array(labels)

    if task_name == "cola":
        mcc = matthews_corrcoef(labs, preds)
        return {"matthews_correlation": float(mcc)}
    elif task_name == "stsb":
        # Regression: Pearson correlation
        corr, _ = pearsonr(preds, labs)
        return {"pearson_correlation": float(corr)}
    else:
        accuracy = float(np.mean(preds == labs))
        return {"accuracy": accuracy}


def make_glue_metrics_fn(task_name: str):
    """Return a metrics function for a given GLUE task."""
    def fn(predictions, labels):
        return compute_glue_metrics(task_name, predictions, labels)
    return fn
