"""Evaluation utilities for LoRA-SB experiments.

Supports:
- GSM8K evaluation (exact match of final numeric answer)
- MATH evaluation (exact match of final answer)
- GLUE evaluation (task-specific metrics)
- Commonsense reasoning evaluation (accuracy on 8 tasks)
"""

import re
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from typing import Dict, Optional
from tqdm import tqdm

from data import (
    load_gsm8k,
    load_math,
    COMMONSENSE_TASKS,
    COMMONSENSE_DATASET_MAP,
    COMMONSENSE_CONFIG_MAP,
    COMMONSENSE_PROMPTS,
    COMMONSENSE_ANSWER_FIELDS,
    COMMONSENSE_CHOICES_FIELDS,
)
from datasets import load_dataset


def extract_number(text: str) -> Optional[float]:
    """Extract the last number from a text string.

    This is used for GSM8K and MATH evaluation where the model
    generates reasoning followed by a final numeric answer.

    Args:
        text: Generated text from the model.

    Returns:
        The extracted number or None.
    """
    matches = re.findall(r'-?\d+\.?\d*', text)
    if not matches:
        return None
    return float(matches[-1])


def extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract answer from GSM8K model output.

    Looks for patterns like '#### 42' or just the last number.
    """
    hash_match = re.search(r'####\s*(-?\d+\.?\d*)', text)
    if hash_match:
        return float(hash_match.group(1))

    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return float(numbers[-1])
    return None


def extract_math_answer(text: str) -> Optional[str]:
    """Extract answer from MATH model output.

    MATH answers are often in LaTeX boxed format or plain numbers.
    """
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.replace("%", "")
    try:
        num = float(answer)
        return str(int(num) if num == int(num) else round(num, 4))
    except ValueError:
        return answer.lower()


def evaluate_math_task(
    model,
    tokenizer: AutoTokenizer,
    task: str,
    device: torch.device,
    max_new_tokens: int = 256,
) -> float:
    """Evaluate on GSM8K or MATH dataset.

    Args:
        model: The trained model.
        tokenizer: The tokenizer.
        task: 'gsm8k' or 'math'.
        device: Device for computation.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Accuracy as a percentage.
    """
    if task == "gsm8k":
        dataset = load_gsm8k(tokenizer)
        answer_key = "answer"
        extract_fn = extract_gsm8k_answer
    elif task == "math":
        dataset = load_math(tokenizer)
        answer_key = "solution"
        extract_fn = extract_math_answer
    else:
        raise ValueError(f"Unknown math task: {task}")

    model.eval()
    correct = 0
    total = 0

    for i in tqdm(range(len(dataset)), desc=f"Evaluating {task.upper()}"):
        example = dataset[i]
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,
            )

        generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        predicted_answer = extract_fn(generated)

        if predicted_answer is not None:
            true_answer = example[answer_key]
            true_extracted = extract_fn(str(true_answer))
            if true_extracted is not None:
                if abs(predicted_answer - true_extracted) < 1e-6:
                    correct += 1
            else:
                if normalize_answer(str(predicted_answer)) == normalize_answer(str(true_answer)):
                    correct += 1

        total += 1
        if total % 50 == 0:
            print(f"  Progress: {correct}/{total} = {100*correct/total:.2f}%")

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy


def evaluate_commonsense(
    model,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_new_tokens: int = 64,
) -> Dict[str, float]:
    """Evaluate on all 8 commonsense reasoning tasks.

    For each task, formats examples as multiple-choice questions
    and measures accuracy of the model's selected answer.

    Args:
        model: The trained model.
        tokenizer: The tokenizer.
        device: Device for computation.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Dictionary mapping task names to accuracy percentages.
    """
    results = {}

    for task in COMMONSENSE_TASKS:
        ds_name = COMMONSENSE_DATASET_MAP[task]
        config = COMMONSENSE_CONFIG_MAP.get(task)

        if config:
            eval_dataset = load_dataset(ds_name, config, split="validation")
        else:
            eval_dataset = load_dataset(ds_name, split="validation")

        prompt_template = COMMONSENSE_PROMPTS[task]
        answer_field = COMMONSENSE_ANSWER_FIELDS[task]
        choices_field = COMMONSENSE_CHOICES_FIELDS[task]

        correct = 0
        total = 0

        for example in tqdm(eval_dataset, desc=f"Evaluating {task}"):
            prompt = prompt_template.format(**{k: example.get(k, "") for k in example.keys()})

            if choices_field is None:
                choices = None
            elif isinstance(choices_field, list):
                choices = [example.get(c, "") for c in choices_field]
                choices_text = " ".join([f"({j}) {c}" for j, c in enumerate(choices)])
                prompt = prompt + choices_text + "\n"
            elif choices_field == "endings":
                choices = example.get(choices_field, [])
                choices_text = " ".join([f"({j}) {c}" for j, c in enumerate(choices)])
                prompt = prompt + choices_text + "\n"
            elif choices_field == "choices":
                choices_data = example.get(choices_field, {})
                if isinstance(choices_data, dict):
                    text_choices = choices_data.get("text", [])
                    label_choices = choices_data.get("label", [])
                else:
                    text_choices = [c.get("text", str(c)) for c in choices_data]
                    label_choices = [c.get("label", str(c)) for c in choices_data]
                choices_text = " ".join([f"({l}) {t}" for l, t in zip(label_choices, text_choices)])
                prompt = prompt + choices_text + "\n"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=1.0,
                )

            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            true_answer = str(example.get(answer_field, "")).strip()
            predicted_answer = extract_predicted_choice(generated)

            if normalize_answer(predicted_answer) == normalize_answer(true_answer):
                correct += 1
            elif true_answer.lower() in predicted_answer.lower():
                correct += 1

            total += 1

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        results[task] = accuracy
        print(f"  {task}: {accuracy:.2f}%")

    avg = sum(results.values()) / len(results) if results else 0.0
    print(f"  Average: {avg:.2f}%")
    return results


def extract_predicted_choice(text: str) -> str:
    """Extract predicted choice letter/number from generated text.

    Looks for patterns like '0', '1', 'A', 'B', etc.
    """
    text = text.strip()
    for match in re.findall(r'\b([A-D0-3])\b', text):
        return match
    return text[:5]


def evaluate_glue_detail(
    model,
    tokenizer: AutoTokenizer,
    task_name: str,
    device: torch.device,
) -> Dict[str, float]:
    """Detailed GLUE evaluation with per-class metrics.

    Args:
        model: Trained model.
        tokenizer: Tokenizer.
        task_name: GLUE task name.
        device: Device.

    Returns:
        Dictionary of metric values.
    """
    from sklearn.metrics import (
        matthews_corrcoef,
        accuracy_score,
        precision_recall_fscore_support,
    )
    from scipy.stats import pearsonr

    from data import load_glue_dataset

    _, eval_dataset, _, data_collator = load_glue_dataset(
        task_name=task_name,
        tokenizer=tokenizer,
        max_seq_length=512,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=32,
        collate_fn=data_collator,
    )

    is_regression = task_name == "stsb"
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)

            if is_regression:
                preds = outputs.logits.squeeze()
            else:
                preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    results = {}
    if task_name == "cola":
        results["matthews_correlation"] = float(matthews_corrcoef(all_labels, all_preds))
    elif task_name == "stsb":
        results["pearson"] = float(pearsonr(all_labels, all_preds)[0])
    else:
        results["accuracy"] = float(accuracy_score(all_labels, all_preds))
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        results["precision"] = float(precision)
        results["recall"] = float(recall)
        results["f1"] = float(f1)

    return results
