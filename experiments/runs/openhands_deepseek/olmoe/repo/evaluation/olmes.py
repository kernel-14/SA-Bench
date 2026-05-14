"""
OLMES (Open Language Model Evaluation Standard) implementation.

Implements evaluation setup from Appendix C (Table 11):
    - During pretraining: 0-shot Completion/Cloze Formulation (CF)
    - After pretraining: 5-shot max(MCF, CF) with appropriate probability normalization

Tasks:
    - MMLU, HellaSwag, ARC-Challenge, ARC-Easy, PIQA, Winogrande
    - BoolQ, CommonsenseQA, OpenBookQA, SocialIQA, SciQ, COPA
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable


def compute_token_logprob(
    model,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> float:
    """
    Compute log probability of target_ids given input_ids as context.
    Used for Cloze Formulation (CF) evaluation.
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Concatenate context and target
        full_ids = torch.cat([input_ids, target_ids.unsqueeze(0)], dim=-1).to(device)
        logits, _, _ = model(full_ids)

        # Get logits for target positions
        target_logits = logits[:, input_ids.size(-1) - 1:-1, :]  # predict each target token
        target = full_ids[:, input_ids.size(-1):]

        log_probs = F.log_softmax(target_logits, dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.sum().item()


def completion_formulation_eval(
    model,
    tokenizer,
    task_data: List[Dict],
    num_shots: int = 0,
    normalize_by_chars: bool = False,
    max_seq_len: int = 4096,
) -> float:
    """
    Completion/Cloze Formulation (CF) evaluation.

    Each question has a context and multiple answer choices. We compute
    the probability of each answer choice given the context and pick the
    highest probability one.

    Args:
        model: OLMoE model
        tokenizer: tokenizer
        task_data: list of {"context": str, "choices": List[str], "correct": int}
        num_shots: number of few-shot examples
        normalize_by_chars: if True, normalize by character count (MMLU)
        max_seq_len: max sequence length
    """
    correct = 0
    total = 0

    for item in task_data:
        context = item["context"]
        choices = item["choices"]
        correct_idx = item["correct"]

        choice_scores = []
        for choice in choices:
            # Build prompt with few-shot examples
            if num_shots > 0:
                prompt = build_few_shot_prompt(task_data, item, num_shots, context, choice)
            else:
                prompt = f"{context} {choice}"

            tokens = tokenizer.encode(prompt)
            choice_tokens = tokenizer.encode(choice)

            if len(tokens) > max_seq_len:
                tokens = tokens[-max_seq_len:]

            input_ids = torch.tensor(tokens[:-len(choice_tokens)], dtype=torch.long).unsqueeze(0)
            target_ids = torch.tensor(choice_tokens, dtype=torch.long)

            score = compute_token_logprob(model, input_ids, target_ids)

            if normalize_by_chars:
                score = score / len(choice)

            choice_scores.append(score)

        predicted = max(range(len(choice_scores)), key=lambda i: choice_scores[i])
        if predicted == correct_idx:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def multiple_choice_formulation_eval(
    model,
    tokenizer,
    task_data: List[Dict],
    num_shots: int = 5,
    max_seq_len: int = 4096,
) -> float:
    """
    Multiple-Choice Formulation (MCF) evaluation.

    Model scores answer labels like A/B/C/D directly rather than
    the full answer string.

    Args:
        model: OLMoE model
        tokenizer: tokenizer
        task_data: list of {"context": str, "choices": List[str], "correct": int}
        num_shots: number of few-shot examples
        max_seq_len: max sequence length
    """
    correct = 0
    total = 0
    label_tokens = ["A", "B", "C", "D", "E", "F"]

    for item in task_data:
        context = item["context"]
        choices = item["choices"]
        correct_idx = item["correct"]

        # Build prompt with labels
        prompt = context + "\n\n"
        for i, choice in enumerate(choices[:len(label_tokens)]):
            prompt += f"{label_tokens[i]}. {choice}\n"
        prompt += "Answer:"

        tokens = tokenizer.encode(prompt)
        if len(tokens) > max_seq_len:
            tokens = tokens[-max_seq_len:]

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        with torch.no_grad():
            logits, _, _ = model(input_ids)
            next_logits = logits[:, -1, :]  # logits for next token

        label_scores = []
        for i in range(len(choices)):
            label_id = tokenizer.encode(label_tokens[i])[0]
            label_scores.append(next_logits[0, label_id].item())

        predicted = max(range(len(label_scores)), key=lambda i: label_scores[i])
        if predicted == correct_idx:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def max_mcf_cf_eval(
    model,
    tokenizer,
    task_data: List[Dict],
    num_shots: int = 5,
) -> float:
    """
    Combined evaluation: max of MCF and CF scores (OLMES standard).
    """
    cf_score = completion_formulation_eval(
        model, tokenizer, task_data, num_shots=num_shots,
        normalize_by_chars=False,
    )
    mcf_score = multiple_choice_formulation_eval(
        model, tokenizer, task_data, num_shots=num_shots,
    )
    return max(cf_score, mcf_score)


def build_few_shot_prompt(
    task_data: List[Dict],
    current_item: Dict,
    num_shots: int,
    context: str,
    choice: str,
) -> str:
    """Build a few-shot prompt with in-context examples."""
    # Use the first num_shots examples (excluding current)
    examples = [d for d in task_data[:num_shots + 1] if d is not current_item][:num_shots]

    prompt_parts = []
    for ex in examples:
        ex_context = ex["context"]
        ex_correct = ex["choices"][ex["correct"]]
        prompt_parts.append(f"{ex_context} {ex_correct}")

    prompt_parts.append(f"{context} {choice}")
    return "\n\n".join(prompt_parts)


# Dataset loaders for standard benchmarks
def load_mmlu(subset: str = "all", split: str = "test") -> List[Dict]:
    """Load MMLU benchmark data."""
    try:
        from datasets import load_dataset
        if subset == "all":
            data = load_dataset("cais/mmlu", "all", split=split)
        else:
            data = load_dataset("cais/mmlu", subset, split=split)
        return [
            {
                "context": item["question"],
                "choices": item["choices"],
                "correct": item["answer"],
            }
            for item in data
        ]
    except ImportError:
        return []


def load_hellaswag(split: str = "validation") -> List[Dict]:
    """Load HellaSwag benchmark data."""
    try:
        from datasets import load_dataset
        data = load_dataset("Rowan/hellaswag", split=split)
        return [
            {
                "context": item["ctx"],
                "choices": item["endings"],
                "correct": int(item["label"]),
            }
            for item in data
        ]
    except ImportError:
        return []


def load_arc_challenge(split: str = "test") -> List[Dict]:
    """Load ARC-Challenge benchmark data."""
    try:
        from datasets import load_dataset
        data = load_dataset("ai2_arc", "ARC-Challenge", split=split)
        return [
            {
                "context": item["question"],
                "choices": item["choices"]["text"],
                "correct": item["choices"]["label"].index(item["answerKey"]) if "answerKey" in item else 0,
            }
            for item in data
        ]
    except ImportError:
        return []


def load_arc_easy(split: str = "test") -> List[Dict]:
    """Load ARC-Easy benchmark data."""
    try:
        from datasets import load_dataset
        data = load_dataset("ai2_arc", "ARC-Easy", split=split)
        return [
            {
                "context": item["question"],
                "choices": item["choices"]["text"],
                "correct": item["choices"]["label"].index(item["answerKey"]) if "answerKey" in item else 0,
            }
            for item in data
        ]
    except ImportError:
        return []


def load_piqa(split: str = "validation") -> List[Dict]:
    """Load PIQA benchmark data."""
    try:
        from datasets import load_dataset
        data = load_dataset("piqa", split=split)
        return [
            {
                "context": item["goal"],
                "choices": [item["sol1"], item["sol2"]],
                "correct": item["label"],
            }
            for item in data
        ]
    except ImportError:
        return []


def load_winogrande(split: str = "validation") -> List[Dict]:
    """Load Winogrande benchmark data."""
    try:
        from datasets import load_dataset
        data = load_dataset("winogrande", "winogrande_xl", split=split)
        return [
            {
                "context": item["sentence"].replace("_", "{}"),
                "choices": [item["option1"], item["option2"]],
                "correct": int(item["answer"]) - 1,
            }
            for item in data
        ]
    except ImportError:
        return []


EVALUATION_TASKS = {
    "mmlu": {
        "loader": load_mmlu,
        "evaluator": max_mcf_cf_eval,
        "num_shots": 5,
        "normalization": "char",  # for MMLU CF
    },
    "hellaswag": {
        "loader": load_hellaswag,
        "evaluator": max_mcf_cf_eval,
        "num_shots": 5,
        "normalization": "char",
    },
    "arc_challenge": {
        "loader": load_arc_challenge,
        "evaluator": max_mcf_cf_eval,
        "num_shots": 5,
        "normalization": "pmi",
    },
    "arc_easy": {
        "loader": load_arc_easy,
        "evaluator": max_mcf_cf_eval,
        "num_shots": 5,
        "normalization": "char",
    },
    "piqa": {
        "loader": load_piqa,
        "evaluator": max_mcf_cf_eval,
        "num_shots": 5,
        "normalization": "char",
    },
    "winogrande": {
        "loader": load_winogrande,
        "evaluator": max_mcf_cf_eval,
        "num_shots": 5,
        "normalization": "none",
    },
}


def run_olmes_evaluation(
    model,
    tokenizer,
    tasks: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Run full OLMES evaluation suite.

    Returns dict of task_name -> accuracy.
    """
    if tasks is None:
        tasks = list(EVALUATION_TASKS.keys())

    results = {}
    for task_name in tasks:
        if task_name not in EVALUATION_TASKS:
            print(f"Unknown task: {task_name}")
            continue

        task_config = EVALUATION_TASKS[task_name]
        print(f"Evaluating {task_name}...")

        data = task_config["loader"]()
        if not data:
            print(f"  No data loaded for {task_name}, skipping.")
            continue

        score = task_config["evaluator"](
            model=model,
            tokenizer=tokenizer,
            task_data=data,
            num_shots=task_config["num_shots"],
        )
        results[task_name] = score
        print(f"  {task_name}: {score:.4f}")

    return results
