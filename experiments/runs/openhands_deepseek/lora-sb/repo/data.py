"""Dataset loading and preprocessing for LoRA-SB experiments.

Supports:
- GLUE benchmark (CoLA, RTE, MRPC, STS-B, QNLI, SST-2)
- MetaMathQA (arithmetic reasoning, evaluation on GSM8K and MATH)
- COMMONSENSE170K (8 commonsense reasoning tasks)
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding
from typing import Optional, Dict, List, Tuple
import numpy as np


GLUE_TASK_KEYS = {
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "qnli": ("question", "sentence"),
    "stsb": ("sentence1", "sentence2"),
}

GLUE_NUM_LABELS = {
    "cola": 2,
    "mrpc": 2,
    "rte": 2,
    "sst2": 2,
    "qnli": 2,
    "stsb": 1,
}


def load_glue_dataset(
    task_name: str,
    tokenizer: AutoTokenizer,
    max_seq_length: int = 512,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    max_predict_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset, DataCollatorWithPadding]:
    """Load and preprocess a GLUE task.

    Args:
        task_name: GLUE task name (cola, mrpc, rte, sst2, qnli, stsb).
        tokenizer: HuggingFace tokenizer.
        max_seq_length: Maximum sequence length for tokenization.
        max_train_samples: Limit training samples.
        max_eval_samples: Limit validation samples.
        max_predict_samples: Limit test samples.
        seed: Random seed.

    Returns:
        (train_dataset, eval_dataset, predict_dataset, data_collator)
    """
    raw_datasets = load_dataset("glue", task_name)
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    sentence1_key, sentence2_key = GLUE_TASK_KEYS[task_name]

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=False, max_length=max_seq_length, truncation=True)
        result["labels"] = examples["label"]
        return result

    train_dataset = raw_datasets["train"]
    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)

    eval_dataset = raw_datasets["validation_matched" if task_name == "mnli" else "validation"]
    if max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    predict_dataset = raw_datasets["test_matched" if task_name == "mnli" else "test"]
    if max_predict_samples is not None:
        predict_dataset = predict_dataset.select(range(min(max_predict_samples, len(predict_dataset))))
    predict_dataset = predict_dataset.map(preprocess_function, batched=True, remove_columns=predict_dataset.column_names)

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    return train_dataset, eval_dataset, predict_dataset, data_collator


def load_metamathqa(
    tokenizer: AutoTokenizer,
    max_seq_length: int = 512,
    max_train_samples: int = 50000,
    seed: int = 42,
) -> Tuple[Dataset, DataCollatorForSeq2Seq]:
    """Load MetaMathQA dataset for arithmetic reasoning fine-tuning.

    MetaMathQA contains mathematical problems with solutions.
    Uses DataCollatorForSeq2Seq for causal LM training.

    Returns:
        (train_dataset, data_collator)
    """
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    if max_train_samples is not None:
        dataset = dataset.select(range(min(max_train_samples, len(dataset))))

    def preprocess_function(examples):
        queries = examples["query"]
        responses = examples["response"]
        texts = [f"Question: {q}\nAnswer: {r}" for q, r in zip(queries, responses)]

        tokenized = tokenizer(
            texts,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None, padding=True)
    return dataset, data_collator


COMMONSENSE_TASKS = [
    "boolq",
    "piqa",
    "siqa",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
]

COMMONSENSE_DATASET_MAP = {
    "boolq": "boolq",
    "piqa": "piqa",
    "siqa": "social_i_qa",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "arc_easy": "ai2_arc",
    "arc_challenge": "ai2_arc",
    "openbookqa": "openbookqa",
}

COMMONSENSE_CONFIG_MAP = {
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
}

COMMONSENSE_PROMPTS = {
    "boolq": "{passage}\nQuestion: {question}?\nAnswer: ",
    "piqa": "{goal}\nAnswer: ",
    "siqa": "{context}\nQuestion: {question}\nAnswer: ",
    "hellaswag": "{ctx}\nAnswer: ",
    "winogrande": "{sentence}\nAnswer: ",
    "arc_easy": "{question}\nAnswer: ",
    "arc_challenge": "{question}\nAnswer: ",
    "openbookqa": "{question_stem}\nAnswer: ",
}

COMMONSENSE_ANSWER_FIELDS = {
    "boolq": "answer",
    "piqa": "label",
    "siqa": "label",
    "hellaswag": "label",
    "winogrande": "answer",
    "arc_easy": "answerKey",
    "arc_challenge": "answerKey",
    "openbookqa": "answerKey",
}

COMMONSENSE_CHOICES_FIELDS = {
    "boolq": None,
    "piqa": ["sol1", "sol2"],
    "siqa": ["answerA", "answerB", "answerC"],
    "hellaswag": "endings",
    "winogrande": ["option1", "option2"],
    "arc_easy": "choices",
    "arc_challenge": "choices",
    "openbookqa": "choices",
}


def load_commonsense_dataset(
    tokenizer: AutoTokenizer,
    max_seq_length: int = 256,
    max_train_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[Dataset, DataCollatorForSeq2Seq]:
    """Load and preprocess the COMMONSENSE170K dataset.

    Combines 8 commonsense reasoning tasks into a single training set.
    Each example is formatted as a multiple-choice question.

    Returns:
        (train_dataset, data_collator)
    """
    all_datasets = []

    for task in COMMONSENSE_TASKS:
        ds_name = COMMONSENSE_DATASET_MAP[task]
        config = COMMONSENSE_CONFIG_MAP.get(task)

        if config:
            raw = load_dataset(ds_name, config, split="train")
        else:
            raw = load_dataset(ds_name, split="train")

        prompt_template = COMMONSENSE_PROMPTS[task]
        answer_field = COMMONSENSE_ANSWER_FIELDS[task]
        choices_field = COMMONSENSE_CHOICES_FIELDS[task]

        def make_preprocess(template, answer_key, choices_key):
            def preprocess(examples):
                texts = []
                labels = []
                for i in range(len(examples[answer_key])):
                    example = {k: examples[k][i] for k in examples.keys()}

                    if choices_key is None:
                        prompt = template.format(**example)
                        label_str = str(example[answer_key]).strip()
                    elif isinstance(choices_key, list):
                        prompt = template.format(**example)
                        choices = [example[c] for c in choices_key]
                        choices_text = " ".join([f"({j}) {c}" for j, c in enumerate(choices)])
                        prompt = prompt + choices_text + "\n"
                        label_str = str(int(example[answer_key]))
                    elif choices_key == "endings":
                        prompt = template.format(**example)
                        choices = example[choices_key]
                        choices_text = " ".join([f"({j}) {c}" for j, c in enumerate(choices)])
                        prompt = prompt + choices_text + "\n"
                        label_str = str(int(example[answer_key]))
                    elif choices_key == "choices":
                        prompt = template.format(**example)
                        choices = example[choices_key]
                        if isinstance(choices, dict):
                            text_choices = choices.get("text", [])
                            label_choices = choices.get("label", [])
                        else:
                            text_choices = [c.get("text", str(c)) for c in choices]
                            label_choices = [c.get("label", str(c)) for c in choices]
                        choices_text = " ".join([f"({l}) {t}" for l, t in zip(label_choices, text_choices)])
                        prompt = prompt + choices_text + "\n"
                        label_str = str(example[answer_key]).strip()
                    else:
                        prompt = template.format(**example)
                        label_str = str(example[answer_key]).strip()

                    texts.append(prompt)
                    labels.append(label_str)

                return {"text": texts, "label": labels}
            return preprocess

        preprocess_fn = make_preprocess(prompt_template, answer_field, choices_field)
        processed = raw.map(preprocess_fn, batched=True, remove_columns=raw.column_names)
        all_datasets.append(processed)

    combined = concatenate_datasets(all_datasets)
    if max_train_samples is not None:
        combined = combined.select(range(min(max_train_samples, len(combined))))

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            max_length=max_seq_length,
            truncation=True,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    combined = combined.map(tokenize_fn, batched=True, remove_columns=combined.column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None, padding=True)
    return combined, data_collator


def load_gsm8k(tokenizer: AutoTokenizer, max_seq_length: int = 512) -> Dataset:
    """Load GSM8K evaluation dataset."""
    dataset = load_dataset("gsm8k", "main", split="test")

    def preprocess(examples):
        texts = [f"Question: {q}\nAnswer: " for q in examples["question"]]
        tokenized = tokenizer(texts, max_length=max_seq_length, truncation=True, padding=False)
        tokenized["answer"] = examples["answer"]
        return tokenized

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)


def load_math(tokenizer: AutoTokenizer, max_seq_length: int = 512) -> Dataset:
    """Load MATH evaluation dataset."""
    dataset = load_dataset("hendrycks/competition_math", split="test")

    def preprocess(examples):
        texts = [f"Question: {q}\nAnswer: " for q in examples["problem"]]
        tokenized = tokenizer(texts, max_length=max_seq_length, truncation=True, padding=False)
        tokenized["solution"] = examples["solution"]
        return tokenized

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)


def create_init_dataloader(
    dataset: Dataset,
    num_samples: int,
    batch_size: int = 8,
    collate_fn: Optional[callable] = None,
    seed: int = 42,
) -> DataLoader:
    """Create a small DataLoader for LoRA-SB initialization.

    Randomly samples num_samples from the dataset.

    Args:
        dataset: Full training dataset.
        num_samples: Number of samples for init (0.1% of full dataset).
        batch_size: Batch size for gradient computation.
        collate_fn: Optional custom collate function.
        seed: Random seed.

    Returns:
        DataLoader with sampled subset.
    """
    total = len(dataset)
    num_samples = min(num_samples, total)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:num_samples]
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def load_dataset_for_task(
    task_type: str,
    task_name: Optional[str],
    tokenizer: AutoTokenizer,
    max_seq_length: int = 512,
    max_train_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple:
    """Unified interface for loading datasets.

    Args:
        task_type: 'glue', 'math', or 'commonsense'.
        task_name: Task name for GLUE, else None.
        tokenizer: HuggingFace tokenizer.
        max_seq_length: Max sequence length.
        max_train_samples: Limit training samples.
        seed: Random seed.

    Returns:
        Tuple appropriate for the task type.
    """
    if task_type == "glue":
        if task_name is None:
            raise ValueError("task_name is required for GLUE tasks")
        return load_glue_dataset(
            task_name=task_name,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_train_samples=max_train_samples,
            seed=seed,
        )
    elif task_type == "math":
        train_dataset, collator = load_metamathqa(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_train_samples=max_train_samples,
            seed=seed,
        )
        gsm8k_dataset = load_gsm8k(tokenizer, max_seq_length)
        math_dataset = load_math(tokenizer, max_seq_length)
        return train_dataset, gsm8k_dataset, math_dataset, collator
    elif task_type == "commonsense":
        train_dataset, collator = load_commonsense_dataset(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_train_samples=max_train_samples,
            seed=seed,
        )
        return train_dataset, collator
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
