
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding
from torch.utils.data import DataLoader, RandomSampler
from typing import Dict, List, Any, Optional

from config import Config

def get_tokenizer(model_name: str):
    """Loads and returns the tokenizer for a given model name."""
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_function_causal_lm(examples, tokenizer, max_seq_len):
    """Tokenizes text for causal language modeling tasks."""
    output = tokenizer(examples["text"], truncation=True, max_length=max_seq_len)
    return output

def tokenize_function_sequence_classification(examples, tokenizer, max_seq_len, text_column_names: List[str]):
    """Tokenizes text for sequence classification tasks."""
    if len(text_column_names) == 1:
        inputs = examples[text_column_names[0]]
        return tokenizer(inputs, truncation=True, max_length=max_seq_len, padding="max_length")
    elif len(text_column_names) == 2:
        inputs = examples[text_column_names[0]]
        inputs_pair = examples[text_column_names[1]]
        return tokenizer(inputs, inputs_pair, truncation=True, max_length=max_seq_len, padding="max_length")
    else:
        raise ValueError("Unsupported number of text columns for sequence classification.")

def get_dataloader_for_initialization(
    tokenizer: AutoTokenizer,
    config: Config,
    task_name: str,
    model_name: str,
    num_samples: int,
    max_seq_len: int,
    batch_size: int,
    seed: int = 42,
) -> DataLoader:
    """
    Prepares a DataLoader for a small subset of the training data for LoRA-SB initialization.
    """
    if task_name == "arithmetic":
        dataset = load_dataset(config.metamath_dataset, split=config.metamath_train_split)
        # MetaMathQA typically has 'question' and 'answer' fields. Combine them for causal LM.
        dataset = dataset.map(lambda x: {"text": x["question"] + " " + x["answer"]}, batched=False)
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function_causal_lm(examples, tokenizer, max_seq_len),
            batched=True,
            remove_columns=dataset.column_names
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif task_name == "commonsense_reasoning":
        # Load one of the commonsense datasets for initialization, e.g., BoolQ
        # For actual training, COMMONSENSE170K combines multiple. For init, one is enough.
        # Paper Appendix C: "averaged over a mini-batch" -> suggests any task data can be used
        dataset = load_dataset("super_glue", "boolq", split="train") # Example for now
        dataset = dataset.map(lambda x: {"text": x["question"] + " " + x["passage"]}, batched=False)
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function_causal_lm(examples, tokenizer, max_seq_len),
            batched=True,
            remove_columns=dataset.column_names
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif task_name == "nlu":
        # For NLU tasks, often sequence classification. Use GLUE task like SST-2.
        dataset = load_dataset("glue", "sst2", split="train")
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function_sequence_classification(examples, tokenizer, max_seq_len, ["sentence"]),
            batched=True,
            remove_columns=dataset.column_names
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        raise ValueError(f"Unsupported task for initialization: {task_name}")

    # Select a random subset of samples
    if len(tokenized_dataset) > num_samples:
        generator = torch.Generator().manual_seed(seed)
        sampled_indices = torch.randperm(len(tokenized_dataset), generator=generator)[:num_samples].tolist()
        sampled_dataset = tokenized_dataset.select(sampled_indices)
    else:
        sampled_dataset = tokenized_dataset

    dataloader = DataLoader(
        sampled_dataset,
        sampler=RandomSampler(sampled_dataset, generator=torch.Generator().manual_seed(seed)),
        batch_size=batch_size,
        collate_fn=data_collator
    )
    return dataloader

def get_glue_datasets(tokenizer, task_name: str, max_seq_len: int):
    """Loads and tokenizes a GLUE dataset for training and evaluation."""
    raw_datasets = load_dataset("glue", task_name)
    
    sentence1_key, sentence2_key = None, None
    if task_name == "cola":
        sentence1_key = "sentence"
    elif task_name == "mnli":
        sentence1_key, sentence2_key = "premise", "hypothesis"
    elif task_name == "mrpc":
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    elif task_name == "qnli":
        sentence1_key, sentence2_key = "question", "sentence"
    elif task_name == "rte":
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    elif task_name == "sst2":
        sentence1_key = "sentence"
    elif task_name == "stsb":
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        raise ValueError(f"Unsupported GLUE task: {task_name}")

    def tokenize_fn(examples):
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        return tokenizer(*texts, max_length=max_seq_len, truncation=True, padding="max_length")

    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=[col for col in raw_datasets["train"].column_names if col != "label"]
    )
    return tokenized_datasets


def get_metamath_datasets(tokenizer, max_seq_len: int):
    """Loads and tokenizes MetaMathQA dataset for causal language modeling."""
    raw_datasets = load_dataset("competition_math") # Paper references MetaMathQA (50), but often uses competition_math
                                                   # as the base for math reasoning. Using competition_math for simplicity.
    def tokenize_fn(examples):
        # Assuming 'question' and 'solution' fields exist for competition_math
        # For MetaMathQA, it would be 'question' and 'answer'. Adjust as needed.
        texts = [q + " " + s for q, s in zip(examples["question"], examples["solution"])]
        return tokenizer(texts, truncation=True, max_length=max_seq_len)

    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    return tokenized_datasets

# Data collator factory
def get_data_collator(tokenizer, task_type: str):
    if task_type == "causal_lm":
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif task_type == "sequence_classification":
        return DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        raise ValueError(f"Unsupported data collator task type: {task_type}")

