import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    AutoTokenizer
)

from config import Config # Assuming config.py is in the same directory


class DatasetLoader:
    """
    Handles loading, preprocessing, and sampling of datasets for LoRA-SB experiments.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: Config):
        """
        Initializes the DatasetLoader.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to use for preprocessing.
            config (Config): The global configuration object.
        """
        self._tokenizer = tokenizer
        self._config = config
        self._task_to_keys: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {
            "MetaMathQA": ("question", "answer", None), # Special handling for causal LM
            "COMMONSENSE170K": ("question", "choices", "answer"), # Special handling for MCQs
            "cola": ("sentence", None, "label"),
            "mnli": ("premise", "hypothesis", "label"),
            "mrpc": ("sentence1", "sentence2", "label"),
            "qnli": ("question", "sentence", "label"),
            "rte": ("sentence1", "sentence2", "label"),
            "sst2": ("sentence", None, "label"),
            "stsb": ("sentence1", "sentence2", "label"),
        }

    def _preprocess_function_causal_lm(self, examples: Dict[str, List[str]], template: str) -> Dict:
        """
        Preprocesses examples for Causal Language Models (e.g., MetaMathQA, COMMONSENSE170K).
        Applies a prompt template and tokenizes the combined text.
        """
        # Apply prompt template to each example
        processed_texts = [
            template.format(**{k: examples[k][i] for k in examples})
            for i in range(len(examples[list(examples.keys())[0]]))
        ]

        # Tokenize the processed texts
        tokenized_inputs = self._tokenizer(
            processed_texts,
            max_length=self._config.max_seq_len,
            padding="max_length", # Pad to max_seq_len
            truncation=True,
            return_tensors="pt" # Return PyTorch tensors
        )
        return tokenized_inputs
    
    def _preprocess_function_glue(self, examples: Dict[str, List[str]], task_name: str) -> Dict:
        """
        Preprocesses examples for GLUE tasks.
        Handles single-sentence and sentence-pair tasks, tokenizes, and maps labels.
        """
        sentence1_key, sentence2_key, label_key = self._task_to_keys[task_name]

        # Tokenize sentence pairs or single sentences
        if sentence2_key:
            texts = (examples[sentence1_key], examples[sentence2_key])
        else:
            texts = examples[sentence1_key]

        tokenized_inputs = self._tokenizer(
            *texts,
            max_length=self._config.max_seq_len,
            padding="max_length", # Pad to max_seq_len
            truncation=True,
            return_tensors="pt" # Return PyTorch tensors
        )

        # Map labels if they exist
        if label_key and label_key in examples:
            tokenized_inputs["labels"] = examples[label_key]

        return tokenized_inputs

    def load_and_preprocess_dataset(self, task_name: str) -> DatasetDict:
        """
        Loads and preprocesses a dataset based on the task name.

        Args:
            task_name (str): The name of the task (e.g., "MetaMathQA", "COMMONSENSE170K", "cola").

        Returns:
            DatasetDict: A dictionary containing the processed 'train', 'validation', and 'test' splits.
        """
        raw_datasets: DatasetDict
        model_type: str = "causal_lm" # Default model type

        if task_name == "MetaMathQA":
            # Using 'math_qa' from HuggingFace datasets as a proxy for MetaMathQA
            # Adjust if a specific 'metamathqa' dataset becomes available.
            raw_datasets = load_dataset("math_qa")
            # Example template for CausalLM, adjust as per specific MetaMathQA format in paper if needed
            # For math_qa, 'problem' and 'answer' are keys
            prompt_template = "Question: {problem}\nAnswer: {answer}"
            preprocess_func = lambda examples: self._preprocess_function_causal_lm(
                {"problem": examples["problem"], "answer": examples["answer"]},
                prompt_template
            )
            raw_datasets = raw_datasets.filter(lambda x: x['problem'] is not None and x['answer'] is not None)
            
            # Rename splits to standard 'train', 'validation', 'test' if they differ
            if "validation" not in raw_datasets and "test" in raw_datasets:
                raw_datasets["validation"] = raw_datasets["test"]

            # Limit to 50k samples for MetaMathQA as mentioned in paper's experiments for consistency
            if "train" in raw_datasets and len(raw_datasets["train"]) > 50000:
                raw_datasets["train"] = raw_datasets["train"].select(range(50000))
            if "validation" in raw_datasets and len(raw_datasets["validation"]) > 5000: # Example limit for validation
                raw_datasets["validation"] = raw_datasets["validation"].select(range(5000))


        elif task_name == "COMMONSENSE170K":
            # As COMMONSENSE170K is a consolidation of 8 datasets, we will load a few
            # representative ones and concatenate. This is a simplification from the paper's
            # full consolidation but demonstrates the approach.
            # Example: PIQA, HellaSwag, BoolQ
            # For a full reproduction, all 8 (PIQA, SIQA, HellaS., WinoG., ARC-e, ARC-c, OBQA, BoolQ)
            # would need to be loaded and combined.
            print("Loading representative datasets for COMMONSENSE170K (PIQA, HellaSwag, BoolQ)...")
            datasets_to_combine = []

            # PIQA
            piqa_ds = load_dataset("piqa")
            def piqa_format(example):
                choices = [example['sol1'], example['sol2']]
                return {
                    "question": example['goal'],
                    "choices": choices,
                    "answer": example['label'] # 0 or 1
                }
            piqa_ds = piqa_ds.map(piqa_format, remove_columns=piqa_ds["train"].column_names)
            datasets_to_combine.append(piqa_ds)

            # HellaSwag
            hellaswag_ds = load_dataset("Rowan/hellaswag")
            def hellaswag_format(example):
                # HellaSwag's endings are multiple choices for a given context
                return {
                    "question": example['ctx_a'] + " " + example['ctx_b'],
                    "choices": example['endings'],
                    "answer": int(example['label']) # Label is string '0', '1', '2', '3'
                }
            hellaswag_ds = hellaswag_ds.map(hellaswag_format, remove_columns=hellaswag_ds["train"].column_names)
            datasets_to_combine.append(hellaswag_ds)
            
            # BoolQ
            boolq_ds = load_dataset("boolq")
            def boolq_format(example):
                return {
                    "question": example['question'],
                    "choices": ["yes", "no"],
                    "answer": 0 if example['answer'] else 1 # map true/false to 0/1
                }
            boolq_ds = boolq_ds.map(boolq_format, remove_columns=boolq_ds["train"].column_names)
            datasets_to_combine.append(boolq_ds)

            # Consolidate into a single DatasetDict
            raw_datasets = DatasetDict({
                "train": Dataset.from_list([item for ds in datasets_to_combine for item in ds["train"]]),
                "validation": Dataset.from_list([item for ds in datasets_to_combine for item in ds["validation"]]),
                "test": Dataset.from_list([item for ds in datasets_to_combine for item in ds["test"]]),
            })

            # Prompt template for multiple-choice questions (similar to LLM-adapters)
            # e.g., "Question: {question}\nChoices:\n- {choice_0}\n- {choice_1}\nAnswer: {answer}"
            def commonsense_prompt_template(example):
                question = example['question']
                choices = example['choices']
                correct_answer_idx = example['answer'] # This is 0, 1, ...
                
                # Format choices
                choice_str = "\n".join([f"- {chr(65 + i)}. {c}" for i, c in enumerate(choices)])
                
                # Format the full prompt for generation
                # The label should be the text of the correct choice
                formatted_label = choices[correct_answer_idx] if 0 <= correct_answer_idx < len(choices) else ""
                
                # Input for CausalLM, where model generates the answer.
                # The labels will be created by the DataCollatorForLanguageModeling
                # from this combined text.
                return {
                    "text": f"Question: {question}\nChoices:\n{choice_str}\nAnswer: {formatted_label}"
                }

            processed_datasets = raw_datasets.map(
                commonsense_prompt_template,
                remove_columns=[col for col in raw_datasets["train"].column_names if col != "text"],
                desc="Applying Commonsense prompt template"
            )

            preprocess_func = lambda examples: self._tokenizer(
                examples["text"],
                max_length=self._config.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )


        elif task_name.startswith("glue_"):
            model_type = "sequence_classification"
            actual_task_name = task_name.split("_")[1]
            raw_datasets = load_dataset("glue", actual_task_name)
            
            # Adjust max_seq_len for specific GLUE tasks if necessary, based on ambiguity notes.
            # Default to config.max_seq_len (256) for most, 30 for CoLA.
            if actual_task_name == "cola":
                current_max_seq_len = 30 # As per Table 9
            else:
                current_max_seq_len = 256 # Inferred for other GLUE tasks

            self._config.max_seq_len = current_max_seq_len # Temporarily set for preprocessing

            preprocess_func = lambda examples: self._preprocess_function_glue(examples, actual_task_name)

        else:
            raise ValueError(f"Unsupported task_name: {task_name}")

        with self._config._tokenizer.as_target_tokenizer():
            processed_datasets = raw_datasets.map(
                preprocess_func,
                batched=True,
                # remove_columns=[col for col in raw_datasets["train"].column_names if col not in ["labels", "input_ids", "attention_mask"]],
                desc="Tokenizing and preprocessing dataset",
            )
        
        # Reset max_seq_len for config after preprocessing
        self._config.max_seq_len = self._config.max_seq_len 

        # Remove columns that are not needed for training/evaluation
        # This will depend on the `preprocess_func` output, but `input_ids`, `attention_mask`, `labels` are common.
        # Other columns are typically removed during mapping.
        column_names = {k: v.column_names for k, v in processed_datasets.items()}
        for split in processed_datasets:
            processed_datasets[split] = processed_datasets[split].remove_columns(
                [col for col in column_names[split] if col not in ["input_ids", "attention_mask", "labels", "token_type_ids"]]
            )
            processed_datasets[split].set_format("torch") # Set format to torch

        print(f"Processed dataset splits: {processed_datasets}")
        return processed_datasets

    def get_init_subset(self, dataset: Dataset) -> Dataset:
        """
        Retrieves a randomly sampled subset from the dataset for LoRA-SB initialization.

        Args:
            dataset (Dataset): The full training dataset.

        Returns:
            Dataset: A randomly sampled subset of the dataset.
        """
        num_samples_for_init = int(len(dataset) * self._config.lora_sb.init_sample_ratio)

        # Apply minimum sample threshold as per "Anything UNCLEAR" section
        # The paper's ablation suggests performance plateaus around 25-50 samples.
        num_samples_for_init = max(50, num_samples_for_init)

        # Ensure the number of samples does not exceed the total dataset size
        num_samples_for_init = min(len(dataset), num_samples_for_init)

        print(f"Selecting {num_samples_for_init} samples for LoRA-SB initialization (0.1% of dataset or minimum 50).")

        # Shuffle and select subset
        # Using a fixed seed for reproducibility of the subset selection itself
        shuffled_dataset = dataset.shuffle(seed=self._config.random_seed)
        init_subset = shuffled_dataset.select(range(num_samples_for_init))

        return init_subset

    def get_data_collator(self, task_name: str, model_type: str) -> Callable:
        """
        Provides the appropriate data collator instance for batching data.

        Args:
            task_name (str): The name of the task.
            model_type (str): The type of model ("causal_lm" or "sequence_classification").

        Returns:
            Callable: An instance of a data collator.
        """
        if model_type == "causal_lm":
            # DataCollatorForLanguageModeling handles token shifting for labels internally.
            return DataCollatorForLanguageModeling(self._tokenizer, mlm=False)
        elif model_type == "sequence_classification":
            # DataCollatorWithPadding handles dynamic padding for classification tasks.
            return DataCollatorWithPadding(self._tokenizer, padding="longest")
        else:
            raise ValueError(f"Unsupported model type for data collator: {model_type}")

