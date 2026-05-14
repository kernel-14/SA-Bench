"""
dataset.py – DatasetLoader class for LoRA‑SB reproduction.

Handles loading, tokenizing, and preparing DataLoader instances for
- Arithmetic reasoning (MetaMathQA → GSM8K, MATH)
- Commonsense reasoning (COMMONSENSE170K → 8 tasks)
- GLUE natural language understanding (6 tasks)

All configuration is driven by ExperimentConfig (from config.py).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from config import ExperimentConfig  # our configuration dataclass


class DatasetLoader:
    """
    Manages all data preprocessing and DataLoader creation for LoRA‑SB experiments.

    Args:
        config: ExperimentConfig instance containing task, model, and training parameters.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.seed: int = config.seed
        # Reproducibility
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Load tokenizer
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path
        )
        # For causal LMs (Mistral, Llama, Gemma) ensure a pad token is set.
        # For RoBERTa, the tokenizer already has a pad token.
        self.is_causal_lm: bool = self._is_causal_model()
        if self.is_causal_lm and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Some models may need the pad_token_id attribute as well
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Determine the task family from config.task
        if config.task.startswith("arithmetic"):
            self.family: str = "arithmetic"
        elif config.task.startswith("commonsense"):
            self.family = "commonsense"
        elif config.task.startswith("glue_"):
            self.family = "glue"
            self.sub_task: str = config.task[5:]   # e.g. "cola"
        else:
            raise ValueError(f"Unknown task: {config.task}")

        # Internal storage for preprocessed datasets
        self._train_dataset: Optional[Dataset] = None
        self._eval_datasets: Dict[str, Dataset] = {}

        # Collator selection
        if self.is_causal_lm:
            # For autoregressive training we use a simple LM collator (mlm=False)
            # that expects input_ids and labels (already masked) and pads them.
            self._collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False, return_tensors="pt"
            )
            # Override the default behavior: we provide our own collate_fn that
            # ensures labels padding uses -100 (already handled by DataCollatorForLanguageModeling)
        else:
            # Classification (GLUE)
            self._collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer, return_tensors="pt"
            )

    def _is_causal_model(self) -> bool:
        """
        Determine if the model is a causal (autoregressive) LM based on its name.
        This avoids initialising the whole model just for the tokenizer.
        """
        name = self.config.model_name_or_path.lower()
        # Common causal model families used in the paper
        if any(kw in name for kw in ("mistral", "gemma", "llama")):
            return True
        if "roberta" in name or "bert" in name:
            return False
        # Default to causal LM if unknown (safe for most recent LLMs)
        return True

    # --------------------------------------------------------------------------
    # Public interface
    # --------------------------------------------------------------------------

    def get_init_dataloader(self, n_samples: int) -> DataLoader:
        """
        Returns a DataLoader containing `n_samples` random training examples.
        Each sample is processed with batch_size=1 (needed for per‑sample gradient accumulation).
        """
        if self._train_dataset is None:
            self._load_train_dataset()
        dataset = self._train_dataset
        if n_samples > len(dataset):
            n_samples = len(dataset)
        indices = random.sample(range(len(dataset)), n_samples)
        subset = Subset(dataset, indices)

        return DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def get_train_dataloader(self) -> DataLoader:
        """DataLoader for full training set."""
        if self._train_dataset is None:
            self._load_train_dataset()
        return DataLoader(
            self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=False,
        )

    def get_eval_dataloader(self) -> Dict[str, DataLoader]:
        """Return a dict mapping eval dataset name to its DataLoader."""
        if not self._eval_datasets:
            self._load_eval_datasets()
        eval_loaders: Dict[str, DataLoader] = {}
        for name, dataset in self._eval_datasets.items():
            # Use same batch_size as training, or could be separate; we reuse config.batch_size
            # For evaluation, larger batches are okay.
            eval_loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                pin_memory=True,
            )
            eval_loaders[name] = eval_loader
        return eval_loaders

    # --------------------------------------------------------------------------
    # Collation function (dispatches to the internal collator)
    # --------------------------------------------------------------------------

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a list of examples into a batch."""
        # For GLUE classification, the labels are just integers; the DataCollatorWithPadding
        # handles them automatically.
        # For causal LM, we already have input_ids, attention_mask, and labels.
        # The collator will pad appropriately.
        return self._collator(batch)

    # --------------------------------------------------------------------------
    # Dataset loading and preprocessing
    # --------------------------------------------------------------------------

    def _load_train_dataset(self) -> None:
        """Load and preprocess the training dataset based on family."""
        if self.family == "arithmetic":
            self._train_dataset = self._preprocess_arithmetic_train()
        elif self.family == "commonsense":
            self._train_dataset = self._preprocess_commonsense_train()
        elif self.family == "glue":
            self._train_dataset = self._preprocess_glue(split="train")

    def _load_eval_datasets(self) -> None:
        """Load and preprocess all required evaluation datasets."""
        if self.family == "arithmetic":
            self._eval_datasets = self._preprocess_arithmetic_eval()
        elif self.family == "commonsense":
            self._eval_datasets = self._preprocess_commonsense_eval()
        elif self.family == "glue":
            # For GLUE we usually evaluate on the validation split
            self._eval_datasets = {self.sub_task: self._preprocess_glue(split="validation")}

    # --------------------------------------------------------------------------
    # Arithmetic: MetaMathQA training, GSM8K & MATH evaluation
    # --------------------------------------------------------------------------

    def _preprocess_arithmetic_train(self) -> Dataset:
        """Load and preprocess MetaMathQA for causal LM training."""
        ds = load_dataset(
            self.config.dataset_kwargs.get("path", "meta-math/MetaMathQA"),
            split="train",
            trust_remote_code=True,
        )
        # We assume columns: 'query', 'response'
        # Create prompt + response string
        prompt_template = "Question: {query}\nAnswer: "
        full_template = "Question: {query}\nAnswer: {response}"

        def tokenize_and_mask(example: Dict[str, Any]) -> Dict[str, Any]:
            prompt = prompt_template.format(query=example["query"])
            # Tokenize prompt to know its length
            prompt_enc = self.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
            prompt_len = prompt_enc.input_ids.shape[1]

            # Tokenize full text
            full_text = prompt + example["response"]
            full_enc = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )
            # Set labels: copy input_ids, then mask prompt
            labels = full_enc.input_ids.clone()
            labels[:, :prompt_len] = -100
            return {
                "input_ids": full_enc.input_ids[0],
                "attention_mask": full_enc.attention_mask[0],
                "labels": labels[0],
            }

        ds = ds.map(tokenize_and_mask, remove_columns=ds.column_names)
        return ds

    def _preprocess_arithmetic_eval(self) -> Dict[str, Dataset]:
        """Preprocess GSM8K and MATH test sets for generation‑based evaluation."""
        # GSM8K
        gsm8k = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
        # MATH (from hendrycks)
        math = load_dataset("hendrycks/MATH", "default", split="test", trust_remote_code=True)

        prompt_gsm = "Question: {question}\nAnswer: "
        prompt_math = "Problem: {problem}\nAnswer: "

        def preproc_gsm(example: Dict[str, Any]) -> Dict[str, Any]:
            prompt = prompt_gsm.format(question=example["question"])
            enc = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )
            # Keep the gold answer for evaluation
            answer = example["answer"].split("#### ")[-1].strip()
            return {
                "input_ids": enc.input_ids[0],
                "attention_mask": enc.attention_mask[0],
                "answer": answer,
                "original_answer": example["answer"],  # in case needed
            }

        def preproc_math(example: Dict[str, Any]) -> Dict[str, Any]:
            prompt = prompt_math.format(problem=example["problem"])
            enc = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )
            # For MATH, the answer is inside \boxed{...}
            solution = example["solution"]
            # Extract last boxed expression
            import re
            match = re.findall(r"\\boxed\{([^}]*)\}", solution)
            answer = match[-1] if match else solution.strip().split("\n")[-1]
            return {
                "input_ids": enc.input_ids[0],
                "attention_mask": enc.attention_mask[0],
                "answer": answer,
                "solution": solution,
            }

        gsm8k = gsm8k.map(preproc_gsm, remove_columns=gsm8k.column_names)
        math = math.map(preproc_math, remove_columns=math.column_names)

        return {"gsm8k": gsm8k, "math": math}

    # --------------------------------------------------------------------------
    # Commonsense: COMMONSENSE170K (merge of eight datasets)
    # --------------------------------------------------------------------------

    # Map of dataset names to HuggingFace identifiers and configuration
    COMMONSENSE_DATASETS = [
        ("boolq", "boolq", None),
        ("piqa", "piqa", None),
        ("siqa", "social_i_qa", None),
        ("hellaswag", "hellaswag", None),
        ("winogrande", "winogrande", "winogrande_debiased"),  # common config
        ("arc_e", "ai2_arc", "ARC-Easy"),
        ("arc_c", "ai2_arc", "ARC-Challenge"),
        ("obqa", "openbookqa", "main"),
    ]

    # Template from Hu et al. (2023) LLM‑Adapters
    COMMONSENSE_TEMPLATE = (
        "When you respond, provide only the label (A, B, C, D) corresponding to the correct answer.\n"
        "{context}"
        "Question: {question}\n"
        "Options:\n"
        "{options_str}\n"
        "Answer:"
    )

    def _preprocess_commonsense_train(self) -> Dataset:
        """Load and preprocess all eight commonsense datasets for training."""
        all_parts = []
        for short_name, hf_name, config_name in self.COMMONSENSE_DATASETS:
            try:
                # Determine split: most use 'train', but ai2_arc needs 'train' split (exists)
                ds = load_dataset(hf_name, config_name, split="train", trust_remote_code=True)
            except Exception:
                # Fallback to default config
                ds = load_dataset(hf_name, split="train", trust_remote_code=True)
            processed = self._process_single_commonsense(short_name, ds)
            all_parts.append(processed)
        # Concatenate all datasets
        return concatenate_datasets(all_parts)

    def _preprocess_commonsense_eval(self) -> Dict[str, Dataset]:
        """Preprocess evaluation (test/validation) splits for each commonsense dataset."""
        eval_ds: Dict[str, Dataset] = {}
        for short_name, hf_name, config_name in self.COMMONSENSE_DATASETS:
            # For most, test split is 'validation' (e.g., boolq 'validation', piqa 'validation',
            # siqa 'validation', hellaswag 'validation', winogrande 'validation',
            # arc_e 'test', arc_c 'test', obqa 'validation').
            # We'll try 'test' first, then 'validation', then maybe the only split.
            ds = None
            try:
                ds = load_dataset(hf_name, config_name, split="test", trust_remote_code=True)
            except Exception:
                try:
                    ds = load_dataset(hf_name, config_name, split="validation", trust_remote_code=True)
                except Exception:
                    ds = load_dataset(hf_name, split="validation", trust_remote_code=True)
            processed = self._process_single_commonsense(short_name, ds, is_eval=True)
            eval_ds[short_name] = processed
        return eval_ds

    def _process_single_commonsense(
        self, short_name: str, ds: Dataset, is_eval: bool = False
    ) -> Dataset:
        """Unify fields and tokenize a single commonsense dataset."""
        # First, map to standardized fields: question, options (list), answer_idx (int)
        def extract_fields(example: Dict[str, Any]) -> Dict[str, Any]:
            if short_name == "boolq":
                # Options: ["yes", "no"]; answer is 0 for True, 1 for False? BoolQ label 0=True,1=False
                # But template expects A,B,... Usually 'yes'=A, 'no'=B? Paper uses multiple choice format.
                # We'll map label 0 -> A (yes), 1 -> B (no).
                question = example.get("question", example.get("passage", ""))
                context = example.get("passage", "")
                options = ["yes", "no"]   # A: yes, B: no
                # label 0 -> True -> yes (A), label 1 -> False -> no (B)
                ans_idx = example["label"]  # 0 or 1
                return {
                    "question": question,
                    "context": context,
                    "options": options,
                    "answer_idx": ans_idx,
                }
            elif short_name == "piqa":
                # goal, sol1, sol2, label 0/1 -> sol1/sol2
                options = [example["sol1"], example["sol2"]]
                return {
                    "question": example["goal"],
                    "context": "",
                    "options": options,
                    "answer_idx": example["label"],  # 0 or 1
                }
            elif short_name == "siqa":
                # context, question, answerA, answerB, answerC, label -> 1,2,3 (index 0-based)
                options = [
                    example["answerA"],
                    example["answerB"],
                    example["answerC"],
                ]
                ans_idx = int(example["label"]) - 1  # convert from 1,2,3 to 0,1,2
                return {
                    "question": example["question"],
                    "context": example.get("context", ""),
                    "options": options,
                    "answer_idx": ans_idx,
                }
            elif short_name == "hellaswag":
                # ctx, endings list, label index
                options = example["endings"]
                return {
                    "question": example["ctx"],
                    "context": "",
                    "options": options,
                    "answer_idx": example["label"],
                }
            elif short_name == "winogrande":
                # sentence, option1, option2, answer (1 or 2)
                # We'll adapt: question = sentence with blank? Usually format: "sentence"
                # But template expects question; we'll use "Fill in the blank: " + sentence
                options = [example["option1"], example["option2"]]
                ans_idx = int(example["answer"]) - 1
                return {
                    "question": example["sentence"],
                    "context": "",
                    "options": options,
                    "answer_idx": ans_idx,
                }
            elif short_name in ("arc_e", "arc_c"):
                # question, choices (dict of text/label), answerKey
                choices = example["choices"]["text"]
                labels = example["choices"]["label"]  # list of "A","B",...
                # Map answerKey to index
                ans_key = example["answerKey"]
                try:
                    ans_idx = labels.index(ans_key)
                except ValueError:
                    # fallback numeric
                    ans_idx = ord(ans_key) - ord("A")
                return {
                    "question": example["question"],
                    "context": "",
                    "options": choices,
                    "answer_idx": ans_idx,
                }
            elif short_name == "obqa":
                # question, choices (dict of text/label), answerKey
                choices = example["choices"]["text"]
                labels = example["choices"]["label"]
                ans_key = example["answerKey"]
                ans_idx = labels.index(ans_key)
                return {
                    "question": example["question_stem"],
                    "context": "",
                    "options": choices,
                    "answer_idx": ans_idx,
                }
            else:
                raise ValueError(f"Unknown commonsense dataset {short_name}")

        ds = ds.map(extract_fields, remove_columns=ds.column_names)

        # Now tokenize with causal LM masking
        def tokenize_and_mask(example: Dict[str, Any]) -> Dict[str, Any]:
            context = example.get("context", "")
            options_str = "\n".join(
                f"{chr(65 + i)}. {opt}" for i, opt in enumerate(example["options"])
            )
            prompt = self.COMMONSENSE_TEMPLATE.format(
                context=(context + "\n") if context else "",
                question=example["question"],
                options_str=options_str,
            )
            # The answer is a letter, e.g., "A", "B", ...
            answer_letter = chr(65 + example["answer_idx"])  # 0->A
            completion = answer_letter

            # Tokenize prompt separately to get its length
            prompt_enc = self.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
            prompt_len = prompt_enc.input_ids.shape[1]

            full_text = prompt + completion
            full_enc = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )
            labels = full_enc.input_ids.clone()
            labels[:, :prompt_len] = -100
            # For evaluation, we may want to keep the answer_idx and options
            return {
                "input_ids": full_enc.input_ids[0],
                "attention_mask": full_enc.attention_mask[0],
                "labels": labels[0],
                "answer_idx": example["answer_idx"],
                "options": example["options"],
                "question": example["question"],
                "context": example.get("context", ""),
            }

        ds = ds.map(tokenize_and_mask, remove_columns=ds.column_names)
        return ds

    # --------------------------------------------------------------------------
    # GLUE tasks (RoBERTa‑large)
    # --------------------------------------------------------------------------

    def _preprocess_glue(self, split: str) -> Dataset:
        """Load and tokenize a single GLUE task (e.g., cola, rte)."""
        task = self.sub_task
        # Load dataset
        try:
            ds = load_dataset("glue", task, split=split, trust_remote_code=True)
        except Exception:
            # Some tasks have 'validation' instead of 'test'; we handle outside.
            ds = load_dataset("glue", task, split=split, trust_remote_code=True)

        # Determine whether it's a sentence-pair task
        is_pair = task in ["mrpc", "qnli", "rte", "stsb", "wnli"]

        def tokenize_fn(example: Dict[str, Any]) -> Dict[str, Any]:
            if is_pair:
                tokenized = self.tokenizer(
                    example["sentence1"],
                    example["sentence2"],
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    padding=False,  # done by collator
                    return_tensors=None,  # dict of lists
                )
            else:
                tokenized = self.tokenizer(
                    example["sentence"],
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    padding=False,
                    return_tensors=None,
                )
            tokenized["label"] = example["label"]
            return tokenized

        # Remove all columns except those we will keep
        columns_to_remove = [
            col for col in ds.column_names if col not in ("sentence1", "sentence2", "sentence", "label")
        ]
        ds = ds.map(tokenize_fn, remove_columns=columns_to_remove)
        ds = ds.with_format("torch")
        return ds

    # --------------------------------------------------------------------------
    # Utility properties
    # --------------------------------------------------------------------------

    @property
    def train_dataset_size(self) -> int:
        if self._train_dataset is None:
            self._load_train_dataset()
        return len(self._train_dataset)
