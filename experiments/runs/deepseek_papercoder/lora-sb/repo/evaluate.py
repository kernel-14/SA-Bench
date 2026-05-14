"""
evaluate.py – Evaluator for LoRA‑SB reproduction.

Provides the Evaluator class that computes performance metrics for
- Arithmetic reasoning (GSM8K, MATH)
- Commonsense reasoning (multiple-choice)
- GLUE tasks (classification / regression)

All evaluation logic follows the protocols described in the paper.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import ExperimentConfig
from modeling import ModelWrapper


class Evaluator:
    """
    Computes evaluation metrics for a LoRA‑SB‑tuned model.

    Args:
        model_wrapper: ModelWrapper containing the HuggingFace model.
        config: ExperimentConfig specifying task, device, model, etc.
    """

    def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig) -> None:
        self.model = model_wrapper.model
        self.config = config
        self.device = config.device

        # Load tokenizer for decoding (generative tasks)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine evaluation mode
        task_str: str = config.task
        if task_str.startswith("arithmetic"):
            self.mode: str = "arithmetic"
        elif task_str.startswith("commonsense"):
            self.mode = "commonsense"
        elif task_str.startswith("glue_"):
            self.mode = "glue"
            self.glue_subtask: str = task_str[5:]  # e.g., "cola"
        else:
            raise ValueError(f"Unknown task: {task_str}")

        # Maximum new tokens for generation; defaults from the paper's setting
        if self.mode == "arithmetic":
            self.max_new_tokens_generation: int = getattr(
                config, "max_new_tokens_generation", 512
            )
        elif self.mode == "commonsense":
            self.max_new_tokens_generation: int = getattr(
                config, "max_new_tokens_generation", 50
            )
        else:
            self.max_new_tokens_generation: int = 0  # not used

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Run evaluation on the given DataLoader and return a dictionary of metrics.

        Args:
            loader: DataLoader yielding batches appropriate for the task.
                - For arithmetic: expects keys `input_ids`, `attention_mask`, `raw_answer`.
                - For commonsense: same structure.
                - For GLUE: `input_ids`, `attention_mask`, `label`.

        Returns:
            Dictionary mapping metric names to float values.
        """
        self.model.eval()
        if self.mode in ("arithmetic", "commonsense"):
            return self._evaluate_generative(loader)
        else:
            return self._evaluate_glue(loader)

    # ------------------------------------------------------------------
    # Generative evaluation (Arithmetic & Commonsense)
    # ------------------------------------------------------------------

    def _evaluate_generative(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate generative tasks using greedy decoding and exact-match accuracy.

        Returns:
            dict with key "accuracy".
        """
        all_extracted: List[str] = []
        all_labels: List[str] = []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            ground_truths: List[str] = batch["raw_answer"]  # list of strings

            # Greedy generation
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens_generation,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            input_len = input_ids.shape[1]
            new_tokens = generated[:, input_len:]  # strip the prompt portion

            for i in range(generated.size(0)):
                output_str = self.tokenizer.decode(
                    new_tokens[i],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()

                if self.mode == "arithmetic":
                    extracted = self._extract_arithmetic_answer(output_str)
                else:  # commonsense
                    extracted = self._extract_commonsense_answer(output_str)

                all_extracted.append(extracted)
                all_labels.append(ground_truths[i])

        # Compute exact-match accuracy
        correct = sum(
            1 for e, l in zip(all_extracted, all_labels) if self._answer_equal(e, l)
        )
        accuracy = correct / len(all_labels) if all_labels else 0.0
        return {"accuracy": float(accuracy)}

    def _extract_arithmetic_answer(self, text: str) -> str:
        """
        Extract the final numeric answer from a GSM8K‑style output.

        Looks for the `####` pattern; if absent, falls back to the last number.
        """
        # Primary pattern: #### <number>
        match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
        if match:
            num_str = match.group(1).replace(",", "")
            return num_str.strip()

        # Fallback: find all numbers and return the last one
        numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1].replace(",", "").strip()

        return ""  # no number found

    def _extract_commonsense_answer(self, text: str) -> str:
        """
        Extract the multiple‑choice answer from model output.

        The output is expected to contain a single answer token (e.g., 'A', 'yes').
        """
        # Normalise: lower‑case, strip
        cleaned = text.strip().lower()

        # If the whole output is a single answer token, return it.
        if cleaned in ("a", "b", "c", "d", "yes", "no"):
            return cleaned

        # Look for the first token that matches an answer token
        for token in cleaned.split():
            if token in ("a", "b", "c", "d", "yes", "no"):
                return token

        # Fallback: return the whole text (likely incorrect)
        return cleaned

    def _answer_equal(self, extracted: str, ground: str) -> bool:
        """
        Compare two answer strings after normalisation.
        """
        ext_norm = self._normalize_answer(extracted)
        grd_norm = self._normalize_answer(ground)
        return ext_norm == grd_norm

    def _normalize_answer(self, s: str) -> str:
        """
        Lower‑case, strip, collapse whitespace.
        """
        import string

        # Lowercase and remove punctuation that is often extraneous
        s = s.lower()
        s = re.sub(r"[^\w\s]", "", s)
        return " ".join(s.split())

    # ------------------------------------------------------------------
    # GLUE evaluation (classification / regression)
    # ------------------------------------------------------------------

    def _evaluate_glue(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate a GLUE task (classification or regression).

        Returns:
            dict with the appropriate metric key.
        """
        all_preds: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"]  # column name used in preprocessing

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            logits = outputs.logits.cpu()
            all_preds.append(logits)
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute metric per GLUE subtask
        task = self.glue_subtask

        if task == "cola":
            # Matthews correlation coefficient (binary classification)
            preds = all_preds.argmax(dim=-1).numpy()
            labels = all_labels.numpy()
            mcc = matthews_corrcoef(labels, preds)
            return {"matthews_correlation": float(mcc)}

        elif task == "stsb":
            # Regression: Pearson correlation
            # Logits are shape (N, 1) after squeeze
            preds = all_preds.squeeze(-1).numpy()
            labels = all_labels.numpy()
            pearson, _ = pearsonr(preds, labels)
            return {"pearson": float(pearson)}

        else:
            # Accuracy for RTE, MRPC, QNLI, SST‑2
            preds = all_preds.argmax(dim=-1).numpy()
            labels = all_labels.numpy()
            acc = accuracy_score(labels, preds)
            return {"accuracy": float(acc)}
