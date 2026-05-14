```python
# evaluation/in_loop_eval.py
"""
In‑loop downstream evaluation for the OLMoE pretraining run.

This module implements the ``InLoopEvaluator`` class that evaluates the model
on a set of language understanding tasks (ARC, HellaSwag, MMLU, etc.) using
the Completion/Cloze (CF) formulation, as described in the OLMoE paper
(Appendix C – “During pretraining”).  The evaluator is called by
``PretrainTrainer`` at regular intervals and logs the resulting metrics.

All evaluation logic follows the paper's setup:
  - 0‑shot for most tasks, character‑length normalisation where indicated.
  - MMLU Var aggregates accuracy across 0‑5 shots.
  - Few‑shot demonstrations are sampled deterministically from the
    corresponding training (or dev) splits.

Dependencies: datasets, transformers, torch, numpy.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import (
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
from transformers import PreTrainedTokenizer

import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helper: chunk a list of few‑shot demonstrations into a single string
# ---------------------------------------------------------------------------
def _join_demos(demos: List[str]) -> str:
    """Concatenate few‑shot demonstration strings with a newline separator."""
    if not demos:
        return ""
    return "\n\n".join(demos) + "\n\n"


# ---------------------------------------------------------------------------
#  Main evaluator class
# ---------------------------------------------------------------------------
class InLoopEvaluator:
    """
    Evaluates a given model on a set of predefined tasks using the Cloze
    (completion) formulation.

    Args:
        tokenizer:        HuggingFace tokenizer (GPT‑NeoX for OLMoE).
        tasks:            List of task name strings (e.g. ``["hellaswag", "mmlu_var"]``).
        eval_config:      Optional dictionary with evaluation settings (unused for now).
        max_seq_length:   Maximum sequence length for tokenization (default 4096).
        seed:             Random seed for deterministic few‑shot example selection.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        tasks: List[str],
        eval_config: Optional[Dict] = None,
        max_seq_length: int = 4096,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.tasks = tasks
        self.eval_config = eval_config or {}
        self.max_seq_length = max_seq_length
        self.seed = seed

        # Cache few‑shot demonstration strings: {task_key: {shots: [demo_str, ...]}}
        self._fewshot_cache: Dict[str, Dict[int, List[str]]] = {}

        # Build task configuration lookup (dataset name, splits, norm type)
        self._task_config = self._build_task_config()

    # ======================================================================
    #  Public API
    # ======================================================================
    def evaluate(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation on all configured tasks.  Should be called only on
        rank 0 (the trainer handles this, but we guard anyway).

        Args:
            model:  The Lu model in evaluation mode.
            step:   Current training step (for logging, not used here).

        Returns:
            Dictionary mapping task name → accuracy (or NaN on failure).
        """
        # Only master rank performs evaluation to avoid duplicated work.
        if dist.is_initialized() and dist.get_rank() != 0:
            return {}

        model.eval()
        metrics: Dict[str, float] = {}

        with torch.no_grad():
            for task in self.tasks:
                try:
                    task_info = self._task_config[task]
                    logger.info("Evaluating %s ...", task)

                    if task == "mmlu_var":
                        # Average accuracy over 0‑5 shots
                        accs = []
                        for shots_val in range(0, 6):
                            acc = self._eval_task(model, task_info, shots=shots_val)
                            accs.append(acc)
                        metrics["mmlu_var"] = float(np.mean(accs))
                    else:
                        shots = task_info.get("shots", 0)
                        metrics[task] = self._eval_task(model, task_info, shots)

                    logger.info("  %s: %.4f", task, metrics[task])
                except Exception:
                    logger.exception("Evaluation of %s failed.", task)
                    metrics[task] = float("nan")

        return metrics

    # ======================================================================
    #  Per‑task evaluation implementation
    # ======================================================================
    def _eval_task(
        self,
        model: torch.nn.Module,
        task_info: Dict,
        shots: int,
    ) -> float:
        """Evaluate a single task with a given number of few‑shot examples."""
        dataset = self._load_val_split(task_info)
        fewshot_demos = self._get_fewshot_demos(task_info, shots)

        correct, total = 0, 0
        for example in dataset:
            prompt, choices, label = self._format_example(
                task_info, example, fewshot_demos
            )
            if prompt is None:
                continue

            norm = task_info.get("norm", "none")
            scores = self._compute_scores(model, prompt, choices, norm)
            pred = np.argmax(scores)
            if pred == label:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    # ======================================================================
    #  Dataset loading
    # ======================================================================
    def _load_val_split(self, task_info: Dict):
        """Load the validation split for a task."""
        ds_name = task_info["dataset"]
        subset = task_info.get("subset")
        split = task_info["val_split"]
        if subset and subset not in (None, "all"):
            return load_dataset(ds_name, subset, split=split, trust_remote_code=True)
        else:
            return load_dataset(ds_name, split=split, trust_remote_code=True)

    def _load_train_subset(self, task_info: Dict, num_examples: int = 0):
        """
        Load the train (or few‑shot) split.  For MMLU with ``subset="all"``,
        the dev splits of all subjects are concatenated.
        """
        ds_name = task_info["dataset"]
        subset = task_info.get("subset")
        split = task_info["train_split"]

        if ds_name == "mmlu" and subset == "all":
            # Combine the dev split of every MMLU subject.
            return self._load_mmlu_combined_dev(num_examples)

        if subset and subset not in (None, "all"):
            ds = load_dataset(ds_name, subset, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(ds_name, split=split, trust_remote_code=True)

        if num_examples > 0 and len(ds) > num_examples:
            rng = np.random.default_rng(self.seed + 100)
            indices = rng.choice(len(ds), size=num_examples, replace=False)
            ds = ds.select(indices.tolist())
        return ds

    def _load_mmlu_combined_dev(self, num_examples: int = 0):
        """
        Load the ``dev`` split of every MMLU subject and concatenate them.
        If ``num_examples > 0``, a deterministic sample is returned.
        """
        all_subjects = get_dataset_config_names("mmlu")
        dev_sets = []
        for subj in all_subjects:
            try:
                ds = load_dataset("mmlu", subj, split="dev", trust_remote_code=True)
                dev_sets.append(ds)
            except Exception:
                logger.debug("Could not load MMLU dev for subject %s", subj)
        if not dev_sets:
            raise RuntimeError("No MMLU dev splits could be loaded.")
        combined = concatenate_datasets(dev_sets)
        if num_examples > 0 and len(combined) > num_examples:
            rng = np.random.default_rng(self.seed + 200)
            indices = rng.choice(len(combined), size=num_examples, replace=False)
            combined = combined.select(indices.tolist())
        return combined

    # ======================================================================
    #  Few‑shot demonstration cache
    # ======================================================================
    def _get_fewshot_demos(self, task_info: Dict, shots: int) -> List[str]:
        """Return formatted demonstration strings (prompt + answer) for the task."""
        if shots <= 0:
            return []

        cache_key = (
            f"{task_info['dataset']}_{task_info.get('subset','')}_{shots}"
        )
        if cache_key not in self._fewshot_cache:
            train_ds = self._load_train_subset(task_info, num_examples=shots)
            demos = []
            for example in train_ds:
                demo = self._format_demonstration(task_info, example)
                demos.append(demo)
            self._fewshot_cache[cache_key] = demos
        return self._fewshot_cache[cache_key]

    # ======================================================================
    #  Task‑specific formatting
    # ======================================================================
    def _format_example(
        self,
        task_info: Dict,
        example: Dict,
        fewshot_demos: List[str],
    ) -> Tuple[Optional[str], List[str], int]:
        """
        Produce the prompt (without final answer) and the list of answer
        choice strings together with the ground‑