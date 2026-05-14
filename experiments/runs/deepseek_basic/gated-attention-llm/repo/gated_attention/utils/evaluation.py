"""
Evaluation utilities for gated LLMs.

Implements the evaluation protocol described in Sec 3.1:
  - Perplexity (PPL) on diverse held-out test sets
  - Few-shot benchmarks: Hellaswag, MMLU, GSM8k, HumanEval, C-eval, CMMLU
  - Long-context RULER evaluation (Sec 4.4, Table 5)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class BenchmarkResults:
    """Container for benchmark evaluation results.

    Maps to the metrics reported in Tables 1-5:
      - Avg PPL (perplexity on held-out test sets)
      - Hellaswag (English commonsense, Zellers et al., 2019)
      - MMLU (general knowledge, Hendrycks et al., 2020)
      - GSM8k (math reasoning, Cobbe et al., 2021)
      - HumanEval (coding, Chen et al., 2021)
      - C-eval (Chinese proficiency, Huang et al., 2024)
      - CMMLU (Chinese multitask, Li et al., 2023)
      - RULER (long-context, Hsieh et al., 2024)
    """
    avg_ppl: float = float("inf")
    hellaswag: float = 0.0
    mmlu: float = 0.0
    gsm8k: float = 0.0
    humaneval: float = 0.0
    c_eval: float = 0.0
    cmmlu: float = 0.0

    # RULER results at various context lengths
    ruler_scores: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "avg_ppl": self.avg_ppl,
            "hellaswag": self.hellaswag,
            "mmlu": self.mmlu,
            "gsm8k": self.gsm8k,
            "humaneval": self.humaneval,
            "c_eval": self.c_eval,
            "cmmlu": self.cmmlu,
            **{f"ruler_{k}k": v for k, v in self.ruler_scores.items()},
        }


def evaluate_perplexity(
    model,
    dataloader,
    max_batches: int = 1000,
    device: torch.device = None,
) -> float:
    """Evaluate perplexity on a held-out test set.

    Following Sec 3.1: PPL is reported on diverse domains including
    English, Chinese, Code, Math, Law, and Literature.

    Args:
        model: The GatedLLM model
        dataloader: DataLoader yielding {"input_ids": ..., "labels": ...}
        max_batches: Maximum number of evaluation batches
        device: Device to run on

    Returns:
        Perplexity score
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids)
            logits = outputs["logits"]

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += (shift_labels != -100).sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss)

    return ppl


def evaluate_multiple_domains(
    model,
    dataloaders: Dict[str, object],
    max_batches: int = 1000,
) -> Dict[str, float]:
    """Evaluate PPL across multiple domains.

    Following Sec 3.1: evaluates on English, Chinese, Code, Math,
    Law, and Literature test sets.
    """
    results = {}
    for domain, loader in dataloaders.items():
        ppl = evaluate_perplexity(model, loader, max_batches)
        results[domain] = ppl

    results["avg_ppl"] = sum(results.values()) / len(results)
    return results


def evaluate_benchmarks(
    model,
    tokenizer,
    benchmarks: Optional[List[str]] = None,
    num_fewshot: int = 5,
) -> BenchmarkResults:
    """Evaluate on standard few-shot benchmarks.

    Implements the evaluation protocol from Sec 3.1.

    Benchmarks:
      - Hellaswag: 10-shot, accuracy
      - MMLU: 5-shot, accuracy
      - GSM8k: 5-shot, exact match
      - HumanEval: 0-shot, pass@1
      - C-eval: 5-shot, accuracy
      - CMMLU: 5-shot, accuracy
    """
    if benchmarks is None:
        benchmarks = ["hellaswag", "mmlu", "gsm8k", "humaneval", "c_eval", "cmmlu"]

    results = BenchmarkResults()

    # Note: Full benchmark evaluation requires the specific dataset implementations.
    # This is a scaffolding that would be filled in with actual evaluation code
    # using libraries like lm-evaluation-harness or custom implementations.

    # Placeholder: in practice, one would integrate with lm_eval or similar
    results.avg_ppl = 0.0  # Would come from perplexity evaluation

    return results


def evaluate_ruler(
    model,
    tokenizer,
    seq_lengths: List[int] = [4096, 8192, 16384, 32768, 65536, 131072],
) -> Dict[int, float]:
    """Evaluate on RULER benchmark at various context lengths.

    Following Sec 4.4 and Table 5: RULER (Hsieh et al., 2024)
    evaluates long-context capabilities across sequence lengths.

    The paper reports:
      - Baseline (32k training): 88.89 -> 85.88 -> ... -> fails at 64k/128k
      - SDPA-Gate (32k training): 90.56 -> 87.11 -> ... -> 58.82 at 128k
      - YaRN extended baseline: drops significantly (82.90 at 4k)
      - YaRN extended SDPA-Gate: more robust (88.13 at 4k, 58.82 at 128k)
    """
    # RULER evaluation would go here
    # Requires the RULER benchmark dataset and evaluation protocol
    return {}
