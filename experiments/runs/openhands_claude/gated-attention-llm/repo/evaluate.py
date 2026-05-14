"""
Evaluation utilities for gated-attention language models.

Covers the benchmarks reported in the paper (Sec. 3.1):
  - Perplexity on diverse held-out test sets (English, Chinese, Code, Math, Law, Literature)
  - Few-shot evaluation via lm-eval harness:
      Hellaswag, MMLU, GSM8k, HumanEval, C-eval, CMMLU
  - RULER long-context benchmark (Sec. 4.4)

Usage:
    python evaluate.py --checkpoint checkpoints/moe_15a2b_G1_elementwise/step_0100000.pt
                       --eval_data_dir /data/eval
                       --benchmarks hellaswag mmlu gsm8k
"""

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ModelConfig
from model import GatedTransformerLM, build_model
from data import EvalDataset, make_eval_dataloader


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: GatedTransformerLM,
    data_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> float:
    """Compute perplexity on a held-out dataset."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type=device.type, dtype=dtype):
            out = model(input_ids=input_ids, labels=labels)

        # out["loss"] is mean CE over non-ignored tokens
        n_tokens = (labels != -100).sum().item()
        total_nll += out["loss"].item() * n_tokens
        total_tokens += n_tokens

    avg_nll = total_nll / max(total_tokens, 1)
    return math.exp(avg_nll)


def evaluate_ppl_suite(
    model: GatedTransformerLM,
    eval_data_dir: str,
    seq_len: int = 4096,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Evaluate perplexity on all available domain test sets.

    Expects files named: english.bin, chinese.bin, code.bin, math.bin,
    law.bin, literature.bin  (uint16 token arrays).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domains = ["english", "chinese", "code", "math", "law", "literature"]
    results: dict[str, float] = {}

    for domain in domains:
        path = Path(eval_data_dir) / f"{domain}.bin"
        if not path.exists():
            continue
        dataset = EvalDataset.from_file(path, seq_len=seq_len)
        loader = make_eval_dataloader(dataset, batch_size=batch_size)
        ppl = compute_perplexity(model, loader, device)
        results[domain] = ppl
        print(f"  {domain:12s} PPL: {ppl:.4f}")

    if results:
        avg_ppl = sum(results.values()) / len(results)
        results["avg"] = avg_ppl
        print(f"  {'avg':12s} PPL: {avg_ppl:.4f}")

    return results


# ---------------------------------------------------------------------------
# lm-eval harness integration
# ---------------------------------------------------------------------------

def evaluate_lm_eval(
    model: GatedTransformerLM,
    tokenizer,
    tasks: list[str],
    num_fewshot: int = 5,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> dict[str, float]:
    """Run few-shot evaluation using the lm-evaluation-harness.

    Tasks used in the paper:
      - hellaswag (10-shot)
      - mmlu (5-shot)
      - gsm8k (8-shot)
      - humaneval (0-shot)
      - ceval-valid (5-shot)
      - cmmlu (5-shot)
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError("Install lm-eval: pip install lm-eval>=0.4.2")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Wrap our model in an lm-eval compatible interface
    lm = _WrappedLM(model, tokenizer, device=device, batch_size=batch_size)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )

    scores: dict[str, float] = {}
    for task, task_results in results["results"].items():
        # Extract the primary metric for each task
        if "acc_norm,none" in task_results:
            scores[task] = task_results["acc_norm,none"] * 100
        elif "acc,none" in task_results:
            scores[task] = task_results["acc,none"] * 100
        elif "exact_match,strict-match" in task_results:
            scores[task] = task_results["exact_match,strict-match"] * 100
        elif "pass@1,none" in task_results:
            scores[task] = task_results["pass@1,none"] * 100

    return scores


class _WrappedLM:
    """Minimal lm-eval wrapper for GatedTransformerLM."""

    def __init__(self, model, tokenizer, device: str = "cuda", batch_size: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self._device = torch.device(device)
        self._batch_size = batch_size
        self.model.eval()

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self.model.cfg.max_seq_len

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def tok_encode(self, string: str) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        results = []
        for ctx, cont in requests:
            ctx_ids = self.tok_encode(ctx)
            cont_ids = self.tok_encode(cont)
            input_ids = torch.tensor(
                [ctx_ids + cont_ids], dtype=torch.long, device=self._device
            )
            labels = input_ids.clone()
            labels[0, : len(ctx_ids)] = -100

            out = self.model(input_ids=input_ids, labels=labels)
            log_prob = -out["loss"].item() * len(cont_ids)
            is_greedy = self._is_greedy(ctx_ids, cont_ids)
            results.append((log_prob, is_greedy))
        return results

    def _is_greedy(self, ctx_ids: list[int], cont_ids: list[int]) -> bool:
        input_ids = torch.tensor([ctx_ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids)
        logits = out["logits"][0, -1, :]
        greedy_token = logits.argmax().item()
        return greedy_token == cont_ids[0]

    @torch.no_grad()
    def loglikelihood_rolling(self, requests) -> list[float]:
        results = []
        for (string,) in requests:
            ids = self.tok_encode(string)
            input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
            labels = input_ids.clone()
            out = self.model(input_ids=input_ids, labels=labels)
            results.append(-out["loss"].item() * len(ids))
        return results

    @torch.no_grad()
    def generate_until(self, requests) -> list[str]:
        results = []
        for ctx, gen_kwargs in requests:
            ctx_ids = self.tok_encode(ctx)
            input_ids = torch.tensor([ctx_ids], dtype=torch.long, device=self._device)
            max_new = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_new,
                eos_token_id=self.eot_token_id,
            )
            new_tokens = generated[0, len(ctx_ids) :].tolist()
            results.append(self.tok_decode(new_tokens))
        return results


# ---------------------------------------------------------------------------
# RULER long-context evaluation (Sec. 4.4)
# ---------------------------------------------------------------------------

def evaluate_ruler(
    model: GatedTransformerLM,
    tokenizer,
    context_lengths: list[int] = [4096, 8192, 16384, 32768, 65536, 131072],
    device: Optional[torch.device] = None,
) -> dict[int, float]:
    """Evaluate on RULER benchmark at multiple context lengths (Hsieh et al., 2024).

    RULER tests the model's ability to retrieve and reason over information
    at various positions within long contexts.  This function provides the
    evaluation harness; the actual RULER tasks require the official dataset.

    Returns a dict mapping context_length → accuracy (%).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from ruler_eval import RulerEvaluator  # type: ignore
        evaluator = RulerEvaluator(model, tokenizer, device=device)
        results = {}
        for ctx_len in context_lengths:
            if ctx_len > model.cfg.max_seq_len:
                print(f"  Skipping ctx_len={ctx_len} (exceeds model max_seq_len)")
                continue
            acc = evaluator.evaluate(context_length=ctx_len)
            results[ctx_len] = acc
            print(f"  RULER ctx={ctx_len:7d}: {acc:.2f}%")
        return results
    except ImportError:
        print(
            "RULER evaluator not found. Install from "
            "https://github.com/hsiehjackson/RULER and add to PYTHONPATH."
        )
        return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate gated-attention LM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_data_dir", type=str, default=None)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["hellaswag", "mmlu", "gsm8k"],
        help="lm-eval task names",
    )
    parser.add_argument("--num_fewshot", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--ruler", action="store_true", help="Run RULER evaluation")
    parser.add_argument(
        "--ruler_lengths",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384, 32768, 65536, 131072],
    )
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2-7B")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg: ModelConfig = ckpt["model_cfg"]
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Tokenizer
    from data import load_tokenizer
    tokenizer = load_tokenizer(args.tokenizer)

    # Perplexity
    if args.eval_data_dir is not None:
        print("\n=== Perplexity ===")
        evaluate_ppl_suite(model, args.eval_data_dir, seq_len=args.seq_len, device=device)

    # Few-shot benchmarks
    if args.benchmarks:
        print("\n=== Few-shot Benchmarks ===")
        scores = evaluate_lm_eval(
            model, tokenizer, args.benchmarks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            device=str(device),
        )
        for task, score in scores.items():
            print(f"  {task:20s}: {score:.2f}")

    # RULER
    if args.ruler:
        print("\n=== RULER Long-Context ===")
        evaluate_ruler(model, tokenizer, args.ruler_lengths, device=device)


if __name__ == "__main__":
    main()
