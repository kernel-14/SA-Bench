"""
Downstream task evaluation for nGPT / GPT.

The paper (Section 3, Figure 3) evaluates on a set of standard downstream
tasks using few-shot prompting in the style of GPT-2 (Radford et al., 2018).
Tasks reported in the paper include:
  - HellaSwag
  - PIQA
  - WinoGrande
  - ARC-Easy / ARC-Challenge
  - WMT14 FR→EN (BLEU, Figure 10)

Evaluation strategy: likelihood-based multiple-choice (rank classification).
For each candidate completion, compute the log-likelihood under the model
and select the highest-scoring one.

Usage:
    python evaluate.py --checkpoint checkpoints/ckpt_0100000.pt \
                       --tasks hellaswag piqa winogrande arc_easy arc_challenge
"""

import os
import copy
import json
import math
import argparse
import logging
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import ModelConfig, MODEL_CONFIGS
from model import build_model, NGPT

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Likelihood-based evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_completions(
    model: torch.nn.Module,
    context_ids: torch.Tensor,
    completion_ids_list: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> List[float]:
    """Score each completion by its average log-likelihood given the context.

    Args:
        model:                Language model.
        context_ids:          Token IDs for the context (1-D tensor).
        completion_ids_list:  List of 1-D tensors, one per candidate.
        device:               Compute device.
        dtype:                Autocast dtype.

    Returns:
        List of average log-likelihoods (higher = more likely).
    """
    model.eval()
    scores = []
    ctx = torch.autocast(device_type=device.type, dtype=dtype)

    for completion_ids in completion_ids_list:
        input_ids = torch.cat([context_ids, completion_ids]).unsqueeze(0).to(device)
        targets = input_ids[:, 1:].clone()
        input_ids = input_ids[:, :-1]

        with ctx:
            logits, _ = model(input_ids)

        # Only score the completion tokens
        n_ctx = len(context_ids)
        comp_logits = logits[0, n_ctx - 1 : n_ctx - 1 + len(completion_ids)]
        comp_targets = targets[0, n_ctx - 1 : n_ctx - 1 + len(completion_ids)]

        log_probs = F.log_softmax(comp_logits, dim=-1)
        token_log_probs = log_probs.gather(1, comp_targets.unsqueeze(1)).squeeze(1)
        avg_log_prob = token_log_probs.mean().item()
        scores.append(avg_log_prob)

    return scores


# ---------------------------------------------------------------------------
# Task-specific dataset loaders
# ---------------------------------------------------------------------------

class HellaSwagDataset(Dataset):
    """HellaSwag: 4-way sentence completion."""

    def __init__(self, data_path: str, tokenizer):
        self.tokenizer = tokenizer
        with open(data_path) as f:
            self.examples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        context = ex["ctx"]
        endings = ex["endings"]
        label = int(ex["label"])
        return context, endings, label


class PIQADataset(Dataset):
    """PIQA: 2-way physical intuition QA."""

    def __init__(self, data_path: str, labels_path: str, tokenizer):
        self.tokenizer = tokenizer
        with open(data_path) as f:
            self.examples = [json.loads(line) for line in f]
        with open(labels_path) as f:
            self.labels = [int(line.strip()) for line in f]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        goal = ex["goal"]
        solutions = [ex["sol1"], ex["sol2"]]
        label = self.labels[idx]
        return goal, solutions, label


class WinoGrandeDataset(Dataset):
    """WinoGrande: 2-way commonsense reasoning."""

    def __init__(self, data_path: str, tokenizer):
        self.tokenizer = tokenizer
        with open(data_path) as f:
            self.examples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        sentence = ex["sentence"]
        options = [ex["option1"], ex["option2"]]
        label = int(ex["answer"]) - 1   # 1-indexed → 0-indexed
        # Replace "_" placeholder with each option
        completions = [sentence.replace("_", opt) for opt in options]
        return "", completions, label


class ARCDataset(Dataset):
    """ARC Easy / Challenge: 4-way science QA."""

    def __init__(self, data_path: str, tokenizer):
        self.tokenizer = tokenizer
        with open(data_path) as f:
            self.examples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        question = ex["question"]["stem"]
        choices = ex["question"]["choices"]
        texts = [c["text"] for c in choices]
        answer_key = ex["answerKey"]
        labels = [c["label"] for c in choices]
        label = labels.index(answer_key)
        context = f"Question: {question}\nAnswer:"
        return context, texts, label


# ---------------------------------------------------------------------------
# Generic evaluation loop
# ---------------------------------------------------------------------------

def evaluate_task(
    model: torch.nn.Module,
    dataset: Dataset,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    max_examples: Optional[int] = None,
) -> Dict[str, float]:
    """Run likelihood-based evaluation on a multiple-choice dataset.

    Returns:
        dict with "accuracy" key.
    """
    correct = 0
    total = 0

    for i in range(len(dataset)):
        if max_examples is not None and i >= max_examples:
            break

        context, completions, label = dataset[i]

        context_ids = torch.tensor(
            tokenizer.encode(context), dtype=torch.long
        )
        completion_ids_list = [
            torch.tensor(tokenizer.encode(c), dtype=torch.long)
            for c in completions
        ]

        scores = score_completions(model, context_ids, completion_ids_list, device, dtype)
        pred = scores.index(max(scores))
        correct += int(pred == label)
        total += 1

    return {"accuracy": correct / total if total > 0 else 0.0}


# ---------------------------------------------------------------------------
# Perplexity evaluation (validation loss)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    data_path: str,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    n_batches: int = 100,
) -> float:
    """Compute perplexity on a binary token file."""
    from data import TokenizedDataset

    dataset = TokenizedDataset(data_path, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model.eval()
    ctx = torch.autocast(device_type=device.type, dtype=dtype)
    total_loss = 0.0
    count = 0

    for x, y in loader:
        if count >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / max(count, 1)
    return math.exp(avg_loss)


# ---------------------------------------------------------------------------
# Analysis utilities (Section 3.2 / Figure 4-6)
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_embedding_norms(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Compute norm statistics for input and output embeddings (Figure 4)."""
    raw = model.module if hasattr(model, "module") else model
    stats = {}

    if hasattr(raw, "E_input"):
        norms_in = raw.E_input.weight.norm(dim=1)
        stats["input_embedding_norms"] = norms_in
    if hasattr(raw, "E_output"):
        w = raw.E_output.weight if hasattr(raw.E_output, "weight") else raw.E_output
        norms_out = w.norm(dim=1)
        stats["output_embedding_norms"] = norms_out

    return stats


@torch.no_grad()
def analyze_attention_condition_numbers(model: torch.nn.Module) -> List[Dict[str, float]]:
    """Compute median condition numbers for attention matrices per layer (Figure 5)."""
    raw = model.module if hasattr(model, "module") else model
    results = []

    for layer_idx, layer in enumerate(raw.layers):
        attn = layer.attn if hasattr(layer, "attn") else None
        if attn is None:
            continue

        cond_numbers = []
        for proj in [attn.Wq, attn.Wk, attn.Wv, attn.Wo]:
            w = proj.weight.float()
            try:
                sv = torch.linalg.svdvals(w)
                cond = (sv.max() / sv.min().clamp(min=1e-10)).item()
                cond_numbers.append(cond)
            except Exception:
                pass

        results.append({
            "layer": layer_idx,
            "median_cond": float(torch.tensor(cond_numbers).median()) if cond_numbers else float("nan"),
        })

    return results


@torch.no_grad()
def analyze_eigen_learning_rates(model: torch.nn.Module) -> List[Dict[str, float]]:
    """Extract eigen learning rate statistics per layer (Figure 6)."""
    raw = model.module if hasattr(model, "module") else model
    if not isinstance(raw, NGPT):
        return []

    results = []
    for layer_idx, layer in enumerate(raw.layers):
        alpha_A = layer.alpha_A().abs()
        alpha_M = layer.alpha_M().abs()
        results.append({
            "layer": layer_idx,
            "alpha_A_mean": alpha_A.mean().item(),
            "alpha_A_std": alpha_A.std().item(),
            "alpha_M_mean": alpha_M.mean().item(),
            "alpha_M_std": alpha_M.std().item(),
        })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT / nGPT on downstream tasks")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default=None,
                        help="Override model type (inferred from checkpoint if not set)")
    parser.add_argument("--model_size", type=str, default="0.5B")
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--tasks", nargs="+",
                        default=["hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge"],
                        help="Tasks to evaluate")
    parser.add_argument("--data_dir", type=str, default="data/eval",
                        help="Directory containing task data files")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--val_bin", type=str, default=None,
                        help="Path to val.bin for perplexity evaluation")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_examples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    model_type = args.model_type or cfg_dict.get("model_type", "ngpt")
    model_size = args.model_size or cfg_dict.get("model_size", "0.5B")

    model_cfg = copy.deepcopy(MODEL_CONFIGS[model_size])
    model_cfg.model_type = model_type
    model_cfg.max_seq_len = args.seq_len

    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Loaded {model_type.upper()} {model_size} from {args.checkpoint}")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    results = {}

    # Perplexity
    if args.val_bin and os.path.exists(args.val_bin):
        ppl = evaluate_perplexity(model, args.val_bin, args.seq_len, args.batch_size, device, dtype)
        results["perplexity"] = ppl
        logger.info(f"Perplexity: {ppl:.3f}")

    # Downstream tasks
    task_loaders = {
        "hellaswag": lambda: HellaSwagDataset(
            os.path.join(args.data_dir, "hellaswag_val.jsonl"), tokenizer
        ),
        "piqa": lambda: PIQADataset(
            os.path.join(args.data_dir, "piqa_val.jsonl"),
            os.path.join(args.data_dir, "piqa_val_labels.lst"),
            tokenizer,
        ),
        "winogrande": lambda: WinoGrandeDataset(
            os.path.join(args.data_dir, "winogrande_val.jsonl"), tokenizer
        ),
        "arc_easy": lambda: ARCDataset(
            os.path.join(args.data_dir, "arc_easy_val.jsonl"), tokenizer
        ),
        "arc_challenge": lambda: ARCDataset(
            os.path.join(args.data_dir, "arc_challenge_val.jsonl"), tokenizer
        ),
    }

    for task in args.tasks:
        if task not in task_loaders:
            logger.warning(f"Unknown task: {task}")
            continue
        try:
            dataset = task_loaders[task]()
            task_results = evaluate_task(
                model, dataset, tokenizer, device, dtype, args.max_examples
            )
            results[task] = task_results["accuracy"]
            logger.info(f"{task}: {task_results['accuracy'] * 100:.2f}%")
        except FileNotFoundError as e:
            logger.warning(f"Skipping {task}: {e}")

    # Average accuracy (Figure 3 bottom-right)
    acc_values = [v for k, v in results.items() if k != "perplexity"]
    if acc_values:
        avg_acc = sum(acc_values) / len(acc_values)
        results["avg_accuracy"] = avg_acc
        logger.info(f"Average accuracy: {avg_acc * 100:.2f}%")

    # nGPT-specific analysis
    if model_type == "ngpt":
        eigen_stats = analyze_eigen_learning_rates(model)
        if eigen_stats:
            avg_alpha_A = sum(s["alpha_A_mean"] for s in eigen_stats) / len(eigen_stats)
            avg_alpha_M = sum(s["alpha_M_mean"] for s in eigen_stats) / len(eigen_stats)
            logger.info(f"Mean |alpha_A|: {avg_alpha_A:.4f}, Mean |alpha_M|: {avg_alpha_M:.4f}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
