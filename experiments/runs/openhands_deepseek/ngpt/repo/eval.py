"""Evaluation on downstream tasks.

Evaluates nGPT and GPT models on standard benchmarks after training,
matching the evaluation setup described in Section 3 and Figure 3.
"""

import os
import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

from config import TrainConfig
from model import create_model


def compute_perplexity(model, input_ids: torch.Tensor) -> float:
    """Compute perplexity on a text sequence."""
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean',
        )
    return math.exp(loss.item())


@torch.no_grad()
def evaluate_lambada(model, tokenizer, data_path: str) -> Dict[str, float]:
    """Evaluate on LAMBADA dataset (last-word prediction)."""
    # Simplified implementation
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    total_tokens = 0

    with open(data_path, 'r') as f:
        for line in f:
            text = line.strip()
            if not text:
                continue

            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue

            input_ids = torch.tensor([tokens[:-1]], device=model.device)
            target_id = torch.tensor([tokens[-1]], device=model.device)

            logits = model(input_ids)
            last_logits = logits[0, -1, :]
            pred = torch.argmax(last_logits)

            if pred.item() == target_id.item():
                total_correct += 1
            total_samples += 1

            loss = F.cross_entropy(last_logits.unsqueeze(0), target_id)
            total_loss += loss.item()
            total_tokens += 1

    acc = total_correct / max(total_samples, 1)
    ppl = math.exp(total_loss / max(total_tokens, 1))
    return {"lambada_acc": acc, "lambada_ppl": ppl}


@torch.no_grad()
def evaluate_wikitext(model, tokenizer, data_path: str) -> Dict[str, float]:
    """Evaluate on WikiText-2 dataset."""
    total_loss = 0.0
    total_tokens = 0

    with open(data_path, 'r') as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    seq_len = model.max_seq_len

    for i in range(0, len(tokens) - seq_len, seq_len):
        chunk = tokens[i:i + seq_len]
        input_ids = torch.tensor([chunk], device=model.device)

        logits = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
        )
        total_loss += loss.item()
        total_tokens += (len(chunk) - 1)

    ppl = math.exp(total_loss / max(total_tokens, 1))
    return {"wikitext_ppl": ppl}


@torch.no_grad()
def evaluate_pg19(model, tokenizer, data_path: str, max_seq_len: int = 32768) -> Dict[str, float]:
    """Evaluate on PG19 dataset with length extrapolation (Appendix A.8).

    Tests the model's ability to handle sequences longer than training length.
    """
    results = {}
    context_lengths = [1024, 2048, 4096, 8192, 16384, 32768]

    with open(data_path, 'r') as f:
        text = f.read()

    tokens = tokenizer.encode(text)

    for ctx_len in context_lengths:
        if ctx_len > len(tokens):
            continue
        total_loss = 0.0
        total_tokens = 0

        # Take first ctx_len tokens for evaluation
        chunk = tokens[:ctx_len]
        input_ids = torch.tensor([chunk], device=model.device)

        logits = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
        )
        total_loss += loss.item()
        total_tokens += (len(chunk) - 1)

        ppl = math.exp(total_loss / max(total_tokens, 1))
        results[f"pg19_ppl_{ctx_len}"] = ppl

    return results


def evaluate_model(
    checkpoint_path: str,
    config: TrainConfig,
    eval_datasets: Dict[str, str],
    device: str = "cuda",
):
    """Main evaluation function.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Training configuration
        eval_datasets: Dict mapping dataset name to file path
        device: Device to run evaluation on
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Create model
    model = create_model(config, use_ngpt=config.use_ngpt)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = {}

    for dataset_name, data_path in eval_datasets.items():
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, skipping {dataset_name}")
            continue

        if dataset_name == "lambada":
            # Requires a tokenizer - simplified here
            pass
        elif dataset_name == "wikitext":
            # Requires a tokenizer - simplified here
            pass
        elif dataset_name == "pg19":
            # Length extrapolation
            pass

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate GPT/nGPT models")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--preset", type=str, default="0.5B", choices=["0.5B", "1.0B"])
    parser.add_argument("--use-ngpt", action="store_true", default=True)
    parser.add_argument("--eval-data-dir", type=str, default="./eval_data")
    parser.add_argument("--output", type=str, default="./eval_results.json")
    args = parser.parse_args()

    from config import ModelConfig, TrainConfig

    model_cfg = ModelConfig.presets()[args.preset]
    config = TrainConfig(model=model_cfg, use_ngpt=args.use_ngpt)

    eval_datasets = {
        "lambada": os.path.join(args.eval_data_dir, "lambada_test.jsonl"),
        "wikitext": os.path.join(args.eval_data_dir, "wikitext-2-test.txt"),
        "pg19": os.path.join(args.eval_data_dir, "pg19_test.txt"),
    }

    results = evaluate_model(args.checkpoint, config, eval_datasets)

    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("Evaluation results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
