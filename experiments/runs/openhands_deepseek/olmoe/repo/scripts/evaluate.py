#!/usr/bin/env python3
"""
OLMES evaluation script for OLMoE-1B-7B.

Usage:
    python scripts/evaluate.py --model_path ./checkpoints/final

Evaluates model on standard benchmarks per Appendix C:
    - MMLU, HellaSwag, ARC-Challenge, ARC-Easy, PIQA, Winogrande
    - Uses 5-shot max(MCF, CF) evaluation
"""
import argparse
import os
import sys
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.olmoe_model import OLMoEModel
from evaluation.olmes import run_olmes_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="OLMoE OLMES Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="Specific tasks to evaluate (default: all)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples per task (for quick testing)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print(f"Loading model from: {args.model_path}")
    model = OLMoEModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    active, total = model.get_num_params()
    print(f"Model: {active/1e9:.2f}B active / {total/1e9:.2f}B total params")

    results = run_olmes_evaluation(model, tokenizer, tasks=args.tasks)

    print("\n" + "=" * 50)
    print("OLMoE-1B-7B OLMES Results:")
    print("-" * 50)
    for task, score in results.items():
        print(f"  {task:20s}: {score:.4f}")
    if results:
        avg = sum(results.values()) / len(results)
        print(f"  {'Average':20s}: {avg:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
