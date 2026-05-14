"""
Script to evaluate agent performance with varying numbers of thinking steps.
Reproduces Figure 45 / Appendix E.5 behavioral evidence.
"""
import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path

from config import Config
from training.train import load_checkpoint
from data.boxoban import load_dataset
from utils.metrics import evaluate_agent


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent with thinking steps")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--boxoban_path", type=str, default="data/boxoban-levels")
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_levels", type=int, default=1000)
    parser.add_argument("--max_thinking_steps", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = Config()
    config.data.boxoban_path = args.boxoban_path

    agent = load_checkpoint(args.checkpoint, config, device=str(device))
    agent.eval()

    results = {}

    for split_name, split_path in [
        ("unfiltered_test", config.data.test_split),
        ("medium", config.data.medium_split),
        ("hard", config.data.hard_split),
    ]:
        try:
            dataset = load_dataset(args.boxoban_path, split_path)
        except FileNotFoundError:
            print(f"Split {split_name} not found, skipping")
            continue

        levels = [dataset[i] for i in range(min(args.num_levels, len(dataset)))]
        print(f"\n=== {split_name} ({len(levels)} levels) ===")

        results[split_name] = {}
        for n_think in range(args.max_thinking_steps + 1):
            metrics = evaluate_agent(
                agent=agent,
                levels=levels,
                env_config=config.env,
                device=device,
                thinking_steps=n_think,
            )
            results[split_name][n_think] = metrics
            print(f"  Thinking steps {n_think}: solve rate = {metrics['solve_rate'] * 100:.1f}%")

    output_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
