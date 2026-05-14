"""
Script to train and evaluate DRC agents of different sizes (Appendix F).

Trains DRC(1,9) and DRC(9,1) agents and runs the full probing + intervention
pipeline on each.
"""
import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from dataclasses import replace

from config import Config, DRCConfig
from model.drc import DRCAgent
from training.train import train, load_checkpoint
from data.boxoban import load_dataset
from probing.concepts import compute_agent_approach_direction, compute_box_push_direction
from probing.evaluate import run_probing_experiment
from utils.metrics import evaluate_agent, compute_extra_levels_solved


AGENT_CONFIGS = {
    "DRC_1_9": DRCConfig(num_layers=1, num_ticks=9),
    "DRC_9_1": DRCConfig(num_layers=9, num_ticks=1),
    "DRC_3_3": DRCConfig(num_layers=3, num_ticks=3),
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate DRC agents of different sizes")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="evaluate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--boxoban_path", type=str, default="data/boxoban-levels")
    parser.add_argument("--output_dir", type=str, default="results/agent_sizes")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--agent_type", choices=list(AGENT_CONFIGS.keys()), default="DRC_1_9")
    parser.add_argument("--num_seeds", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = Config()
    config.data.boxoban_path = args.boxoban_path
    config.drc = AGENT_CONFIGS[args.agent_type]
    config.train.total_transitions = 100_000_000

    if args.mode == "train":
        print(f"Training {args.agent_type} agent...")
        agent_ckpt_dir = os.path.join(args.checkpoint_dir, args.agent_type)
        train(config, checkpoint_dir=agent_ckpt_dir)
        return

    ckpt_path = os.path.join(args.checkpoint_dir, args.agent_type, "final_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    agent = load_checkpoint(ckpt_path, config, device=str(device))
    agent.eval()

    train_dataset = load_dataset(args.boxoban_path, config.data.train_split)
    test_dataset = load_dataset(args.boxoban_path, config.data.valid_split)
    medium_dataset = load_dataset(args.boxoban_path, config.data.medium_split)
    medium_levels = [medium_dataset[i] for i in range(min(1000, len(medium_dataset)))]

    print(f"\n=== {args.agent_type} Agent Evaluation ===")

    base_metrics = evaluate_agent(agent, medium_levels, config.env, device, thinking_steps=0)
    think_metrics = evaluate_agent(agent, medium_levels, config.env, device, thinking_steps=5)
    print(f"Solve rate (no thinking): {base_metrics['solve_rate'] * 100:.1f}%")
    print(f"Solve rate (5 thinking):  {think_metrics['solve_rate'] * 100:.1f}%")

    results = {
        "agent_type": args.agent_type,
        "base_solve_rate": base_metrics["solve_rate"],
        "think_solve_rate": think_metrics["solve_rate"],
        "probing": {},
    }

    for concept_name, concept_fn in [
        ("CA", compute_agent_approach_direction),
        ("CB", compute_box_push_direction),
    ]:
        results["probing"][concept_name] = {}
        for probe_size in [1, 3]:
            results["probing"][concept_name][f"probe_{probe_size}x{probe_size}"] = {}
            for layer_idx in range(config.drc.num_layers):
                print(f"\nProbing {concept_name}, {probe_size}x{probe_size}, Layer {layer_idx + 1}")
                r = run_probing_experiment(
                    agent=agent,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    concept_fn=concept_fn,
                    layer_idx=layer_idx,
                    probe_size=probe_size,
                    config=config,
                    device=device,
                    num_seeds=args.num_seeds,
                )
                results["probing"][concept_name][f"probe_{probe_size}x{probe_size}"][f"layer_{layer_idx + 1}"] = r
                print(f"  Macro F1: {r['mean_macro_f1']:.4f} ± {r['std_macro_f1']:.4f}")

    output_path = os.path.join(args.output_dir, f"{args.agent_type}_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
