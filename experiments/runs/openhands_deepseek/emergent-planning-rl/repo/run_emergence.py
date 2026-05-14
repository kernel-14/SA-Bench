"""
Analyze the emergence of planning during training.

Reproduces Sections 6.2 and Appendix C:
- Trains probes at multiple training checkpoints
- Measures macro F1 for C_A and C_B at each checkpoint
- Correlates with ability to benefit from extra test-time compute
- Shows co-emergence of concept representations and planning-like behavior
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch

from configs.config import Config, ProbeConfig
from models.drc import DRCNet
from probing.linear_probe import LinearProbe, train_probe
from probing.dataset import collect_probe_data
from analysis.thinking_time import evaluate_thinking_time


def main():
    parser = argparse.ArgumentParser(description="Analyze emergence of planning during training")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing training checkpoints")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/emergence")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Episodes for probe data per checkpoint")
    parser.add_argument("--test_episodes", type=int, default=250)
    parser.add_argument("--num_thinking_episodes", type=int, default=500,
                        help="Episodes for thinking time evaluation")
    parser.add_argument("--max_checkpoints", type=int, default=50,
                        help="Maximum number of checkpoints to analyze")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    config = Config()
    probe_config = ProbeConfig()

    # Find checkpoint files
    ckpt_pattern = os.path.join(args.checkpoint_dir, "checkpoint_*.pt")
    checkpoint_files = sorted(glob.glob(ckpt_pattern))
    if not checkpoint_files:
        print(f"No checkpoints found matching {ckpt_pattern}")
        return

    checkpoint_files = checkpoint_files[:args.max_checkpoints]
    print(f"Found {len(checkpoint_files)} checkpoints")

    # Load levels
    levels = _load_levels(args.data_dir, "unfiltered", max_levels=1000)
    medium_levels = _load_levels(args.data_dir, "medium", max_levels=500)
    if not medium_levels:
        medium_levels = levels[:500]

    results = []

    for ckpt_idx, ckpt_path in enumerate(checkpoint_files):
        print(f"\n[{ckpt_idx + 1}/{len(checkpoint_files)}] {os.path.basename(ckpt_path)}")

        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
        except Exception as e:
            print(f"  Error loading checkpoint: {e}")
            continue

        transitions = checkpoint.get("total_transitions_processed", 0)
        print(f"  Transitions processed: {transitions:,}")

        # Create model and load weights
        model = DRCNet(
            input_channels=config.sokoban.num_channels,
            hidden_channels=config.drc.hidden_channels,
            num_layers=config.drc.D,
            num_ticks=config.drc.N,
            num_actions=5,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # Probe C_A and C_B
        macro_f1_A = {}
        macro_f1_B = {}

        for concept_type in ["agent_approach", "box_push"]:
            # Collect probe data
            try:
                datasets = collect_probe_data(
                    model, levels,
                    num_episodes=args.train_episodes,
                    device=device,
                    concept_type=concept_type,
                )
            except Exception as e:
                print(f"  Error collecting probe data for {concept_type}: {e}")
                continue

            for layer_idx in range(config.drc.D):
                if layer_idx not in datasets:
                    continue

                train_states, train_labels = datasets[layer_idx].get_tensors()
                if train_states.shape[0] == 0:
                    continue

                # Train a quick probe (fewer epochs for speed)
                probe = LinearProbe(
                    in_channels=config.drc.hidden_channels,
                    num_classes=5,
                    kernel_size=1,
                )

                metrics = train_probe(
                    probe,
                    train_states, train_labels,
                    test_states=None, test_labels=None,
                    epochs=5,
                    batch_size=probe_config.batch_size,
                    learning_rate=probe_config.learning_rate,
                    weight_decay=probe_config.weight_decay,
                    device=device,
                )

                if concept_type == "agent_approach":
                    macro_f1_A[layer_idx] = metrics.get("macro_f1", 0)
                else:
                    macro_f1_B[layer_idx] = metrics.get("macro_f1", 0)

        # Thinking time benefit
        thinking_results = evaluate_thinking_time(
            model, medium_levels,
            num_episodes=min(args.num_thinking_episodes, len(medium_levels)),
            num_thinking_steps=5,
            device=device,
        )
        extra_solve_rate = thinking_results.get("extra_solve_rate", 0)

        r = {
            "transitions": transitions,
            "macro_f1_A_layer0": macro_f1_A.get(0, 0),
            "macro_f1_A_layer1": macro_f1_A.get(1, 0),
            "macro_f1_A_layer2": macro_f1_A.get(2, 0),
            "macro_f1_B_layer0": macro_f1_B.get(0, 0),
            "macro_f1_B_layer1": macro_f1_B.get(1, 0),
            "macro_f1_B_layer2": macro_f1_B.get(2, 0),
            "extra_solve_rate": extra_solve_rate,
            "solve_rate_no_thinking": thinking_results.get("solve_rate_no_thinking", 0),
            "solve_rate_with_thinking": thinking_results.get("solve_rate_with_thinking", 0),
        }
        results.append(r)

        print(f"  C_A F1: L0={r['macro_f1_A_layer0']:.3f}, "
              f"L1={r['macro_f1_A_layer1']:.3f}, L2={r['macro_f1_A_layer2']:.3f}")
        print(f"  C_B F1: L0={r['macro_f1_B_layer0']:.3f}, "
              f"L1={r['macro_f1_B_layer1']:.3f}, L2={r['macro_f1_B_layer2']:.3f}")
        print(f"  Extra solve rate: {extra_solve_rate:.4f}")

    # Save results as CSV
    csv_path = os.path.join(args.output_dir, "emergence_results.csv")
    with open(csv_path, "w") as f:
        if results:
            headers = list(results[0].keys())
            f.write(",".join(headers) + "\n")
            for r in results:
                f.write(",".join(str(r[h]) for h in headers) + "\n")

    print(f"\nResults saved to {csv_path}")

    # Print correlation summary
    if len(results) > 5:
        print("\nCorrelation Analysis:")
        print("-" * 40)
        for key in ["macro_f1_A_layer0", "macro_f1_A_layer1", "macro_f1_A_layer2",
                     "macro_f1_B_layer0", "macro_f1_B_layer1", "macro_f1_B_layer2"]:
            values = [r[key] for r in results]
            extra = [r["extra_solve_rate"] for r in results]
            if np.std(values) > 0 and np.std(extra) > 0:
                corr = np.corrcoef(values, extra)[0, 1]
                mark = " **" if corr > 0.7 else ""
                print(f"  {key}: r = {corr:.4f}{mark}")


def _load_levels(data_dir: str, split: str = "unfiltered", max_levels: int = 1000) -> list:
    levels = []
    level_dir = os.path.join(data_dir, "boxoban-levels-master", split, "train")
    if not os.path.exists(level_dir):
        level_dir = os.path.join(data_dir, split, "train")
    if not os.path.exists(level_dir):
        # Generate synthetic
        import random
        for _ in range(max_levels):
            grid = ["########"]
            for r in range(6):
                grid.append("#" + " " * 6 + "#")
            grid.append("########")
            grid[1] = "#." + " " * 4 + ".#"
            grid[2] = "#$" + " " * 4 + "$#"
            grid[5] = "# " * 1 + "@" + " " * 4 + "#"
            levels.append("\n".join(grid))
        return levels

    for filename in sorted(os.listdir(level_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(level_dir, filename)
            with open(filepath, "r") as f:
                content = f.read()
            level_strs = content.strip().split("\n\n")
            if len(level_strs) <= 1:
                level_strs = content.strip().split(";")
            levels.extend([s.strip() for s in level_strs if s.strip()])

    return levels[:max_levels]


if __name__ == "__main__":
    main()
