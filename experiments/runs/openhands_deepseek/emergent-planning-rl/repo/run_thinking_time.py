"""
Run thinking time analysis experiments.

Reproduces Sections 5 and 6.2:
- Evaluates how extra test-time compute (thinking steps) improves performance
- Analyzes plan refinement across internal ticks
- Correlates planning-relevant concept emergence with planning-like behavior
"""

import os
import sys
import argparse
import numpy as np
import torch

from configs.config import Config
from models.drc import DRCNet
from probing.linear_probe import LinearProbe
from analysis.thinking_time import (
    evaluate_thinking_time,
    analyze_plan_refinement,
    analyze_plan_refinement_across_ticks,
)


def load_levels(data_dir: str, split: str = "unfiltered", max_levels: int = 1000) -> list:
    """Load Boxoban levels."""
    levels = []
    level_dir = os.path.join(data_dir, "boxoban-levels-master", split, "train")
    if not os.path.exists(level_dir):
        level_dir = os.path.join(data_dir, split, "train")
    if not os.path.exists(level_dir):
        return _generate_synthetic(max_levels)

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


def _generate_synthetic(n: int) -> list:
    import random
    levels = []
    for _ in range(n):
        grid = ["########"]
        for r in range(6):
            grid.append("#" + " " * 6 + "#")
        grid.append("########")
        grid[1] = "#." + " " * 4 + ".#"
        grid[2] = "#$" + " " * 4 + "$#"
        grid[5] = "# " * 1 + "@" + " " * 4 + "#"
        levels.append("\n".join(grid))
    return levels


def main():
    parser = argparse.ArgumentParser(description="Run thinking time analysis")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--probe_A", type=str, default=None,
                        help="Path to C_A probe checkpoint")
    parser.add_argument("--probe_B", type=str, default=None,
                        help="Path to C_B probe checkpoint")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/thinking_time")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--num_thinking_steps", type=int, default=5)
    parser.add_argument("--analysis_type", type=str, default="all",
                        choices=["behavioral", "plan_refinement", "all"],
                        help="Type of analysis to run")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    config = Config()
    model = DRCNet(
        input_channels=config.sokoban.num_channels,
        hidden_channels=config.drc.hidden_channels,
        num_layers=config.drc.D,
        num_ticks=config.drc.N,
        num_actions=5,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded from {args.checkpoint}")

    # Load levels
    levels = load_levels(args.data_dir, "unfiltered", max_levels=args.num_episodes * 2)
    print(f"Loaded {len(levels)} levels")

    # Behavioral analysis: how many extra levels solved with thinking time?
    if args.analysis_type in ("behavioral", "all"):
        print("\n" + "=" * 60)
        print("Behavioral Analysis: Extra Test-Time Compute")
        print("=" * 60)

        for think_steps in [0, 1, 3, 5]:
            results = evaluate_thinking_time(
                model, levels,
                num_episodes=min(args.num_episodes, len(levels)),
                num_thinking_steps=think_steps,
                device=device,
            )
            print(f"\n  Thinking steps: {think_steps}")
            print(f"    Solve rate: {results['solve_rate_with_thinking']:.4f}")
            print(f"    Extra solved: {results['extra_solved']}")

            if think_steps == 5:
                # Save behavioral results
                behav_path = os.path.join(args.output_dir, "thinking_time_behavioral.txt")
                with open(behav_path, "w") as f:
                    f.write("Thinking Time Behavioral Analysis\n")
                    f.write("=" * 60 + "\n")
                    for k, v in results.items():
                        f.write(f"{k}: {v}\n")

    # Plan refinement analysis: how does plan quality improve across ticks?
    if args.analysis_type in ("plan_refinement", "all") and args.probe_A is not None:
        print("\n" + "=" * 60)
        print("Plan Refinement Analysis: Concept C_A")
        print("=" * 60)

        probe = LinearProbe(
            in_channels=config.drc.hidden_channels,
            num_classes=5,
            kernel_size=1,
        )
        probe.load_state_dict(
            torch.load(args.probe_A, map_location=device)["model_state_dict"]
        )
        probe.to(device)
        probe.eval()

        for layer_idx in range(config.drc.D):
            print(f"\n  Layer {layer_idx}:")
            tick_metrics = analyze_plan_refinement(
                model, probe, levels,
                num_episodes=min(args.num_episodes // 5, len(levels)),
                num_thinking_steps=args.num_thinking_steps,
                device=device,
                concept_type="agent_approach",
                layer_idx=layer_idx,
            )
            for tick, metrics in sorted(tick_metrics.items()):
                print(f"    Tick {tick}: macro_f1 = {metrics['macro_f1']:.4f}, "
                      f"accuracy = {metrics['accuracy']:.4f}")

        # Save plan refinement results
        refine_path = os.path.join(args.output_dir, "plan_refinement_C_A.txt")
        with open(refine_path, "w") as f:
            f.write("Plan Refinement Analysis (C_A)\n")
            f.write("=" * 60 + "\n")
            for layer_idx in range(config.drc.D):
                f.write(f"\nLayer {layer_idx}:\n")
                tick_metrics = analyze_plan_refinement(
                    model, probe, levels,
                    num_episodes=min(args.num_episodes // 5, len(levels)),
                    num_thinking_steps=args.num_thinking_steps,
                    device=device,
                    concept_type="agent_approach",
                    layer_idx=layer_idx,
                )
                for tick, metrics in sorted(tick_metrics.items()):
                    f.write(f"  Tick {tick}: macro_f1 = {metrics['macro_f1']:.4f}, "
                            f"accuracy = {metrics['accuracy']:.4f}\n")

    if args.analysis_type in ("plan_refinement", "all") and args.probe_B is not None:
        print("\n" + "=" * 60)
        print("Plan Refinement Analysis: Concept C_B")
        print("=" * 60)

        probe_B = LinearProbe(
            in_channels=config.drc.hidden_channels,
            num_classes=5,
            kernel_size=1,
        )
        probe_B.load_state_dict(
            torch.load(args.probe_B, map_location=device)["model_state_dict"]
        )
        probe_B.to(device)
        probe_B.eval()

        for layer_idx in range(config.drc.D):
            print(f"\n  Layer {layer_idx}:")
            tick_metrics = analyze_plan_refinement(
                model, probe_B, levels,
                num_episodes=min(args.num_episodes // 5, len(levels)),
                num_thinking_steps=args.num_thinking_steps,
                device=device,
                concept_type="box_push",
                layer_idx=layer_idx,
            )
            for tick, metrics in sorted(tick_metrics.items()):
                print(f"    Tick {tick}: macro_f1 = {metrics['macro_f1']:.4f}, "
                      f"accuracy = {metrics['accuracy']:.4f}")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
