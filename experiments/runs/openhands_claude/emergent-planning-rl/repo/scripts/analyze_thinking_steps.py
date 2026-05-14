"""
Script to analyze test-time plan refinement (Section 5, Figure 6).

Forces the agent to perform N thinking steps and measures how probe
macro F1 improves across the additional internal ticks.
"""
import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict

from config import Config
from model.drc import DRCAgent
from training.train import load_checkpoint
from data.boxoban import load_dataset
from probing.concepts import compute_agent_approach_direction, compute_box_push_direction
from probing.evaluate import (
    collect_probe_data,
    ProbeDataset,
    train_probe,
    collect_probe_data_all_ticks,
)
from probing.probes import create_probe
from utils.metrics import compute_probe_f1_over_ticks, macro_f1
from environment.sokoban import SokobanEnv


def run_thinking_steps_analysis(
    agent: DRCAgent,
    train_dataset,
    test_dataset,
    concept_fn,
    concept_name: str,
    layer_idx: int,
    config: Config,
    device: torch.device,
    num_episodes: int = 1000,
    thinking_steps: int = 5,
    num_seeds: int = 5,
) -> Dict:
    """
    Measure probe macro F1 at each internal tick during thinking steps.
    
    Reproduces Figure 6 / Figure 22 analysis.
    """
    print(f"Training probe for {concept_name} at layer {layer_idx + 1}...")
    cell_states, labels, _ = collect_probe_data(
        agent=agent,
        dataset=train_dataset,
        num_episodes=config.probe.train_episodes,
        concept_fn=concept_fn,
        layer_idx=layer_idx,
        device=device,
        grid_size=config.env.grid_size,
    )
    train_ds = ProbeDataset(cell_states, labels)

    probes = []
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        probe = create_probe(1, config.drc.hidden_channels, config.probe.num_classes)
        probe = train_probe(
            probe, train_ds,
            epochs=config.probe.epochs,
            batch_size=config.probe.batch_size,
            learning_rate=config.probe.learning_rate,
            weight_decay=config.probe.weight_decay,
            device=device,
        )
        probes.append(probe)

    print(f"Collecting tick-level data ({num_episodes} episodes, {thinking_steps} thinking steps)...")
    tick_data = collect_probe_data_all_ticks(
        agent=agent,
        dataset=test_dataset,
        num_episodes=num_episodes,
        concept_fn=concept_fn,
        layer_idx=layer_idx,
        device=device,
        grid_size=config.env.grid_size,
        thinking_steps=thinking_steps,
    )

    tick_f1s_per_seed = []
    for probe in probes:
        tick_f1s = compute_probe_f1_over_ticks(
            probe=probe,
            tick_data=tick_data,
            device=device,
            num_classes=config.probe.num_classes,
        )
        tick_f1s_per_seed.append(tick_f1s)

    num_ticks = thinking_steps * agent.num_ticks
    mean_f1s = {}
    for tick in range(num_ticks):
        f1_vals = [seed_f1s.get(tick, 0.0) for seed_f1s in tick_f1s_per_seed]
        mean_f1s[tick] = float(np.mean(f1_vals))

    return {
        "concept": concept_name,
        "layer": layer_idx + 1,
        "thinking_steps": thinking_steps,
        "num_ticks_per_step": agent.num_ticks,
        "tick_macro_f1": mean_f1s,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze test-time plan refinement")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--boxoban_path", type=str, default="data/boxoban-levels")
    parser.add_argument("--output_dir", type=str, default="results/thinking_steps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--thinking_steps", type=int, default=5)
    parser.add_argument("--num_seeds", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = Config()
    config.data.boxoban_path = args.boxoban_path

    agent = load_checkpoint(args.checkpoint, config, device=str(device))
    agent.eval()

    train_dataset = load_dataset(args.boxoban_path, config.data.train_split)
    test_dataset = load_dataset(args.boxoban_path, config.data.valid_split)

    all_results = []

    for concept_name, concept_fn in [
        ("CA", compute_agent_approach_direction),
        ("CB", compute_box_push_direction),
    ]:
        for layer_idx in range(config.drc.num_layers):
            print(f"\n=== {concept_name}, Layer {layer_idx + 1} ===")
            result = run_thinking_steps_analysis(
                agent=agent,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                concept_fn=concept_fn,
                concept_name=concept_name,
                layer_idx=layer_idx,
                config=config,
                device=device,
                num_episodes=args.num_episodes,
                thinking_steps=args.thinking_steps,
                num_seeds=args.num_seeds,
            )
            all_results.append(result)

            tick_f1s = result["tick_macro_f1"]
            first_f1 = tick_f1s.get(0, 0.0)
            last_f1 = tick_f1s.get(max(tick_f1s.keys()), 0.0)
            print(f"  First tick F1: {first_f1:.4f}")
            print(f"  Last tick F1:  {last_f1:.4f}")
            print(f"  Improvement:   {last_f1 - first_f1:.4f}")

    output_path = os.path.join(args.output_dir, "thinking_steps_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n=== Figure 6 Summary (Final Layer) ===")
    for concept_name in ["CA", "CB"]:
        for r in all_results:
            if r["concept"] == concept_name and r["layer"] == config.drc.num_layers:
                tick_f1s = r["tick_macro_f1"]
                print(f"\n{concept_name} (Layer {r['layer']}):")
                for tick in sorted(tick_f1s.keys()):
                    step = tick // agent.num_ticks + 1
                    tick_in_step = tick % agent.num_ticks + 1
                    print(f"  Step {step}, Tick {tick_in_step}: F1 = {tick_f1s[tick]:.4f}")


if __name__ == "__main__":
    main()
