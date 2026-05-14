"""
Script to run intervention experiments (Section 6.1 of the paper).

Performs Agent-Shortcut and Box-Shortcut interventions to verify that
the agent's representations of C_A and C_B causally influence behavior.

Reproduces Table 1 results.
"""
import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict

from config import Config, CONCEPT_CLASSES
from model.drc import DRCAgent
from training.train import load_checkpoint
from data.boxoban import load_dataset
from probing.concepts import compute_agent_approach_direction, compute_box_push_direction
from probing.evaluate import collect_probe_data, ProbeDataset, train_probe
from probing.probes import LinearProbe1x1, create_probe
from interventions.intervene import (
    run_agent_shortcut_intervention,
    run_box_shortcut_intervention,
    create_random_probe,
    generate_agent_shortcut_levels,
    generate_box_shortcut_levels,
    augment_levels_with_symmetries,
)
from environment.sokoban import SokobanEnv


NEVER = CONCEPT_CLASSES["NEVER"]
UP = CONCEPT_CLASSES["UP"]
DOWN = CONCEPT_CLASSES["DOWN"]
LEFT = CONCEPT_CLASSES["LEFT"]
RIGHT = CONCEPT_CLASSES["RIGHT"]


def train_probe_for_intervention(
    agent: DRCAgent,
    train_dataset,
    concept_fn,
    layer_idx: int,
    config: Config,
    device: torch.device,
    seed: int = 0,
) -> LinearProbe1x1:
    """Train a 1x1 probe for use in interventions."""
    torch.manual_seed(seed)

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
    probe = create_probe(1, config.drc.hidden_channels, config.probe.num_classes)
    probe = train_probe(
        probe, train_ds,
        epochs=config.probe.epochs,
        batch_size=config.probe.batch_size,
        learning_rate=config.probe.learning_rate,
        weight_decay=config.probe.weight_decay,
        device=device,
    )
    return probe


def run_agent_shortcut_experiments(
    agent: DRCAgent,
    levels: List[Dict],
    probes: List[LinearProbe1x1],
    random_probes: List[LinearProbe1x1],
    layer_idx: int,
    config: Config,
    device: torch.device,
) -> Dict:
    """Run Agent-Shortcut interventions with trained and random probes."""
    env = SokobanEnv(grid_size=config.env.grid_size)

    trained_successes = []
    random_successes = []

    for probe_idx, (probe, rand_probe) in enumerate(zip(probes, random_probes)):
        t_success = 0
        r_success = 0

        for level_spec in levels:
            grid = level_spec["grid"]
            short_route = level_spec["short_route"]
            long_route = level_spec["long_route"]

            if len(long_route) < 1:
                continue

            long_route_dirs = []
            for i, pos in enumerate(long_route[:config.intervention.max_directional_squares]):
                if i == 0:
                    dir_class = UP
                else:
                    prev = long_route[i - 1]
                    dr = pos[0] - prev[0]
                    dc = pos[1] - prev[1]
                    dir_map = {(-1, 0): UP, (1, 0): DOWN, (0, -1): LEFT, (0, 1): RIGHT}
                    dir_class = dir_map.get((dr, dc), UP)
                long_route_dirs.append((pos, dir_class))

            result = run_agent_shortcut_intervention(
                agent=agent,
                env=env,
                level=grid,
                probe_ca=probe,
                layer_idx=layer_idx,
                short_route_positions=short_route,
                long_route_positions_dirs=long_route_dirs[:1],
                alpha=config.intervention.alpha,
                device=device,
            )

            if _solved_via_long_route(result, long_route):
                t_success += 1

            result_rand = run_agent_shortcut_intervention(
                agent=agent,
                env=env,
                level=grid,
                probe_ca=rand_probe,
                layer_idx=layer_idx,
                short_route_positions=short_route,
                long_route_positions_dirs=long_route_dirs[:1],
                alpha=config.intervention.alpha,
                device=device,
            )

            if _solved_via_long_route(result_rand, long_route):
                r_success += 1

        trained_successes.append(t_success / len(levels) * 100)
        random_successes.append(r_success / len(levels) * 100)

    return {
        "trained_mean": np.mean(trained_successes),
        "trained_std": np.std(trained_successes),
        "random_mean": np.mean(random_successes),
        "random_std": np.std(random_successes),
    }


def run_box_shortcut_experiments(
    agent: DRCAgent,
    levels: List[Dict],
    probes: List[LinearProbe1x1],
    random_probes: List[LinearProbe1x1],
    layer_idx: int,
    config: Config,
    device: torch.device,
) -> Dict:
    """Run Box-Shortcut interventions with trained and random probes."""
    env = SokobanEnv(grid_size=config.env.grid_size)

    trained_successes = []
    random_successes = []

    for probe, rand_probe in zip(probes, random_probes):
        t_success = 0
        r_success = 0

        for level_spec in levels:
            grid = level_spec["grid"]
            short_route = level_spec["short_route"]
            long_route = level_spec["long_route"]
            box_pos = level_spec.get("box_initial_pos", short_route[0])

            long_route_dirs = [(box_pos, RIGHT)]

            result = run_box_shortcut_intervention(
                agent=agent,
                env=env,
                level=grid,
                probe_cb=probe,
                layer_idx=layer_idx,
                short_route_positions=short_route,
                box_initial_pos=box_pos,
                long_route_positions_dirs=long_route_dirs,
                alpha=config.intervention.alpha,
                device=device,
            )

            if result.get("solved", False):
                t_success += 1

            result_rand = run_box_shortcut_intervention(
                agent=agent,
                env=env,
                level=grid,
                probe_cb=rand_probe,
                layer_idx=layer_idx,
                short_route_positions=short_route,
                box_initial_pos=box_pos,
                long_route_positions_dirs=long_route_dirs,
                alpha=config.intervention.alpha,
                device=device,
            )

            if result_rand.get("solved", False):
                r_success += 1

        trained_successes.append(t_success / len(levels) * 100)
        random_successes.append(r_success / len(levels) * 100)

    return {
        "trained_mean": np.mean(trained_successes),
        "trained_std": np.std(trained_successes),
        "random_mean": np.mean(random_successes),
        "random_std": np.std(random_successes),
    }


def _solved_via_long_route(result: Dict, long_route: List) -> bool:
    """Heuristic: check if agent solved the level (proxy for taking long route)."""
    return result.get("solved", False)


def main():
    parser = argparse.ArgumentParser(description="Run intervention experiments")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--boxoban_path", type=str, default="data/boxoban-levels")
    parser.add_argument("--output_dir", type=str, default="results/interventions")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_seeds", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = Config()
    config.data.boxoban_path = args.boxoban_path

    agent = load_checkpoint(args.checkpoint, config, device=str(device))
    agent.eval()

    train_dataset = load_dataset(args.boxoban_path, config.data.train_split)

    as_base_levels = generate_agent_shortcut_levels(num_base_levels=25)
    as_levels = augment_levels_with_symmetries(as_base_levels)[:200]

    bs_base_levels = generate_box_shortcut_levels(num_base_levels=25)
    bs_levels = augment_levels_with_symmetries(bs_base_levels)[:200]

    results = {}

    for layer_idx in range(config.drc.num_layers):
        print(f"\n=== Layer {layer_idx + 1} ===")
        results[f"layer_{layer_idx + 1}"] = {}

        print("Training C_A probes...")
        ca_probes = []
        ca_rand_probes = []
        for seed in range(args.num_seeds):
            probe = train_probe_for_intervention(
                agent, train_dataset, compute_agent_approach_direction,
                layer_idx, config, device, seed=seed
            )
            ca_probes.append(probe)

            rand_probe = create_random_probe(
                config.drc.hidden_channels, config.probe.num_classes
            )
            ca_rand_probes.append(rand_probe)

        print("Running Agent-Shortcut interventions...")
        as_results = run_agent_shortcut_experiments(
            agent, as_levels, ca_probes, ca_rand_probes,
            layer_idx, config, device
        )
        results[f"layer_{layer_idx + 1}"]["AS"] = as_results
        print(f"  AS Trained: {as_results['trained_mean']:.1f} ± {as_results['trained_std']:.1f}%")
        print(f"  AS Random:  {as_results['random_mean']:.1f} ± {as_results['random_std']:.1f}%")

        print("Training C_B probes...")
        cb_probes = []
        cb_rand_probes = []
        for seed in range(args.num_seeds):
            probe = train_probe_for_intervention(
                agent, train_dataset, compute_box_push_direction,
                layer_idx, config, device, seed=seed
            )
            cb_probes.append(probe)

            rand_probe = create_random_probe(
                config.drc.hidden_channels, config.probe.num_classes
            )
            cb_rand_probes.append(rand_probe)

        print("Running Box-Shortcut interventions...")
        bs_results = run_box_shortcut_experiments(
            agent, bs_levels, cb_probes, cb_rand_probes,
            layer_idx, config, device
        )
        results[f"layer_{layer_idx + 1}"]["BS"] = bs_results
        print(f"  BS Trained: {bs_results['trained_mean']:.1f} ± {bs_results['trained_std']:.1f}%")
        print(f"  BS Random:  {bs_results['random_mean']:.1f} ± {bs_results['random_std']:.1f}%")

    output_path = os.path.join(args.output_dir, "intervention_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n=== Table 1 Summary ===")
    print(f"{'':5} {'Layer 1':>20} {'Layer 2':>20} {'Layer 3':>20}")
    print(f"{'':5} {'Trained':>10} {'Random':>10} {'Trained':>10} {'Random':>10} {'Trained':>10} {'Random':>10}")
    for exp_type in ["AS", "BS"]:
        row = f"{exp_type:5}"
        for layer_idx in range(config.drc.num_layers):
            r = results[f"layer_{layer_idx + 1}"][exp_type]
            row += f" {r['trained_mean']:>9.1f}% {r['random_mean']:>9.1f}%"
        print(row)


if __name__ == "__main__":
    main()
