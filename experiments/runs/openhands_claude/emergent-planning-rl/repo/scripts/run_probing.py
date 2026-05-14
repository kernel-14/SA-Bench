"""
Script to run the full probing experiment (Section 4 of the paper).

Trains 1x1 and 3x3 probes to predict C_A and C_B from the agent's cell states
at each layer, and compares to baseline probes using raw observations.

Reproduces Figure 4 results.
"""
import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path

from config import Config
from model.drc import DRCAgent
from training.train import load_checkpoint
from data.boxoban import load_dataset
from probing.concepts import compute_agent_approach_direction, compute_box_push_direction
from probing.evaluate import (
    run_probing_experiment,
    run_baseline_probing_experiment,
)


def main():
    parser = argparse.ArgumentParser(description="Run probing experiments")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--boxoban_path", type=str, default="data/boxoban-levels")
    parser.add_argument("--output_dir", type=str, default="results/probing")
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
    test_dataset = load_dataset(args.boxoban_path, config.data.valid_split)

    print(f"Train levels: {len(train_dataset)}, Test levels: {len(test_dataset)}")

    results = {}

    for concept_name, concept_fn in [
        ("CA", compute_agent_approach_direction),
        ("CB", compute_box_push_direction),
    ]:
        results[concept_name] = {}

        for probe_size in [1, 3, 5, 7]:
            results[concept_name][f"probe_{probe_size}x{probe_size}"] = {}

            for layer_idx in range(config.drc.num_layers):
                print(f"\nConcept: {concept_name}, Probe: {probe_size}x{probe_size}, Layer: {layer_idx + 1}")

                layer_results = run_probing_experiment(
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

                results[concept_name][f"probe_{probe_size}x{probe_size}"][f"layer_{layer_idx + 1}"] = layer_results
                print(f"  Macro F1: {layer_results['mean_macro_f1']:.4f} ± {layer_results['std_macro_f1']:.4f}")

            print(f"\nBaseline probe {probe_size}x{probe_size} for {concept_name}:")
            baseline_results = run_baseline_probing_experiment(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                concept_fn=concept_fn,
                probe_size=probe_size,
                config=config,
                device=device,
                num_seeds=args.num_seeds,
            )
            results[concept_name][f"probe_{probe_size}x{probe_size}"]["baseline"] = baseline_results
            print(f"  Baseline Macro F1: {baseline_results['mean_macro_f1']:.4f} ± {baseline_results['std_macro_f1']:.4f}")

    output_path = os.path.join(args.output_dir, "probing_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n=== Summary (1x1 probes) ===")
    for concept_name in ["CA", "CB"]:
        print(f"\nConcept {concept_name}:")
        for layer_idx in range(config.drc.num_layers):
            r = results[concept_name]["probe_1x1"][f"layer_{layer_idx + 1}"]
            print(f"  Layer {layer_idx + 1}: {r['mean_macro_f1']:.4f} ± {r['std_macro_f1']:.4f}")
        baseline = results[concept_name]["probe_1x1"]["baseline"]
        print(f"  Baseline: {baseline['mean_macro_f1']:.4f} ± {baseline['std_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
