"""
Script to analyze the emergence of planning during training (Section 6.2).

For each training checkpoint:
  1. Measures macro F1 of probes for C_A and C_B
  2. Measures the number of additional levels solved with thinking steps

Reproduces Figure 9 results.
"""
import argparse
import json
import os
import glob
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
)
from probing.probes import create_probe
from utils.metrics import compute_extra_levels_solved, macro_f1
from environment.sokoban import SokobanEnv


def evaluate_checkpoint(
    checkpoint_path: str,
    config: Config,
    train_dataset,
    test_dataset,
    medium_levels: List[np.ndarray],
    device: torch.device,
    num_probe_seeds: int = 1,
    thinking_steps: int = 5,
) -> Dict:
    """
    Evaluate a single checkpoint for:
    1. Probe macro F1 for C_A and C_B at each layer
    2. Extra levels solved with thinking steps
    """
    agent = DRCAgent(
        obs_channels=config.env.obs_channels,
        num_actions=config.env.num_actions,
        num_layers=config.drc.num_layers,
        num_ticks=config.drc.num_ticks,
        hidden_channels=config.drc.hidden_channels,
        encoder_channels=config.drc.encoder_channels,
        kernel_size=config.drc.kernel_size,
        padding=config.drc.padding,
        grid_size=config.env.grid_size,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent = agent.to(device)
    agent.eval()

    total_transitions = ckpt.get("total_transitions", 0)

    probe_results = {}
    for concept_name, concept_fn in [
        ("CA", compute_agent_approach_direction),
        ("CB", compute_box_push_direction),
    ]:
        probe_results[concept_name] = {}
        for layer_idx in range(config.drc.num_layers):
            cell_states, labels, _ = collect_probe_data(
                agent=agent,
                dataset=train_dataset,
                num_episodes=config.probe.checkpoint_train_episodes,
                concept_fn=concept_fn,
                layer_idx=layer_idx,
                device=device,
                grid_size=config.env.grid_size,
            )
            test_cs, test_labels, _ = collect_probe_data(
                agent=agent,
                dataset=test_dataset,
                num_episodes=config.probe.checkpoint_test_episodes,
                concept_fn=concept_fn,
                layer_idx=layer_idx,
                device=device,
                grid_size=config.env.grid_size,
            )

            f1s = []
            for seed in range(num_probe_seeds):
                torch.manual_seed(seed)
                probe = create_probe(1, config.drc.hidden_channels, config.probe.num_classes)
                train_ds = ProbeDataset(cell_states, labels)
                probe = train_probe(
                    probe, train_ds,
                    epochs=config.probe.epochs,
                    batch_size=config.probe.batch_size,
                    learning_rate=config.probe.learning_rate,
                    weight_decay=config.probe.weight_decay,
                    device=device,
                )

                probe.eval()
                test_ds = ProbeDataset(test_cs, test_labels)
                from torch.utils.data import DataLoader
                loader = DataLoader(test_ds, batch_size=64, shuffle=False)
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for cs, lb in loader:
                        cs = cs.to(device)
                        logits = probe(cs)
                        preds = logits.argmax(dim=1).cpu().numpy().reshape(-1)
                        all_preds.append(preds)
                        all_labels.append(lb.numpy().reshape(-1))
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                f1s.append(macro_f1(all_labels, all_preds, config.probe.num_classes))

            probe_results[concept_name][f"layer_{layer_idx + 1}"] = {
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s)),
            }

    extra_solved = compute_extra_levels_solved(
        agent=agent,
        levels=medium_levels,
        env_config=config.env,
        device=device,
        thinking_steps=thinking_steps,
    )
    extra_pct = extra_solved / len(medium_levels) * 100

    return {
        "total_transitions": total_transitions,
        "probe_results": probe_results,
        "extra_levels_solved_pct": extra_pct,
        "extra_levels_solved": extra_solved,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze emergence of planning during training")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--boxoban_path", type=str, default="data/boxoban-levels")
    parser.add_argument("--output_dir", type=str, default="results/emergence")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_medium_levels", type=int, default=1000)
    parser.add_argument("--thinking_steps", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = Config()
    config.data.boxoban_path = args.boxoban_path

    train_dataset = load_dataset(args.boxoban_path, config.data.train_split)
    test_dataset = load_dataset(args.boxoban_path, config.data.valid_split)
    medium_dataset = load_dataset(args.boxoban_path, config.data.medium_split)

    medium_levels = [medium_dataset[i] for i in range(min(args.num_medium_levels, len(medium_dataset)))]
    print(f"Using {len(medium_levels)} medium levels for evaluation")

    checkpoints = sorted(glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_*.pt")))
    print(f"Found {len(checkpoints)} checkpoints")

    all_results = []

    for ckpt_path in checkpoints:
        print(f"\nEvaluating: {ckpt_path}")
        try:
            result = evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                config=config,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                medium_levels=medium_levels,
                device=device,
                thinking_steps=args.thinking_steps,
            )
            result["checkpoint"] = ckpt_path
            all_results.append(result)
            print(f"  Transitions: {result['total_transitions']:,}")
            print(f"  Extra levels solved: {result['extra_levels_solved_pct']:.1f}%")
            for concept in ["CA", "CB"]:
                for layer in range(config.drc.num_layers):
                    f1 = result["probe_results"][concept][f"layer_{layer + 1}"]["mean_f1"]
                    print(f"  {concept} Layer {layer + 1} F1: {f1:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            continue

    output_path = os.path.join(args.output_dir, "emergence_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n=== Correlation Summary (Final Layer) ===")
    ca_f1s = [r["probe_results"]["CA"]["layer_3"]["mean_f1"] for r in all_results]
    cb_f1s = [r["probe_results"]["CB"]["layer_3"]["mean_f1"] for r in all_results]
    extra_pcts = [r["extra_levels_solved_pct"] for r in all_results]

    if len(ca_f1s) > 1:
        ca_corr = np.corrcoef(ca_f1s, extra_pcts)[0, 1]
        cb_corr = np.corrcoef(cb_f1s, extra_pcts)[0, 1]
        print(f"Correlation C_A F1 vs extra levels solved: {ca_corr:.4f}")
        print(f"Correlation C_B F1 vs extra levels solved: {cb_corr:.4f}")


if __name__ == "__main__":
    main()
