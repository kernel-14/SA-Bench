"""
Script to train and evaluate the ResNet agent (Appendix G).
"""
import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict

from config import Config
from model.drc import ResNetAgent
from data.boxoban import load_dataset
from environment.sokoban import SokobanEnv
from probing.concepts import compute_agent_approach_direction, compute_box_push_direction
from probing.probes import create_probe
from probing.evaluate import ProbeDataset, train_probe, evaluate_probe
from utils.metrics import evaluate_agent, macro_f1


def collect_resnet_probe_data(
    agent: ResNetAgent,
    dataset,
    num_episodes: int,
    concept_fn,
    layer_idx: int,
    device: torch.device,
    grid_size: int = 8,
):
    """Collect intermediate activations from ResNet for probing."""
    from probing.concepts import build_trajectory_from_episode

    env = SokobanEnv(grid_size=grid_size)
    agent.eval()

    cell_states_list = []
    labels_list = []

    for _ in range(num_episodes):
        level = dataset.sample()
        obs = env.reset(level)
        episode_obs = [obs.copy()]
        episode_actions = []
        episode_activations = []
        done = False

        while not done:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                out = agent.forward(obs_tensor, return_intermediates=True)

            intermediates = out["intermediates"]
            activation = intermediates[layer_idx].squeeze(0).cpu().numpy()
            episode_activations.append(activation)

            action = out["policy_logits"].argmax(dim=-1).item()
            episode_actions.append(action)
            obs, _, done, _ = env.step(action)
            episode_obs.append(obs.copy())

        if len(episode_actions) < 2:
            continue

        trajectory = build_trajectory_from_episode(episode_obs, episode_actions, grid_size)
        concept_labels = concept_fn(trajectory, grid_size)

        T = min(len(episode_activations), len(concept_labels))
        for t in range(T):
            cell_states_list.append(episode_activations[t])
            labels_list.append(concept_labels[t])

    return cell_states_list, labels_list


def train_resnet(config: Config, checkpoint_dir: str):
    """Train ResNet agent using IMPALA."""
    import torch.optim as optim
    import torch.nn as nn
    from training.impala import impala_loss
    from data.boxoban import create_level_sampler

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    agent = ResNetAgent(
        obs_channels=config.env.obs_channels,
        num_actions=config.env.num_actions,
        num_blocks=24,
        channels=config.drc.hidden_channels,
        grid_size=config.env.grid_size,
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=config.train.learning_rate_start)
    level_sampler = create_level_sampler(
        config.data.boxoban_path, config.data.train_split, seed=config.train.seed
    )

    print("ResNet agent training not fully implemented in this script.")
    print("Use training/train.py with a ResNet-specific config.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--boxoban_path", type=str, default="data/boxoban-levels")
    parser.add_argument("--output_dir", type=str, default="results/resnet")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--num_blocks", type=int, default=24)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = Config()
    config.data.boxoban_path = args.boxoban_path

    agent = ResNetAgent(
        obs_channels=config.env.obs_channels,
        num_actions=config.env.num_actions,
        num_blocks=args.num_blocks,
        channels=config.drc.hidden_channels,
        grid_size=config.env.grid_size,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent = agent.to(device)
    agent.eval()

    train_dataset = load_dataset(args.boxoban_path, config.data.train_split)
    test_dataset = load_dataset(args.boxoban_path, config.data.valid_split)

    results = {"probing": {}}

    for concept_name, concept_fn in [
        ("CA", compute_agent_approach_direction),
        ("CB", compute_box_push_direction),
    ]:
        results["probing"][concept_name] = {}
        for probe_size in [1, 3]:
            results["probing"][concept_name][f"probe_{probe_size}x{probe_size}"] = {}

            for layer_idx in range(args.num_blocks + 1):
                print(f"Probing {concept_name}, {probe_size}x{probe_size}, Layer {layer_idx}")

                train_cs, train_labels = collect_resnet_probe_data(
                    agent, train_dataset, 500, concept_fn, layer_idx, device
                )
                test_cs, test_labels = collect_resnet_probe_data(
                    agent, test_dataset, 200, concept_fn, layer_idx, device
                )

                if not train_cs:
                    continue

                train_ds = ProbeDataset(train_cs, train_labels)
                test_ds = ProbeDataset(test_cs, test_labels)

                f1s = []
                for seed in range(args.num_seeds):
                    torch.manual_seed(seed)
                    probe = create_probe(probe_size, config.drc.hidden_channels, config.probe.num_classes)
                    probe = train_probe(
                        probe, train_ds,
                        epochs=config.probe.epochs,
                        batch_size=config.probe.batch_size,
                        learning_rate=config.probe.learning_rate,
                        weight_decay=config.probe.weight_decay,
                        device=device,
                    )
                    metrics = evaluate_probe(probe, test_ds, device=device)
                    f1s.append(metrics["macro_f1"])

                r = {
                    "mean_macro_f1": float(np.mean(f1s)),
                    "std_macro_f1": float(np.std(f1s)),
                }
                results["probing"][concept_name][f"probe_{probe_size}x{probe_size}"][f"layer_{layer_idx}"] = r
                print(f"  Macro F1: {r['mean_macro_f1']:.4f} ± {r['std_macro_f1']:.4f}")

    output_path = os.path.join(args.output_dir, "resnet_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
