"""
Script to probe for alternative concepts (Appendix D.4, D.5).

Tests:
  - Binary simplifications: Agent Approach, Box Push (NEVER vs AGAIN)
  - Reversed asymmetry: Agent Exit Direction, Box Approach Direction
  - Global probes for future actions (Appendix D.5)
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
from probing.concepts import (
    compute_agent_approach_direction,
    compute_box_push_direction,
    compute_agent_approach_binary,
    compute_box_push_binary,
    compute_agent_exit_direction,
    compute_box_approach_direction,
)
from probing.evaluate import (
    run_probing_experiment,
    collect_probe_data,
    ProbeDataset,
    train_probe,
    evaluate_probe,
)
from probing.probes import create_probe, GlobalLinearProbe
from environment.sokoban import SokobanEnv


def probe_future_actions(
    agent,
    train_dataset,
    test_dataset,
    config: Config,
    device: torch.device,
    max_horizon: int = 10,
    num_seeds: int = 5,
) -> dict:
    """
    Train global probes to predict the agent's action N steps in the future.
    Appendix D.5 - falsifies hypothesis that agent plans via explicit action sequences.
    """
    env = SokobanEnv(grid_size=config.env.grid_size)

    def collect_future_action_data(dataset, num_episodes, horizon):
        cell_states_by_layer = {l: [] for l in range(config.drc.num_layers)}
        obs_list = []
        labels = []

        for _ in range(num_episodes):
            level = dataset.sample()
            obs = env.reset(level)
            h, c = agent.init_hidden(1, device)
            episode_obs = [obs.copy()]
            episode_actions = []
            episode_cell_states = {l: [] for l in range(config.drc.num_layers)}
            done = False

            while not done:
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    out = agent.forward(obs_tensor, h, c)
                h = out["hidden_states"]
                c = out["cell_states"]

                for l in range(config.drc.num_layers):
                    episode_cell_states[l].append(c[l].squeeze(0).cpu().numpy())

                action = out["policy_logits"].argmax(dim=-1).item()
                episode_actions.append(action)
                obs, _, done, _ = env.step(action)
                episode_obs.append(obs.copy())

            T = len(episode_actions)
            for t in range(T - horizon):
                future_action = episode_actions[t + horizon]
                for l in range(config.drc.num_layers):
                    cell_states_by_layer[l].append(episode_cell_states[l][t])
                obs_list.append(episode_obs[t])
                labels.append(future_action)

        return cell_states_by_layer, obs_list, labels

    results = {}
    for horizon in range(1, max_horizon + 1):
        results[horizon] = {}
        train_cs, train_obs, train_labels = collect_future_action_data(
            train_dataset, 500, horizon
        )
        test_cs, test_obs, test_labels = collect_future_action_data(
            test_dataset, 200, horizon
        )

        for layer_idx in range(config.drc.num_layers):
            from probing.evaluate import ProbeDataset
            train_ds = ProbeDataset(
                [cs.reshape(-1) for cs in train_cs[layer_idx]],
                [np.array(l) for l in train_labels],
            )

            class FlatProbeDataset(torch.utils.data.Dataset):
                def __init__(self, cell_states, labels):
                    self.cs = cell_states
                    self.lb = labels
                def __len__(self): return len(self.cs)
                def __getitem__(self, i):
                    return torch.from_numpy(self.cs[i]).float(), torch.tensor(self.lb[i]).long()

            flat_train = FlatProbeDataset(
                [cs.flatten() for cs in train_cs[layer_idx]], train_labels
            )
            flat_test = FlatProbeDataset(
                [cs.flatten() for cs in test_cs[layer_idx]], test_labels
            )

            accs = []
            for seed in range(num_seeds):
                torch.manual_seed(seed)
                probe = GlobalLinearProbe(
                    config.drc.hidden_channels, config.env.grid_size, config.env.num_actions
                )
                probe = train_probe(
                    probe, flat_train,
                    epochs=config.probe.epochs,
                    batch_size=config.probe.batch_size,
                    learning_rate=config.probe.learning_rate,
                    weight_decay=config.probe.weight_decay,
                    device=device,
                )
                metrics = evaluate_probe(
                    probe, flat_test, device=device, num_classes=config.env.num_actions
                )
                accs.append(metrics["accuracy"])

            results[horizon][f"layer_{layer_idx + 1}"] = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Probe for alternative concepts")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--boxoban_path", type=str, default="data/boxoban-levels")
    parser.add_argument("--output_dir", type=str, default="results/alt_probing")
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

    all_results = {}

    concept_experiments = [
        ("CA_main", compute_agent_approach_direction, 5),
        ("CB_main", compute_box_push_direction, 5),
        ("CA_binary", compute_agent_approach_binary, 2),
        ("CB_binary", compute_box_push_binary, 2),
        ("CA_exit", compute_agent_exit_direction, 5),
        ("CB_approach", compute_box_approach_direction, 5),
    ]

    for concept_name, concept_fn, num_classes in concept_experiments:
        print(f"\n=== Probing for {concept_name} ===")
        all_results[concept_name] = {}

        for probe_size in [1, 3]:
            all_results[concept_name][f"probe_{probe_size}x{probe_size}"] = {}

            for layer_idx in range(config.drc.num_layers):
                config_copy = Config()
                config_copy.probe.num_classes = num_classes

                result = run_probing_experiment(
                    agent=agent,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    concept_fn=concept_fn,
                    layer_idx=layer_idx,
                    probe_size=probe_size,
                    config=config_copy,
                    device=device,
                    num_seeds=args.num_seeds,
                )
                all_results[concept_name][f"probe_{probe_size}x{probe_size}"][f"layer_{layer_idx + 1}"] = result
                print(f"  {probe_size}x{probe_size} Layer {layer_idx + 1}: {result['mean_macro_f1']:.4f} ± {result['std_macro_f1']:.4f}")

    print("\n=== Future Action Probing (Appendix D.5) ===")
    future_action_results = probe_future_actions(
        agent, train_dataset, test_dataset, config, device,
        max_horizon=10, num_seeds=args.num_seeds
    )
    all_results["future_actions"] = future_action_results

    for horizon in range(1, 11):
        print(f"\nHorizon {horizon}:")
        for layer_idx in range(config.drc.num_layers):
            r = future_action_results[horizon][f"layer_{layer_idx + 1}"]
            print(f"  Layer {layer_idx + 1}: accuracy = {r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}")

    output_path = os.path.join(args.output_dir, "alt_probing_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
