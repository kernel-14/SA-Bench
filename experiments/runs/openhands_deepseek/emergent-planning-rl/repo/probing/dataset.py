"""
Dataset creation for linear probe training.

Collects cell state activations from the DRC agent and corresponding
concept labels across many episodes.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from ..environment.sokoban import (
    SokobanEnv, parse_boxoban_level, grid_to_symbolic,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_NOOP,
)
from ..models.drc import DRCNet
from .concepts import ConceptLabeler, CLASS_NEVER


class ProbeDataset:
    """
    Dataset for training linear probes.

    Stores (cell_state, concept_label) pairs collected across episodes.
    Cell states come from the DRC agent's ConvLSTM layers.
    """

    def __init__(self):
        self.states = defaultdict(list)  # layer_idx -> list of (C, H, W) arrays
        self.labels = defaultdict(list)  # layer_idx -> list of (H, W) arrays

    def add_sample(
        self,
        layer_idx: int,
        cell_state: np.ndarray,
        label: np.ndarray,
    ) -> None:
        """Add a single sample (one square's cell state and label)."""
        self.states[layer_idx].append(cell_state)
        self.labels[layer_idx].append(label)

    def add_batch(
        self,
        layer_idx: int,
        cell_states: np.ndarray,  # (B, C, H, W)
        labels: np.ndarray,       # (B, H, W)
    ) -> None:
        """Add a batch of samples."""
        for i in range(cell_states.shape[0]):
            self.states[layer_idx].append(cell_states[i])
            self.labels[layer_idx].append(labels[i])

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all data for the first layer as tensors."""
        if not self.states:
            return torch.zeros(0), torch.zeros(0, dtype=torch.long)
        # Use first available layer
        layer_idx = list(self.states.keys())[0]
        states = np.stack(self.states[layer_idx], axis=0)
        labels = np.stack(self.labels[layer_idx], axis=0)
        return (
            torch.from_numpy(states).float(),
            torch.from_numpy(labels).long(),
        )

    def get_class_counts(self) -> Dict[int, int]:
        """Count occurrences of each class across all labels."""
        all_labels = []
        for lbl_list in self.labels.values():
            for lbl in lbl_list:
                all_labels.append(lbl.flatten())
        if not all_labels:
            return {}
        all_labels = np.concatenate(all_labels)
        counts = {}
        for cls_idx in range(5):
            counts[cls_idx] = int((all_labels == cls_idx).sum())
        return counts


def collect_probe_data(
    model: DRCNet,
    levels: List[str],
    num_episodes: int,
    device: torch.device,
    concept_type: str = "agent_approach",
    max_steps: int = 120,
    collect_layer: Optional[int] = None,
) -> Dict[int, ProbeDataset]:
    """
    Collect probe training data by running the agent on many episodes.

    Args:
        model: trained DRC agent
        levels: list of level strings
        num_episodes: number of episodes to collect
        device: torch device
        concept_type: "agent_approach" or "box_push"
        max_steps: maximum episode length
        collect_layer: specific layer to collect from (None = all layers)

    Returns:
        datasets: dict mapping layer_idx to ProbeDataset
    """
    model.eval()
    env = SokobanEnv(max_steps=max_steps)
    labeler = ConceptLabeler(env, concept_type=concept_type)

    datasets = defaultdict(ProbeDataset)

    for ep in range(num_episodes):
        level_idx = ep % len(levels)
        grid = parse_boxoban_level(levels[level_idx])
        env.load_level(grid)
        obs = env.reset()
        done = False

        # Episode storage
        episode_states = defaultdict(list)  # layer -> list of cell_states per step
        episode_actions = []
        episode_grids = [env.get_grid()]

        model_states = None
        step = 0

        while not done and step < max_steps:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, value, model_states = model(obs_tensor, model_states)

                # Extract cell states from all layers
                for d in range(model.num_layers):
                    if collect_layer is not None and d != collect_layer:
                        continue
                    if model_states[d] is not None:
                        cell_state = model_states[d][1]  # (1, C, H, W)
                        episode_states[d].append(cell_state.cpu().numpy()[0])

            # Greedy action selection
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()

            episode_actions.append(action)
            obs, reward, done, info = env.step(action)
            episode_grids.append(env.get_grid())
            step += 1

        # Compute concept labels for the episode
        episode_labels = labeler.compute_episode_labels(
            episode_actions[:step], grid
        )

        # Match labels with collected states
        for d, states_list in episode_states.items():
            for t in range(min(len(states_list), len(episode_labels))):
                if t in episode_labels:
                    cell_state = states_list[t]
                    label = episode_labels[t]
                    datasets[d].add_sample(d, cell_state, label)

    return dict(datasets)
