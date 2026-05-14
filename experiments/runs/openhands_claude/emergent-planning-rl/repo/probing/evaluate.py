import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Callable
from sklearn.metrics import f1_score, precision_score, recall_score

from config import Config, CONCEPT_CLASSES
from model.drc import DRCAgent
from probing.probes import LinearProbe1x1, LinearProbe3x3, create_probe
from probing.concepts import (
    compute_agent_approach_direction,
    compute_box_push_direction,
    build_trajectory_from_episode,
    Transition,
)
from environment.sokoban import SokobanEnv
from data.boxoban import BoxobanDataset


class ProbeDataset(Dataset):
    """
    Dataset of (cell_state, label) pairs for probe training.
    
    Each sample corresponds to a single (transition, square) pair.
    cell_state: (C, H, W) tensor
    label: (H, W) integer tensor of concept class labels
    """

    def __init__(
        self,
        cell_states: List[np.ndarray],
        labels: List[np.ndarray],
    ):
        assert len(cell_states) == len(labels)
        self.cell_states = cell_states
        self.labels = labels

    def __len__(self) -> int:
        return len(self.cell_states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cs = torch.from_numpy(self.cell_states[idx]).float()
        lb = torch.from_numpy(self.labels[idx]).long()
        return cs, lb


class ObsProbeDataset(Dataset):
    """Dataset using raw observations as input (baseline probes)."""

    def __init__(
        self,
        observations: List[np.ndarray],
        labels: List[np.ndarray],
    ):
        self.observations = observations
        self.labels = labels

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = torch.from_numpy(self.observations[idx]).float().permute(2, 0, 1)
        lb = torch.from_numpy(self.labels[idx]).long()
        return obs, lb


def collect_probe_data(
    agent: DRCAgent,
    dataset: BoxobanDataset,
    num_episodes: int,
    concept_fn: Callable,
    layer_idx: int,
    device: torch.device,
    grid_size: int = 8,
    thinking_steps: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Collect cell states and concept labels by running the agent on episodes.
    
    Returns:
        cell_states: list of (C, H, W) arrays, one per transition
        labels: list of (H, W) integer arrays
        observations: list of (H, W, 7) arrays (for baseline)
    """
    env = SokobanEnv(grid_size=grid_size)
    agent.eval()

    cell_states_list = []
    labels_list = []
    obs_list = []

    for ep_idx in range(num_episodes):
        level = dataset.sample()
        obs = env.reset(level)

        h, c = agent.init_hidden(1, device)
        episode_obs = []
        episode_actions = []
        episode_cell_states = []
        done = False

        for _ in range(thinking_steps):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                out = agent.forward(obs_tensor, h, c)
            h = out["hidden_states"]
            c = out["cell_states"]

        while not done:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                out = agent.forward(obs_tensor, h, c)

            h = out["hidden_states"]
            c = out["cell_states"]

            cell_state = c[layer_idx].squeeze(0).cpu().numpy()
            episode_cell_states.append(cell_state)
            episode_obs.append(obs.copy())

            logits = out["policy_logits"]
            action = logits.argmax(dim=-1).item()
            episode_actions.append(action)

            obs, _, done, _ = env.step(action)

        if len(episode_obs) < 2:
            continue

        episode_obs.append(obs.copy())

        trajectory = build_trajectory_from_episode(episode_obs, episode_actions, grid_size)
        concept_labels = concept_fn(trajectory, grid_size)

        T = min(len(episode_cell_states), len(concept_labels))
        for t in range(T):
            cell_states_list.append(episode_cell_states[t])
            labels_list.append(concept_labels[t])
            obs_list.append(episode_obs[t])

    return cell_states_list, labels_list, obs_list


def collect_probe_data_all_ticks(
    agent: DRCAgent,
    dataset: BoxobanDataset,
    num_episodes: int,
    concept_fn: Callable,
    layer_idx: int,
    device: torch.device,
    grid_size: int = 8,
    thinking_steps: int = 5,
) -> Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Collect cell states at each internal tick during thinking steps.
    Used for Figure 6 / test-time plan refinement analysis.
    
    The agent is forced to remain stationary for `thinking_steps` steps.
    At each of the N*thinking_steps internal ticks, we record the cell state
    and the concept labels (computed from the subsequent episode behavior).
    
    Returns dict mapping tick_idx -> (cell_states, labels)
    """
    env = SokobanEnv(grid_size=grid_size)
    agent.eval()

    num_ticks = agent.num_ticks
    total_ticks = thinking_steps * num_ticks

    tick_data: Dict[int, Tuple[List, List]] = {
        tick: ([], []) for tick in range(total_ticks)
    }

    for ep_idx in range(num_episodes):
        level = dataset.sample()

        # First, run the episode normally to get concept labels
        obs = env.reset(level)
        h, c = agent.init_hidden(1, device)
        episode_obs = [obs.copy()]
        episode_actions = []
        done = False

        while not done:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                out = agent.forward(obs_tensor, h, c)
            h = out["hidden_states"]
            c = out["cell_states"]
            action = out["policy_logits"].argmax(dim=-1).item()
            episode_actions.append(action)
            obs, _, done, _ = env.step(action)
            episode_obs.append(obs.copy())

        if len(episode_actions) < 2:
            continue

        trajectory = build_trajectory_from_episode(episode_obs, episode_actions, grid_size)
        concept_labels = concept_fn(trajectory, grid_size)

        if len(concept_labels) == 0:
            continue

        # Use the label at t=0 (start of episode) as the target for thinking steps
        first_label = concept_labels[0]

        # Now run thinking steps from the start of the episode
        obs_start = env.reset(level)
        h_think, c_think = agent.init_hidden(1, device)
        obs_tensor = torch.from_numpy(obs_start).float().unsqueeze(0).to(device)

        global_tick = 0
        for step in range(thinking_steps):
            for tick in range(num_ticks):
                with torch.no_grad():
                    out = agent.forward(obs_tensor, h_think, c_think)
                h_think = out["hidden_states"]
                c_think = out["cell_states"]

                cell_state = c_think[layer_idx].squeeze(0).cpu().numpy()
                tick_data[global_tick][0].append(cell_state)
                tick_data[global_tick][1].append(first_label)
                global_tick += 1

    return tick_data


def train_probe(
    probe: nn.Module,
    train_dataset: Dataset,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Train a linear probe using AdamW optimizer.
    
    From Appendix D.1:
      - AdamW optimizer
      - 10 epochs
      - Batch size 16
      - Learning rate 0.001
      - Weight decay 0.001
    """
    probe = probe.to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    probe.train()
    for epoch in range(epochs):
        for cell_states, labels in loader:
            cell_states = cell_states.to(device)
            labels = labels.to(device)

            logits = probe(cell_states)

            if logits.dim() == 4:
                B, C, H, W = logits.shape
                logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
                labels_flat = labels.reshape(-1)
            else:
                logits_flat = logits
                labels_flat = labels

            loss = nn.CrossEntropyLoss()(logits_flat, labels_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    return probe


def evaluate_probe(
    probe: nn.Module,
    test_dataset: Dataset,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
    num_classes: int = 5,
) -> Dict[str, float]:
    """
    Evaluate probe performance using macro F1 score.
    Also computes per-class precision, recall, F1.
    """
    probe.eval()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for cell_states, labels in loader:
            cell_states = cell_states.to(device)
            logits = probe(cell_states)

            if logits.dim() == 4:
                B, C, H, W = logits.shape
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy().reshape(-1))
                all_labels.append(labels.numpy().reshape(-1))
            else:
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = (all_preds == all_labels).mean()

    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0
    )
    per_class_precision = precision_score(
        all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0
    )
    per_class_recall = recall_score(
        all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0
    )

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "per_class_f1": per_class_f1.tolist(),
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
    }


def run_probing_experiment(
    agent: DRCAgent,
    train_dataset: BoxobanDataset,
    test_dataset: BoxobanDataset,
    concept_fn: Callable,
    layer_idx: int,
    probe_size: int,
    config: Config,
    device: torch.device,
    num_seeds: int = 5,
) -> Dict[str, float]:
    """
    Full probing experiment: collect data, train probes with multiple seeds,
    evaluate and return mean/std of macro F1.
    """
    print(f"Collecting training data (layer {layer_idx + 1}, probe {probe_size}x{probe_size})...")
    train_cs, train_labels, train_obs = collect_probe_data(
        agent=agent,
        dataset=train_dataset,
        num_episodes=config.probe.train_episodes,
        concept_fn=concept_fn,
        layer_idx=layer_idx,
        device=device,
        grid_size=config.env.grid_size,
    )

    print(f"Collecting test data...")
    test_cs, test_labels, test_obs = collect_probe_data(
        agent=agent,
        dataset=test_dataset,
        num_episodes=config.probe.test_episodes,
        concept_fn=concept_fn,
        layer_idx=layer_idx,
        device=device,
        grid_size=config.env.grid_size,
    )

    train_ds = ProbeDataset(train_cs, train_labels)
    test_ds = ProbeDataset(test_cs, test_labels)

    macro_f1s = []
    for seed in range(num_seeds):
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
        metrics = evaluate_probe(probe, test_ds, device=device, num_classes=config.probe.num_classes)
        macro_f1s.append(metrics["macro_f1"])

    return {
        "mean_macro_f1": np.mean(macro_f1s),
        "std_macro_f1": np.std(macro_f1s),
        "all_macro_f1s": macro_f1s,
    }


def run_baseline_probing_experiment(
    train_dataset: BoxobanDataset,
    test_dataset: BoxobanDataset,
    concept_fn: Callable,
    probe_size: int,
    config: Config,
    device: torch.device,
    num_seeds: int = 5,
    num_train_episodes: int = 3000,
    num_test_episodes: int = 1000,
) -> Dict[str, float]:
    """
    Baseline probing experiment using raw observations as input.
    """
    env = SokobanEnv(grid_size=config.env.grid_size)

    def collect_obs_data(dataset, num_episodes):
        obs_list, labels_list = [], []
        for _ in range(num_episodes):
            level = dataset.sample()
            obs = env.reset(level)
            episode_obs = [obs.copy()]
            episode_actions = []
            done = False

            while not done:
                action = np.random.randint(0, 5)
                obs, _, done, _ = env.step(action)
                episode_obs.append(obs.copy())
                episode_actions.append(action)

            if len(episode_actions) < 2:
                continue

            trajectory = build_trajectory_from_episode(
                episode_obs, episode_actions, config.env.grid_size
            )
            concept_labels = concept_fn(trajectory, config.env.grid_size)
            T = min(len(episode_obs) - 1, len(concept_labels))
            for t in range(T):
                obs_list.append(episode_obs[t])
                labels_list.append(concept_labels[t])

        return obs_list, labels_list

    train_obs, train_labels = collect_obs_data(train_dataset, num_train_episodes)
    test_obs, test_labels = collect_obs_data(test_dataset, num_test_episodes)

    train_ds = ObsProbeDataset(train_obs, train_labels)
    test_ds = ObsProbeDataset(test_obs, test_labels)

    obs_channels = config.env.obs_channels
    macro_f1s = []
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        probe = create_probe(probe_size, obs_channels, config.probe.num_classes)
        probe = train_probe(
            probe, train_ds,
            epochs=config.probe.epochs,
            batch_size=config.probe.batch_size,
            learning_rate=config.probe.learning_rate,
            weight_decay=config.probe.weight_decay,
            device=device,
        )
        metrics = evaluate_probe(probe, test_ds, device=device, num_classes=config.probe.num_classes)
        macro_f1s.append(metrics["macro_f1"])

    return {
        "mean_macro_f1": np.mean(macro_f1s),
        "std_macro_f1": np.std(macro_f1s),
        "all_macro_f1s": macro_f1s,
    }
