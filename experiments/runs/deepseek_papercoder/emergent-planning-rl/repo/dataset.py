## dataset.py
"""
Concept labeling and probe dataset classes for the interpretability pipeline.

Provides:
- ConceptLabeler : computes ground‑truth labels (`C_A` and `C_B`) for an entire
  recorded episode.
- ProbeDataset   : runs a trained DRC agent on a set of Sokoban levels, collects
  internal cell states and concept labels, and provides a PyTorch DataLoader for
  training/evaluating linear probes.
"""

from typing import List, Dict, Tuple, Optional, Any
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Project‑internal imports (avoid circular dependencies)
# ---------------------------------------------------------------------------
from environment import SokobanEnv
from model import DRCNetwork
from utils import Config, set_seed


# ---------------------------------------------------------------------------
# Helper: map environment action index -> direction index for labels
# ---------------------------------------------------------------------------
ACTION2LABEL = {
    0: 4,  # NOOP         -> NEVER (not used for approach directions, but safe)
    1: 0,  # UP           -> UP   (index 0)
    2: 1,  # DOWN         -> DOWN (index 1)
    3: 2,  # LEFT         -> LEFT (index 2)
    4: 3,  # RIGHT        -> RIGHT (index 3)
}

LABEL2STRING = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "NEVER",
}


# ===========================================================================
# 1.  ConceptLabeler
# ===========================================================================
class ConceptLabeler:
    """
    Computes per‑square, per‑timestep labels for Agent Approach Direction (`C_A`)
    and Box Push Direction (`C_B`).

    Usage:
        labeler = ConceptLabeler()
        labels_A, labels_B = labeler.label_episode(trajectory)

    where `trajectory` is a list of dicts, each having keys:
        - 'action'      : int (0‑4, see ACTION2LABEL)
        - 'agent_pos'   : (r, c) coordinates of the agent *after* the step
        - 'push_event'  : (from_r, from_c, to_r, to_c) or None if no push occurred
    """

    NEVER = 4
    BOARD_SIZE = 8

    @staticmethod
    def _build_agent_events(
        traj: List[Dict[str, Any]]
    ) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        For each square, record the (step, direction_index) of every time the
        agent *steps onto* that square (i.e., the square becomes its new position).
        """
        events: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for t, step in enumerate(traj):
            direction = ACTION2LABEL.get(step['action'], 4)
            # only record if the action is a valid movement and the agent moved
            if direction == 4:
                continue
            pos = tuple(step['agent_pos'])
            events.setdefault(pos, []).append((t, direction))
        return events

    @staticmethod
    def _build_box_events(
        traj: List[Dict[str, Any]]
    ) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        For each square, record the (step, direction_index) of every time a box
        is pushed *off* that square.  The push_event tuple must be provided.
        """
        events: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for t, step in enumerate(traj):
            push = step.get('push_event')
            if push is not None:
                from_r, from_c, _, _ = push
                direction = ACTION2LABEL.get(step['action'], 4)
                if direction == 4:
                    continue  # should not happen, but safe
                pos = (from_r, from_c)
                events.setdefault(pos, []).append((t, direction))
        return events

    @staticmethod
    def _label_for_square(
        t: int,
        event_list: List[Tuple[int, int]]
    ) -> int:
        """
        Given a sorted (by step) list of future events for a square, return the
        direction of the first event with step > t, or NEVER if none.
        """
        # events are appended in chronological order during building,
        # so the list is already sorted.
        for step, direction in event_list:
            if step > t:
                return direction
        return ConceptLabeler.NEVER

    def label_episode(
        self,
        trajectory: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute C_A and C_B labels for each step of the episode.

        Args:
            trajectory: list of per‑step dicts (see class docstring).

        Returns:
            labels_A : (T, 8, 8) int array  –  class indices for C_A
            labels_B : (T, 8, 8) int array  –  class indices for C_B
        """
        T = len(trajectory)
        if T == 0:
            raise ValueError("Empty trajectory")

        # Build event lists (sorted by step)
        agent_events = self._build_agent_events(trajectory)
        box_events   = self._build_box_events(trajectory)

        labels_A = np.full((T, self.BOARD_SIZE, self.BOARD_SIZE),
                           self.NEVER, dtype=np.int8)
        labels_B = np.full_like(labels_A, self.NEVER)

        for t in range(T):
            for r in range(self.BOARD_SIZE):
                for c in range(self.BOARD_SIZE):
                    pos = (r, c)
                    # Agent approach
                    if pos in agent_events:
                        labels_A[t, r, c] = self._label_for_square(t, agent_events[pos])
                    # Box push
                    if pos in box_events:
                        labels_B[t, r, c] = self._label_for_square(t, box_events[pos])

        return labels_A, labels_B


# ===========================================================================
# 2.  ProbeDataset
# ===========================================================================
class ProbeDataset:
    """
    Collects and stores cell state activations and concept labels from episodes
    played by a trained DRC agent.

    Typical workflow:
        dataset = ProbeDataset(model, env, levels, labeler, config)
        dataset.generate(num_episodes=3000, split='train')
        # later:
        dataset.load(split='train')
        loader = dataset.get_dataloader(layer_idx=1, concept='C_A', batch_size=128)
        for x, y in loader:
            ...
    """

    def __init__(
        self,
        model: DRCNetwork,
        env: SokobanEnv,
        levels: List[str],
        labeler: ConceptLabeler,
        config: Config,                     # full configuration object
    ):
        """
        Args:
            model  : trained DRCNetwork (weights loaded, eval mode).
            env    : SokobanEnv instance (used for episode resets / step).
            levels : list of level strings to run on.
            labeler: ConceptLabeler instance.
            config : Config object (provides probing hyperparameters and paths).
        """
        self.model = model
        self.env = env
        self.levels = levels
        self.labeler = labeler
        self.config = config

        # Cache directory for storing generated datasets
        self.cache_dir = config.probing.get('dataset_cache_dir', './probe_datasets')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Storage for loaded arrays (set by load())
        self.obs: torch.Tensor = None          # (N, 8, 8, 7)
        self.cell_l0: torch.Tensor = None      # (N, 32, 8, 8)
        self.cell_l1: torch.Tensor = None
        self.cell_l2: torch.Tensor = None
        self.label_A: torch.Tensor = None      # (N, 8, 8)
        self.label_B: torch.Tensor = None
        self.length: int = 0

    # ------------------------------------------------------------------
    #  Data generation
    # ------------------------------------------------------------------
    def generate(
        self,
        num_episodes: int,
        split: str = 'train',
        use_greedy: bool = True
    ) -> None:
        """
        Run the agent on `num_episodes` randomly selected levels, record cell states
        and compute concept labels, and save the dataset to disk.

        Args:
            num_episodes : number of episodes to collect (e.g., 3000 for train).
            split        : 'train' or 'test' (determines the subdirectory and level subset).
            use_greedy   : if True, select actions greedily (argmax). Otherwise sample
                           from policy (for data augmentation). Paper uses greedy.
        """
        # Reproducibility
        seed = self.config.seed
        set_seed(seed)

        # Select levels (allow repetition if fewer levels than episodes)
        rng = np.random.RandomState(seed)
        level_indices = rng.choice(len(self.levels), size=num_episodes, replace=True)
        chosen_levels = [self.levels[idx] for idx in level_indices]

        # Accumulators for raw data
        all_obs_list = []
        all_cell_l0_list = []
        all_cell_l1_list = []
        all_cell_l2_list = []
        all_label_A_list = []
        all_label_B_list = []

        for episode_idx, level_str in enumerate(chosen_levels):
            # Reset environment and recurrent state
            obs = self.env.set_level(level_str)   # returns symbolic obs (8,8,7)
            state = self.model.initial_state(batch_size=1)

            # Per‑episode storage (we need full trace for labeling)
            trajectory = []   # will hold dicts with 'action', 'agent_pos', 'push_event'
            obs_per_step = []
            cell_l0_per_step = []
            cell_l1_per_step = []
            cell_l2_per_step = []

            while not self.env.done:
                # Record current observation before action
                obs_per_step.append(obs.copy())
                # Agent forward pass (we need cell states after last internal tick)
                # Use the method that returns cell states
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1,8,8,7)
                logits, value, new_state = self.model(obs_tensor, state)
                # Greedy action selection
                if use_greedy:
                    action = logits.argmax(dim=-1).item()
                else:
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                # Retrieve cell states after final tick (last computed forward)
                cell_states = self.model.get_final_cell_states(obs_tensor, state)
                # cell_states is a list of tensors, one per layer: (1,32,8,8)
                c_l0 = cell_states[0].squeeze(0).cpu()   # shape (32,8,8)
                c_l1 = cell_states[1].squeeze(0).cpu()
                c_l2 = cell_states[2].squeeze(0).cpu()
                cell_l0_per_step.append(c_l0)
                cell_l1_per_step.append(c_l1)
                cell_l2_per_step.append(c_l2)

                # Step environment
                next_obs, _, _, _ = self.env.step(action)
                # Record trajectory info *after* step
                agent_pos = self.env.agent_pos   # current agent position (after step)
                push_event = self.env.last_push_event   # we need to add this to environment
                trajectory.append({
                    'action': action,
                    'agent_pos': agent_pos,
                    'push_event': push_event,
                })

                # Update for next iteration
                obs = next_obs
                state = new_state

            # Episode finished – label the whole trajectory
            labels_A, labels_B = self.labeler.label_episode(trajectory)  # (T,8,8)

            # Append to global lists
            all_obs_list.append(np.stack(obs_per_step, axis=0))          # (T,8,8,7)
            all_cell_l0_list.append(torch.stack(cell_l0_per_step, dim=0)) # (T,32,8,8)
            all_cell_l1_list.append(torch.stack(cell_l1_per_step, dim=0))
            all_cell_l2_list.append(torch.stack(cell_l2_per_step, dim=0))
            all_label_A_list.append(labels_A)
            all_label_B_list.append(labels_B)

            if (episode_idx + 1) % 100 == 0:
                print(f"[ProbeDataset] Generated {episode_idx+1}/{num_episodes} episodes")

        # Concatenate across episodes (time‑major)
        self.obs      = torch.cat([torch.as_tensor(o, dtype=torch.float32) for o in all_obs_list], dim=0)
        self.cell_l0  = torch.cat(all_cell_l0_list, dim=0)
        self.cell_l1  = torch.cat(all_cell_l1_list, dim=0)
        self.cell_l2  = torch.cat(all_cell_l2_list, dim=0)
        self.label_A  = torch.as_tensor(np.concatenate(all_label_A_list, axis=0), dtype=torch.long)
        self.label_B  = torch.as_tensor(np.concatenate(all_label_B_list, axis=0), dtype=torch.long)
        self.length    = self.obs.size(0)

        # Save to disk
        save_dir = os.path.join(self.cache_dir, split)
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'obs':      self.obs,
            'cell_l0':  self.cell_l0,
            'cell_l1':  self.cell_l1,
            'cell_l2':  self.cell_l2,
            'label_A':  self.label_A,
            'label_B':  self.label_B,
        }, os.path.join(save_dir, 'probe_data.pt'))
        print(f"[ProbeDataset] Saved dataset to {save_dir}")

    # ------------------------------------------------------------------
    #  Loading pre‑generated dataset
    # ------------------------------------------------------------------
    def load(self, split: str = 'train') -> None:
        """
        Load a previously generated dataset from disk.

        Args:
            split: 'train' or 'test' (the subdirectory under cache_dir).
        """
        load_path = os.path.join(self.cache_dir, split, 'probe_data.pt')
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"Dataset file not found: {load_path}")
        data = torch.load(load_path)
        self.obs      = data['obs']
        self.cell_l0  = data['cell_l0']
        self.cell_l1  = data['cell_l1']
        self.cell_l2  = data['cell_l2']
        self.label_A  = data['label_A']
        self.label_B  = data['label_B']
        self.length    = self.obs.size(0)
        print(f"[ProbeDataset] Loaded dataset from {load_path} ({self.length} transitions)")

    # ------------------------------------------------------------------
    #  Dataloader factory
    # ------------------------------------------------------------------
    def get_dataloader(
        self,
        layer_idx: int,
        concept: str,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create a DataLoader for training/evaluating a linear probe on a specific
        layer and concept.

        Args:
            layer_idx : 0, 1, or 2 (which ConvLSTM layer to probe).
            concept   : 'C_A' or 'C_B'.
            batch_size: number of samples per batch.
            shuffle   : whether to shuffle the dataset (True for training).
            num_workers: number of subprocesses for data loading.

        Returns:
            torch.utils.data.DataLoader yielding tuples (x, y) where
                x : (batch, 32, 8, 8)  – cell state of chosen layer
                y : (batch, 8, 8)      – ground‑truth labels (class indices)
        """
        if self.obs is None or self.length == 0:
            raise RuntimeError("Dataset not loaded or generated. Call generate() or load() first.")

        # Select the appropriate cell state tensor
        if layer_idx == 0:
            x_tensor = self.cell_l0
        elif layer_idx == 1:
            x_tensor = self.cell_l1
        elif layer_idx == 2:
            x_tensor = self.cell_l2
        else:
            raise ValueError(f"Invalid layer_idx {layer_idx}; must be 0,1,2")

        # Select the appropriate label tensor
        if concept == 'C_A':
            y_tensor = self.label_A
        elif concept == 'C_B':
            y_tensor = self.label_B
        else:
            raise ValueError(f"Unknown concept '{concept}' (use 'C_A' or 'C_B')")

        # Ensure both tensors have the same first dimension
        assert x_tensor.size(0) == y_tensor.size(0), "Mismatched dataset size"

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False
        )
        return loader

    # ------------------------------------------------------------------
    #  PyTorch Dataset interface (not strictly required but provided)
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Returns a tuple of all arrays for the given index, useful for debugging.
        """
        return (
            self.obs[idx],
            self.cell_l0[idx],
            self.cell_l1[idx],
            self.cell_l2[idx],
            self.label_A[idx],
            self.label_B[idx],
        )


# ===========================================================================
#  Additional small helper to be added to environment: last_push_event
# ===========================================================================
# In environment.py, after a successful push we must store the last push event.
# We'll add it as an attribute `last_push_event` in the environment. For dataset
# completeness, we assume SokobanEnv already has it. (Modify environment.py if needed.)
# For this file, we rely on that attribute.
