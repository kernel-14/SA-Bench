"""
Dataset utilities for RWM training.

Creates sliding window datasets from trajectory data for autoregressive training.
Window size = M + N (history + forecast horizons).
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple


class TrajectoryDataset(Dataset):
    """
    Dataset of sliding windows over trajectory data.

    Each sample consists of:
      - obs_history: (M, obs_size) - historical observations
      - action_history: (M+N, action_size) - historical + future actions
      - obs_targets: (N, obs_size) - target future observations
      - priv_targets: (N, priv_size) or None - target privileged info

    Args:
        observations: list of trajectories, each (T, obs_size)
        actions: list of trajectories, each (T, action_size)
        history_horizon: M
        forecast_horizon: N
        privileged_info: list of trajectories, each (T, priv_size) or None
    """

    def __init__(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        history_horizon: int = 32,
        forecast_horizon: int = 8,
        privileged_info: Optional[List[np.ndarray]] = None,
    ):
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.window_size = history_horizon + forecast_horizon

        self.windows = []

        for traj_idx, (obs_traj, act_traj) in enumerate(zip(observations, actions)):
            T = len(obs_traj)
            priv_traj = privileged_info[traj_idx] if privileged_info is not None else None

            # Slide window over trajectory
            for t in range(T - self.window_size):
                obs_hist = obs_traj[t: t + history_horizon]
                act_hist = act_traj[t: t + history_horizon + forecast_horizon]
                obs_tgt = obs_traj[t + history_horizon: t + self.window_size]

                priv_tgt = None
                if priv_traj is not None:
                    priv_tgt = priv_traj[t + history_horizon: t + self.window_size]

                self.windows.append((obs_hist, act_hist, obs_tgt, priv_tgt))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        obs_hist, act_hist, obs_tgt, priv_tgt = self.windows[idx]

        obs_hist = torch.tensor(obs_hist, dtype=torch.float32)
        act_hist = torch.tensor(act_hist, dtype=torch.float32)
        obs_tgt = torch.tensor(obs_tgt, dtype=torch.float32)

        if priv_tgt is not None:
            priv_tgt = torch.tensor(priv_tgt, dtype=torch.float32)
            return obs_hist, act_hist, obs_tgt, priv_tgt
        else:
            return obs_hist, act_hist, obs_tgt


class NoisyTrajectoryDataset(TrajectoryDataset):
    """
    Dataset with Gaussian noise augmentation for robustness evaluation.

    Used in Section 4.2 to evaluate robustness under noise perturbations.
    """

    def __init__(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        history_horizon: int = 32,
        forecast_horizon: int = 8,
        privileged_info: Optional[List[np.ndarray]] = None,
        obs_noise_std: float = 0.0,
        action_noise_std: float = 0.0,
    ):
        super().__init__(
            observations, actions, history_horizon, forecast_horizon, privileged_info
        )
        self.obs_noise_std = obs_noise_std
        self.action_noise_std = action_noise_std

    def __getitem__(self, idx):
        result = super().__getitem__(idx)

        if isinstance(result, tuple) and len(result) == 4:
            obs_hist, act_hist, obs_tgt, priv_tgt = result
        else:
            obs_hist, act_hist, obs_tgt = result
            priv_tgt = None

        # Add noise to history
        if self.obs_noise_std > 0:
            obs_hist = obs_hist + torch.randn_like(obs_hist) * self.obs_noise_std
        if self.action_noise_std > 0:
            act_hist = act_hist + torch.randn_like(act_hist) * self.action_noise_std

        if priv_tgt is not None:
            return obs_hist, act_hist, obs_tgt, priv_tgt
        else:
            return obs_hist, act_hist, obs_tgt


def create_dataloaders(
    observations: List[np.ndarray],
    actions: List[np.ndarray],
    history_horizon: int = 32,
    forecast_horizon: int = 8,
    privileged_info: Optional[List[np.ndarray]] = None,
    batch_size: int = 1024,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        observations: list of observation trajectories
        actions: list of action trajectories
        history_horizon: M
        forecast_horizon: N
        privileged_info: list of privileged info trajectories
        batch_size: training batch size
        val_split: fraction of data for validation
        num_workers: dataloader workers
        seed: random seed

    Returns:
        train_loader, val_loader
    """
    dataset = TrajectoryDataset(
        observations, actions, history_horizon, forecast_horizon, privileged_info
    )

    n = len(dataset)
    n_val = int(n * val_split)
    n_train = n - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def generate_synthetic_trajectories(
    n_trajectories: int = 100,
    trajectory_length: int = 500,
    obs_size: int = 45,
    action_size: int = 12,
    priv_size: int = 8,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Generate synthetic trajectory data for testing.

    Simulates simple linear dynamics with noise.

    Returns:
        observations, actions, privileged_info
    """
    np.random.seed(seed)

    observations = []
    actions = []
    privileged_info = []

    for _ in range(n_trajectories):
        obs_traj = np.zeros((trajectory_length, obs_size))
        act_traj = np.random.randn(trajectory_length, action_size) * 0.1
        priv_traj = np.zeros((trajectory_length, priv_size))

        # Simple dynamics: obs_{t+1} = 0.9 * obs_t + 0.1 * action + noise
        obs_traj[0] = np.random.randn(obs_size) * 0.1
        for t in range(1, trajectory_length):
            obs_traj[t] = (
                0.9 * obs_traj[t - 1]
                + 0.1 * act_traj[t - 1, :obs_size] if action_size >= obs_size
                else np.pad(0.1 * act_traj[t - 1], (0, obs_size - action_size))
                + np.random.randn(obs_size) * 0.01
            )
            priv_traj[t] = (np.abs(obs_traj[t, :priv_size]) > 0.5).astype(float)

        observations.append(obs_traj)
        actions.append(act_traj)
        privileged_info.append(priv_traj)

    return observations, actions, privileged_info
