"""
Training utilities for Robotic World Model (RWM).

Implements:
- Sliding window data construction (M + N steps)
- Autoregressive training loop
- Teacher-forcing training (baseline comparison)
- Evaluation metrics for autoregressive prediction error
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time
from .world_model import RoboticWorldModel


class TrajectoryDataset(Dataset):
    """
    Dataset that constructs sliding windows of size M+N over trajectories.
    """
    
    def __init__(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        history_horizon: int = 32,
        forecast_horizon: int = 8,
    ):
        """
        Args:
            trajectories: List of trajectory dicts with keys:
                - 'obs': (T, obs_dim)
                - 'act': (T-1, act_dim) or (T, act_dim)
                - 'priv': (T, priv_dim) optional
            history_horizon: M
            forecast_horizon: N
        """
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.window_size = history_horizon + forecast_horizon
        
        # Build all valid windows
        self.windows = []
        for traj in trajectories:
            T = traj['obs'].shape[0]
            # We need obs[t:t+M+N] and act[t:t+M+N-1]
            if T < self.window_size + 1:
                continue
            num_windows = T - self.window_size
            for i in range(num_windows):
                window = {
                    'obs': traj['obs'][i:i + self.window_size],
                    'act': traj['act'][i:i + self.window_size - 1],
                }
                if 'priv' in traj:
                    window['priv'] = traj['priv'][i:i + self.window_size]
                self.windows.append(window)
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx]
        return {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in window.items()
        }


class WorldModelTrainer:
    """
    Trainer for RWM and baseline world models with autoregressive training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        forecast_decay: float = 1.0,
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.forecast_decay = forecast_decay
        
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        mode: str = 'autoregressive',
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a batch.
        
        Args:
            batch: Dict with 'obs', 'act', optional 'priv'
            mode: 'autoregressive' or 'teacher_forcing'
        """
        obs = batch['obs'].to(self.device)
        act = batch['act'].to(self.device)
        priv = batch.get('priv', None)
        if priv is not None:
            priv = priv.to(self.device)
        
        M = self.model.history_horizon
        N = self.model.forecast_horizon
        
        if mode == 'autoregressive':
            # Full autoregressive forward
            obs_history = obs[:, :M, :]
            act_history = act[:, :M, :]
            act_future = act[:, M-1:M+N-1, :]  # N actions for N predictions
            
            predictions = self.model.autoregressive_forward_with_sampling(
                obs_history, act_history, act_future, use_reparameterization=True
            )
            
            targets = {'obs': obs[:, M:M+N, :]}
            if priv is not None:
                targets['priv'] = priv[:, M:M+N, :]
            
        elif mode == 'teacher_forcing':
            # Teacher forcing: use ground truth observations
            # Equivalent to autoregressive with N=1 rolled over the sequence
            obs_full = obs[:, :M+N, :]
            act_full = act[:, :M+N-1, :]
            
            all_means = []
            all_stds = []
            
            for k in range(N):
                obs_hist = obs_full[:, k:k+M, :]
                act_hist = act_full[:, k:k+M, :]
                act_next = act_full[:, k+M-1:k+M, :]
                
                pred = self.model.autoregressive_forward(
                    obs_hist, act_hist, act_next
                )
                all_means.append(pred['obs_means'][:, 0, :])
                all_stds.append(pred['obs_stds'][:, 0, :])
            
            predictions = {
                'obs_means': torch.stack(all_means, dim=1),
                'obs_stds': torch.stack(all_stds, dim=1),
            }
            
            targets = {'obs': obs[:, M:M+N, :]}
            if priv is not None:
                targets['priv'] = priv[:, M:M+N, :]
        
        return self.model.compute_loss(predictions, targets)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        mode: str = 'autoregressive',
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss_dict = self.compute_loss(batch, mode)
        loss = loss_dict['total_loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        return {k: v.item() for k, v in loss_dict.items()}
    
    @torch.no_grad()
    def evaluate_prediction_error(
        self,
        dataset: Dataset,
        num_steps: int = 100,
        batch_size: int = 1024,
    ) -> Dict[str, float]:
        """
        Evaluate autoregressive prediction error over multiple rollout steps.
        
        Computes relative prediction error (normalized MSE) per forecast step.
        """
        self.model.eval()
        
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        
        all_errors = []
        max_steps = min(num_steps, self.model.forecast_horizon)
        
        for batch in dataloader:
            obs = batch['obs'].to(self.device)
            act = batch['act'].to(self.device)
            
            M = self.model.history_horizon
            N = min(max_steps, obs.shape[1] - M - 1)
            
            obs_history = obs[:, :M, :]
            act_history = act[:, :M, :]
            act_future = act[:, M-1:M+N-1, :]  # N actions for N predictions
            
            predictions = self.model.autoregressive_forward(
                obs_history, act_history, act_future
            )
            
            obs_targets = obs[:, M:M+N, :]
            obs_means = predictions['obs_means'][:, :N, :]
            
            # Compute MSE per step
            mse = ((obs_targets - obs_means) ** 2).mean(dim=(0, 2))  # (N,)
            # Normalize by observation variance
            obs_var = obs_targets.var(dim=(0, 2), unbiased=False)
            relative_error = mse / (obs_var + 1e-8)
            
            all_errors.append(relative_error.cpu())
        
        # Average over batches
        errors = torch.stack(all_errors).mean(dim=0)
        
        return {
            'per_step_error': errors.numpy(),
            'mean_error': errors.mean().item(),
        }


def create_dataloader(
    trajectories: List[Dict[str, np.ndarray]],
    history_horizon: int,
    forecast_horizon: int,
    batch_size: int = 1024,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from trajectories."""
    dataset = TrajectoryDataset(
        trajectories,
        history_horizon=history_horizon,
        forecast_horizon=forecast_horizon,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0,
    )


def compute_relative_prediction_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    obs_var: float = None,
) -> torch.Tensor:
    """
    Compute relative prediction error e.
    
    e = MSE(predicted, target) / var(target)
    """
    mse = ((predicted - target) ** 2).mean(dim=-1)  # per-step MSE
    if obs_var is None:
        obs_var = target.var(dim=(0, 1), unbiased=False).sum()
    return mse / (obs_var + 1e-8)
