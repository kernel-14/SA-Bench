"""Autoregressive training loop for Robotic World Model (Sec 3.2, Algorithm 1).

Training follows the self-supervised autoregressive scheme:
1. Construct training data by sliding a window of size M+N over trajectories
2. For each batch, warm up the model over M historical steps (inner autoregression)
3. Predict N future steps autoregressively, feeding predictions back (outer autoregression)
4. Compute multi-step loss with decay factor α

Key hyperparameters (Table S10):
  - History horizon M = 32
  - Forecast horizon N = 8
  - Learning rate 1e-4, weight decay 1e-5
  - Batch size 1024
  - Max iterations 2500
"""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.rwm import RoboticWorldModel, RWMLoss
from data.dataset import SlidingWindowDataset, TrajectoryBuffer
from config import RWMConfig


class WorldModelTrainer:
    """Trainer for Robotic World Model with dual-autoregressive mechanism."""

    def __init__(self, model: RoboticWorldModel, config: RWMConfig, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.loss_fn = RWMLoss(forecast_decay=config.forecast_decay)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.M = config.history_horizon
        self.N = config.forecast_horizon
        self.batch_size = config.batch_size

        self.writer: Optional[SummaryWriter] = None

    def set_writer(self, writer: SummaryWriter):
        self.writer = writer

    def train(
        self,
        dataloader: DataLoader,
        num_iterations: int,
        log_interval: int = 50,
    ) -> Dict[str, list]:
        """Run the autoregressive training loop.

        Args:
            dataloader: Provides batches of (observations, actions, privileged)
            num_iterations: Number of training iterations
            log_interval: Log metrics every N iterations

        Returns:
            dict of logged metrics over training
        """
        history: Dict[str, list] = {
            "iteration": [],
            "total_loss": [],
            "obs_loss": [],
            "priv_loss": [],
        }

        data_iter = iter(dataloader)

        self.model.train()
        pbar = tqdm(range(num_iterations), desc="RWM training")

        for iteration in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            observations = batch["observations"].to(self.device)
            actions = batch["actions"].to(self.device)
            privileged = batch.get("privileged")
            if privileged is not None:
                privileged = privileged.to(self.device)

            loss, losses = self._train_step(observations, actions, privileged)

            if iteration % log_interval == 0:
                history["iteration"].append(iteration)
                history["total_loss"].append(losses["total_loss"])
                history["obs_loss"].append(losses["obs_loss"])
                history["priv_loss"].append(losses["priv_loss"])
                pbar.set_postfix({"loss": f"{losses['total_loss']:.4f}"})

                if self.writer is not None:
                    for key, val in losses.items():
                        self.writer.add_scalar(f"rwm/{key}", val, iteration)

        return history

    def _train_step(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        privileged: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step with autoregressive loss.

        observations: (B, M+N, obs_dim) — full window
        actions: (B, M+N-1, action_dim) — actions for all transitions
        privileged: (B, M+N, priv_dim) — optional privileged info
        """
        batch_size = observations.shape[0]

        # Split into history (M) and forecast target (N)
        obs_history = observations[:, :self.M]  # (B, M, obs_dim)
        obs_target = observations[:, self.M:self.M + self.N]  # (B, N, obs_dim)

        # Actions: M-1 for history warming + N for forecast
        acts_history = actions[:, :self.M - 1 + self.N]  # (B, M-1+N, action_dim)

        priv_target = None
        if privileged is not None:
            priv_target = privileged[:, self.M:self.M + self.N]

        self.optimizer.zero_grad()

        predictions = self.model.forecast_autoregressive(
            observations=obs_history,
            actions=acts_history,
            forecast_horizon=self.N,
        )

        total_loss, losses = self.loss_fn(predictions, obs_target, priv_target)
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        self.optimizer.step()

        return total_loss, losses

    def evaluate_autoregressive_error(
        self,
        dataloader: DataLoader,
        max_horizon: int = 100,
    ) -> Dict[str, np.ndarray]:
        """Evaluate autoregressive prediction error over long horizons.

        Computes relative prediction error per forecast step (as in Fig 3, Fig 4).

        Returns:
            dict with relative_errors: (max_horizon,) array
        """
        self.model.eval()
        errors = []
        counts = np.zeros(max_horizon)

        with torch.no_grad():
            for batch in dataloader:
                observations = batch["observations"].to(self.device)
                actions = batch["actions"].to(self.device)

                batch_size = observations.shape[0]
                M = self.M
                # Use as much horizon as available
                effective_horizon = min(
                    max_horizon, observations.shape[1] - M - 1
                )

                obs_history = observations[:, :M]
                obs_target = observations[:, M:M + effective_horizon]

                preds = self.model.forecast_autoregressive(
                    observations=obs_history,
                    actions=actions[:, :M - 1 + effective_horizon],
                    forecast_horizon=effective_horizon,
                )

                # Relative error: ||pred - target|| / ||target||
                err = torch.norm(
                    preds["obs_means"] - obs_target, dim=-1
                ) / (torch.norm(obs_target, dim=-1) + 1e-8)  # (B, H)

                errors.append(err.cpu().numpy())
                counts[:effective_horizon] += batch_size

        errors = np.concatenate(errors, axis=0)  # (total_B, max_H)
        mean_errors = errors.mean(axis=0)

        return {"relative_errors": mean_errors}

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def create_dataloader(
    buffer: TrajectoryBuffer,
    config: RWMConfig,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for autoregressive training.

    Uses SlidingWindowDataset with window size (M+N).
    """
    dataset = SlidingWindowDataset(
        buffer=buffer,
        history_horizon=config.history_horizon,
        forecast_horizon=config.forecast_horizon,
        use_privileged=(config.robot.privileged_dim > 0),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader
