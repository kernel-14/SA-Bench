"""
Self-supervised autoregressive training for RWM.

Training scheme (Section 3.2):
  - Sliding window of size M + N over trajectories
  - History horizon M: context for inner autoregression
  - Forecast horizon N: outer autoregressive prediction steps
  - Loss: weighted multi-step prediction error with decay factor alpha
  - Reparameterization trick for gradient propagation through stochastic predictions

Loss (Eq. 2):
  L = (1/N) * sum_{k=1}^{N} alpha^k * [L_o(o'_{t+k}, o_{t+k}) + L_c(c'_{t+k}, c_{t+k})]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List, Tuple
import numpy as np


class WorldModelTrainer:
    """
    Trainer for RWM using self-supervised autoregressive training.

    Args:
        model: RoboticWorldModel instance
        optimizer: PyTorch optimizer
        history_horizon: M - number of historical steps as context
        forecast_horizon: N - number of future steps to predict
        forecast_decay: alpha - decay factor for multi-step loss weighting
        device: torch device
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        history_horizon: int = 32,
        forecast_horizon: int = 8,
        forecast_decay: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.optimizer = optimizer
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.forecast_decay = forecast_decay
        self.device = device

        # Precompute decay weights: alpha^k for k=1..N
        self.decay_weights = torch.tensor(
            [forecast_decay ** k for k in range(1, forecast_horizon + 1)],
            dtype=torch.float32,
            device=device,
        )

    def compute_loss(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        obs_targets: torch.Tensor,
        priv_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute autoregressive multi-step prediction loss.

        Args:
            obs_history: (batch, M, obs_size) - historical observations
            action_history: (batch, M+N, action_size) - historical + future actions
            obs_targets: (batch, N, obs_size) - target future observations
            priv_targets: (batch, N, priv_size) or None - target privileged info

        Returns:
            loss: scalar tensor
            metrics: dict of loss components
        """
        batch_size = obs_history.shape[0]
        M = self.history_horizon
        N = self.forecast_horizon
        device = obs_history.device

        # Inner autoregression: process history to build hidden state
        hidden = self.model.gru_base.init_hidden(batch_size, device)
        x_hist = torch.cat([obs_history, action_history[:, :M, :]], dim=-1)
        _, hidden = self.model.gru_base(x_hist, hidden)

        # Outer autoregression: predict N future steps
        current_obs = obs_history[:, -1, :]  # Start from last observed obs
        obs_loss_total = torch.tensor(0.0, device=device)
        priv_loss_total = torch.tensor(0.0, device=device)

        for k in range(N):
            action_k = action_history[:, M + k, :]
            obs_mean, obs_std, priv_mean, priv_std, hidden = self.model.predict_step(
                current_obs, action_k, hidden
            )

            # Observation loss: negative log-likelihood
            obs_target_k = obs_targets[:, k, :]
            obs_dist = torch.distributions.Normal(obs_mean, obs_std)
            obs_nll = -obs_dist.log_prob(obs_target_k).mean()
            obs_loss_total = obs_loss_total + self.decay_weights[k] * obs_nll

            # Privileged information loss
            if priv_mean is not None and priv_targets is not None:
                priv_target_k = priv_targets[:, k, :]
                priv_dist = torch.distributions.Normal(priv_mean, priv_std)
                priv_nll = -priv_dist.log_prob(priv_target_k).mean()
                priv_loss_total = priv_loss_total + self.decay_weights[k] * priv_nll

            # Reparameterization: sample for next step (outer autoregression)
            eps = torch.randn_like(obs_mean)
            current_obs = obs_mean + obs_std * eps  # reparameterized sample

        # Normalize by N
        obs_loss = obs_loss_total / N
        priv_loss = priv_loss_total / N
        total_loss = obs_loss + priv_loss

        metrics = {
            "loss": total_loss.item(),
            "obs_loss": obs_loss.item(),
            "priv_loss": priv_loss.item(),
        }

        return total_loss, metrics

    def train_step(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        obs_targets: torch.Tensor,
        priv_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        loss, metrics = self.compute_loss(
            obs_history, action_history, obs_targets, priv_targets
        )
        loss.backward()
        self.optimizer.step()

        return metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {"loss": 0.0, "obs_loss": 0.0, "priv_loss": 0.0}
        num_batches = 0

        for batch in dataloader:
            if len(batch) == 3:
                obs_history, action_history, obs_targets = batch
                priv_targets = None
            else:
                obs_history, action_history, obs_targets, priv_targets = batch
                priv_targets = priv_targets.to(self.device)

            obs_history = obs_history.to(self.device)
            action_history = action_history.to(self.device)
            obs_targets = obs_targets.to(self.device)

            metrics = self.train_step(
                obs_history, action_history, obs_targets, priv_targets
            )

            for k, v in metrics.items():
                epoch_metrics[k] += v
            num_batches += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= max(num_batches, 1)

        return epoch_metrics

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        eval_horizon: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate autoregressive prediction error over long horizons.

        Computes relative prediction error e = ||o' - o||_2 / ||o||_2
        averaged over the evaluation horizon.
        """
        self.model.eval()
        all_errors = []

        for batch in dataloader:
            if len(batch) == 3:
                obs_history, action_history, obs_targets = batch
            else:
                obs_history, action_history, obs_targets, _ = batch

            obs_history = obs_history.to(self.device)
            action_history = action_history.to(self.device)
            obs_targets = obs_targets.to(self.device)

            batch_size = obs_history.shape[0]
            M = self.history_horizon

            # Get hidden state from history
            hidden = self.model.gru_base.init_hidden(batch_size, self.device)
            x_hist = torch.cat([obs_history, action_history[:, :M, :]], dim=-1)
            _, hidden = self.model.gru_base(x_hist, hidden)

            current_obs = obs_history[:, -1, :]
            step_errors = []

            for k in range(min(eval_horizon, obs_targets.shape[1])):
                action_k = action_history[:, M + k, :]
                obs_mean, obs_std, _, _, hidden = self.model.predict_step(
                    current_obs, action_k, hidden
                )

                # Relative prediction error
                target = obs_targets[:, k, :]
                error = torch.norm(obs_mean - target, dim=-1) / (
                    torch.norm(target, dim=-1) + 1e-8
                )
                step_errors.append(error.mean().item())

                # Feed prediction back
                current_obs = obs_mean  # deterministic for evaluation

            all_errors.append(step_errors)

        # Average over batches
        all_errors = np.array(all_errors)
        mean_errors = all_errors.mean(axis=0)

        return {
            "mean_relative_error": float(mean_errors.mean()),
            "final_step_error": float(mean_errors[-1]) if len(mean_errors) > 0 else 0.0,
            "per_step_errors": mean_errors.tolist(),
        }


class TeacherForcingTrainer(WorldModelTrainer):
    """
    Teacher-forcing training (special case of autoregressive with N=1).

    This is the baseline training scheme used by many existing architectures.
    Equivalent to autoregressive training with forecast_horizon=1.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs):
        # Override forecast_horizon to 1 for teacher forcing
        kwargs["forecast_horizon"] = 1
        super().__init__(model, optimizer, **kwargs)

    def compute_loss(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        obs_targets: torch.Tensor,
        priv_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Teacher-forcing loss: predict next step from ground truth history.

        This can be parallelized over the sequence dimension.
        """
        batch_size = obs_history.shape[0]
        M = self.history_horizon
        device = obs_history.device

        # Process full sequence in parallel (teacher forcing)
        obs_mean, obs_std, priv_mean, priv_std, _ = self.model(
            obs_history, action_history[:, :M, :]
        )

        # Only predict the last step (next observation)
        obs_mean_last = obs_mean[:, -1, :]
        obs_std_last = obs_std[:, -1, :]
        obs_target = obs_targets[:, 0, :]  # First target = next step

        obs_dist = torch.distributions.Normal(obs_mean_last, obs_std_last)
        obs_loss = -obs_dist.log_prob(obs_target).mean()

        priv_loss = torch.tensor(0.0, device=device)
        if priv_mean is not None and priv_targets is not None:
            priv_mean_last = priv_mean[:, -1, :]
            priv_std_last = priv_std[:, -1, :]
            priv_target = priv_targets[:, 0, :]
            priv_dist = torch.distributions.Normal(priv_mean_last, priv_std_last)
            priv_loss = -priv_dist.log_prob(priv_target).mean()

        total_loss = obs_loss + priv_loss

        metrics = {
            "loss": total_loss.item(),
            "obs_loss": obs_loss.item(),
            "priv_loss": priv_loss.item(),
        }

        return total_loss, metrics
