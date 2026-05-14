# rwm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict


class RWMModel(nn.Module):
    """
    Robotic World Model (RWM) implements a GRU-based dual-autoregressive structure to
    predict next-step observations and privileged information. It supports autoregressive
    rollouts for sequential predictions and computes multi-step prediction losses during training.
    """

    def __init__(
        self,
        history_horizon: int = 32,
        forecast_horizon: int = 8,
        input_dim: int = 48,
        hidden_dim: int = 256,
        output_dim: int = 48,
        privileged_dim: int = 8,
        decay_factor: float = 1.0,
    ):
        """
        Initialize the RWM Model with GRU-based architecture and MLP heads.

        Args:
            history_horizon (int): Length of historical observation-action sequence (M).
            forecast_horizon (int): Length of autoregressive forecast horizon (N).
            input_dim (int): Dimension of each input observation.
            hidden_dim (int): Dimension of GRU hidden state.
            output_dim (int): Dimension of predicted output observations.
            privileged_dim (int): Dimension of privileged information predictions.
            decay_factor (float): Forecast loss decay factor for prioritizing earlier steps.
        """
        super(RWMModel, self).__init__()
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.privileged_dim = privileged_dim
        self.decay_factor = decay_factor

        # GRU Base
        self.gru = nn.GRU(input_dim * 2, hidden_dim, num_layers=2, batch_first=True)

        # MLP Heads
        # Prediction for next observations
        self.observation_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim * 2) # Gaussian: mean and standard deviation
        )

        # Prediction for privileged information
        self.privileged_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, privileged_dim)
        )

    def forward(
        self, history: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single forward pass for the Robotic World Model (RWM).

        Args:
            history (torch.Tensor): Historical observations of shape (batch_size, M, input_dim).
            actions (torch.Tensor): Historical actions of shape (batch_size, M, action_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted observations (mean, std) and privileged information
                - Observations: (batch_size, N, output_dim * 2) (where last dimension includes mean and std dev)
                - Privileged information: (batch_size, N, privileged_dim)
        """
        # Combine inputs: concatenate observations and actions along feature dimension
        inputs = torch.cat([history, actions], dim=-1)  # Shape: (B, M, input_dim * 2)

        # Initialize GRU hidden state
        batch_size = inputs.size(0)
        h_0 = torch.zeros(2, batch_size, self.hidden_dim).to(inputs.device)  # 2 layers of GRU

        # Process historical sequence through GRU
        _, h_t = self.gru(inputs, h_0)  # h_t contains the last hidden state

        # Autoregressive rollout using forecast horizon N
        predictions_obs = []
        predictions_privileged = []

        # Initialize autoregressive input as last timestep in the history
        autoreg_input = inputs[:, -1, :]  # Shape: (B, input_dim * 2)

        for _ in range(self.forecast_horizon):
            # Pass input through GRU and update hidden state
            _, h_t = self.gru(autoreg_input.unsqueeze(1), h_t)

            # Predict next observation (mean and std) and privileged info
            obs_pred = self.observation_head(h_t[-1])  # Use the last layer's hidden state
            privileged_pred = self.privileged_head(h_t[-1])

            # Split observations into mean and std for Gaussian predictions
            pred_mean, pred_log_std = torch.chunk(obs_pred, 2, dim=-1)
            pred_std = torch.exp(pred_log_std)  # Convert log variance to std deviation
            predictions_obs.append(torch.cat([pred_mean, pred_std], dim=-1))  # (mean, std)

            # Store privileged info predictions
            predictions_privileged.append(privileged_pred)

            # Autoregressively feed back predictions as inputs
            autoreg_input = torch.cat([pred_mean, actions[:, -1, :]], dim=-1)

        # Concatenate predictions across the forecast horizon
        predictions_obs = torch.stack(predictions_obs, dim=1)  # (B, N, output_dim * 2)
        predictions_privileged = torch.stack(predictions_privileged, dim=1)  # (B, N, privileged_dim)

        return predictions_obs, predictions_privileged

    def compute_loss(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the multi-step prediction loss for the Robotic World Model (RWM).

        Args:
            predictions (Tuple[torch.Tensor, torch.Tensor]): Predicted observations and privileged information.
                - Observations: (batch_size, N, output_dim * 2) (mean and std dev)
                - Privileged information: (batch_size, N, privileged_dim)
            targets (Tuple[torch.Tensor, torch.Tensor]): Ground truth observations and privileged information.
                - Observations: (batch_size, N, output_dim)
                - Privileged information: (batch_size, N, privileged_dim)

        Returns:
            torch.Tensor: Total multi-step loss.
        """
        predicted_obs, predicted_privileged = predictions
        target_obs, target_privileged = targets

        # Decompose predicted observations into mean and std
        pred_mean, pred_std = torch.chunk(predicted_obs, 2, dim=-1)  # (mean, std)
        
        # Observation loss (Gaussian negative log-likelihood)
        mse_loss = ((pred_mean - target_obs) ** 2) / (pred_std**2 + 1e-8)
        gaussian_nll = mse_loss + 2 * torch.log(pred_std + 1e-8)
        obs_loss = gaussian_nll.mean(dim=-1)  # Average over feature dimensions

        # Privileged information loss (mean squared error)
        priv_loss = F.mse_loss(predicted_privileged, target_privileged, reduction="none")
        priv_loss = priv_loss.mean(dim=-1)  # Average over feature dimensions

        # Apply decay factor across the forecast horizon
        forecast_weights = torch.tensor(
            [self.decay_factor**k for k in range(1, self.forecast_horizon + 1)],
            device=obs_loss.device,
        )
        forecast_weights /= forecast_weights.sum()  # Normalize decay factors

        # Weighted loss across forecast horizon
        total_obs_loss = torch.sum(obs_loss * forecast_weights, dim=1).mean()  # Batch mean
        total_priv_loss = torch.sum(priv_loss * forecast_weights, dim=1).mean()  # Batch mean

        # Total loss
        total_loss = total_obs_loss + total_priv_loss
        return total_loss

