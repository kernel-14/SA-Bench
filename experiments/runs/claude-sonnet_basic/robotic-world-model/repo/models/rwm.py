"""
Robotic World Model (RWM) - GRU-based world model with dual-autoregressive mechanism.

Architecture:
  - GRU base: 2 layers, hidden size 256
  - MLP heads: hidden size 128, ReLU activation
    - observation head: predicts mean and log_std of next observation
    - privileged info head: predicts mean and log_std of privileged info (contacts, etc.)

Dual-autoregressive mechanism:
  (i)  Inner autoregression: GRU hidden states updated autoregressively over history horizon M
  (ii) Outer autoregression: predicted observations fed back into the network over forecast horizon N
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class GRUBase(nn.Module):
    """Multi-layer GRU base for RWM."""

    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
            hidden: (num_layers, batch, hidden_size) or None

        Returns:
            output: (batch, seq_len, hidden_size)
            hidden: (num_layers, batch, hidden_size)
        """
        output, hidden = self.gru(x, hidden)
        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class MLPHead(nn.Module):
    """MLP head predicting mean and log_std of a Gaussian distribution."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size * 2),  # mean + log_std
        )
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (..., input_size)

        Returns:
            mean: (..., output_size)
            std: (..., output_size)  -- softplus-activated
        """
        out = self.net(x)
        mean, log_std = out.chunk(2, dim=-1)
        std = torch.nn.functional.softplus(log_std) + 1e-5
        return mean, std


class RoboticWorldModel(nn.Module):
    """
    Robotic World Model (RWM).

    Predicts the distribution of the next observation and privileged information
    given a history of observation-action pairs.

    The model uses a dual-autoregressive mechanism:
      - Inner autoregression: GRU processes the history horizon M step by step,
        updating hidden states autoregressively.
      - Outer autoregression: During the forecast horizon N, predicted observations
        are fed back as inputs for subsequent predictions.

    Args:
        obs_size: Dimension of the observation space.
        action_size: Dimension of the action space.
        priv_size: Dimension of the privileged information space (e.g., contacts).
        hidden_size: GRU hidden size (default: 256).
        num_gru_layers: Number of GRU layers (default: 2).
        head_hidden_size: MLP head hidden size (default: 128).
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        priv_size: int,
        hidden_size: int = 256,
        num_gru_layers: int = 2,
        head_hidden_size: int = 128,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.priv_size = priv_size
        self.hidden_size = hidden_size

        # Input: observation + action concatenated
        input_size = obs_size + action_size

        self.gru_base = GRUBase(input_size, hidden_size, num_gru_layers)

        # Observation prediction head
        self.obs_head = MLPHead(hidden_size, obs_size, head_hidden_size)

        # Privileged information prediction head (e.g., contacts)
        if priv_size > 0:
            self.priv_head = MLPHead(hidden_size, priv_size, head_hidden_size)
        else:
            self.priv_head = None

    def forward(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Single forward pass over a sequence (used for teacher-forcing / next-step prediction).

        Args:
            obs_history: (batch, seq_len, obs_size)
            action_history: (batch, seq_len, action_size)
            hidden: (num_layers, batch, hidden_size) or None

        Returns:
            obs_mean: (batch, seq_len, obs_size)
            obs_std: (batch, seq_len, obs_size)
            priv_mean: (batch, seq_len, priv_size) or None
            priv_std: (batch, seq_len, priv_size) or None
            hidden: (num_layers, batch, hidden_size)
        """
        x = torch.cat([obs_history, action_history], dim=-1)
        gru_out, hidden = self.gru_base(x, hidden)

        obs_mean, obs_std = self.obs_head(gru_out)

        priv_mean, priv_std = None, None
        if self.priv_head is not None:
            priv_mean, priv_std = self.priv_head(gru_out)

        return obs_mean, obs_std, priv_mean, priv_std, hidden

    def predict_step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Single-step prediction (used during autoregressive rollout).

        Args:
            obs: (batch, obs_size)
            action: (batch, action_size)
            hidden: (num_layers, batch, hidden_size) or None

        Returns:
            obs_mean: (batch, obs_size)
            obs_std: (batch, obs_size)
            priv_mean: (batch, priv_size) or None
            priv_std: (batch, priv_size) or None
            hidden: (num_layers, batch, hidden_size)
        """
        x = torch.cat([obs, action], dim=-1).unsqueeze(1)  # (batch, 1, input_size)
        gru_out, hidden = self.gru_base(x, hidden)
        gru_out = gru_out.squeeze(1)  # (batch, hidden_size)

        obs_mean, obs_std = self.obs_head(gru_out)

        priv_mean, priv_std = None, None
        if self.priv_head is not None:
            priv_mean, priv_std = self.priv_head(gru_out)

        return obs_mean, obs_std, priv_mean, priv_std, hidden

    def sample_obs(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Sample observation from Gaussian distribution using reparameterization."""
        eps = torch.randn_like(mean)
        return mean + std * eps

    def autoregressive_rollout(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        future_actions: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Full dual-autoregressive rollout.

        Inner autoregression: process history horizon M to build hidden state.
        Outer autoregression: predict N future steps, feeding predictions back.

        Args:
            obs_history: (batch, M, obs_size) - historical observations
            action_history: (batch, M, action_size) - historical actions
            future_actions: (batch, N, action_size) - future actions for forecast
            deterministic: if True, use mean predictions; else sample

        Returns:
            pred_obs_means: (batch, N, obs_size)
            pred_obs_stds: (batch, N, obs_size)
            pred_priv_means: (batch, N, priv_size) or None
            pred_priv_stds: (batch, N, priv_size) or None
        """
        batch_size = obs_history.shape[0]
        N = future_actions.shape[1]
        device = obs_history.device

        # Inner autoregression: process history to build hidden state
        hidden = self.gru_base.init_hidden(batch_size, device)
        x_hist = torch.cat([obs_history, action_history], dim=-1)
        _, hidden = self.gru_base(x_hist, hidden)

        # Outer autoregression: predict N future steps
        pred_obs_means = []
        pred_obs_stds = []
        pred_priv_means = []
        pred_priv_stds = []

        # Start with the last observed observation
        current_obs = obs_history[:, -1, :]  # (batch, obs_size)

        for k in range(N):
            action = future_actions[:, k, :]  # (batch, action_size)
            obs_mean, obs_std, priv_mean, priv_std, hidden = self.predict_step(
                current_obs, action, hidden
            )

            pred_obs_means.append(obs_mean)
            pred_obs_stds.append(obs_std)

            if priv_mean is not None:
                pred_priv_means.append(priv_mean)
                pred_priv_stds.append(priv_std)

            # Feed prediction back (outer autoregression)
            if deterministic:
                current_obs = obs_mean
            else:
                current_obs = self.sample_obs(obs_mean, obs_std)

        pred_obs_means = torch.stack(pred_obs_means, dim=1)   # (batch, N, obs_size)
        pred_obs_stds = torch.stack(pred_obs_stds, dim=1)

        if pred_priv_means:
            pred_priv_means = torch.stack(pred_priv_means, dim=1)
            pred_priv_stds = torch.stack(pred_priv_stds, dim=1)
        else:
            pred_priv_means = None
            pred_priv_stds = None

        return pred_obs_means, pred_obs_stds, pred_priv_means, pred_priv_stds

    def get_hidden_from_history(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process history to obtain GRU hidden state (for policy rollout initialization).

        Args:
            obs_history: (batch, M, obs_size)
            action_history: (batch, M, action_size)

        Returns:
            hidden: (num_layers, batch, hidden_size)
        """
        batch_size = obs_history.shape[0]
        device = obs_history.device
        hidden = self.gru_base.init_hidden(batch_size, device)
        x_hist = torch.cat([obs_history, action_history], dim=-1)
        _, hidden = self.gru_base(x_hist, hidden)
        return hidden
