"""Robotic World Model (RWM) with dual-autoregressive mechanism.

Core architecture (Table S7):
  - Base: GRU with hidden shape [256, 256] (2 layers)
  - Heads: MLP with hidden [128], ReLU activation
  - Outputs: Gaussian mean + log_std for observations and privileged information

Training uses the autoregressive scheme described in Section 3.2 and Fig. S6:
  - Inner autoregression: GRU hidden states updated autoregressively over M historical steps
  - Outer autoregression: Predicted observations fed back into network over N forecast steps

Loss (Eq. 2): L = (1/N) * sum_{k=1}^{N} α^k [L_o(o'_{t+k}, o_{t+k}) + L_c(c'_{t+k}, c_{t+k})]
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import GaussianHead


class RoboticWorldModel(nn.Module):
    """GRU-based world model with dual-autoregressive mechanism."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        privileged_dim: int = 0,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 2,
        head_hidden_size: int = 128,
        head_activation: str = "relu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.privileged_dim = privileged_dim
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        input_dim = obs_dim + action_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
        )

        self.obs_head = GaussianHead(
            input_dim=gru_hidden_size,
            output_dim=obs_dim,
            hidden_dim=head_hidden_size,
            activation=head_activation,
        )

        self.privileged_head: Optional[GaussianHead] = None
        if privileged_dim > 0:
            self.privileged_head = GaussianHead(
                input_dim=gru_hidden_size,
                output_dim=privileged_dim,
                hidden_dim=head_hidden_size,
                activation=head_activation,
            )

    def _init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            self.gru_num_layers, batch_size, self.gru_hidden_size, device=device
        )

    def _gru_step(
        self, obs: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single step: concatenate observation and action, pass through GRU."""
        inp = torch.cat([obs, action], dim=-1).unsqueeze(1)  # (B, 1, D)
        out, new_hidden = self.gru(inp, hidden)
        return out.squeeze(1), new_hidden  # (B, H)

    def _predict_step(
        self, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Predict next observation (and privileged info) from hidden state."""
        obs_mean, obs_log_std = self.obs_head(hidden_state)
        privileged_out = None
        if self.privileged_head is not None:
            privileged_out = self.privileged_head(hidden_state)
        return obs_mean, obs_log_std, privileged_out

    def warmup(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Inner autoregression: process M historical steps sequentially.

        Args:
            observations: (B, M, obs_dim) — ground truth history
            actions: (B, M-1, action_dim) — actions for transitions between steps

        Returns:
            hidden: (num_layers, B, H) — final hidden state after processing history
        """
        batch_size = observations.shape[0]
        device = observations.device
        hidden = self._init_hidden(batch_size, device)

        for t in range(actions.shape[1]):
            _, hidden = self._gru_step(observations[:, t], actions[:, t], hidden)

        return hidden

    def forward_history(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Process M historical steps and predict next observation.

        Args:
            observations: (B, M, obs_dim)
            actions: (B, M-1, action_dim) — includes the action at step M-1 leading to step M

        Returns:
            obs_mean: (B, obs_dim)
            obs_log_std: (B, obs_dim)
            privileged_out: optional (mean, log_std) for privileged info
        """
        hidden = self.warmup(observations, actions)
        obs_mean, obs_log_std, privileged_out = self._predict_step(hidden)
        return obs_mean, obs_log_std, privileged_out

    def forecast_autoregressive(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        forecast_horizon: int,
    ) -> Dict[str, torch.Tensor]:
        """Outer autoregression: predict N steps into the future by feeding
        predictions back as input.

        Args:
            observations: (B, M, obs_dim) — historical observations
            actions: (B, M-1+N, action_dim) — actions for all transitions
                   (history + forecast)
            forecast_horizon: N, number of steps to predict

        Returns:
            dict with:
              - obs_means: (B, N, obs_dim)
              - obs_log_stds: (B, N, obs_dim)
              - priv_means: (B, N, priv_dim) if privileged head exists
              - priv_log_stds: (B, N, priv_dim) if privileged head exists
              - predicted_obs: (B, N, obs_dim) — sampled predictions (for rollout)
        """
        batch_size = observations.shape[0]
        device = observations.device
        M = observations.shape[1]

        hidden = self._init_hidden(batch_size, device)

        # Inner autoregression: process all M steps sequentially
        for t in range(M - 1):
            _, hidden = self._gru_step(observations[:, t], actions[:, t], hidden)

        # Last feed uses observation M-1 and action M-1
        last_obs = observations[:, -1]
        _, hidden = self._gru_step(last_obs, actions[:, M - 1], hidden)

        obs_means = []
        obs_log_stds = []
        priv_means = []
        priv_log_stds = []
        predicted_obs = []

        current_obs = observations[:, 0]  # dummy, overwritten

        for k in range(forecast_horizon):
            obs_mean, obs_log_std, priv_out = self._predict_step(hidden)
            obs_means.append(obs_mean)
            obs_log_stds.append(obs_log_std)

            if priv_out is not None:
                priv_means.append(priv_out[0])
                priv_log_stds.append(priv_out[1])

            # Sample prediction (reparameterization)
            std = torch.exp(obs_log_std)
            eps = torch.randn_like(obs_mean)
            sampled_obs = obs_mean + std * eps
            predicted_obs.append(sampled_obs)

            # Feed prediction back as next input
            action_idx = M - 1 + k + 1 if (M - 1 + k + 1) < actions.shape[1] else -1
            next_action = actions[:, M - 1 + k + 1] if (M - 1 + k + 1) < actions.shape[1] else actions[:, -1]

            _, hidden = self._gru_step(sampled_obs, next_action, hidden)

        result = {
            "obs_means": torch.stack(obs_means, dim=1),
            "obs_log_stds": torch.stack(obs_log_stds, dim=1),
            "predicted_obs": torch.stack(predicted_obs, dim=1),
        }

        if self.privileged_head is not None:
            result["priv_means"] = torch.stack(priv_means, dim=1)
            result["priv_log_stds"] = torch.stack(priv_log_stds, dim=1)

        return result

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        forecast_horizon: int,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: warmup + autoregressive forecast."""
        return self.forecast_autoregressive(observations, actions, forecast_horizon)


class RWMLoss(nn.Module):
    """Autoregressive training loss for RWM (Eq. 2).

    L = (1/N) * sum_{k=1}^{N} α^k [L_o(o'_{t+k}, o_{t+k}) + L_c(c'_{t+k}, c_{t+k})]

    Where L_o and L_c are Gaussian negative log-likelihood losses.
    """

    def __init__(self, forecast_decay: float = 1.0):
        super().__init__()
        self.forecast_decay = forecast_decay

    def gaussian_nll(
        self,
        pred_mean: torch.Tensor,
        pred_log_std: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood under Gaussian distribution with diagonal covariance."""
        var = torch.exp(2 * pred_log_std)
        nll = 0.5 * (
            torch.log(var + 1e-8) + ((target - pred_mean) ** 2) / (var + 1e-8)
        ).mean(dim=-1)
        return nll.mean()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets_obs: torch.Tensor,
        targets_priv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-step autoregressive loss.

        Args:
            predictions: from model.forecast_autoregressive()
            targets_obs: (B, N, obs_dim) — ground truth future observations
            targets_priv: (B, N, priv_dim) — ground truth privileged info

        Returns:
            total_loss: scalar tensor
            losses: dict with individual loss terms
        """
        N = predictions["obs_means"].shape[1]
        obs_loss = torch.tensor(0.0, device=predictions["obs_means"].device)
        priv_loss = torch.tensor(0.0, device=predictions["obs_means"].device)

        for k in range(N):
            decay_weight = self.forecast_decay ** k
            obs_loss_k = self.gaussian_nll(
                predictions["obs_means"][:, k],
                predictions["obs_log_stds"][:, k],
                targets_obs[:, k],
            )
            obs_loss = obs_loss + decay_weight * obs_loss_k

            if targets_priv is not None and "priv_means" in predictions:
                priv_loss_k = self.gaussian_nll(
                    predictions["priv_means"][:, k],
                    predictions["priv_log_stds"][:, k],
                    targets_priv[:, k],
                )
                priv_loss = priv_loss + decay_weight * priv_loss_k

        total_loss = (obs_loss + priv_loss) / N

        losses = {
            "obs_loss": obs_loss.item() / N,
            "priv_loss": priv_loss.item() / N if targets_priv is not None else 0.0,
            "total_loss": total_loss.item(),
        }

        return total_loss, losses
