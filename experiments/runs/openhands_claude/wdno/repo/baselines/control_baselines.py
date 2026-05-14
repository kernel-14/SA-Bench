"""
Control baselines: PID, SAC, BC, BPPO, SL (Supervised Learning).

References:
  - PID: Li et al. (2006), ANN-PID from Slama et al. (2019), Ding et al. (2022)
  - SAC: Haarnoja et al. (2018)
  - BC: Pomerleau (1988)
  - BPPO: Zhuang et al. (2023)
  - SL: Hwang et al. (2022)

Hyperparameters from Appendix I.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ANN-PID Controller (Appendix I.1)
# ---------------------------------------------------------------------------

class ANNPIDController(nn.Module):
    """
    Neural network-based PID parameter planner for MIMO control.

    Architecture (Table 21):
      - 2 Conv1d layers + 2 FC layers
      - Activation: Softsign
      - Kernel size: 3, padding: 1, stride: 1

    The network takes the error signal and outputs PID parameters (Kp, Ki, Kd)
    for each spatial point, enabling MIMO PID control.
    """

    def __init__(self, nx: int = 120):
        super().__init__()
        self.nx = nx

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, stride=1),
            nn.Softsign(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.Softsign(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * nx, 256),
            nn.Softsign(),
            nn.Linear(256, nx * 3),  # Kp, Ki, Kd for each spatial point
        )

    def forward(self, error: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            error: [B, X] current error (u_current - u_target)
        Returns:
            Kp, Ki, Kd: each [B, X]
        """
        B = error.shape[0]
        h = self.conv_layers(error.unsqueeze(1))  # [B, 64, X]
        h = h.reshape(B, -1)
        params = self.fc_layers(h).reshape(B, 3, self.nx)
        Kp = params[:, 0]
        Ki = params[:, 1]
        Kd = params[:, 2]
        return Kp, Ki, Kd

    def compute_control(
        self,
        error: torch.Tensor,
        integral: torch.Tensor,
        derivative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PID control output.

        Args:
            error: [B, X] current error
            integral: [B, X] accumulated error
            derivative: [B, X] error derivative
        Returns:
            f: [B, X] control force
        """
        Kp, Ki, Kd = self.forward(error)
        return Kp * error + Ki * integral + Kd * derivative


# ---------------------------------------------------------------------------
# SAC (Soft Actor-Critic) (Appendix I.2)
# ---------------------------------------------------------------------------

class SACPolicy(nn.Module):
    """
    SAC policy network for PDE control.

    Outputs mean and log_std of Gaussian policy.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-20, 2)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)


class SACQNetwork(nn.Module):
    """Twin Q-networks for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class SAC:
    """
    Soft Actor-Critic for PDE control.

    Hyperparameters (Table 22):
      - discount: 0.5
      - target_smoothing: 0.05
      - lr_critic: 3e-4
      - lr_entropy: 3e-3
      - lr_policy: 3e-3
      - batch_size: 8192
      - n_episodes: 1500
      - updates_per_step: 50
      - target_updates_per_step: 15
      - replay_buffer_size: 1e6
      - energy_weight: 2e-5
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        discount: float = 0.5,
        tau: float = 0.05,
        lr_critic: float = 3e-4,
        lr_entropy: float = 3e-3,
        lr_policy: float = 3e-3,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.discount = discount
        self.tau = tau

        self.policy = SACPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = SACQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = SACQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_entropy)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, _ = self.policy.sample(state.to(self.device))
        return action

    def update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            next_action, next_log_pi = self.policy.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = reward + (1 - done) * self.discount * q_next

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_pi = self.policy.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_pi - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }


# ---------------------------------------------------------------------------
# Behavior Cloning (BC) (Appendix I.5)
# ---------------------------------------------------------------------------

class BCPolicy(nn.Module):
    """
    Behavior Cloning policy network.

    Hyperparameters (Table 25):
      - n_layers: 2
      - hidden_dim: 1024
      - activation: ReLU
      - lr: 1e-4
      - batch_size: 512
      - episodes: 5e5
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 1024, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def compute_loss(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pred = self.forward(state)
        return F.mse_loss(pred, action)


# ---------------------------------------------------------------------------
# BPPO (Behavior Proximal Policy Optimization) (Appendix I.4)
# ---------------------------------------------------------------------------

class BPPOPolicy(nn.Module):
    """
    BPPO policy network.

    Hyperparameters (Table 24):
      - n_layers: 2
      - hidden_dim: 1024
      - activation: ReLU
      - lr: 1e-5
      - batch_size: 512
      - clip_ratio: 0.25
      - weight_decay: 0.96
      - advantage_weight: 0.9
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 1024, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(state)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(-1)

    def compute_ppo_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        advantage: torch.Tensor,
        old_log_prob: torch.Tensor,
        clip_ratio: float = 0.25,
    ) -> torch.Tensor:
        log_prob = self.get_log_prob(state, action)
        ratio = (log_prob - old_log_prob).exp()
        clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        loss = -torch.min(ratio * advantage, clipped * advantage).mean()
        return loss


# ---------------------------------------------------------------------------
# Supervised Learning (SL) (Appendix I.3)
# ---------------------------------------------------------------------------

class SLController:
    """
    Supervised Learning controller (Hwang et al. 2022).

    Uses a neural surrogate model to optimize control via backpropagation.
    The control force f is treated as a learnable parameter and optimized
    using L-BFGS.

    Hyperparameters (Table 23):
      - lr: 0.1
      - n_epochs: 100
      - weight_state_loss: 1.0
      - weight_recon_loss: 0.01
      - lbfgs_tol: 1e-5
    """

    def __init__(
        self,
        surrogate_model: nn.Module,
        lr: float = 0.1,
        n_epochs: int = 100,
        weight_state: float = 1.0,
        weight_recon: float = 0.01,
    ):
        self.surrogate = surrogate_model
        self.lr = lr
        self.n_epochs = n_epochs
        self.weight_state = weight_state
        self.weight_recon = weight_recon

    def optimize(
        self,
        u0: torch.Tensor,
        u_target: torch.Tensor,
        f_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Optimize control force f to minimize control objective.

        Args:
            u0: initial condition [B, X]
            u_target: target state [B, X]
            f_init: initial guess for f [B, T, X]
        Returns:
            f_opt: optimized control force [B, T, X]
        """
        B, X = u0.shape
        T = f_init.shape[1] if f_init is not None else 80

        if f_init is None:
            f = torch.zeros(B, T, X, device=u0.device, requires_grad=True)
        else:
            f = f_init.clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [f],
            lr=self.lr,
            max_iter=self.n_epochs,
            tolerance_grad=1e-5,
            tolerance_change=1e-5,
        )

        def closure():
            optimizer.zero_grad()
            # Predict state using surrogate
            u_pred = self.surrogate(u0, f)
            u_T_pred = u_pred[:, -1]  # final state

            # State loss
            state_loss = self.weight_state * F.mse_loss(u_T_pred, u_target)

            # Reconstruction loss (regularization)
            recon_loss = self.weight_recon * torch.mean(f ** 2)

            loss = state_loss + recon_loss
            loss.backward()
            return loss

        optimizer.step(closure)
        return f.detach()
