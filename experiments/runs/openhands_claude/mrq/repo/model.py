"""
MR.Q – Model-based Representations for Q-learning
Neural network components as described in Fujimoto et al. (2024).

Architecture summary
--------------------
  f_ω  : s          → z_s          (state encoder)
  g_ω  : (z_s, a)   → z_sa         (state-action encoder)
  m    : z_sa        → (z̃_s', r̃, d̃) (linear MDP predictor, part of g_ω)
  Q_θ  : z_sa        → ℝ            (value network, ×2)
  π_φ  : z_s         → a            (policy network)
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    build_reward_bins,
    reward_cross_entropy,
    two_hot_decode,
    huber_loss,
    xavier_uniform_init,
)
from config import MRQConfig


# ── Shared layer-norm + activation helper ────────────────────────────────────

class LNActiv(nn.Module):
    """LayerNorm followed by an activation function."""

    def __init__(self, dim: int, activ: nn.Module) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.activ = activ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activ(self.ln(x))


# ── State encoder f_ω ────────────────────────────────────────────────────────

class StateEncoderCNN(nn.Module):
    """
    Convolutional state encoder for image observations (84×84).

    Four conv layers: 32 channels, kernel 3, strides [2, 2, 2, 1].
    Flattened output (1568) → Linear → LayerNorm + ELU → z_s (512).
    """

    def __init__(self, in_channels: int, zs_dim: int = 512) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1)
        # 84×84 → 41×41 → 20×20 → 9×9 → 7×7  ⟹  32×7×7 = 1568
        self.linear = nn.Linear(1568, zs_dim)
        self.ln_activ = LNActiv(zs_dim, nn.ELU())
        self.apply(xavier_uniform_init)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state.float() / 255.0 - 0.5
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.reshape(x.size(0), -1)
        return self.ln_activ(self.linear(x))


class StateEncoderMLP(nn.Module):
    """
    Three-layer MLP state encoder for vector observations.
    Each layer: Linear → LayerNorm + ELU.
    """

    def __init__(self, state_dim: int, zs_dim: int = 512, hidden_dim: int = 512) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, zs_dim)
        self.ln1 = LNActiv(hidden_dim, nn.ELU())
        self.ln2 = LNActiv(hidden_dim, nn.ELU())
        self.ln3 = LNActiv(zs_dim, nn.ELU())
        self.apply(xavier_uniform_init)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.ln1(self.l1(state))
        x = self.ln2(self.l2(x))
        return self.ln3(self.l3(x))


# ── State-action encoder g_ω + linear MDP predictor m ────────────────────────

class StateActionEncoder(nn.Module):
    """
    State-action encoder g_ω and linear MDP predictor m.

    Architecture:
      za  = ELU(Linear(action))                          [za_dim]
      zsa = concat([z_s, za])
      zsa = LN+ELU(Linear(zsa))  ×2  then  Linear(zsa)  [zsa_dim]
      out = Linear(zsa)                                  [zs_dim + reward_bins + 1]

    Returns (model_output, zsa) where model_output = [z̃_s', r̃_logits, d̃].
    """

    def __init__(
        self,
        action_dim: int,
        zs_dim: int = 512,
        za_dim: int = 256,
        zsa_dim: int = 512,
        hidden_dim: int = 512,
        reward_bins: int = 65,
    ) -> None:
        super().__init__()
        self.zs_dim = zs_dim
        self.reward_bins = reward_bins

        # Action embedding
        self.za = nn.Linear(action_dim, za_dim)

        # State-action MLP
        self.zsa1 = nn.Linear(zs_dim + za_dim, hidden_dim)
        self.zsa2 = nn.Linear(hidden_dim, hidden_dim)
        self.zsa3 = nn.Linear(hidden_dim, zsa_dim)

        # Linear MDP predictor: z_sa → (z̃_s', r̃_logits, d̃)
        output_dim = zs_dim + reward_bins + 1
        self.model = nn.Linear(zsa_dim, output_dim)

        self.ln1 = LNActiv(hidden_dim, nn.ELU())
        self.ln2 = LNActiv(hidden_dim, nn.ELU())

        self.apply(xavier_uniform_init)

    def forward(
        self, zs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            model_out: (B, zs_dim + reward_bins + 1)
            zsa:       (B, zsa_dim)
        """
        za = F.elu(self.za(action))
        zsa = torch.cat([zs, za], dim=-1)
        zsa = self.ln1(self.zsa1(zsa))
        zsa = self.ln2(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        model_out = self.model(zsa)
        return model_out, zsa

    def split_model_output(
        self, model_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split model output into (z̃_s', r̃_logits, d̃)."""
        zs_pred = model_out[:, : self.zs_dim]
        r_logits = model_out[:, self.zs_dim : self.zs_dim + self.reward_bins]
        d_pred = model_out[:, -1:]
        return zs_pred, r_logits, d_pred


# ── Value network Q_θ ─────────────────────────────────────────────────────────

class ValueNetwork(nn.Module):
    """
    Four-layer MLP value network.
    Layers 1–3: Linear → LayerNorm + ELU.
    Layer 4:    Linear → scalar.
    """

    def __init__(self, zsa_dim: int = 512, hidden_dim: int = 512) -> None:
        super().__init__()
        self.l1 = nn.Linear(zsa_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)
        self.ln1 = LNActiv(hidden_dim, nn.ELU())
        self.ln2 = LNActiv(hidden_dim, nn.ELU())
        self.ln3 = LNActiv(hidden_dim, nn.ELU())
        self.apply(xavier_uniform_init)

    def forward(self, zsa: torch.Tensor) -> torch.Tensor:
        q = self.ln1(self.l1(zsa))
        q = self.ln2(self.l2(q))
        q = self.ln3(self.l3(q))
        return self.l4(q)


# ── Policy network π_φ ────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    Three-layer MLP policy network.
    Layers 1–2: Linear → LayerNorm + ReLU.
    Layer 3:    Linear → action (Tanh for continuous, Gumbel-Softmax for discrete).
    """

    def __init__(
        self,
        zs_dim: int = 512,
        action_dim: int = 1,
        hidden_dim: int = 512,
        discrete: bool = False,
        gumbel_tau: float = 10.0,
    ) -> None:
        super().__init__()
        self.discrete = discrete
        self.gumbel_tau = gumbel_tau

        self.l1 = nn.Linear(zs_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.ln1 = LNActiv(hidden_dim, nn.ReLU())
        self.ln2 = LNActiv(hidden_dim, nn.ReLU())
        self.apply(xavier_uniform_init)

    def forward(self, zs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action:       (B, action_dim) – activated output.
            pre_activ:    (B, action_dim) – raw logits before final activation.
        """
        a = self.ln1(self.l1(zs))
        a = self.ln2(self.l2(a))
        pre_activ = self.l3(a)

        if self.discrete:
            action = F.gumbel_softmax(pre_activ, tau=self.gumbel_tau, hard=False)
        else:
            action = torch.tanh(pre_activ)

        return action, pre_activ


# ── Full MR.Q agent ───────────────────────────────────────────────────────────

class MRQAgent(nn.Module):
    """
    MR.Q agent encapsulating all networks and update logic.

    Networks
    --------
    encoder (f_ω + g_ω + m):  state/state-action encoder + MDP predictor
    Q1, Q2:                    twin value networks
    pi:                        policy network
    encoder_target:            target encoder (ω')
    Q1_target, Q2_target:      target value networks (θ')
    pi_target:                 target policy (φ')
    """

    def __init__(self, cfg: MRQConfig, state_dim_or_channels: int, action_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        self.discrete = cfg.action_type == "discrete"
        self.image_obs = cfg.obs_type == "image"

        # ── Encoder ──────────────────────────────────────────────────────────
        if self.image_obs:
            self.state_enc = StateEncoderCNN(state_dim_or_channels, cfg.zs_dim)
            self.state_enc_target = StateEncoderCNN(state_dim_or_channels, cfg.zs_dim)
        else:
            self.state_enc = StateEncoderMLP(state_dim_or_channels, cfg.zs_dim, cfg.hidden_dim)
            self.state_enc_target = StateEncoderMLP(
                state_dim_or_channels, cfg.zs_dim, cfg.hidden_dim
            )

        # For discrete actions the policy outputs a one-hot vector of size action_dim,
        # so the action input to g_ω is also action_dim-dimensional.
        self.sa_enc = StateActionEncoder(
            action_dim=action_dim,
            zs_dim=cfg.zs_dim,
            za_dim=cfg.za_dim,
            zsa_dim=cfg.zsa_dim,
            hidden_dim=cfg.hidden_dim,
            reward_bins=cfg.reward_bins,
        )
        self.sa_enc_target = StateActionEncoder(
            action_dim=action_dim,
            zs_dim=cfg.zs_dim,
            za_dim=cfg.za_dim,
            zsa_dim=cfg.zsa_dim,
            hidden_dim=cfg.hidden_dim,
            reward_bins=cfg.reward_bins,
        )

        # ── Value networks ────────────────────────────────────────────────────
        self.Q1 = ValueNetwork(cfg.zsa_dim, cfg.hidden_dim)
        self.Q2 = ValueNetwork(cfg.zsa_dim, cfg.hidden_dim)
        self.Q1_target = ValueNetwork(cfg.zsa_dim, cfg.hidden_dim)
        self.Q2_target = ValueNetwork(cfg.zsa_dim, cfg.hidden_dim)

        # ── Policy networks ───────────────────────────────────────────────────
        self.pi = PolicyNetwork(
            cfg.zs_dim, action_dim, cfg.hidden_dim, self.discrete, cfg.gumbel_tau
        )
        self.pi_target = PolicyNetwork(
            cfg.zs_dim, action_dim, cfg.hidden_dim, self.discrete, cfg.gumbel_tau
        )

        # ── Reward bins (fixed) ───────────────────────────────────────────────
        # Registered as buffer so they move with .to(device)
        bins = build_reward_bins(cfg.reward_bins, cfg.reward_range, device=torch.device("cpu"))
        self.register_buffer("reward_bins_tensor", bins)

        # ── Reward scaling ────────────────────────────────────────────────────
        self.register_buffer("mean_abs_reward", torch.tensor(1.0))
        self.register_buffer("mean_abs_reward_target", torch.tensor(1.0))

        # ── Terminal loss gate ────────────────────────────────────────────────
        # Set to False until the first terminal transition is observed
        self.terminal_loss_active: bool = False

        # Hard-copy parameters to target networks
        self._hard_update_targets()

    # ── Target network management ─────────────────────────────────────────────

    def _hard_update_targets(self) -> None:
        """θ' ← θ, φ' ← φ, ω' ← ω."""
        for src, tgt in [
            (self.state_enc, self.state_enc_target),
            (self.sa_enc, self.sa_enc_target),
            (self.Q1, self.Q1_target),
            (self.Q2, self.Q2_target),
            (self.pi, self.pi_target),
        ]:
            tgt.load_state_dict(src.state_dict())

    def update_targets(self, mean_abs_reward: float) -> None:
        """
        Periodic synchronisation (called every T_target steps).
        Also updates reward scaling: r̄' ← r̄, r̄ ← new_mean_abs_reward.
        """
        self._hard_update_targets()
        self.mean_abs_reward_target.copy_(self.mean_abs_reward)
        self.mean_abs_reward.fill_(max(mean_abs_reward, 1e-8))

    # ── Encoding helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """z_s = f_ω(s)  (no gradient)."""
        return self.state_enc(state)

    def encode_state_action(
        self, zs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(model_out, z_sa) = g_ω(z_s, a)."""
        return self.sa_enc(zs, action)

    @torch.no_grad()
    def encode_state_target(self, state: torch.Tensor) -> torch.Tensor:
        """z̄_s' = f_{ω'}(s')  (target encoder, no gradient)."""
        return self.state_enc_target(state)

    # ── Action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        state: torch.Tensor,
        explore: bool = False,
        expl_noise_std: float = 0.2,
    ) -> torch.Tensor:
        """
        Select an action given a state observation.

        For continuous actions: tanh output + optional Gaussian noise, clipped to [-1, 1].
        For discrete actions:   argmax of (Gumbel-Softmax output + optional noise).
        """
        zs = self.state_enc(state)
        action, _ = self.pi(zs)

        if explore:
            noise = torch.randn_like(action) * expl_noise_std
            if self.discrete:
                action = action + noise
                action = F.one_hot(action.argmax(dim=-1), self.action_dim).float()
            else:
                action = (action + noise).clamp(-1.0, 1.0)

        return action

    # ── Encoder update ────────────────────────────────────────────────────────

    def encoder_loss(
        self,
        states: torch.Tensor,       # (B, *obs_shape)
        actions: torch.Tensor,      # (B, H_Enc, action_dim)
        rewards: torch.Tensor,      # (B, H_Enc)
        dones: torch.Tensor,        # (B, H_Enc)  1 = terminal
        next_states: torch.Tensor,  # (B, H_Enc, *obs_shape)
    ) -> torch.Tensor:
        """
        Unrolled encoder loss over H_Enc steps (Equation 14).

        L_Enc = Σ_{t=1}^{H_Enc} [λ_R * L_Reward(r̃^t)
                                 + λ_D * L_Dynamics(z̃_s'^t)
                                 + λ_T * L_Terminal(d̃^t)]

        Gradients propagate through the entire unroll (z̃^t feeds into step t+1).
        """
        cfg = self.cfg
        B, H = actions.shape[:2]

        # Initial state embedding – gradients flow through f_ω
        zs_pred = self.state_enc(states)  # (B, zs_dim)

        total_loss = torch.tensor(0.0, device=states.device)

        for t in range(H):
            a_t = actions[:, t]                    # (B, action_dim)
            r_t = rewards[:, t]                    # (B,)
            d_t = dones[:, t]                      # (B,)
            s_next_t = next_states[:, t]           # (B, *obs_shape)

            # Forward through state-action encoder + linear predictor
            # Gradients flow through zs_pred → g_ω → m
            model_out, _ = self.sa_enc(zs_pred, a_t)
            zs_next_pred, r_logits, d_pred = self.sa_enc.split_model_output(model_out)

            # Target next-state embedding from target encoder (no gradient)
            with torch.no_grad():
                zs_next_target = self.state_enc_target(s_next_t)  # (B, zs_dim)

            # Reward loss (cross-entropy with two-hot)
            r_loss = reward_cross_entropy(r_logits, r_t, self.reward_bins_tensor)

            # Dynamics loss (MSE)
            dyn_loss = F.mse_loss(zs_next_pred, zs_next_target)

            # Terminal loss (MSE) – gated until first terminal is seen
            if self.terminal_loss_active:
                term_loss = F.mse_loss(d_pred.squeeze(-1), d_t)
            else:
                term_loss = torch.tensor(0.0, device=states.device)

            total_loss = (
                total_loss
                + cfg.lambda_reward * r_loss
                + cfg.lambda_dynamics * dyn_loss
                + cfg.lambda_terminal * term_loss
            )

            # Predicted next-state embedding feeds into the next unroll step.
            # Gradients propagate through the full unroll chain.
            zs_pred = zs_next_pred

        return total_loss

    # ── Value update ──────────────────────────────────────────────────────────

    def value_loss(
        self,
        states: torch.Tensor,           # (B, *obs_shape)
        actions: torch.Tensor,          # (B, action_dim)  – action at t=0
        multi_rewards: torch.Tensor,    # (B, H_Q)
        multi_dones: torch.Tensor,      # (B, H_Q)
        next_states_hq: torch.Tensor,   # (B, *obs_shape)  – state at t=H_Q
        priorities: Optional[torch.Tensor] = None,  # (B,) for LAP weighting
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TD3-style value loss with multi-step returns (Equation 19).

        Multi-step return with done masking:
            y = (1/r̄) * (Σ_{t=0}^{H_Q-1} γ^t * not_done_t * r_t
                          + γ^{H_Q} * not_done_{H_Q-1} * r̄' * Q'_min)

        Returns (loss, td_errors) where td_errors are used to update LAP priorities.
        """
        cfg = self.cfg
        B = states.shape[0]
        r_bar = self.mean_abs_reward.item()
        r_bar_prime = self.mean_abs_reward_target.item()

        # ── Compute target value ──────────────────────────────────────────────
        with torch.no_grad():
            # Target policy action at s_{H_Q}
            zs_hq = self.state_enc_target(next_states_hq)
            a_pi_target, _ = self.pi_target(zs_hq)

            # Add clipped Gaussian noise to target action
            noise = (
                torch.randn_like(a_pi_target) * cfg.target_noise_std
            ).clamp(-cfg.target_noise_clip, cfg.target_noise_clip)

            if self.discrete:
                a_pi_target = a_pi_target + noise
                a_pi_target = F.one_hot(
                    a_pi_target.argmax(dim=-1), self.action_dim
                ).float()
            else:
                a_pi_target = (a_pi_target + noise).clamp(-1.0, 1.0)

            # Target Q value (min of two target networks)
            _, zsa_hq = self.sa_enc_target(zs_hq, a_pi_target)
            q1_target = self.Q1_target(zsa_hq)
            q2_target = self.Q2_target(zsa_hq)
            q_target_min = torch.min(q1_target, q2_target)  # (B, 1)

            # Multi-step discounted return with done masking.
            # not_done tracks whether the episode is still running.
            discounted_return = torch.zeros(B, 1, device=states.device)
            running_discount = 1.0
            not_done = torch.ones(B, 1, device=states.device)

            for t in range(cfg.q_horizon):
                r_t = multi_rewards[:, t : t + 1]
                d_t = multi_dones[:, t : t + 1]
                discounted_return = discounted_return + running_discount * not_done * r_t
                not_done = not_done * (1.0 - d_t)
                running_discount *= cfg.discount

            # Bootstrap only if episode has not terminated
            bootstrap = running_discount * not_done * r_bar_prime * q_target_min
            y = (discounted_return + bootstrap) / max(r_bar, 1e-8)  # (B, 1)

        # ── Current Q predictions ─────────────────────────────────────────────
        # Encoder is fixed during value update; stop gradients.
        with torch.no_grad():
            zs = self.state_enc(states)
            _, zsa = self.sa_enc(zs, actions)

        q1 = self.Q1(zsa)  # (B, 1)
        q2 = self.Q2(zsa)  # (B, 1)

        # TD errors for LAP priority update
        td_errors = (0.5 * (torch.abs(q1 - y) + torch.abs(q2 - y))).squeeze(-1).detach()

        # Huber loss (element-wise)
        loss1 = huber_loss(q1, y)  # (B, 1)
        loss2 = huber_loss(q2, y)  # (B, 1)
        loss = (loss1 + loss2).squeeze(-1).mean()  # scalar

        return loss, td_errors

    # ── Policy update ─────────────────────────────────────────────────────────

    def policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Deterministic policy gradient loss (Equation 20).

        L_Policy = -0.5 * Σ_{i=1,2} Q̃_i(z_sa_π) + λ_pre-activ * ||z_π||²

        Gradients flow: Q_i → z_sa_π → a_π → π_φ.
        The encoder (f_ω, g_ω) parameters are not in policy_opt so they are
        not updated, but gradients do flow through g_ω to reach a_π.
        """
        cfg = self.cfg

        # State embedding – encoder fixed, no gradient to f_ω
        with torch.no_grad():
            zs = self.state_enc(states)

        # Policy action – gradients flow through π_φ
        a_pi, pre_activ = self.pi(zs)

        # State-action embedding – gradients flow through a_pi → π_φ
        # (g_ω parameters receive gradients but are not updated by policy_opt)
        _, zsa_pi = self.sa_enc(zs, a_pi)

        q1 = self.Q1(zsa_pi)
        q2 = self.Q2(zsa_pi)

        policy_grad = -0.5 * (q1 + q2).mean()
        reg = cfg.lambda_pre_activ * (pre_activ ** 2).mean()

        return policy_grad + reg
