"""MR.Q Agent: Model-based Representations for Q-learning.

Implementation of the algorithm described in:
"Towards General-Purpose Model-Free Reinforcement Learning" (Fujimoto et al., 2024).
"""

import copy
import numpy as np
from typing import Tuple

import torch
import torch.nn.functional as F

from mrq.config import MRQConfig
from mrq.networks import StateEncoder, StateActionEncoder, ValueNetwork, PolicyNetwork
from mrq.utils import two_hot_encode, clip_action, RewardScaler
from mrq.replay_buffer import ReplayBuffer


class MRQAgent:
    """MR.Q: model-based representations for Q-learning.

    Architecture:
        f_ω: s → z_s                          (state encoder)
        g_ω: (z_s, a) → (preds, z_sa)         (state-action encoder + MDP predictor)
        Q_θ: z_sa → R                          (twin value functions)
        π_φ: z_s → a                           (policy)

    Losses:
        Encoder (Eq 14): unrolled dynamics + reward + terminal prediction
        Value (Eq 19): multi-step return, clipped double Q, Huber loss
        Policy (Eq 20): DPG + pre-activation regularization
    """

    def __init__(
        self, config: MRQConfig, state_dim: int, action_dim: int,
        action_space_low=None, action_space_high=None, device: str = "cuda",
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.action_low = action_space_low
        self.action_high = action_space_high
        if action_space_low is not None:
            self.action_range = action_space_high - action_space_low
        else:
            self.action_range = np.ones(action_dim) * 2.0

        # ---- Networks ----
        self.state_encoder = StateEncoder(config, state_dim).to(self.device)
        self.state_action_encoder = StateActionEncoder(config, action_dim).to(self.device)
        self.q1 = ValueNetwork(config).to(self.device)
        self.q2 = ValueNetwork(config).to(self.device)
        self.policy_net = PolicyNetwork(config, action_dim).to(self.device)

        # ---- Target networks (hard-copied every T_target steps) ----
        self.state_encoder_target = copy.deepcopy(self.state_encoder)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.policy_target = copy.deepcopy(self.policy_net)
        for net in [self.state_encoder_target, self.q1_target, self.q2_target, self.policy_target]:
            for p in net.parameters():
                p.requires_grad_(False)

        # ---- Optimizers ----
        enc_params = list(self.state_encoder.parameters()) + list(self.state_action_encoder.parameters())
        self.encoder_optim = torch.optim.AdamW(enc_params, lr=config.encoder_lr, weight_decay=config.encoder_weight_decay)
        self.value_optim = torch.optim.AdamW(list(self.q1.parameters()) + list(self.q2.parameters()), lr=config.value_lr)
        self.policy_optim = torch.optim.AdamW(self.policy_net.parameters(), lr=config.policy_lr)

        # ---- Reward scaling (Eq 19) ----
        self.reward_scaler = RewardScaler()
        self.target_reward_scale = 1.0  # r̄', updated with target networks
        self.total_it = 0

    # ==================================================================
    # Action selection
    # ==================================================================

    def select_action(
        self, state: np.ndarray, deterministic: bool = False, explore: bool = False,
    ) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            zs = self.state_encoder(s)
            action, _ = self.policy_net(zs, deterministic=deterministic)
            action = action.squeeze(0).cpu().numpy()

        if explore and not deterministic:
            noise = np.random.randn(self.action_dim).astype(np.float32) * self.config.exploration_noise
            if self.config.discrete_actions:
                action = action + noise
                action = self._to_one_hot(action)
            else:
                noise = noise * self.action_range * 0.5
                action = action + noise
                if self.action_low is not None:
                    action = np.clip(action, self.action_low, self.action_high)
                else:
                    action = np.clip(action, -1.0, 1.0)
        return action

    def _to_one_hot(self, action: np.ndarray) -> np.ndarray:
        a_idx = np.argmax(action)
        one_hot = np.zeros(self.action_dim, dtype=np.float32)
        one_hot[a_idx] = 1.0
        return one_hot

    # ==================================================================
    # Encoder loss (Eq 13–17)
    # ==================================================================

    def _encoder_loss(self, buffer: ReplayBuffer) -> Tuple[torch.Tensor, dict]:
        """
        Encoder loss with unrolled dynamics over H_Enc steps.

        Unrolls the model from s_0 using actual actions a_0, ..., a_{H_Enc-1}.
        At each step t, predicts z_s'_{t+1}, r_{t+1}, d_{t+1} and compares
        against target encoder output and ground truth.
        """
        cfg = self.config
        result = buffer.sample_subsequences(cfg.batch_size, cfg.h_enc + 1)
        if result is None:
            states, actions, rewards, dones, next_states, _ = buffer.sample(cfg.batch_size)
            states = states.unsqueeze(1)
            actions = actions.unsqueeze(1)
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1)
            next_states = next_states.unsqueeze(1)
            h_actual = 1
        else:
            states, actions, rewards, dones, next_states = result
            h_actual = cfg.h_enc

        # Encode initial state
        zs_current = self.state_encoder(states[:, 0])
        with torch.no_grad():
            zs_target = self.state_encoder_target(next_states[:, 0])

        total_dyn = 0.0
        total_rew = 0.0
        total_term = 0.0

        for t in range(h_actual):
            preds, _ = self.state_action_encoder(zs_current, actions[:, t])
            zs_pred = preds[:, :cfg.zs_dim]
            r_logits = preds[:, cfg.zs_dim:cfg.zs_dim + cfg.reward_bins]
            d_pred = preds[:, cfg.zs_dim + cfg.reward_bins:]

            # Dynamics loss: MSE with target encoder (Eq 16)
            total_dyn = total_dyn + F.mse_loss(zs_pred, zs_target)

            # Reward loss: categorical CE with two-hot encoding (Eq 15)
            two_hot = two_hot_encode(
                rewards[:, t].squeeze(-1), cfg.reward_bins, cfg.reward_range[0], cfg.reward_range[1],
            )
            total_rew = total_rew + (-(two_hot * F.log_softmax(r_logits, dim=-1)).sum(dim=-1).mean())

            # Terminal loss: MSE (Eq 17); λ_terminal=0 until first terminal seen
            if buffer.has_seen_terminal:
                total_term = total_term + F.mse_loss(d_pred, dones[:, t])

            # Detach for next unroll step
            zs_current = zs_pred.detach()
            if t + 1 < h_actual:
                with torch.no_grad():
                    zs_target = self.state_encoder_target(next_states[:, t + 1])

        loss = cfg.lambda_dynamics * total_dyn + cfg.lambda_reward * total_rew + cfg.lambda_terminal * total_term
        info = {
            "enc_dyn": total_dyn / h_actual,
            "enc_rew": total_rew / h_actual,
            "enc_term": total_term / h_actual if buffer.has_seen_terminal else 0.0,
        }
        return loss, info

    # ==================================================================
    # Value loss (Eq 18–19)
    # ==================================================================

    def _value_loss(self, buffer: ReplayBuffer) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        Multi-step (H_Q=3) clipped double Q-learning with Huber loss.

        Reward scaling: target = n_step_return / r̄
        where Q_j' = r̄' * min(Q1', Q2') uses the target reward scale.
        """
        cfg = self.config

        # Try multi-step subsequence; fall back to LAP-sampled transitions
        result = buffer.sample_subsequences(cfg.batch_size, cfg.h_q + 1)
        if result is not None:
            states_seq, actions_seq, rewards_seq, dones_seq, next_states_seq = result
            s_0 = states_seq[:, 0]
            a_0 = actions_seq[:, 0]
            s_H = states_seq[:, cfg.h_q]  # s after H_Q actions
        else:
            # Fallback: single-step from LAP sampling
            states, actions, rewards, dones, next_states, indices = buffer.sample(cfg.batch_size)
            s_0 = states
            a_0 = actions
            s_H = next_states
            rewards_seq = rewards.unsqueeze(1)
            dones_seq = dones.unsqueeze(1)
            result = None

        # Current Q values
        zs = self.state_encoder(s_0)
        _, zsa = self.state_action_encoder(zs, a_0)
        q1 = self.q1(zsa)
        q2 = self.q2(zsa)

        with torch.no_grad():
            # Target action with noise (Eq 18 / TD3 style)
            zs_H = self.state_encoder_target(s_H)
            a_H, _ = self.policy_target(zs_H)
            if cfg.discrete_actions:
                noise = torch.randn_like(a_H) * cfg.target_noise_std
                a_H = a_H + noise
            else:
                noise = torch.clamp(
                    torch.randn_like(a_H) * cfg.target_noise_std,
                    -cfg.target_noise_clip, cfg.target_noise_clip,
                )
                a_H = clip_action(a_H + noise, -1.0, 1.0)
            _, zsa_H = self.state_action_encoder(zs_H, a_H)

            # Clipped double Q target, scaled by target reward scale
            tq1 = self.q1_target(zsa_H)
            tq2 = self.q2_target(zsa_H)
            tq_min = torch.min(tq1, tq2)
            tq_scaled = tq_min * self.target_reward_scale  # r̄' * Q_j'

            # Multi-step return
            if result is not None:
                n_step_return = self._n_step_return(rewards_seq, dones_seq, tq_scaled, cfg.h_q)
            else:
                n_step_return = rewards_seq[:, 0] + cfg.discount * (1 - dones_seq[:, 0]) * tq_scaled

            # Scale by current reward scale
            r_scale = self.reward_scaler.get_scale()
            target = n_step_return / r_scale

        # Huber loss (LAP uses this to eliminate bias from prioritized sampling)
        td1 = q1 - target
        td2 = q2 - target
        loss = (
            F.huber_loss(q1, target, delta=cfg.huber_delta, reduction="none") +
            F.huber_loss(q2, target, delta=cfg.huber_delta, reduction="none")
        ).mean()

        td_errors = ((td1.abs() + td2.abs()) / 2.0).squeeze(-1).cpu().numpy()

        # Return indices from LAP sampling for priority update
        if result is None:
            return loss, td_errors, indices
        else:
            _, _, _, _, _, indices = buffer.sample(cfg.batch_size)
            return loss, td_errors, indices[:cfg.batch_size]

    def _n_step_return(self, rewards: torch.Tensor, dones: torch.Tensor,
                       final_q: torch.Tensor, n: int) -> torch.Tensor:
        """Compute n-step discounted return recursively."""
        ret = final_q
        for t in reversed(range(n)):
            ret = rewards[:, t] + self.config.discount * (1 - dones[:, t]) * ret
        return ret

    # ==================================================================
    # Policy loss (Eq 20)
    # ==================================================================

    def _policy_loss(self, states: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        cfg = self.config
        zs = self.state_encoder(states)
        action, z_pi = self.policy_net(zs)
        _, zsa = self.state_action_encoder(zs, action)
        q1 = self.q1(zsa)
        q2 = self.q2(zsa)
        pre_activ_reg = cfg.lambda_pre_activ * (z_pi ** 2).mean()
        loss = -0.5 * (q1 + q2).mean() + pre_activ_reg
        return loss, {"pol_loss": loss.item()}

    # ==================================================================
    # Training update
    # ==================================================================

    def update(self, buffer: ReplayBuffer) -> dict:
        cfg = self.config

        # ---- Encoder update ----
        self.encoder_optim.zero_grad()
        enc_loss, enc_info = self._encoder_loss(buffer)
        enc_loss.backward()
        enc_params = list(self.state_encoder.parameters()) + list(self.state_action_encoder.parameters())
        torch.nn.utils.clip_grad_norm_(enc_params, cfg.gradient_clip_norm)
        self.encoder_optim.step()

        # ---- Value update ----
        self.value_optim.zero_grad()
        val_loss, td_errors, indices = self._value_loss(buffer)
        val_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), cfg.gradient_clip_norm,
        )
        self.value_optim.step()

        # Update LAP priorities and reward scaler
        buffer.update_priorities(indices, td_errors)
        with torch.no_grad():
            sampled_rewards = torch.as_tensor(
                buffer.rewards[indices], device=self.device, dtype=torch.float32,
            )
            self.reward_scaler.update(sampled_rewards)

        # ---- Policy update ----
        self.policy_optim.zero_grad()
        states, _, _, _, _, _ = buffer.sample(cfg.batch_size)
        pol_loss, pol_info = self._policy_loss(states)
        pol_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), cfg.gradient_clip_norm)
        self.policy_optim.step()

        # ---- Target network update (periodic) ----
        self.total_it += 1
        if self.total_it % cfg.target_update_freq == 0:
            self._update_targets()
            self.target_reward_scale = self.reward_scaler.get_scale()

        return {"val_loss": val_loss.item(), **enc_info, **pol_info}

    def _update_targets(self):
        self.state_encoder_target.load_state_dict(self.state_encoder.state_dict())
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.policy_target.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save({
            "state_encoder": self.state_encoder.state_dict(),
            "state_action_encoder": self.state_action_encoder.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "policy": self.policy_net.state_dict(),
            "state_encoder_target": self.state_encoder_target.state_dict(),
            "q1_target": self.q1_target.state_dict(), "q2_target": self.q2_target.state_dict(),
            "policy_target": self.policy_target.state_dict(),
            "total_it": self.total_it,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.state_action_encoder.load_state_dict(ckpt["state_action_encoder"])
        self.q1.load_state_dict(ckpt["q1"]); self.q2.load_state_dict(ckpt["q2"])
        self.policy_net.load_state_dict(ckpt["policy"])
        self.state_encoder_target.load_state_dict(ckpt["state_encoder_target"])
        self.q1_target.load_state_dict(ckpt["q1_target"]); self.q2_target.load_state_dict(ckpt["q2_target"])
        self.policy_target.load_state_dict(ckpt["policy_target"])
        self.total_it = ckpt["total_it"]
