"""
MR.Q: Model-based Representations for Q-learning

Main algorithm implementation.
Based on: "Towards General-Purpose Model-Free RL (MR.Q)"
Fujimoto et al., 2025

Key components:
1. Encoder (StateEncoder + StateActionEncoder + linear MDP predictor)
   - Trained with reward, dynamics, and terminal losses
   - Unrolled over H_enc steps
2. Value function (twin critics, TD3-style)
   - Multi-step returns (H_Q steps)
   - Reward scaling
   - LAP prioritized sampling + Huber loss
3. Policy (deterministic policy gradient)
   - Gumbel-Softmax for discrete, Tanh for continuous
   - Pre-activation regularization
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mrq.networks import StateEncoder, StateActionEncoder, ValueNetwork, PolicyNetwork, ZS_DIM, ZSA_DIM
from utils.reward_encoding import get_reward_bins, two_hot_encode, decode_reward


class MRQ:
    """
    MR.Q Algorithm.
    
    Hyperparameters follow Table 3 of the paper and are kept fixed
    across all benchmarks.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        # Environment settings
        discrete=False,
        image_obs=False,
        state_channels=None,
        action_scale=1.0,       # Scale for continuous action range
        # Encoder hyperparameters
        enc_horizon=5,          # H_enc: encoder unroll horizon
        lambda_reward=0.1,      # lambda_Reward
        lambda_dynamics=1.0,    # lambda_Dynamics (default 1.0, not specified in table)
        lambda_terminal=0.1,    # lambda_Terminal
        reward_bins=65,
        reward_range=(-10.0, 10.0),
        # Value hyperparameters
        q_horizon=3,            # H_Q: multi-step return horizon
        gamma=0.99,
        target_noise_std=0.2,   # sigma for target policy noise
        target_noise_clip=0.3,  # c for target policy noise clipping
        # Policy hyperparameters
        lambda_preactiv=1e-5,   # Pre-activation regularization
        gumbel_tau=10.0,
        # LAP hyperparameters
        lap_alpha=0.4,
        min_priority=1.0,
        # Training hyperparameters
        batch_size=256,
        enc_lr=1e-4,
        value_lr=3e-4,
        policy_lr=3e-4,
        weight_decay=1e-4,
        grad_clip_value=20.0,   # Gradient clipping for value network
        target_update_freq=250, # T_target
        # Exploration
        expl_noise_std=0.2,
        # Device
        device="cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.image_obs = image_obs
        self.action_scale = action_scale

        self.enc_horizon = enc_horizon
        self.lambda_reward = lambda_reward
        self.lambda_dynamics = lambda_dynamics
        self.lambda_terminal = lambda_terminal
        self.reward_bins = reward_bins

        self.q_horizon = q_horizon
        self.gamma = gamma
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip

        self.lambda_preactiv = lambda_preactiv
        self.gumbel_tau = gumbel_tau

        self.lap_alpha = lap_alpha
        self.min_priority = min_priority
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.expl_noise_std = expl_noise_std
        self.grad_clip_value = grad_clip_value

        self.device = torch.device(device)
        self.total_steps = 0

        # Reward scaling: mean absolute reward in buffer
        self.reward_scale = 1.0       # r_bar (current)
        self.reward_scale_target = 1.0  # r_bar' (target, updated with target nets)

        # Terminal loss weight: set to 0 until first terminal seen
        self.seen_terminal = False
        self.lambda_terminal_eff = 0.0

        # Reward bins for categorical reward prediction
        self.bins = get_reward_bins(reward_bins, reward_range).to(self.device)

        # ---- Networks ----
        # Encoder (trained end-to-end)
        self.encoder = StateEncoder(state_dim, image_obs, state_channels).to(self.device)
        self.sa_encoder = StateActionEncoder(action_dim, reward_bins).to(self.device)

        # Target encoder (for dynamics target z_s')
        self.encoder_target = copy.deepcopy(self.encoder).to(self.device)
        self.sa_encoder_target = copy.deepcopy(self.sa_encoder).to(self.device)

        # Twin value networks
        self.Q1 = ValueNetwork().to(self.device)
        self.Q2 = ValueNetwork().to(self.device)
        self.Q1_target = copy.deepcopy(self.Q1).to(self.device)
        self.Q2_target = copy.deepcopy(self.Q2).to(self.device)

        # Policy network
        self.policy = PolicyNetwork(action_dim, discrete, gumbel_tau).to(self.device)
        self.policy_target = copy.deepcopy(self.policy).to(self.device)

        # ---- Optimizers ----
        # Encoder optimizer (enc_lr, weight_decay)
        enc_params = list(self.encoder.parameters()) + list(self.sa_encoder.parameters())
        self.enc_optimizer = torch.optim.AdamW(
            enc_params, lr=enc_lr, weight_decay=weight_decay, eps=1e-8
        )

        # Value optimizer (value_lr, weight_decay)
        value_params = list(self.Q1.parameters()) + list(self.Q2.parameters())
        self.value_optimizer = torch.optim.AdamW(
            value_params, lr=value_lr, weight_decay=weight_decay, eps=1e-8
        )

        # Policy optimizer (policy_lr, weight_decay)
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=policy_lr, weight_decay=weight_decay, eps=1e-8
        )

        # Freeze target networks
        for net in [self.encoder_target, self.sa_encoder_target,
                    self.Q1_target, self.Q2_target, self.policy_target]:
            for p in net.parameters():
                p.requires_grad = False

    # =========================================================================
    # Action selection
    # =========================================================================

    @torch.no_grad()
    def select_action(self, state, explore=True):
        """
        Select action given state.
        
        For exploration: adds Gaussian noise to action.
        For discrete: argmax of noisy one-hot.
        For continuous: clip to [-1, 1] * action_scale.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        zs = self.encoder(state_t)
        _, a_pi = self.policy(zs)

        if explore:
            if self.discrete:
                # Add noise to each dimension of the one-hot
                noise = torch.randn_like(a_pi) * self.expl_noise_std
                a_pi = a_pi + noise
                action = a_pi.argmax(dim=-1).item()
            else:
                noise = torch.randn_like(a_pi) * self.expl_noise_std
                a_pi = torch.clamp(a_pi + noise, -1.0, 1.0)
                action = (a_pi * self.action_scale).squeeze(0).cpu().numpy()
        else:
            if self.discrete:
                action = a_pi.argmax(dim=-1).item()
            else:
                action = (a_pi * self.action_scale).squeeze(0).cpu().numpy()

        return action

    # =========================================================================
    # Target network update
    # =========================================================================

    def update_target_networks(self):
        """
        Hard copy of current networks to target networks.
        Also updates reward scaling.
        Called every T_target steps.
        """
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.sa_encoder_target.load_state_dict(self.sa_encoder.state_dict())
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())
        # Update target reward scale
        self.reward_scale_target = self.reward_scale

    # =========================================================================
    # Encoder update
    # =========================================================================

    def update_encoder(self, seq_batch):
        """
        Update encoder using unrolled dynamics over H_enc steps.
        
        Loss = sum_{t=1}^{H_enc} [
            lambda_R * CE(r_pred_t, TwoHot(r_t)) +
            lambda_D * MSE(z_s_pred_t, z_s_target_t) +
            lambda_T * MSE(d_pred_t, d_t)
        ]
        
        seq_batch: dict with keys states, actions, rewards, dones
                   each of shape (H_enc+1, batch, ...)
        """
        states = seq_batch["states"]    # (H_enc+1, B, ...)
        actions = seq_batch["actions"]  # (H_enc+1, B, action_dim)
        rewards = seq_batch["rewards"]  # (H_enc+1, B, 1)
        dones = seq_batch["dones"]      # (H_enc+1, B, 1)

        batch_size = states.shape[1]
        total_loss = torch.tensor(0.0, device=self.device)

        # Encode initial state
        zs_tilde = self.encoder(states[0])  # (B, ZS_DIM)

        for t in range(self.enc_horizon):
            # Apply state-action encoder and linear MDP predictor
            model_out, zsa = self.sa_encoder(zs_tilde, actions[t])
            # model_out: (B, ZS_DIM + reward_bins + 1)

            # Split predictions
            zs_pred = model_out[:, :ZS_DIM]                          # predicted next state emb
            r_logits = model_out[:, ZS_DIM:ZS_DIM + self.reward_bins]  # reward logits
            d_pred = model_out[:, -1:]                                # terminal prediction

            # --- Reward loss (categorical cross-entropy with two-hot target) ---
            r_target = rewards[t + 1].squeeze(-1)  # (B,)
            two_hot = two_hot_encode(r_target, self.bins)  # (B, reward_bins)
            reward_loss = F.cross_entropy(r_logits, two_hot)

            # --- Dynamics loss (MSE with target encoder output) ---
            with torch.no_grad():
                zs_target = self.encoder_target(states[t + 1])  # (B, ZS_DIM)
            dynamics_loss = F.mse_loss(zs_pred, zs_target)

            # --- Terminal loss (MSE with binary terminal signal) ---
            terminal_loss = F.mse_loss(d_pred, dones[t + 1])

            # Accumulate losses
            step_loss = (
                self.lambda_reward * reward_loss
                + self.lambda_dynamics * dynamics_loss
                + self.lambda_terminal_eff * terminal_loss
            )
            total_loss = total_loss + step_loss

            # Update z_s_tilde for next step (use predicted next state embedding)
            zs_tilde = zs_pred.detach()  # stop gradient through time

        self.enc_optimizer.zero_grad()
        total_loss.backward()
        self.enc_optimizer.step()

        return total_loss.item()

    # =========================================================================
    # Value update
    # =========================================================================

    def update_value(self, seq_batch, indices):
        """
        Update twin value networks using multi-step TD targets.
        
        Uses Huber loss (to eliminate bias from LAP prioritized sampling).
        Reward scaling: divide by r_bar, multiply target by r_bar'.
        
        Returns TD errors for priority update.
        """
        states = seq_batch["states"]    # (H_Q+1, B, ...)
        actions = seq_batch["actions"]  # (H_Q+1, B, action_dim)
        rewards = seq_batch["rewards"]  # (H_Q+1, B, 1)
        dones = seq_batch["dones"]      # (H_Q+1, B, 1)

        batch_size = states.shape[1]

        # Compute multi-step return target
        # Target: (1/r_bar) * [sum_{t=0}^{H_Q-1} gamma^t * r_t + gamma^{H_Q} * Q_target]
        with torch.no_grad():
            # Get target action at s_{H_Q}
            zs_HQ = self.encoder_target(states[self.q_horizon])
            _, a_target = self.policy_target(zs_HQ)

            if self.discrete:
                # Add noise to each dimension, then argmax -> one-hot
                noise = torch.randn_like(a_target) * self.target_noise_std
                a_noisy = a_target + noise
                a_idx = a_noisy.argmax(dim=-1)
                a_target_noisy = F.one_hot(a_idx, self.action_dim).float()
            else:
                noise = torch.clamp(
                    torch.randn_like(a_target) * self.target_noise_std,
                    -self.target_noise_clip, self.target_noise_clip
                )
                a_target_noisy = torch.clamp(a_target + noise, -1.0, 1.0)

            # Get target Q values
            _, zsa_target = self.sa_encoder_target(zs_HQ, a_target_noisy)
            Q1_target = self.Q1_target(zsa_target)
            Q2_target = self.Q2_target(zsa_target)
            Q_target_min = torch.min(Q1_target, Q2_target)

            # Multi-step discounted return
            # Note: reward scaling - divide accumulated rewards by r_bar,
            # multiply bootstrap by r_bar' (target scale)
            discounted_return = torch.zeros(batch_size, 1, device=self.device)
            not_done = torch.ones(batch_size, 1, device=self.device)

            for t in range(self.q_horizon):
                discounted_return = discounted_return + not_done * (self.gamma ** t) * rewards[t]
                not_done = not_done * (1.0 - dones[t])

            # Scale rewards and add bootstrap
            # y = (1/r_bar) * [sum gamma^t r_t + gamma^H * r_bar' * Q_target]
            r_scale = max(self.reward_scale, 1e-8)
            target = (1.0 / r_scale) * (
                discounted_return + not_done * (self.gamma ** self.q_horizon) * self.reward_scale_target * Q_target_min
            )

        # Current Q values (stop gradient through encoder)
        with torch.no_grad():
            zs_0 = self.encoder(states[0])
            _, zsa_0 = self.sa_encoder(zs_0, actions[0])

        Q1_pred = self.Q1(zsa_0.detach())
        Q2_pred = self.Q2(zsa_0.detach())

        # Huber loss (eliminates bias from LAP)
        td_error1 = Q1_pred - target
        td_error2 = Q2_pred - target
        value_loss = F.huber_loss(Q1_pred, target) + F.huber_loss(Q2_pred, target)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        # Gradient clipping for value network
        nn.utils.clip_grad_norm_(
            list(self.Q1.parameters()) + list(self.Q2.parameters()),
            self.grad_clip_value
        )
        self.value_optimizer.step()

        # TD errors for priority update (use mean of both critics)
        td_errors = ((td_error1 + td_error2) / 2.0).abs().detach().cpu().numpy().squeeze()

        return value_loss.item(), td_errors

    # =========================================================================
    # Policy update
    # =========================================================================

    def update_policy(self, batch):
        """
        Update policy using deterministic policy gradient.
        
        Loss = -0.5 * sum_i Q_i(z_sa_pi) + lambda_preactiv * z_pi^2
        
        Uses Gumbel-Softmax for discrete, Tanh for continuous.
        Gradients do NOT flow through encoder.
        """
        states = batch["states"]

        with torch.no_grad():
            zs = self.encoder(states)

        z_pi, a_pi = self.policy(zs.detach())

        # Get z_sa for policy action (gradients flow through policy only)
        _, zsa_pi = self.sa_encoder(zs.detach(), a_pi)

        Q1_pi = self.Q1(zsa_pi)
        Q2_pi = self.Q2(zsa_pi)

        # Policy loss: maximize Q, with pre-activation regularization
        policy_loss = -0.5 * (Q1_pi + Q2_pi).mean()
        policy_loss = policy_loss + self.lambda_preactiv * (z_pi ** 2).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item()

    # =========================================================================
    # Main training step
    # =========================================================================

    def train(self, replay_buffer):
        """
        Perform one training iteration.
        
        Every T_target steps:
          1. Update target networks
          2. Update reward scaling
          3. For T_target steps:
             a. Update encoder
             b. Update value
             c. Update policy
        
        Returns dict of losses.
        """
        self.total_steps += 1

        # Check if we should update target networks
        if self.total_steps % self.target_update_freq == 0:
            self.update_target_networks()
            # Update reward scaling from buffer
            self.reward_scale = replay_buffer.get_mean_abs_reward()

        # Sample sequences for encoder + value updates
        # Need H_enc + 1 states for encoder, H_Q + 1 for value
        # Use max of both
        total_seq_len = max(self.enc_horizon, self.q_horizon) + 1

        seq_batch = replay_buffer.sample_sequences(
            batch_size=self.batch_size,
            seq_len=total_seq_len
        )

        # Encoder update
        enc_loss = self.update_encoder(seq_batch)

        # Value update (uses first H_Q+1 steps)
        value_seq = {
            "states": seq_batch["states"][:self.q_horizon + 1],
            "actions": seq_batch["actions"][:self.q_horizon + 1],
            "rewards": seq_batch["rewards"][:self.q_horizon + 1],
            "dones": seq_batch["dones"][:self.q_horizon + 1],
        }
        value_loss, td_errors = self.update_value(value_seq, seq_batch["start_indices"])

        # Update LAP priorities
        replay_buffer.update_priorities(seq_batch["start_indices"], td_errors)

        # Policy update (uses single-step batch)
        single_batch, _ = replay_buffer.sample(self.batch_size)
        policy_loss = self.update_policy(single_batch)

        return {
            "enc_loss": enc_loss,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "reward_scale": self.reward_scale,
        }

    def update_terminal_weight(self, seen_terminal):
        """Enable terminal loss once a terminal transition is seen."""
        if seen_terminal and not self.seen_terminal:
            self.seen_terminal = True
            self.lambda_terminal_eff = self.lambda_terminal

    def save(self, path):
        """Save model state."""
        torch.save({
            "encoder": self.encoder.state_dict(),
            "sa_encoder": self.sa_encoder.state_dict(),
            "Q1": self.Q1.state_dict(),
            "Q2": self.Q2.state_dict(),
            "policy": self.policy.state_dict(),
            "reward_scale": self.reward_scale,
            "total_steps": self.total_steps,
        }, path)

    def load(self, path):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.sa_encoder.load_state_dict(checkpoint["sa_encoder"])
        self.Q1.load_state_dict(checkpoint["Q1"])
        self.Q2.load_state_dict(checkpoint["Q2"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.reward_scale = checkpoint.get("reward_scale", 1.0)
        self.total_steps = checkpoint.get("total_steps", 0)
        self.update_target_networks()
