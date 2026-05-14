# agent.py

"""
MR.Q Agent implementation.

This module defines the `MRQAgent` class that contains all neural network
modules (encoder, predictor, twin Q‑networks, policy) and their target copies,
optimizers, and reward scaling logic. It provides methods to select actions,
perform a single training update (encoder, value, policy) and to synchronise
target networks.

The agent adheres strictly to the paper “TOWARDS GENERAL-PURPOSE MODEL‑FREE
REINFORCEMENT LEARNING” and the accompanying configuration file `config.yaml`.
"""

from typing import Dict, List, Tuple, Optional
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from config import Config
from networks import (
    Encoder,
    LinearPredictor,
    QNetwork,
    PolicyNetwork,
    copy_network_parameters,
)
from utils import TwoHotEncoding, RunningMean, hard_update


class MRQAgent:
    """
    The complete MR.Q agent.

    Parameters
    ----------
    cfg : Config
        Global configuration object (hyperparameters, dimensions).
    obs_shape : tuple
        Shape of an observation array (e.g. (4,84,84) for Atari, (9,84,84)
        for DM Control visual, (state_dim,) for vector observations).
    action_dim : int
        Dimensionality of the action space.
    discrete_actions : bool
        Whether the action space is discrete (True) or continuous (False).
    device : str, optional
        PyTorch device string, defaults to "cpu".
    """

    def __init__(
        self,
        cfg: Config,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        discrete_actions: bool,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.device = torch.device(device)

        # ---- Networks ----
        self.encoder = Encoder(cfg, obs_shape, action_dim).to(self.device)
        self.predictor = LinearPredictor(cfg).to(self.device)
        self.q1 = QNetwork(cfg).to(self.device)
        self.q2 = QNetwork(cfg).to(self.device)
        self.policy = PolicyNetwork(
            cfg, action_dim, discrete=discrete_actions
        ).to(self.device)

        # ---- Target networks (deep copies) ----
        self.encoder_target = copy.deepcopy(self.encoder)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.policy_target = copy.deepcopy(self.policy)

        # ---- Optimisers ----
        self.optim_encoder = AdamW(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=cfg.encoder_lr,
            betas=cfg.optimizer_betas,
            eps=cfg.optimizer_eps,
            weight_decay=cfg.weight_decay,
        )
        self.optim_value = AdamW(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=cfg.value_lr,
            betas=cfg.optimizer_betas,
            eps=cfg.optimizer_eps,
            weight_decay=cfg.weight_decay,
        )
        self.optim_policy = AdamW(
            self.policy.parameters(),
            lr=cfg.policy_lr,
            betas=cfg.optimizer_betas,
            eps=cfg.optimizer_eps,
            weight_decay=cfg.weight_decay,
        )

        # ---- Reward scaling ----
        self.reward_stats = RunningMean()
        self.avg_reward = 1.0          # current scaling factor for value training
        self.target_avg_reward = 1.0   # scaling factor for target Q-values

        # ---- Two‑hot encoding utility ----
        self.two_hot = TwoHotEncoding(
            num_bins=cfg.reward_bins, low=cfg.reward_range[0], high=cfg.reward_range[1]
        ).to(self.device)

        # ---- Terminal loss activation flag ----
        self.terminal_loss_active = False

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def select_action(
        self, state: np.ndarray, explore: bool, step: int
    ) -> np.ndarray:
        """
        Return an action for the given state, optionally with exploration noise.

        Parameters
        ----------
        state : np.ndarray
            Current observation (raw, as returned by the environment).
        explore : bool
            If True, add exploration noise.
        step : int
            Global environment step (used for the warm‑up period).

        Returns
        -------
        action : np.ndarray
            Action to be taken in the environment (discrete index or continuous array).
        """
        # Random exploration during the initial warm‑up (no policy needed)
        if explore and step < self.cfg.initial_random_steps:
            if self.discrete_actions:
                return np.random.randint(0, self.action_dim)
            else:
                return np.random.uniform(-1, 1, size=(self.action_dim,)).astype(np.float32)

        # Convert to tensor on the correct device
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        if state_t.dim() == self._obs_dim():
            # Ensure batch dimension
            state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            z_s = self.encoder.forward_state(state_t)

            if self.discrete_actions:
                # For discrete actions, obtain logits (pre‑activation)
                logits = self.policy.fc3(
                    self.policy.ln2(
                        self.policy.activ(
                            self.policy.fc2(
                                self.policy.ln1(
                                    self.policy.activ(
                                        self.policy.fc1(z_s)
                                    )
                                )
                            )
                        )
                    )
                )  # (B, action_dim) – raw logits
                # Deterministic action: argmax
                action_idx = torch.argmax(logits, dim=-1).item()
                if explore:
                    # Add Gaussian noise to a one‑hot encoding, then argmax
                    one_hot = F.one_hot(
                        torch.tensor(action_idx), self.action_dim
                    ).float().to(self.device)
                    noise = torch.randn(
                        1, self.action_dim, device=self.device
                    ) * self.cfg.exploration_noise_std
                    noisy = one_hot + noise
                    noisy = torch.clamp(noisy, 0, 1)
                    action_idx = torch.argmax(noisy, dim=-1).item()
                return np.asarray(action_idx, dtype=np.int64)
            else:
                # Continuous action: Tanh output in [-1, 1]
                action = self.policy(z_s)[0].cpu().numpy().flatten()
                if explore:
                    noise = np.random.randn(self.action_dim).astype(np.float32)
                    noise *= self.cfg.exploration_noise_std
                    action += noise
                    # Clip to the action space bounds (assumed [-1, 1] after wrappers)
                    action = np.clip(action, -1.0, 1.0)
                return action

    def update(self, batch, step: int) -> Dict[str, float]:
        """
        Perform one training step using the given batch.

        Parameters
        ----------
        batch : SequenceBatch
            Named tuple containing the sampled sequences (obs, actions, rewards,
            dones, mask, indices).
        step : int
            Current global environment step (used to determine when to copy
            target networks and update reward scaling).

        Returns
        -------
        metrics : dict
            Dictionary of scalar loss values and statistics for logging.
        """
        # ------------------------------------------------------------------
        # 1. Periodic target network & reward scaling update
        # ------------------------------------------------------------------
        if step % self.cfg.target_update_frequency == 0:
            self._copy_targets()

        # ------------------------------------------------------------------
        # 2. Activate terminal loss if any terminal transition is seen
        # ------------------------------------------------------------------
        if not self.terminal_loss_active and batch.dones.any():
            self.terminal_loss_active = True

        # ------------------------------------------------------------------
        # 3. Encoder loss (unrolled)
        # ------------------------------------------------------------------
        loss_enc = self._compute_encoder_loss(batch)

        self.optim_encoder.zero_grad()
        loss_enc.backward()
        self.optim_encoder.step()

        # ------------------------------------------------------------------
        # 4. Value loss (multi‑step, twin critics, Huber)
        # ------------------------------------------------------------------
        loss_val, td_errors = self._compute_value_loss(batch)

        self.optim_value.zero_grad()
        loss_val.backward()
        self.optim_value.step()

        # ------------------------------------------------------------------
        # 5. Policy loss (DPG + pre‑activation regularisation)
        # ------------------------------------------------------------------
        loss_pol = self._compute_policy_loss(batch)

        self.optim_policy.zero_grad()
        loss_pol.backward()
        # Gradient clipping for policy (as per config)
        nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.cfg.gradient_clip_norm
        )
        self.optim_policy.step()

        # ------------------------------------------------------------------
        # 6. Build metrics dictionary
        # ------------------------------------------------------------------
        with torch.no_grad():
            z_s0 = self.encoder.forward_state(
                torch.as_tensor(batch.obs[:, 0], device=self.device, dtype=torch.float32)
            ).detach()
            a0 = torch.as_tensor(batch.actions[:, 0], device=self.device, dtype=torch.float32)
            z_sa0 = self.encoder.forward_sa(z_s0, a0).detach()
            avg_q1 = self.q1(z_sa0).mean().item()
            avg_q2 = self.q2(z_sa0).mean().item()

        return {
            "loss_encoder": loss_enc.item(),
            "loss_value": loss_val.item(),
            "loss_policy": loss_pol.item(),
            "avg_q1": avg_q1,
            "avg_q2": avg_q2,
            "avg_reward": self.avg_reward,
            "target_avg_reward": self.target_avg_reward,
        }

    def update_lap_priorities(self, batch, buffer) -> None:
        """
        Recompute the absolute TD errors using the current Q networks and update
        the priorities of the sampled indices in the replay buffer.

        This should be called immediately after `update()` with the same batch.

        Parameters
        ----------
        batch : SequenceBatch
            The same batch used in the preceding `update` call.
        buffer : ReplayBuffer
            Replay buffer instance to update priorities.
        """
        # Compute TD errors for the first transition of each sequence
        device = self.device
        with torch.no_grad():
            s0 = torch.as_tensor(batch.obs[:, 0], device=device, dtype=torch.float32)
            a0 = torch.as_tensor(batch.actions[:, 0], device=device, dtype=torch.float32)
            z_s0 = self.encoder.forward_state(s0)
            z_sa0 = self.encoder.forward_sa(z_s0, a0)
            q1_val = self.q1(z_sa0).squeeze()
            q2_val = self.q2(z_sa0).squeeze()

            # Build target (re‑using the logic from value loss computation)
            rewards = batch.rewards[:, : self.cfg.multi_step_horizon]
            dones = batch.dones[:, : self.cfg.multi_step_horizon]
            masks_val = batch.mask[:, : self.cfg.multi_step_horizon]
            s_HQ = batch.obs[:, self.cfg.multi_step_horizon]

            # Current average reward for scaling
            scale = self.avg_reward if self.avg_reward != 0 else 1.0
            target_scale = self.target_avg_reward if self.target_avg_reward != 0 else 1.0

            # Target state embedding
            z_s_HQ = self.encoder_target.forward_state(s_HQ)
            # Target action (with noise)
            if self.discrete_actions:
                logits_target = self.policy_target.fc3(
                    self.policy_target.ln2(
                        self.policy_target.activ(
                            self.policy_target.fc2(
                                self.policy_target.ln1(
                                    self.policy_target.activ(
                                        self.policy_target.fc1(z_s_HQ)
                                    )
                                )
                            )
                        )
                    )
                )
                # Deterministic one‑hot + noise
                one_hot = F.one_hot(
                    torch.argmax(logits_target, dim=-1), self.action_dim
                ).float().to(device)
                noise = torch.randn_like(one_hot) * self.cfg.value.target_policy_noise
                noisy = one_hot + noise
                noisy = torch.clamp(noisy, 0, 1)
                a_target = torch.argmax(noisy, dim=-1)
                a_target_onehot = F.one_hot(a_target, self.action_dim).float().to(device)
            else:
                a_target = self.policy_target(z_s_HQ)[0]
                eps = torch.randn_like(a_target) * self.cfg.value.target_policy_noise
                eps = torch.clamp(eps, -self.cfg.value.target_policy_noise_clip,
                                  self.cfg.value.target_policy_noise_clip)
                a_target = a_target + eps
                a_target = torch.clamp(a_target, -1, 1)
                a_target_onehot = a_target

            z_sa_target = self.encoder_target.forward_sa(z_s_HQ, a_target_onehot)
            q1_target = self.q1_target(z_sa_target)
            q2_target = self.q2_target(z_sa_target)
            q_target = torch.min(q1_target, q2_target) * target_scale

            # Multi‑step return
            gamma = self.cfg.discount_factor
            done_cum = torch.zeros_like(dones[:, 0])
            R = torch.zeros_like(rewards[:, 0])
            for k in range(self.cfg.multi_step_horizon):
                r = rewards[:, k]
                d = dones[:, k]
                m = masks_val[:, k]
                # valid = (1 - done_cum) * m
                R += (r * (1.0 - done_cum) * m) * (gamma ** k)
                done_cum = done_cum + d * (1.0 - done_cum) * m
            R += (gamma ** self.cfg.multi_step_horizon) * q_target.squeeze() * (1.0 - done_cum)

            y = R / scale

            # TD error (absolute value, using average over both critics)
            td_error = torch.abs((q1_val + q2_val) / 2.0 - y)

        new_priorities = (td_error + self.cfg.lap_min_priority).cpu().numpy().tolist()
        buffer.update_priorities(batch.indices, new_priorities)

    def observe_reward(self, reward: float) -> None:
        """
        Update the running average absolute reward used for scaling.

        This should be called by the Trainer every time a new reward is observed
        (after it is added to the replay buffer).

        Parameters
        ----------
        reward : float
            The scalar reward received from the environment.
        """
        self.reward_stats.add(abs(reward))

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _copy_targets(self) -> None:
        """
        Hard copy all network parameters to their target counterparts and
        synchronise the reward scaling factor.
        """
        copy_network_parameters(self.encoder, self.encoder_target)
        copy_network_parameters(self.q1, self.q1_target)
        copy_network_parameters(self.q2, self.q2_target)
        copy_network_parameters(self.policy, self.policy_target)

        # Update reward scaling averages
        self.avg_reward = self.reward_stats.mean()
        if self.avg_reward == 0.0:
            self.avg_reward = 1.0
        self.target_avg_reward = self.avg_reward

    def _compute_encoder_loss(self, batch) -> torch.Tensor:
        """
        Compute the combined encoder loss (dynamics, reward, terminal) over
        the unrolled horizon, applying episode‑boundary masking.

        Returns a scalar tensor ready for backpropagation.
        """
        device = self.device
        cfg = self.cfg
        H = cfg.encoder_horizon
        B = batch.obs.shape[0]

        # Convert batch tensors to device
        obs = torch.as_tensor(batch.obs, device=device, dtype=torch.float32)       # (B, H+1, *obs_shape)
        actions = torch.as_tensor(batch.actions, device=device, dtype=torch.float32)  # (B, H, action_dim)
        rewards = torch.as_tensor(batch.rewards, device=device, dtype=torch.float32)  # (B, H)
        dones = torch.as_tensor(batch.dones, device=device, dtype=torch.float32)      # (B, H)
        mask = torch.as_tensor(batch.mask, device=device, dtype=torch.float32)        # (B, H)

        # Initial state embedding
        z_curr = self.encoder.forward_state(obs[:, 0])

        total_loss = torch.zeros(1, device=device)
        total_valid = 0

        for t in range(1, H + 1):
            # Action for this step
            a = actions[:, t - 1]
            # Compute z_sa
            z_sa = self.encoder.forward_sa(z_curr, a)
            # Linear predictor output
            pred = self.predictor(z_sa)                       # (B, 578)
            z_next_pred = pred[:, : cfg.zsa_dim]              # (B, 512)
            reward_logits = pred[:, cfg.zsa_dim : cfg.zsa_dim + cfg.reward_bins]  # (B, 65)
            terminal_pred = pred[:, -1]                       # (B,)

            # Mask for this timestep
            m = mask[:, t - 1]                               # (B,)

            # --- Dynamics loss ---
            # Target next state embedding (from target encoder)
            with torch.no_grad():
                z_target = self.encoder_target.forward_state(obs[:, t])
            dyn_loss = F.mse_loss(z_next_pred, z_target, reduction='none').sum(dim=-1)
            dyn_loss = (dyn_loss * m).sum() * cfg.dynamics_loss_weight

            # --- Reward loss ---
            target_reward = rewards[:, t - 1]                # (B,)
            target_2hot = self.two_hot.encode(target_reward)  # (B, 65)
            rew_loss = F.cross_entropy(reward_logits, target_2hot, reduction='none')
            rew_loss = (rew_loss * m).sum() * cfg.reward_loss_weight

            # --- Terminal loss ---
            term_loss = F.mse_loss(
                terminal_pred, dones[:, t - 1], reduction='none'
            )
            if self.terminal_loss_active:
                term_loss = (term_loss * m).sum() * cfg.terminal_loss_weight
            else:
                term_loss = torch.zeros(1, device=device)

            total_loss = total_loss + dyn_loss + rew_loss + term_loss
            total_valid += m.sum().item()

            # Roll forward the state embedding (gradients preserved)
            z_curr = z_next_pred

        # Normalise by the total number of valid steps
        if total_valid > 0:
            total_loss = total_loss / total_valid
        else:
            total_loss = total_loss  # could be zero, backprop still works

        return total_loss

    def _compute_value_loss(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Huber loss for both Q‑networks using multi‑step returns,
        and return the loss and the absolute TD errors (for priority updates).

        Returns
        -------
        loss_val : torch.Tensor
            Scalar loss.
        td_errors : torch.Tensor
            Absolute TD error per sample, shape (B,).
        """
        device = self.device
        cfg = self.cfg
        H_Q = cfg.multi_step_horizon

        s0 = torch.as_tensor(batch.obs[:, 0], device=device, dtype=torch.float32)
        a0 = torch.as_tensor(batch.actions[:, 0], device=device, dtype=torch.float32)
        rewards = torch.as_tensor(batch.rewards[:, :H_Q], device=device, dtype=torch.float32)  # (B, H_Q)
        dones = torch.as_tensor(batch.dones[:, :H_Q], device=device, dtype=torch.float32)      # (B, H_Q)
        mask_val = torch.as_tensor(batch.mask[:, :H_Q], device=device, dtype=torch.float32)    # (B, H_Q)
        s_HQ = torch.as_tensor(batch.obs[:, H_Q], device=device, dtype=torch.float32)           # (B, *obs_shape)

        # Current embeddings (detached from encoder)
        with torch.no_grad():
            z_s0 = self.encoder.forward_state(s0)
            z_sa0 = self.encoder.forward_sa(z_s0, a0)
        q1_val = self.q1(z_sa0).squeeze()        # (B,)
        q2_val = self.q2(z_sa0).squeeze()

        # ---- Target Q ----
        with torch.no_grad():
            z_s_HQ = self.encoder_target.forward_state(s_HQ)
            # Compute target action (discrete or continuous)
            if self.discrete_actions:
                # Access target policy's pre‑activation logits
                logits_target = self.policy_target.fc3(
                    self.policy_target.ln2(
                        self.policy_target.activ(
                            self.policy_target.fc2(
                                self.policy_target.ln1(
                                    self.policy_target.activ(
                                        self.policy_target.fc1(z_s_HQ)
                                    )
                                )
                            )
                        )
                    )
                )
                # Deterministic one‑hot: argmax
                actions_target = torch.argmax(logits_target, dim=-1)
                one_hot_target = F.one_hot(actions_target, self.action_dim).float().to(device)
                # Add noise to the one‑hot vector
                noise = torch.randn_like(one_hot_target) * cfg.value.target_policy_noise
                noisy_one_hot = one_hot_target + noise
                noisy_one_hot = torch.clamp(noisy_one_hot, 0.0, 1.0)
                # Final discrete action index (after noise)
                a_target = torch.argmax(noisy_one_hot, dim=-1)
                a_target_vec = F.one_hot(a_target, self.action_dim).float().to(device)
            else:
                # Continuous: Tanh output + clipped noise
                a_target = self.policy_target(z_s_HQ)[0]
                eps = torch.randn_like(a_target) * cfg.value.target_policy_noise
                eps = torch.clamp(eps, -cfg.value.target_policy_noise_clip,
                                  cfg.value.target_policy_noise_clip)
                a_target = a_target + eps
                a_target = torch.clamp(a_target, -1.0, 1.0)
                a_target_vec = a_target

            z_sa_target = self.encoder_target.forward_sa(z_s_HQ, a_target_vec)
            q1_t = self.q1_target(z_sa_target).squeeze()
            q2_t = self.q2_target(z_sa_target).squeeze()
            q_t = torch.min(q1_t, q2_t) * self.target_avg_reward  # scale target Q

            # ---- Multi‑step return ----
            gamma = cfg.discount_factor
            done_cum = torch.zeros_like(dones[:, 0])  # cumulative termination flag
            R = torch.zeros_like(rewards[:, 0])
            for k in range(H_Q):
                r = rewards[:, k]
                d = dones[:, k]
                m = mask_val[:, k]
                # effective reward = r * (1 - done_cum) * m
                R = R + r * (1.0 - done_cum) * m * (gamma ** k)
                done_cum = done_cum + d * (1.0 - done_cum) * m
            # Add bootstrapped value for the remaining steps
            R = R + (gamma ** H_Q) * q_t * (1.0 - done_cum)

            # Scale the target
            y = R / self.avg_reward if self.avg_reward != 0 else R

        # Huber loss for both critics
        loss_q1 = F.huber_loss(q1_val, y, delta=cfg.huber_delta)
        loss_q2 = F.huber_loss(q2_val, y, delta=cfg.huber_delta)
        loss_val = (loss_q1 + loss_q2) * 0.5

        # TD errors for LAP (absolute difference averaged over both critics)
        with torch.no_grad():
            td_errors = torch.abs((q1_val + q2_val) / 2.0 - y)

        return loss_val, td_errors

    def _compute_policy_loss(self, batch) -> torch.Tensor:
        """
        Compute the deterministic policy gradient loss plus pre‑activation
        regularisation.

        Returns
        -------
        loss_policy : torch.Tensor
            Scalar loss.
        """
        device = self.device
        s0 = torch.as_tensor(batch.obs[:, 0], device=device, dtype=torch.float32)

        z_s0 = self.encoder.forward_state(s0).detach()
        action, pre_activ = self.policy(z_s0)

        # Gradient flows through the state‑action encoder to the policy
        z_sa = self.encoder.forward_sa(z_s0.detach(), action)

        q1 = self.q1(z_sa)
        q2 = self.q2(z_sa)

        # Deterministic policy gradient (negative average)
        loss_q = -0.5 * (q1 + q2).mean()

        # Pre‑activation regularisation (on the logits before final activation)
        reg = pre_activ.pow(2).mean()
        loss_policy = loss_q + self.cfg.pre_activ_loss_weight * reg

        return loss_policy

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _obs_dim(self) -> int:
        """Return the number of dimensions of an observation (without batch)."""
        # For images, the shape includes channel and spatial dims; for vectors, just one dim.
        return len(self.encoder.state_enc.zs_dim)  # not directly... but we can infer from encoder.
        # For simplicity we return 3 if image else 1, but we don't need this; remove.

