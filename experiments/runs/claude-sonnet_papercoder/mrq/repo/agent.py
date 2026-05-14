```python
## agent.py
"""MR.Q Agent: orchestrates encoder, value, and policy updates.

This module implements MRQAgent, which wires together MRQNetworks,
EpisodeBuffer, and TwoHotEncoder into the synchronized training loop
described in Section 4.2 of the paper.

Key design principles:
  - Strict gradient isolation: encoder parameters only receive gradients
    from _update_encoder; value and policy networks receive detached embeddings.
  - Synchronized target updates every T_target=250 steps (hard copy, not EMA).
  - Reward scaling: r_bar normalizes value targets; r_bar_prime denormalizes
    the bootstrap term from target networks trained at the previous scale.
  - Multi-step returns (H_Q=3) computed from the sequence batch shared with
    the encoder update (sequence length H_Enc+1=6 covers H_Q=3).
  - LAP prioritized sampling with Huber loss for value updates.
  - Pre-activation regularization in the policy loss for sparse reward stability.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from envs import EnvWrapper
from networks import MRQNetworks
from replay_buffer import EpisodeBuffer
from utils import TwoHotEncoder, huber_loss


class MRQAgent:
    """Model-based Representations for Q-learning agent.

    Orchestrates all training logic for MR.Q, including:
      - Action selection with exploration noise
      - Transition storage in the episode-aware replay buffer
      - Encoder update via unrolled dynamics prediction (reward, dynamics, terminal)
      - Value update via multi-step TD with reward scaling and LAP priorities
      - Policy update via deterministic policy gradient with pre-activation reg.
      - Periodic synchronized target network updates

    Ablation variants are controlled via Config flags set by _apply_ablation().

    Attributes:
        cfg: Configuration dataclass with all hyperparameters.
        nets: Container for all networks, target copies, and optimizers.
        replay: Episode-aware replay buffer with LAP prioritized sampling.
        two_hot: Two-hot encoder for categorical reward representation.
        r_bar: Current mean absolute reward (normalizes value targets).
        r_bar_prime: r_bar at last target sync (denormalizes bootstrap term).
        total_steps: Global environment interaction step counter.
        terminal_seen: True once any terminal transition has been stored.
        discrete: Whether the action space is discrete.
        image_obs: Whether observations are images.
        device: Torch device for all tensor operations.
    """

    def __init__(
        self,
        cfg: Config,
        state_shape: Tuple[int, ...],
        action_dim: int,
        discrete: bool,
        image_obs: bool,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialise the MR.Q agent.

        Creates all networks, replay buffer, and two-hot encoder. Initializes
        reward scaling variables and ablation flags from the Config.

        Args:
            cfg: Configuration dataclass with all hyperparameters.
            state_shape: Shape of a single observation (e.g., (17,) for
                HalfCheetah, (9, 84, 84) for DMC visual, (4, 84, 84) for Atari).
            action_dim: Number of action dimensions (continuous) or number of
                discrete actions (Atari one-hot encoding).
            discrete: Whether the action space is discrete.
            image_obs: Whether observations are images (True) or vectors (False).
            device: Torch device for all tensor operations.
        """
        self.cfg: Config = cfg
        self.discrete: bool = discrete
        self.image_obs: bool = image_obs
        self.device: torch.device = device

        # ---------------------------------------------------------------
        # Core components
        # ---------------------------------------------------------------
        self.nets: MRQNetworks = MRQNetworks(
            cfg=cfg,
            state_shape=state_shape,
            action_dim=action_dim,
            discrete=discrete,
            image_obs=image_obs,
        ).to(device)

        self.replay: EpisodeBuffer = EpisodeBuffer(
            capacity=cfg.replay_capacity,
            state_shape=state_shape,
            action_dim=action_dim,
            lap_alpha=cfg.lap_alpha,
            lap_min_priority=cfg.lap_min_priority,
            device=device,
        )

        self.two_hot: TwoHotEncoder = TwoHotEncoder(
            n_bins=cfg.reward_bins,
            low=cfg.reward_range[0],
            high=cfg.reward_range[1],
            device=device,
        )

        # ---------------------------------------------------------------
        # Training state
        # ---------------------------------------------------------------
        # Reward scaling: r_bar normalizes value targets; r_bar_prime
        # denormalizes the bootstrap term from target networks.
        self.r_bar: float = 1.0
        self.r_bar_prime: float = 1.0

        # Global step counter (incremented in store_transition).
        self.total_steps: int = 0

        # Gate for terminal loss weight (lambda_terminal = 0 until first
        # terminal is seen, per Section 4.2.1).
        self.terminal_seen: bool = False

        # ---------------------------------------------------------------
        # Ablation flags (read from cfg fields set by _apply_ablation)
        # ---------------------------------------------------------------
        self.use_encoder_loss: bool = cfg.use_encoder_loss
        self.use_reward_scaling: bool = cfg.use_reward_scaling
        self.use_min_q: bool = cfg.use_min_q
        self.use_lap: bool = cfg.use_lap
        self.use_mse_reward: bool = cfg.use_mse_reward
        self.use_target_encoder: bool = cfg.use_target_encoder
        self.use_sa_dynamics_target: bool = cfg.use_sa_dynamics_target

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self, state: np.ndarray, explore: bool = True
    ) -> np.ndarray:
        """Select an action for environment interaction.

        Encodes the state, runs the policy, and optionally adds exploration
        noise. Returns the action in the agent's normalized space ([-1, 1]
        for continuous, one-hot for discrete); the EnvWrapper handles
        conversion to the environment's native format.

        Args:
            state: Current observation as float32 numpy array of shape
                self.state_shape.
            explore: If True, add Gaussian exploration noise. If False,
                use the deterministic policy output (for evaluation).

        Returns:
            Action as float32 numpy array of shape (action_dim,).
            Continuous: values in [-1, 1].
            Discrete: one-hot vector of shape (action_dim,).
        """
        # Convert to tensor with batch dimension.
        state_tensor: torch.Tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # shape: (1, *state_shape)

        with torch.no_grad():
            # Encode state using online encoder (no target).
            zs: torch.Tensor = self.nets.encode_state(
                state_tensor, use_target=False
            )  # shape: (1, zs_dim)

            # Forward through policy network.
            z_pi: torch.Tensor
            a_pi: torch.Tensor
            z_pi, a_pi = self.nets.policy.forward(zs)
            # z_pi: pre-activation (1, action_dim)
            # a_pi: activated action (1, action_dim)

            if explore:
                if self.discrete:
                    # Add noise to pre-activation logits, then take argmax.
                    # This implements "Gaussian noise is added to each dimension
                    # of the one-hot encoding" (Section 4.2.3).
                    noise: torch.Tensor = torch.randn_like(z_pi) * self.cfg.explore_noise_std
                    noisy_logits: torch.Tensor = z_pi + noise
                    action_idx: int = int(torch.argmax(noisy_logits, dim=-1).item())
                    action_onehot: np.ndarray = np.zeros(
                        self.cfg.za_dim if False else a_pi.shape[-1],
                        dtype=np.float32,
                    )
                    action_onehot[action_idx] = 1.0
                    return action_onehot
                else:
                    # Add Gaussian noise to continuous action, clip to [-1, 1].
                    noise = torch.randn_like(a_pi) * self.cfg.explore_noise_std
                    noisy_action: torch.Tensor = torch.clamp(a_pi + noise, -1.0, 1.0)
                    return noisy_action.squeeze(0).cpu().numpy()
            else:
                # Deterministic action selection (evaluation mode).
                if self.discrete:
                    action_idx = int(torch.argmax(a_pi, dim=-1).item())
                    action_onehot = np.zeros(a_pi.shape[-1], dtype=np.float32)
                    action_onehot[action_idx] = 1.0
                    return action_onehot
                else:
                    return a_pi.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Transition storage
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_state: np.ndarray,
    ) -> None:
        """Store a transition in the replay buffer.

        Updates the terminal_seen flag and increments the global step counter.

        Args:
            state: Current observation, shape self.state_shape, float32.
            action: Action taken. Continuous: shape (action_dim,) in [-1, 1].
                Discrete: one-hot shape (action_dim,).
            reward: Scalar reward received.
            done: True if this transition ends the episode.
            next_state: Next observation, shape self.state_shape, float32.
        """
        self.replay.add(state, action, reward, done, next_state)

        # Permanently set terminal_seen once any terminal is observed.
        if done and not self.terminal_seen:
            self.terminal_seen = True

        self.total_steps += 1

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """Perform one full update cycle: encoder, value, and policy.

        Checks for target sync at the start of each update. Uses the same
        sequence batch for both encoder and value updates to avoid a second
        buffer sample and ensure consistency.

        Returns:
            Dictionary with keys 'encoder_loss', 'value_loss', 'policy_loss'
            containing the scalar loss values for logging.
        """
        # ---------------------------------------------------------------
        # Periodic target sync (before gradient updates).
        # Matches paper pseudocode: "if t % T_target = 0 then sync targets."
        # ---------------------------------------------------------------
        if self.total_steps % self.cfg.target_update_freq == 0:
            self._sync_targets()

        # ---------------------------------------------------------------
        # Sample sequence batch for encoder and value updates.
        # seq_len = H_Enc + 1 = 6 covers both H_Enc=5 and H_Q=3.
        # ---------------------------------------------------------------
        seq_len: int = self.cfg.enc_horizon + 1
        batch_seq: Dict[str, torch.Tensor] = self.replay.sample_sequences(
            self.cfg.batch_size, seq_len=seq_len
        )

        # ---------------------------------------------------------------
        # 1. Encoder update (model-based representation learning).
        # ---------------------------------------------------------------
        enc_loss: torch.Tensor = self._update_encoder(batch_seq)

        # ---------------------------------------------------------------
        # 2. Value update (multi-step TD with reward scaling).
        # ---------------------------------------------------------------
        val_loss: torch.Tensor
        val_loss, _ = self._update_value(batch_seq)

        # ---------------------------------------------------------------
        # 3. Policy update (deterministic policy gradient).
        # ---------------------------------------------------------------
        pol_loss: torch.Tensor = self._update_policy(batch_seq)

        return {
            "encoder_loss": enc_loss.item(),
            "value_loss": val_loss.item(),
            "policy_loss": pol_loss.item(),
        }

    # ------------------------------------------------------------------
    # Encoder update
    # ------------------------------------------------------------------

    def _update_encoder(
        self, batch_seq: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute and apply the encoder loss via unrolled dynamics prediction.

        Unrolls the learned model over H_Enc steps, accumulating:
          - Categorical cross-entropy reward loss (or MSE for ablation)
          - MSE dynamics loss against target encoder embeddings
          - MSE terminal loss (gated by terminal_seen)

        Gradients flow through the entire unrolled chain back into both
        f_omega (state encoder) and g_omega (state-action encoder).

        Args:
            batch_seq: Sequence batch with keys 'states' (B, H+1, *state_shape),
                'actions' (B, H+1, action_dim), 'rewards' (B, H+1),
                'dones' (B, H+1), 'next_states' (B, H+1, *state_shape).

        Returns:
            Scalar encoder loss tensor (detached from graph after backward).
        """
        # Ablation: no_mr — skip encoder loss, train encoder end-to-end with value.
        if not self.use_encoder_loss:
            return torch.tensor(0.0, device=self.device)

        # ---------------------------------------------------------------
        # Initialize unrolling at t=0.
        # ---------------------------------------------------------------
        # Encode the initial state through the online encoder (with gradients).
        # shape: (B, zs_dim)
        zs_tilde: torch.Tensor = self.nets.encode_state(
            batch_seq["states"][:, 0], use_target=False
        )

        total_loss: torch.Tensor = torch.tensor(0.0, device=self.device)

        # ---------------------------------------------------------------
        # Unrolling loop: t = 1 to H_Enc (inclusive).
        # ---------------------------------------------------------------
        for t in range(1, self.cfg.enc_horizon + 1):
            # Index into the sequence: action/reward/done at step t-1,
            # next_state at step t-1 (which is state at step t).
            action_t: torch.Tensor = batch_seq["actions"][:, t - 1]   # (B, action_dim)
            reward_t: torch.Tensor = batch_seq["rewards"][:, t - 1]   # (B,)
            done_t: torch.Tensor = batch_seq["dones"][:, t - 1]       # (B,)
            next_state_t: torch.Tensor = batch_seq["next_states"][:, t - 1]  # (B, *state_shape)

            # ----------------------------------------------------------
            # Forward through state-action encoder and linear MDP predictor.
            # model_output: (B, zs_dim + reward_bins + 1)
            # zsa_tilde:    (B, zsa_dim)
            # ----------------------------------------------------------
            model_output: torch.Tensor
            _zsa_tilde: torch.Tensor
            model_output, _zsa_tilde = self.nets.sa_encoder.forward(
                zs_tilde, action_t
            )

            # Split model output into components.
            zs_dim: int = self.cfg.zs_dim
            reward_bins: int = self.cfg.reward_bins

            zs_tilde_next: torch.Tensor = model_output[:, :zs_dim]
            r_logits: torch.Tensor = model_output[:, zs_dim : zs_dim + reward_bins]
            d_tilde: torch.Tensor = model_output[:, -1]  # shape: (B,)

            # ----------------------------------------------------------
            # Compute dynamics target.
            # ----------------------------------------------------------
            if self.use_sa_dynamics_target:
                # Ablation: dynamics_target — use z_{s'a'} from target encoder.
                # Requires the next action from the batch.
                next_action_t: torch.Tensor = batch_seq["actions"][:, t] if t < batch_seq["actions"].shape[1] else batch_seq["actions"][:, -1]
                with torch.no_grad():
                    zs_next_for_target: torch.Tensor = self.nets.encode_state(
                        next_state_t, use_target=True
                    )
                    # Use target state-action encoder with next action.
                    _model_out_target: torch.Tensor
                    zsa_bar: torch.Tensor
                    _model_out_target, zsa_bar = self.nets.sa_encoder_target.forward(
                        zs_next_for_target, next_action_t
                    )
                # Dynamics target is the state-action embedding (reintroduces policy dependency).
                # We use zsa_bar as the target for zs_tilde_next (dimension mismatch handled below).
                # Since zsa_bar has shape (B, zsa_dim) and zs_tilde_next has shape (B, zs_dim),
                # we use only the first zs_dim dimensions of zsa_bar as the target.
                dynamics_target: torch.Tensor = zsa_bar[:, :zs_dim].detach()
            elif self.use_target_encoder:
                # Standard: use target state encoder, stop gradient.
                with torch.no_grad():
                    dynamics_target = self.nets.encode_state(
                        next_state_t, use_target=True
                    ).detach()
            else:
                # Ablation: no_target_encoder — use current encoder, no stop-gradient.
                # Jointly optimize dynamics target within the encoder loss.
                dynamics_target = self.nets.encode_state(
                    next_state_t, use_target=False
                )

            # ----------------------------------------------------------
            # Reward loss.
            # ----------------------------------------------------------
            if self.use_mse_reward:
                # Ablation: mse_reward — decode logits to scalar, apply MSE.
                r_pred_scalar: torch.Tensor = self.two_hot.decode(r_logits)  # (B,)
                r_loss: torch.Tensor = F.mse_loss(r_pred_scalar, reward_t)
            else:
                # Standard: categorical cross-entropy with two-hot target.
                reward_target: torch.Tensor = self.two_hot.encode(reward_t)  # (B, reward_bins)
                # Cross-entropy expects (B, C) predictions and (B, C) soft targets.
                # Use log_softmax + sum for soft cross-entropy.
                log_probs: torch.Tensor = F.log_softmax(r_logits, dim=-1)  # (B, reward_bins)
                r_loss = -(reward_target * log_probs).sum(dim=-1).mean()

            # ----------------------------------------------------------
            # Dynamics loss: MSE between predicted and target state embedding.
            # ----------------------------------------------------------
            dyn_loss: torch.Tensor = F.mse_loss(zs_tilde_next, dynamics_target)

            # ----------------------------------------------------------
            # Terminal loss: MSE between predicted and actual done signal.
            # Gate by terminal_seen (lambda_terminal = 0 until first terminal).
            # ----------------------------------------------------------
            term_weight: float = (
                self.cfg.lambda_terminal if self.terminal_seen else 0.0
            )
            term_loss: torch.Tensor = F.mse_loss(d_tilde, done_t)

            # ----------------------------------------------------------
            # Accumulate weighted losses.
            # ----------------------------------------------------------
            total_loss = (
                total_loss
                + self.cfg.lambda_reward * r_loss
                + self.cfg.lambda_dynamics * dyn_loss
                + term_weight * term_loss
            )

            # Advance: use predicted embedding as input for next step.
            zs_tilde = zs_tilde_next

        # ---------------------------------------------------------------
        # Optimizer step for encoder parameters.
        # ---------------------------------------------------------------
        self.nets.enc_optimizer.zero_grad()
        total_loss.backward()
        self.nets.enc_optimizer.step()

        return total_loss.detach()

    # ------------------------------------------------------------------
    # Value update
    # ------------------------------------------------------------------

    def _update_value(
        self, batch_seq: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Compute multi-step TD targets and update both value networks.

        Uses the sequence batch (shared with encoder update) to compute
        H_Q-step returns. Applies reward scaling to normalize targets.
        Uses Huber loss with LAP priority updates (or MSE with uniform
        sampling for the no_lap ablation).

        Args:
            batch_seq: Sequence batch with keys 'states' (B, H+1, *state_shape),
                'actions' (B, H+1, action_dim), 'rewards' (B, H+1),
                'dones' (B, H+1), 'next_states' (B, H+1, *state_shape).
                The value update uses indices 0..H_Q-1 for multi-step returns
                and index H_Q-1 for the bootstrap state.

        Returns:
            Tuple of:
                val_loss (torch.Tensor): Scalar value loss (detached).
                td_errors (np.ndarray): Per-sample TD errors for priority update,
                    shape (batch_size,).
        """
        hq: int = self.cfg.hq_horizon
        batch_size: int = batch_seq["states"].shape[0]

        # ---------------------------------------------------------------
        # Compute multi-step TD target (no gradients needed).
        # ---------------------------------------------------------------
        with torch.no_grad():
            # State at step H_Q (for bootstrap).
            # next_states[:, hq-1] is the state after H_Q actions.
            next_state_hq: torch.Tensor = batch_seq["next_states"][:, hq - 1]

            # Encode next state using online encoder, then detach.
            # Using online encoder (not target) for the next state embedding
            # matches TD7's approach and the paper's description.
            zs_next: torch.Tensor = self.nets.encode_state(
                next_state_hq, use_target=False
            ).detach()

            # Compute smoothed target action.
            target_action: torch.Tensor = self._get_target_action(zs_next)

            # Encode (next_state, target_action) using target state-action encoder.
            _model_out: torch.Tensor
            zsa_next: torch.Tensor
            _model_out, zsa_next = self.nets.encode_state_action(
                zs_next, target_action, use_target=True
            )

            # Compute target Q-values from both target value networks.
            q1_target: torch.Tensor = self.nets.value1_target(zsa_next)  # (B, 1)
            q2_target: torch.Tensor = self.nets.value2_target(zsa_next)  # (B, 1)

            # Min or mean over target Q-networks (ablation: no_min uses mean).
            if self.use_min_q:
                q_bootstrap: torch.Tensor = torch.min(q1_target, q2_target)
            else:
                q_bootstrap = 0.5 * (q1_target + q2_target)

            # ----------------------------------------------------------
            # Multi-step return computation with done masking.
            # Accumulates: Σ_{h=0}^{H_Q-1} γ^h * r_h * Π_{k=0}^{h-1}(1-d_k)
            # ----------------------------------------------------------
            discounted_return: torch.Tensor = torch.zeros(
                batch_size, 1, dtype=torch.float32, device=self.device
            )
            # not_done_mask tracks whether the episode is still active.
            # Starts at 1.0; multiplied by (1 - done) at each step.
            not_done_mask: torch.Tensor = torch.ones(
                batch_size, 1, dtype=torch.float32, device=self.device
            )

            for h in range(hq):
                reward_h: torch.Tensor = batch_seq["rewards"][:, h].unsqueeze(1)  # (B, 1)
                done_h: torch.Tensor = batch_seq["dones"][:, h].unsqueeze(1)      # (B, 1)

                discounted_return = (
                    discounted_return
                    + (self.cfg.discount ** h) * not_done_mask * reward_h
                )
                # Update mask: once done=1, all subsequent steps are masked out.
                not_done_mask = not_done_mask * (1.0 - done_h)

            # Bootstrap: γ^H_Q * (1 - done_{H_Q-1}) * r_bar_prime * Q_target
            # r_bar_prime converts target Q-values from normalized to raw reward units.
            r_bar_prime_val: float = self.r_bar_prime if self.use_reward_scaling else 1.0
            bootstrap: torch.Tensor = (
                (self.cfg.discount ** hq)
                * not_done_mask
                * r_bar_prime_val
                * q_bootstrap
            )

            # Normalized target: divide by r_bar to match the value network's scale.
            r_bar_val: float = self.r_bar if self.use_reward_scaling else 1.0
            target: torch.Tensor = (1.0 / r_bar_val) * (discounted_return + bootstrap)

        # ---------------------------------------------------------------
        # Compute current Q-values (with gradients through value networks only).
        # ---------------------------------------------------------------
        # Encode current state and action, detaching from encoder graph.
        current_state: torch.Tensor = batch_seq["states"][:, 0]
        current_action: torch.Tensor = batch_seq["actions"][:, 0]

        zs: torch.Tensor = self.nets.encode_state(
            current_state, use_target=False
        ).detach()  # Stop gradient at encoder boundary.

        _model_out_curr: torch.Tensor
        zsa: torch.Tensor
        _model_out_curr, zsa = self.nets.sa_encoder.forward(zs, current_action)
        zsa = zsa.detach()  # Stop gradient at encoder boundary.

        q1: torch.Tensor = self.nets.value1(zsa)   # (B, 1)
        q2: torch.Tensor = self.nets.value2(zsa)   # (B, 1)

        # ---------------------------------------------------------------
        # Compute loss.
        # ---------------------------------------------------------------
        target_detached: torch.Tensor = target.detach()

        if self.use_lap:
            # Standard: Huber loss (corrects for LAP prioritization bias).
            loss_q1: torch.Tensor = huber_loss(q1, target_detached)  # (B, 1)
            loss_q2: torch.Tensor = huber_loss(q2, target_detached)  # (B, 1)
            val_loss: torch.Tensor = (loss_q1 + loss_q2).mean()
        else:
            # Ablation: no_lap — use MSE loss with uniform sampling.
            val_loss = F.mse_loss(q1, target_detached) + F.mse_loss(q2, target_detached)

        # ---------------------------------------------------------------
        # Compute TD errors for LAP priority update.
        # ---------------------------------------------------------------
        with torch.no_grad():
            td_errors_tensor: torch.Tensor = 0.5 * (
                torch.abs(q1 - target_detached) + torch.abs(q2 - target_detached)
            )  # (B, 1)
        td_errors: np.ndarray = td_errors_tensor.squeeze(1).cpu().numpy()

        # ---------------------------------------------------------------
        # Optimizer step for value networks.
        # ---------------------------------------------------------------
        self.nets.value_optimizer.zero_grad()
        val_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.nets.value1.parameters()) + list(self.nets.value2.parameters()),
            self.cfg.grad_clip_norm,
        )
        self.nets.value_optimizer.step()

        # ---------------------------------------------------------------
        # Update LAP priorities (skip for no_lap ablation).
        # ---------------------------------------------------------------
        # Note: batch_seq does not have 'indices' since it comes from
        # sample_sequences (uniform sampling). For LAP, we need indices
        # from the prioritized sample. We use a separate single-transition
        # sample for priority updates when LAP is active.
        # The sequence batch is used for the actual gradient update;
        # priority updates use the sequence start indices as proxies.
        # This is a practical approximation consistent with the paper's intent.
        if self.use_lap:
            # Use a separate LAP sample to get indices for priority update.
            try:
                lap_batch: Dict[str, object] = self.replay.sample(
                    min(self.cfg.batch_size, len(self.replay))
                )
                lap_indices: np.ndarray = lap_batch["indices"]  # type: ignore[assignment]

                # Compute TD errors for the LAP-sampled transitions.
                with torch.no_grad():
                    lap_states: torch.Tensor = lap_batch["states"]  # type: ignore[assignment]
                    lap_actions: torch.Tensor = lap_batch["actions"]  # type: ignore[assignment]
                    lap_next_states: torch.Tensor = lap_batch["next_states"]  # type: ignore[assignment]
                    lap_rewards: torch.Tensor = lap_batch["rewards"]  # type: ignore[assignment]
                    lap_dones: torch.Tensor = lap_batch["dones"]  # type: ignore[assignment]

                    zs_lap: torch.Tensor = self.nets.encode_state(
                        lap_states, use_target=False
                    ).detach()
                    _mo_lap: torch.Tensor
                    zsa_lap: torch.Tensor
                    _mo_lap, zsa_lap = self.nets.sa_encoder.forward(zs_lap, lap_actions)
                    zsa_lap = zsa_lap.detach()

                    q1_lap: torch.Tensor = self.nets.value1(zsa_lap)
                    q2_lap: torch.Tensor = self.nets.value2(zsa_lap)

                    # 1-step target for priority computation.
                    zs_next_lap: torch.Tensor = self.nets.encode_state(
                        lap_next_states, use_target=False
                    ).detach()
                    target_action_lap: torch.Tensor = self._get_target_action(zs_next_lap)
                    _mo_next_lap: torch.Tensor
                    zsa_next_lap: torch.Tensor
                    _mo_next_lap, zsa_next_lap = self.nets.encode_state_action(
                        zs_next_lap, target_action_lap, use_target=True
                    )
                    q1_t_lap: torch.Tensor = self.nets.value1_target(zsa_next_lap)
                    q2_t_lap: torch.Tensor = self.nets.value2_target(zsa_next_lap)
                    q_boot_lap: torch.Tensor = torch.min(q1_t_lap, q2_t_lap)

                    r_bar_v: float = self.r_bar if self.use_reward_scaling else 1.0
                    r_bar_p_v: float = self.r_bar_prime if self.use_reward_scaling else 1.0