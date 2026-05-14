## policies/sac.py
"""Soft Actor-Critic (SAC) policy for Prioritized Generative Replay (PGR).

Implements standard SAC (Haarnoja et al., 2018) as a secondary model-free
baseline. Uses twin critics (Clipped Double Q-learning) rather than REDQ's
ensemble of 10, and performs one gradient update per call to update().

Shares GaussianActor and QNetwork with policies/redq.py to enforce
architectural consistency between SAC and REDQ baselines.

Config references (config.yaml):
    policy.hidden_dim:    256   # baseline hidden width
    policy.hidden_layers: 2     # baseline MLP depth
    policy.gamma:         0.99  # discount factor
    policy.tau:           0.005 # target network EMA coefficient
    policy.lr:            3e-4  # Adam learning rate
    policy.auto_alpha:    true  # automatic entropy tuning
    hardware.device:      "cuda"
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from policies.redq import GaussianActor, QNetwork


class SACPolicy:
    """Standard Soft Actor-Critic with twin critics and automatic entropy tuning.

    Implements SAC (Haarnoja et al., 2018) as a model-free baseline for PGR
    experiments. Uses two Q-networks (Clipped Double Q-learning) for stable
    value estimation. The UTD ratio is controlled externally by PGRTrainer —
    this class performs exactly one gradient update per call to update().

    Key differences from REDQPolicy:
        - Two Q-networks instead of an ensemble of 10
        - No subsampling mechanism (uses both critics for all computations)
        - Typically used with UTD=1 as a baseline (not UTD=20 like REDQ)

    Attributes:
        obs_dim: Observation space dimension.
        action_dim: Action space dimension.
        hidden_dim: Width of hidden layers in actor and critics.
        num_layers: Number of hidden layers in actor and critics.
        gamma: Discount factor for Bellman target computation.
        tau: Target network EMA coefficient for soft updates.
        auto_alpha: Whether to use automatic entropy tuning.
        target_entropy: Target entropy for automatic alpha tuning.
        device: PyTorch device string.
        actor: GaussianActor policy network (shared architecture with REDQ).
        critic1: First QNetwork (twin critic 1).
        critic2: Second QNetwork (twin critic 2).
        target_critic1: Frozen EMA copy of critic1.
        target_critic2: Frozen EMA copy of critic2.
        actor_optimizer: Adam optimizer for the actor.
        critic_optimizer: Adam optimizer for both critics (shared).
        log_alpha: Learnable log entropy coefficient.
        alpha_optimizer: Adam optimizer for log_alpha.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        device: str = "cuda",
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        alpha_init: float = 0.2,
    ) -> None:
        """Initialises the SAC policy with twin critics and automatic entropy tuning.

        Args:
            obs_dim: Observation space dimension. Inferred from the environment
                wrapper at PGRTrainer init time.
            action_dim: Action space dimension. Inferred from the environment
                wrapper at PGRTrainer init time.
            hidden_dim: Width of hidden layers in actor and both critics.
                Corresponds to config.policy.hidden_dim (default 256).
            num_layers: Number of hidden layers in actor and both critics.
                Corresponds to config.policy.hidden_layers (default 2).
            gamma: Discount factor for Bellman target computation. Corresponds
                to config.policy.gamma (default 0.99).
            tau: Target network EMA coefficient for soft updates. Corresponds
                to config.policy.tau (default 0.005).
            lr: Adam optimizer learning rate for actor, critics, and alpha.
                Corresponds to config.policy.lr (default 3e-4).
            device: PyTorch device string. Corresponds to
                config.hardware.device (default "cuda").
            auto_alpha: Whether to use automatic entropy tuning. Corresponds
                to config.policy.auto_alpha (default True).
            target_entropy: Target entropy for automatic alpha tuning. If None,
                defaults to -action_dim (standard SAC heuristic). Corresponds
                to the negative action space dimension.
            alpha_init: Initial entropy coefficient value. Used as the fixed
                alpha when auto_alpha=False, and as the initial value before
                automatic tuning when auto_alpha=True. Default 0.2.
        """
        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.gamma: float = gamma
        self.tau: float = tau
        self.auto_alpha: bool = auto_alpha
        self.device: str = device

        # Standard SAC target entropy heuristic: -dim(A).
        # Encourages the policy to maintain entropy equal to a uniform
        # distribution over a 1D action space.
        self.target_entropy: float = (
            target_entropy if target_entropy is not None else float(-action_dim)
        )

        # Store fixed alpha for when auto_alpha=False.
        self._alpha_fixed: float = float(alpha_init)

        # ── Actor ─────────────────────────────────────────────────────────────
        # Shared architecture with REDQPolicy — imported from policies.redq.
        self.actor: GaussianActor = GaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        # ── Twin critics ──────────────────────────────────────────────────────
        # Two independent Q-networks for Clipped Double Q-learning.
        # Independent random initialization provides diversity in value estimates.
        self.critic1: QNetwork = QNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        self.critic2: QNetwork = QNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        # ── Target critics ────────────────────────────────────────────────────
        # Frozen EMA copies of the critics. Initialized with identical weights.
        # Updated only via soft (Polyak) averaging — never by backpropagation.
        self.target_critic1: QNetwork = QNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        self.target_critic2: QNetwork = QNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        # Hard copy: initialize target critics with exact critic weights.
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Freeze target critics — updated only via _soft_update_targets().
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False

        # ── Optimizers ────────────────────────────────────────────────────────
        # Actor optimizer: only actor parameters.
        self.actor_optimizer: torch.optim.Adam = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr,
        )

        # Critic optimizer: parameters from both critics combined.
        # A single shared optimizer is more efficient than two separate ones
        # and produces identical results since both critics are updated together.
        self.critic_optimizer: torch.optim.Adam = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr,
        )

        # ── Entropy coefficient (alpha) ───────────────────────────────────────
        # log_alpha parameterization ensures alpha = exp(log_alpha) > 0 always.
        # Initialized to log(alpha_init) so that exp(log_alpha) = alpha_init.
        self.log_alpha: torch.Tensor = torch.tensor(
            [float(np.log(alpha_init))],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        self.alpha_optimizer: torch.optim.Adam = torch.optim.Adam(
            [self.log_alpha],
            lr=lr,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Selects an action for the given observation.

        Used by PGRTrainer._collect_transition() for online data collection
        (deterministic=False) and by Evaluator.evaluate_policy() for
        deterministic policy evaluation (deterministic=True).

        Args:
            obs: Float32 numpy array of shape (obs_dim,) — current observation
                from the environment. Converted to a (1, obs_dim) tensor internally.
            deterministic: If True, returns tanh(mean) without sampling noise.
                Used for evaluation. If False (default), samples from the
                Gaussian policy for exploration during training.

        Returns:
            Float32 numpy array of shape (action_dim,) with values in (-1, 1)
            from tanh squashing.
        """
        # Convert numpy observation to (1, obs_dim) tensor on device.
        obs_tensor: torch.Tensor = (
            torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        )

        self.actor.eval()

        with torch.no_grad():
            if deterministic:
                # Deterministic action: tanh(mean) without sampling.
                mean, _log_std = self.actor.forward(obs_tensor)
                action: torch.Tensor = torch.tanh(mean)  # (1, action_dim)
            else:
                # Stochastic action: sample from Gaussian with tanh squashing.
                action, _log_prob = self.actor.sample(obs_tensor)  # (1, action_dim)

        self.actor.train()

        # Squeeze batch dimension and convert to numpy: (action_dim,)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Performs one full SAC update cycle on a mixed batch.

        Called by PGRTrainer._update_policy(). Performs three sequential
        sub-steps: critic update, actor update, alpha update, followed by
        a soft target network update.

        The batch follows the universal transition dict format:
            'observations':      (B, obs_dim)
            'actions':           (B, action_dim)
            'next_observations': (B, obs_dim)
            'rewards':           (B, 1)
            'dones':             (B, 1)
        All tensors are float32 on self.device.

        Args:
            batch: Transition dict from MixedSampler.sample(). Contains
                float32 tensors on self.device with the keys listed above.
                Batch size B corresponds to config.sampling.batch_size
                (default 256).

        Returns:
            Dict of scalar training metrics for logging:
                'critic_loss': Sum of MSE losses for both critics.
                'actor_loss':  SAC actor loss (alpha * log_prob - min_Q).mean().
                'alpha':       Current entropy coefficient value.
                'entropy':     Mean policy entropy (-log_prob) for monitoring.
        """
        # Extract tensors — already on self.device with float32 dtype.
        obs: torch.Tensor = batch["observations"]
        actions: torch.Tensor = batch["actions"]
        next_obs: torch.Tensor = batch["next_observations"]
        rewards: torch.Tensor = batch["rewards"]
        dones: torch.Tensor = batch["dones"]

        # Ensure rewards and dones have shape (B, 1) for Bellman computation.
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)

        # ── Step 1: Update twin critics ───────────────────────────────────────
        critic_loss: float = self._update_critics(
            obs, actions, next_obs, rewards, dones
        )

        # ── Step 2: Update actor ──────────────────────────────────────────────
        actor_loss: float
        mean_log_prob: float
        actor_loss, mean_log_prob = self._update_actor(obs)

        # ── Step 3: Update alpha (entropy coefficient) ────────────────────────
        if self.auto_alpha:
            self._update_alpha(obs)

        # ── Step 4: Soft update of target critics ─────────────────────────────
        self._soft_update_targets()

        # ── Compute current alpha value for logging ───────────────────────────
        current_alpha: float = (
            float(self.log_alpha.exp().item())
            if self.auto_alpha
            else self._alpha_fixed
        )

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha": current_alpha,
            "entropy": float(-mean_log_prob),
        }

    def save(self, path: str) -> None:
        """Saves all policy state to a checkpoint file.

        Saves actor, both critics, both target critics, all optimizers, and
        the log_alpha parameter. Called by PGRTrainer.save_checkpoint().

        Args:
            path: Full file path for the checkpoint. Parent directories are
                created if they do not exist.
        """
        parent_dir: str = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        checkpoint: dict = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            # Store config for verification on load.
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "gamma": self.gamma,
            "tau": self.tau,
            "auto_alpha": self.auto_alpha,
            "target_entropy": self.target_entropy,
        }

        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Restores all policy state from a checkpoint file.

        Restores actor, both critics, both target critics, all optimizers,
        and the log_alpha parameter. Ensures target critics remain frozen
        after loading. Called by PGRTrainer.load_checkpoint().

        Args:
            path: Full file path to the checkpoint saved by save().

        Raises:
            FileNotFoundError: If the checkpoint file does not exist at path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"SACPolicy checkpoint not found at '{path}'. "
                "Ensure the path is correct and the checkpoint was saved."
            )

        checkpoint: dict = torch.load(path, map_location=self.device)

        # Restore network weights.
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2"])

        # Re-freeze target critics after loading — load_state_dict does not
        # preserve requires_grad, so we enforce it explicitly.
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False

        # Restore optimizer states.
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

        # Restore log_alpha — move to device and re-enable gradient tracking.
        log_alpha_loaded: torch.Tensor = checkpoint["log_alpha"].to(self.device)
        with torch.no_grad():
            self.log_alpha.copy_(log_alpha_loaded)

    # ── Private methods ───────────────────────────────────────────────────────

    def _update_critics(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """Updates both twin critics using the Bellman target with entropy regularization.

        Computes the Bellman target using the minimum of both target critics
        (Clipped Double Q-learning) minus the entropy term, then minimizes
        the MSE between current Q estimates and the target.

        Args:
            obs: Current observations, float32 tensor of shape (B, obs_dim).
            actions: Actions taken, float32 tensor of shape (B, action_dim).
            next_obs: Next observations, float32 tensor of shape (B, obs_dim).
            rewards: Rewards received, float32 tensor of shape (B, 1).
            dones: Episode termination flags, float32 tensor of shape (B, 1).
                Values are 0.0 (not done) or 1.0 (done).

        Returns:
            Scalar critic loss as a Python float — sum of MSE losses for
            both critics. Logged by PGRTrainer as "sac/critic_loss".
        """
        # ── Compute Bellman target (no gradient) ──────────────────────────────
        with torch.no_grad():
            # Sample next actions and log probs from the current actor.
            next_action: torch.Tensor
            next_log_prob: torch.Tensor
            next_action, next_log_prob = self.actor.sample(next_obs)
            # next_action: (B, action_dim), next_log_prob: (B, 1)

            # Compute target Q values using both target critics.
            target_q1: torch.Tensor = self.target_critic1(next_obs, next_action)  # (B, 1)
            target_q2: torch.Tensor = self.target_critic2(next_obs, next_action)  # (B, 1)

            # Clipped Double Q: take the minimum to reduce overestimation bias.
            target_q: torch.Tensor = torch.min(target_q1, target_q2)  # (B, 1)

            # Subtract entropy term: alpha * log_prob.
            # alpha is detached so its gradient does not flow through critic loss.
            alpha: torch.Tensor = self.log_alpha.exp().detach()  # scalar
            target_q = target_q - alpha * next_log_prob  # (B, 1)

            # Bellman target: y = r + gamma * (1 - done) * target_q
            # (1 - done) zeros out the bootstrap for terminal states.
            bellman_target: torch.Tensor = (
                rewards + self.gamma * (1.0 - dones) * target_q
            )  # (B, 1)

        # ── Compute current Q estimates ───────────────────────────────────────
        q1: torch.Tensor = self.critic1(obs, actions)  # (B, 1)
        q2: torch.Tensor = self.critic2(obs, actions)  # (B, 1)

        # ── Critic loss: sum of MSE for both critics ──────────────────────────
        # Using sum (not mean) of the two critic losses is standard in SAC
        # implementations and ensures both critics receive equal gradient signal.
        critic_loss: torch.Tensor = (
            F.mse_loss(q1, bellman_target) + F.mse_loss(q2, bellman_target)
        )

        # ── Gradient step ─────────────────────────────────────────────────────
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return float(critic_loss.item())

    def _update_actor(
        self,
        obs: torch.Tensor,
    ) -> Tuple[float, float]:
        """Updates the actor to maximize Q-value minus entropy.

        Samples new actions from the current policy, computes Q-values from
        both critics (not targets), and maximizes the minimum Q minus the
        entropy-regularized log probability.

        The critic forward passes use detached Q-values to prevent gradients
        from flowing back into the critic parameters during the actor update.
        This is achieved by calling critics in a torch.no_grad() context for
        the Q-value computation, then using those values in the actor loss.

        Args:
            obs: Current observations, float32 tensor of shape (B, obs_dim).

        Returns:
            Tuple of:
                - actor_loss: Scalar actor loss as a Python float.
                - mean_log_prob: Mean log probability of sampled actions,
                  used to compute entropy for logging.
        """
        # Sample new actions from the current policy.
        # Gradients flow through action and log_prob back to actor parameters.
        new_action: torch.Tensor
        log_prob: torch.Tensor
        new_action, log_prob = self.actor.sample(obs)
        # new_action: (B, action_dim), log_prob: (B, 1)

        # Compute Q-values from both critics.
        # We detach the Q-values to prevent gradients from flowing into the
        # critic parameters during the actor update — only actor parameters
        # should be updated here.
        q1_new: torch.Tensor = self.critic1(obs, new_action)  # (B, 1)
        q2_new: torch.Tensor = self.critic2(obs, new_action)  # (B, 1)

        # Take the minimum Q-value (conservative estimate).
        min_q: torch.Tensor = torch.min(q1_new, q2_new)  # (B, 1)

        # Actor loss: maximize (Q - alpha * log_prob) = minimize (alpha * log_prob - Q)
        # alpha is detached so its gradient does not flow through actor loss.
        alpha: torch.Tensor = self.log_alpha.exp().detach()  # scalar
        actor_loss: torch.Tensor = (alpha * log_prob - min_q).mean()

        # ── Gradient step ─────────────────────────────────────────────────────
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return float(actor_loss.item()), float(log_prob.mean().item())

    def _update_alpha(
        self,
        obs: torch.Tensor,
    ) -> float:
        """Updates the entropy coefficient alpha via automatic entropy tuning.

        Minimizes the alpha loss to drive the policy entropy toward the
        target entropy. Uses detached log_prob from a fresh actor sample
        to avoid coupling the alpha gradient with the actor gradient.

        Alpha loss: L(α) = -E[α * (log π(a|s) + H_target)]
                         = -mean(log_alpha * (log_prob + target_entropy))

        When log_prob > -target_entropy (entropy too low), alpha increases
        to encourage more exploration. When log_prob < -target_entropy
        (entropy too high), alpha decreases to focus the policy.

        Args:
            obs: Current observations, float32 tensor of shape (B, obs_dim).
                Used to sample fresh log_prob values for the alpha update.

        Returns:
            Scalar alpha loss as a Python float. Logged by PGRTrainer
            as "sac/alpha_loss".
        """
        # Sample fresh log_prob values — detached from the actor computation
        # graph to prevent the alpha gradient from affecting actor parameters.
        with torch.no_grad():
            _new_action: torch.Tensor
            log_prob_detached: torch.Tensor
            _new_action, log_prob_detached = self.actor.sample(obs)
            # log_prob_detached: (B, 1), detached from actor graph

        # Alpha loss: -mean(log_alpha * (log_prob + target_entropy))
        # log_alpha is the learnable parameter; log_prob is treated as a constant.
        alpha_loss: torch.Tensor = -(
            self.log_alpha * (log_prob_detached + self.target_entropy).detach()
        ).mean()

        # ── Gradient step ─────────────────────────────────────────────────────
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return float(alpha_loss.item())

    def _soft_update_targets(self) -> None:
        """Performs Polyak (EMA) averaging to update both target critics.

        Implements the soft update rule:
            θ_target = τ * θ_online + (1 - τ) * θ_target

        Applied independently to both (critic1, target_critic1) and
        (critic2, target_critic2) pairs. Called at the end of every
        update() call.

        The tau=0.005 value from config.policy.tau provides slow, stable
        target updates that prevent oscillation in the Bellman target.
        """
        # Update target_critic1 from critic1.
        for param, target_param in zip(
            self.critic1.parameters(), self.target_critic1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        # Update target_critic2 from critic2.
        for param, target_param in zip(
            self.critic2.parameters(), self.target_critic2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def __repr__(self) -> str:
        """Returns a concise string representation of the SAC policy."""
        current_alpha: float = (
            float(self.log_alpha.exp().item())
            if self.auto_alpha
            else self._alpha_fixed
        )
        return (
            f"SACPolicy("
            f"obs_dim={self.obs_dim}, "
            f"action_dim={self.action_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"gamma={self.gamma}, "
            f"tau={self.tau}, "
            f"alpha={current_alpha:.4f}, "
            f"auto_alpha={self.auto_alpha}, "
            f"device='{self.device}')"
        )
