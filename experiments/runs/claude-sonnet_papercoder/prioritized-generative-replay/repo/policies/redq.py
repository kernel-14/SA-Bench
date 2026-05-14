```python
## policies/redq.py
"""REDQ (Randomized Ensembled Double Q-learning) policy for Prioritized Generative Replay (PGR).

Implements the primary RL backbone used in PGR experiments. REDQ extends SAC
with a large ensemble of Q-networks (ensemble_size=10) and a high update-to-data
(UTD) ratio (utd_ratio=20), enabling sample-efficient learning from mixed batches
of real and synthetic transitions.

Key algorithmic contributions of REDQ (Chen et al., 2021):
    1. Ensemble of 10 Q-networks instead of 2 twin critics
    2. Subsampling subsample_size=2 critics for target computation — prevents
       overestimation bias at high UTD ratios without requiring all 10 critics

Config references (config.yaml):
    policy.hidden_dim:     256   # baseline hidden width (512 for scaling)
    policy.hidden_layers:  2     # baseline depth (3 for scaling)
    policy.utd_ratio:      20    # gradient steps per env step (40 for scaling)
    policy.ensemble_size:  10    # Q-network ensemble size (Chen et al., 2021)
    policy.subsample_size: 2     # critics subsampled for target Q (Chen et al., 2021)
    policy.gamma:          0.99  # discount factor
    policy.tau:            0.005 # target network EMA coefficient
    policy.lr:             3e-4  # Adam learning rate
    policy.auto_alpha:     true  # automatic entropy tuning
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ── Sub-network: Gaussian Actor ───────────────────────────────────────────────


class GaussianActor(nn.Module):
    """Stochastic policy network outputting a Gaussian distribution over actions.

    Applies tanh squashing to enforce action bounds in [-1, 1]. The policy
    is used by REDQPolicy for both action selection (select_action) and
    computing log probabilities for the actor and alpha updates.

    Architecture:
        obs_dim → [hidden_dim → ReLU] × num_layers → hidden_dim
        → mean_layer (hidden_dim → action_dim)
        → log_std_layer (hidden_dim → action_dim), clamped to [-20, 2]

    Attributes:
        obs_dim: Observation space dimension.
        action_dim: Action space dimension.
        hidden_dim: Width of each hidden layer.
        num_layers: Number of hidden layers.
        hidden_layers: ModuleList of hidden linear layers.
        mean_layer: Output head for action mean.
        log_std_layer: Output head for action log standard deviation.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        """Initialises the Gaussian actor network.

        Args:
            obs_dim: Observation space dimension. Corresponds to the flat
                observation dimension from the environment wrapper.
            action_dim: Action space dimension. Corresponds to
                env.action_space_dim().
            hidden_dim: Width of each hidden layer. Corresponds to
                config.policy.hidden_dim (default 256; 512 for scaling).
            num_layers: Number of hidden layers. Corresponds to
                config.policy.hidden_layers (default 2; 3 for scaling).
        """
        super().__init__()

        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers

        # ── Hidden layers ─────────────────────────────────────────────────────
        # First layer: obs_dim → hidden_dim
        # Subsequent layers: hidden_dim → hidden_dim
        hidden_layer_list: List[nn.Linear] = []
        in_dim: int = obs_dim
        for _ in range(num_layers):
            hidden_layer_list.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.hidden_layers: nn.ModuleList = nn.ModuleList(hidden_layer_list)

        # ── Output heads ──────────────────────────────────────────────────────
        # Separate linear layers for mean and log_std — both map hidden_dim → action_dim.
        # Tanh squashing is applied in sample(), not here.
        self.mean_layer: nn.Linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer: nn.Linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the Gaussian distribution parameters for a batch of observations.

        Passes observations through all hidden layers with ReLU activations,
        then computes mean and log_std via separate output heads. The log_std
        is clamped to [-20, 2] to prevent numerical instability from extreme
        standard deviation values.

        Args:
            obs: Float32 tensor of shape (B, obs_dim) — current observations.
                Must be on the same device as the network parameters.

        Returns:
            Tuple of (mean, log_std), both float32 tensors of shape
            (B, action_dim). log_std is clamped to [-20, 2].
            No tanh squashing is applied here — that happens in sample().
        """
        h: torch.Tensor = obs

        # Pass through all hidden layers with ReLU activations.
        for layer in self.hidden_layers:
            h = F.relu(layer(h))

        # Compute mean and log_std from separate output heads.
        mean: torch.Tensor = self.mean_layer(h)  # (B, action_dim)
        log_std: torch.Tensor = self.log_std_layer(h).clamp(-20.0, 2.0)  # (B, action_dim)

        return mean, log_std

    def sample(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples actions using the reparameterization trick with tanh squashing.

        Implements the SAC action sampling procedure:
            1. Compute (mean, log_std) via forward()
            2. Sample pre-tanh action: x_t = mean + std * ε, ε ~ N(0, I)
            3. Apply tanh squashing: action = tanh(x_t)
            4. Compute log probability with tanh correction

        The tanh correction accounts for the change of variables:
            log π(a|s) = log N(x_t; mean, std) - Σ log(1 - tanh²(x_t) + ε)

        The 1e-6 epsilon in the correction prevents log(0) at the boundaries
        of tanh (where tanh(x_t) ≈ ±1).

        Args:
            obs: Float32 tensor of shape (B, obs_dim) — current observations.
                Must be on the same device as the network parameters.

        Returns:
            Tuple of:
                - action: Float32 tensor of shape (B, action_dim) with values
                  in (-1, 1) after tanh squashing.
                - log_prob: Float32 tensor of shape (B, 1) — log probability
                  of the sampled action under the current policy, with tanh
                  correction applied.
        """
        mean, log_std = self.forward(obs)
        std: torch.Tensor = log_std.exp()  # (B, action_dim)

        # Reparameterization trick: x_t = mean + std * ε, ε ~ N(0, I)
        # Using Normal distribution for clean log_prob computation.
        dist: Normal = Normal(mean, std)
        x_t: torch.Tensor = dist.rsample()  # (B, action_dim) — reparameterized sample

        # Apply tanh squashing to enforce action bounds in (-1, 1).
        action: torch.Tensor = torch.tanh(x_t)  # (B, action_dim)

        # Compute log probability with tanh correction.
        # log π(a|s) = log N(x_t; mean, std) - Σ log(1 - tanh²(x_t) + ε)
        # sum(dim=-1, keepdim=True) sums over action dimensions → (B, 1)
        log_prob: torch.Tensor = (
            dist.log_prob(x_t).sum(dim=-1, keepdim=True)
            - torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        )  # (B, 1)

        return action, log_prob


# ── Sub-network: Q-Network ────────────────────────────────────────────────────


class QNetwork(nn.Module):
    """Single Q-network mapping (obs, action) to a scalar Q-value.

    Ten instances of this class form the REDQ ensemble. Each network is
    independently initialized and trained, providing diverse Q-value estimates
    that reduce overestimation bias at high UTD ratios.

    Architecture:
        (obs_dim + action_dim) → [hidden_dim → ReLU] × num_layers → 1

    Attributes:
        obs_dim: Observation space dimension.
        action_dim: Action space dimension.
        hidden_dim: Width of each hidden layer.
        num_layers: Number of hidden layers.
        hidden_layers: ModuleList of hidden linear layers.
        output_layer: Final linear layer mapping hidden_dim → 1.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        """Initialises a single Q-network.

        Args:
            obs_dim: Observation space dimension.
            action_dim: Action space dimension.
            hidden_dim: Width of each hidden layer. Corresponds to
                config.policy.hidden_dim (default 256; 512 for scaling).
            num_layers: Number of hidden layers. Corresponds to
                config.policy.hidden_layers (default 2; 3 for scaling).
        """
        super().__init__()

        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers

        # ── Hidden layers ─────────────────────────────────────────────────────
        # First layer: (obs_dim + action_dim) → hidden_dim
        # Subsequent layers: hidden_dim → hidden_dim
        hidden_layer_list: List[nn.Linear] = []
        in_dim: int = obs_dim + action_dim
        for _ in range(num_layers):
            hidden_layer_list.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.hidden_layers: nn.ModuleList = nn.ModuleList(hidden_layer_list)

        # ── Output layer ──────────────────────────────────────────────────────
        # Maps hidden_dim → 1 (scalar Q-value per sample).
        self.output_layer: nn.Linear = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Computes Q-values for a batch of (observation, action) pairs.

        Args:
            obs: Float32 tensor of shape (B, obs_dim) — current observations.
                Must be on the same device as the network parameters.
            action: Float32 tensor of shape (B, action_dim) — actions taken.
                Must be on the same device as the network parameters.

        Returns:
            Float32 tensor of shape (B, 1) — scalar Q-values for each
            (obs, action) pair in the batch.
        """
        # Concatenate obs and action along the feature dimension.
        x: torch.Tensor = torch.cat([obs, action], dim=-1)  # (B, obs_dim + action_dim)

        # Pass through all hidden layers with ReLU activations.
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Output layer: (B, hidden_dim) → (B, 1)
        return self.output_layer(x)


# ── Main Class: REDQPolicy ────────────────────────────────────────────────────


class REDQPolicy:
    """REDQ policy with ensemble Q-networks and automatic entropy tuning.

    Implements Randomized Ensembled Double Q-learning (Chen et al., 2021) as
    the primary RL backbone for PGR. Key features:
        - Ensemble of ensemble_size=10 Q-networks for diverse value estimates
        - Subsampling subsample_size=2 critics for target Q computation
        - High UTD ratio (utd_ratio=20) for sample-efficient learning
        - Automatic entropy tuning (auto_alpha=True) following SAC

    The policy trains on mixed batches from D_real ∪ D_syn provided by
    MixedSampler. PGRTrainer calls update() exactly utd_ratio=20 times per
    environment step, each time with a freshly sampled batch.

    Attributes:
        obs_dim: Observation space dimension.
        action_dim: Action space dimension.
        hidden_dim: Width of hidden layers in actor and critics.
        num_layers: Number of hidden layers in actor and critics.
        ensemble_size: Number of Q-networks in the ensemble (default 10).
        subsample_size: Number of critics subsampled for target Q (default 2).
        utd_ratio: Update-to-data ratio (stored for reference; loop is external).
        gamma: Discount factor (default 0.99).
        tau: Target network EMA coefficient (default 0.005).
        auto_alpha: Whether to use automatic entropy tuning.
        target_entropy: Target entropy for automatic alpha tuning.
        device: PyTorch device string.
        actor: GaussianActor policy network.
        critics: ModuleList of ensemble_size QNetwork instances.
        target_critics: ModuleList of frozen target QNetwork instances.
        actor_optimizer: Adam optimizer for the actor.
        critic_optimizer: Adam optimizer for all critics (shared).
        log_alpha: Learnable log entropy coefficient.
        alpha_optimizer: Adam optimizer for log_alpha.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        ensemble_size: int = 10,
        subsample_size: int = 2,
        utd_ratio: int = 20,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        lr: float = 3e-4,
        device: str = "cuda",
    ) -> None:
        """Initialises the REDQ policy with all sub-networks and optimizers.

        Args:
            obs_dim: Observation space dimension. Inferred from the environment
                wrapper at PGRTrainer init time.
            action_dim: Action space dimension. Inferred from the environment
                wrapper at PGRTrainer init time.
            hidden_dim: Width of hidden layers in actor and all critics.
                Corresponds to config.policy.hidden_dim (default 256;
                512 for larger-network scaling experiment, Sec. 5.3).
            num_layers: Number of hidden layers in actor and all critics.
                Corresponds to config.policy.hidden_layers (default 2;
                3 for larger-network scaling experiment, Sec. 5.3).
            ensemble_size: Number of Q-networks in the ensemble. Corresponds
                to config.policy.ensemble_size (default 10, Chen et al., 2021).
            subsample_size: Number of critics subsampled for target Q computation.
                Corresponds to config.policy.subsample_size (default 2,
                Chen et al., 2021). Must satisfy subsample_size <= ensemble_size.
            utd_ratio: Update-to-data ratio. Stored as an attribute for reference;
                the actual UTD loop is external (in PGRTrainer._update_policy()).
                Corresponds to config.policy.utd_ratio (default 20; 40 for
                combined scaling experiment, Sec. 5.3).
            gamma: Discount factor for Bellman target computation. Corresponds
                to config.policy.gamma (default 0.99).
            tau: Target network EMA coefficient for soft updates. Corresponds
                to config.policy.tau (default 0.005).
            alpha: Initial entropy coefficient value. Used as the fixed alpha
                when auto_alpha=False. When auto_alpha=True, this is the
                initial value before automatic tuning takes over.
            auto_alpha: Whether to use automatic entropy tuning (SAC-style).
                Corresponds to config.policy.auto_alpha (default True).
            target_entropy: Target entropy for automatic alpha tuning. If None,
                defaults to -action_dim (standard SAC default: -dim(A)).
                Corresponds to the negative action space dimension.
            lr: Adam optimizer learning rate for actor, critics, and alpha.
                Corresponds to config.policy.lr (default 3e-4).
            device: PyTorch device string. Corresponds to
                config.hardware.device (default "cuda").
        """
        if subsample_size > ensemble_size:
            raise ValueError(
                f"subsample_size ({subsample_size}) must be <= "
                f"ensemble_size ({ensemble_size})."
            )

        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.ensemble_size: int = ensemble_size
        self.subsample_size: int = subsample_size
        self.utd_ratio: int = utd_ratio
        self.gamma: float = gamma
        self.tau: float = tau
        self.auto_alpha: bool = auto_alpha
        self.device: str = device

        # ── Target entropy ────────────────────────────────────────────────────
        # Standard SAC default: -dim(A) (negative action space dimension).
        # This encourages the policy to maintain a minimum level of entropy
        # equal to a uniform distribution over a 1D action space.
        self.target_entropy: float = (
            target_entropy if target_entropy is not None else float(-action_dim)
        )

        # ── Actor ─────────────────────────────────────────────────────────────
        self.actor: GaussianActor = GaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        # ── Critic ensemble ───────────────────────────────────────────────────
        # ensemble_size=10 independent Q-networks, each with independent
        # random initialization. Diversity in initialization is critical for
        # REDQ's variance reduction at high UTD ratios.
        self.critics: nn.ModuleList = nn.ModuleList(
            [
                QNetwork(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                )
                for _ in range(ensemble_size)
            ]
        ).to(device)

        # ── Target critic ensemble ────────────────────────────────────────────
        # Identical architecture to critics, initialized with the same weights.
        # Updated only via soft (Polyak) averaging — never by backpropagation.
        self.target_critics: nn.ModuleList = nn.ModuleList(
            [
                QNetwork(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                )
                for _ in range(ensemble_size)
            ]
        ).to(device)

        # Copy weights from critics to target critics.
        for critic, target_critic in zip(self.critics, self.target_critics):
            target_critic.load_state_dict(critic.state_dict())

        # Freeze target critics — they are updated only via soft update.
        # requires_grad=False prevents any accidental gradient accumulation.
        for target_critic in self.target_critics:
            for param in target_critic.parameters():
                param.requires_grad = False

        # ── Optimizers ────────────────────────────────────────────────────────
        # Actor optimizer: only actor parameters.
        self.actor_optimizer: torch.optim.Adam = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr,
        )

        # Critic optimizer: all parameters from all ensemble critics (shared).
        # Using a single optimizer for all critics is more efficient than
        # separate optimizers and produces identical results.
        all_critic_params: List[torch.nn.Parameter] = [
            param
            for critic in self.critics
            for param in critic.parameters()
        ]
        self.critic_optimizer: torch.optim.Adam = torch.optim.Adam(
            all_critic_params,
            lr=lr,
        )

        # ── Entropy coefficient (alpha) ───────────────────────────────────────
        # log_alpha parameterization ensures alpha = exp(log_alpha) > 0 always.
        # Initialized to log(alpha) so that exp(log_alpha) = alpha initially.
        self.log_alpha: torch.Tensor = torch.tensor(
            [float(np.log(alpha))],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.alpha_optimizer: torch.optim.Adam = torch.optim.Adam(
            [self.log_alpha],
            lr=lr,
        )

        # Store initial alpha value for use when auto_alpha=False.
        self._alpha_fixed: float = float(alpha)

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

        For stochastic selection: samples from the Gaussian policy with tanh
        squashing, returning an action in (-1, 1).
        For deterministic selection: returns tanh(mean) without sampling.

        Actions are in (-1, 1) from tanh squashing. The environment wrapper
        (DMCEnv/GymEnv) clips actions to the valid range — no rescaling is
        needed here since dm_control and gymnasium accept actions in [-1, 1]
        for normalized action spaces.

        Args:
            obs: Float32 numpy array of shape (obs_dim,) — current observation
                from the environment. Will be converted to a (1, obs_dim) tensor.
            deterministic: If True, returns tanh(mean) without sampling.
                Used for evaluation. If False (default), samples from the
                Gaussian policy for exploration during training.

        Returns:
            Float32 numpy array of shape (action_dim,) with values in (-1, 1).
        """
        # Convert numpy observation to (1, obs_dim) tensor on device.
        obs_tensor: torch.Tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Set actor to eval mode to disable any dropout (not present in our
        # architecture, but defensive practice for eval-time inference).
        self.actor.eval()

        with torch.no_grad():
            if deterministic:
                # Deterministic action: tanh(mean) without sampling.
                mean, _log_std = self.actor.forward(obs_tensor)
                action: torch.Tensor = torch.tanh(mean)  # (1, action_dim)
            else:
                # Stochastic action: sample from Gaussian with tanh squashing.
                action, _log_prob = self.actor.sample(obs_tensor)  # (1, action_dim)

        # Restore actor to training mode.
        self.actor.train()

        # Convert to numpy: squeeze batch dimension → (action_dim,)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Performs one full REDQ update cycle on a mixed batch.

        Called by PGRTrainer._update_policy() exactly utd_ratio=20 times per
        environment step, each time with a freshly sampled mixed batch from
        MixedSampler. Each call performs:
            1. Critic ensemble update (all ensemble_size critics)
            2. Actor update (using minimum Q over all critics)
            3. Alpha update (if auto_alpha=True)
            4. Soft target update (Polyak averaging)

        The batch follows the universal transition dict format:
            'observations':      (B, obs_dim)
            'actions':           (B, action_dim)
            'next_observations': (B, obs_dim)
            'rewards':           (B, 1)
            'dones':             (B, 1)
        All tensors are float32 on self.device (device placement handled by
        ReplayBuffer.sample() and MixedSampler.sample()).

        Args:
            batch: Transition dict from MixedSampler.sample(). Contains
                float32 tensors on self.device with the keys listed above.
                Batch size B corresponds to config.sampling.batch_size
                (default 256; 512 or 1024 for scaling experiments).

        Returns:
            Dict of scalar training metrics for logging:
                'critic_loss': Mean MSE loss across all ensemble critics.
                'actor_loss':  SAC actor loss (alpha * log_prob - min_Q).mean().
                'alpha_loss':  Entropy coefficient loss (0.0 if auto_alpha=False).
                'alpha':       Current entropy coefficient value.
                'mean_q':      Mean Q-value over the batch (from first critic).
        """
        # Extract tensors from batch — already on self.device with float32 dtype.
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

        # ── Step 1: Update critics ────────────────────────────────────────────
        critic_loss: float = self._update_critics(
            obs, actions, next_obs, rewards, dones
        )

        # ── Step 2: Update actor ──────────────────────────────────────────────
        actor_loss: float
        mean_q: float
        actor_loss, mean_q = self._update_actor(obs)

        # ── Step 3: Update alpha (entropy coefficient) ────────────────────────
        alpha_loss: float = 0.0
        if self.auto_alpha:
            alpha_loss = self._update_alpha(obs)

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
            "alpha_loss": alpha_loss,
            "alpha": current_alpha,
            "mean_q": mean_q,
        }

    def save(self, path: str) -> None:
        """Saves all policy state to a checkpoint file.

        Saves actor, all critics, all target critics, all optimizers, and
        the log_alpha parameter. Called by PGRTrainer.save_checkpoint().

        Args:
            path: Full file path for the checkpoint (e.g.
                "checkpoints/quadruped_walk_curiosity_seed0/policy.pt").
                Parent directories are created if they do not exist.
        """
        parent_dir: str = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        checkpoint: dict = {
            "actor": self.actor.state_dict(),
            "critics": [critic.state_dict() for critic in self.critics],
            "target_critics": [tc.state_dict() for tc in self.target_critics],
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            # Store config for verification on load.
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "ensemble_size": self.ensemble_size,
            "subsample_size": self.subsample_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "auto_alpha": self.auto_alpha,
            "target_entropy": self.target_entropy,
        }

        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Restores all policy state from a checkpoint file.

        Restores actor, all critics, all target critics, all optimizers, and
        the log_alpha parameter. Ensures target critics remain frozen after
        loading. Called by PGRTrainer.load_checkpoint().

        Args:
            path: Full file path to the checkpoint saved by save().

        Raises:
            FileNotFoundError: If the checkpoint file does not exist at path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"REDQPolicy checkpoint not found at '{path}'. "
                "Ensure the path is correct and the checkpoint was saved."
            )

        checkpoint: dict = torch.load(path, map_location=self.device)

        # Restore actor weights.
        self.actor.load_state_dict(checkpoint["actor"])

        # Restore all critic weights.
        for i, critic_state in enumerate(checkpoint["critics"]):
            self.critics[i].load_state_dict(critic_state)

        # Restore all target critic weights.
        for i, target_state in enumerate(checkpoint["target_critics"]):
            self.target_critics[i].load_state_dict(target_state)

        # Re-freeze target critics after loading (defensive — they should
        # already be frozen, but load_state_dict may not preserve requires_grad).
        for target_critic in self.target_critics:
            for param in target_critic.parameters():
                param.requires_grad = False

        # Restore optimizer states.
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

        # Restore log_alpha — move to device and re-enable gradient.
        log_alpha_loaded: torch.Tensor = checkpoint["log_alpha"].to(self.device)
        with torch.no_grad():
            self.