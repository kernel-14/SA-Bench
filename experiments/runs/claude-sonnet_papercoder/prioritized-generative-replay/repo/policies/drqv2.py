```python
## policies/drqv2.py
"""DRQ-v2 (Data-augmented RL v2) policy for pixel-based DMC tasks in PGR.

Implements the DRQ-v2 policy (Yarats et al., 2021) as the RL backbone for
pixel-based experiments in Prioritized Generative Replay. The CNN encoder
f_θ maps stacked pixel observations to latent vectors that are used by both
the diffusion model and ICM, enabling generation in latent space rather than
raw pixel space.

Key design choices aligned with the paper (Section 5):
    - Deterministic actor (DDPG-style) with Gaussian exploration noise
    - CNN encoder shared between actor and critics, updated via critic loss
    - Random shift augmentation applied inside update() for regularization
    - encode() method exposes latent representations for diffusion/ICM use
    - Transitions stored as (f_θ(s), a, f_θ(s'), r) in D_real for pixel tasks

Config references (config.yaml):
    drqv2.feature_dim:  50      # CNN encoder output dimension
    drqv2.hidden_dim:   1024    # actor/critic MLP hidden dimension
    drqv2.lr:           1e-4    # Adam learning rate
    policy.gamma:       0.99    # discount factor
    policy.tau:         0.005   # target network EMA coefficient
    env.image_size:     84      # pixel observation spatial resolution
    env.frame_stack:    3       # number of stacked frames
"""

import copy
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sub-network: CNN Encoder ──────────────────────────────────────────────────


class CNNEncoder(nn.Module):
    """Convolutional encoder f_θ mapping stacked pixel observations to latent vectors.

    Implements the standard DRQ-v2 CNN architecture (Yarats et al., 2021):
        4 × Conv2d layers (32 filters, 3×3 kernels) with ReLU activations
        → Flatten
        → LayerNorm (key DRQ-v2 stabilization trick)
        → Linear projection to feature_dim
        → Tanh activation to bound the latent representation

    The encoder is shared between the actor and critics. It is updated only
    through the critic loss (not the actor loss, where latents are detached).

    For pixel-based PGR experiments, the diffusion model and ICM operate on
    the output of this encoder rather than raw pixels, per the paper:
    "we follow Lu et al. (2024); Esser et al. (2021) and generate data in
    the latent space of the policy's CNN visual encoder."

    Attributes:
        obs_channels: Number of input channels (frame_stack * 3 = 9 by default).
        feature_dim: Output latent dimension. Corresponds to
            config.drqv2.feature_dim (default 50).
        conv_layers: ModuleList of 4 convolutional layers.
        layer_norm: LayerNorm applied to the flattened conv output.
        linear_proj: Linear projection from flattened dim to feature_dim.
        flattened_dim: Computed flattened dimension after all conv layers.
    """

    def __init__(
        self,
        obs_channels: int = 9,
        feature_dim: int = 50,
    ) -> None:
        """Initialises the CNN encoder.

        Computes the flattened dimension after all conv layers by performing
        a dummy forward pass through the conv backbone with a zero tensor of
        the expected input shape (image_size=84 from config.yaml).

        Args:
            obs_channels: Number of input image channels. For frame-stacked
                pixel observations: frame_stack * 3 (e.g. 9 for 3 stacked
                RGB frames). Corresponds to config.env.frame_stack * 3
                (default 3 * 3 = 9).
            feature_dim: Output latent dimension. Corresponds to
                config.drqv2.feature_dim (default 50).
        """
        super().__init__()

        self.obs_channels: int = obs_channels
        self.feature_dim: int = feature_dim

        # ── Convolutional backbone ────────────────────────────────────────────
        # Standard DRQ-v2 architecture: 4 conv layers with 32 filters each.
        # Conv1: stride=2 (halves spatial dims from 84 → 41)
        # Conv2-4: stride=1 (reduces by kernel-1=2 each: 41→39→37→35)
        # All use 3×3 kernels with no padding.
        self.conv_layers: nn.ModuleList = nn.ModuleList([
            nn.Conv2d(obs_channels, 32, kernel_size=3, stride=2),  # 84 → 41
            nn.Conv2d(32, 32, kernel_size=3, stride=1),             # 41 → 39
            nn.Conv2d(32, 32, kernel_size=3, stride=1),             # 39 → 37
            nn.Conv2d(32, 32, kernel_size=3, stride=1),             # 37 → 35
        ])

        # ── Compute flattened dimension via dummy forward pass ─────────────────
        # Use image_size=84 from config.yaml (config.env.image_size).
        # This is more robust than hardcoding 39200 = 32 * 35 * 35.
        with torch.no_grad():
            dummy_input: torch.Tensor = torch.zeros(
                1, obs_channels, 84, 84, dtype=torch.float32
            )
            dummy_out: torch.Tensor = dummy_input
            for conv in self.conv_layers:
                dummy_out = F.relu(conv(dummy_out))
            self.flattened_dim: int = int(dummy_out.view(1, -1).shape[1])

        # ── LayerNorm on flattened conv output ────────────────────────────────
        # Key DRQ-v2 stabilization: normalizes the flattened feature vector
        # before the linear projection. Prevents gradient explosion from
        # large pixel value variations across episodes.
        self.layer_norm: nn.LayerNorm = nn.LayerNorm([self.flattened_dim])

        # ── Linear projection: flattened_dim → feature_dim ───────────────────
        # Maps the normalized conv features to the compact latent space.
        # feature_dim=50 from config.drqv2.feature_dim.
        self.linear_proj: nn.Linear = nn.Linear(self.flattened_dim, feature_dim)

        # ── Weight initialization ─────────────────────────────────────────────
        # Orthogonal initialization for conv layers (standard for DRQ-v2).
        # Default PyTorch initialization for the linear projection.
        for conv in self.conv_layers:
            nn.init.orthogonal_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of stacked pixel observations into latent vectors.

        Normalizes pixel values from [0, 255] to [0, 1] internally. Applies
        4 conv layers with ReLU, LayerNorm on the flattened output, a linear
        projection, and tanh to bound the latent representation.

        Args:
            obs: Pixel observation tensor of shape (B, obs_channels, H, W).
                Values can be in [0, 255] (uint8 or float) — normalized to
                [0, 1] inside this method. Must be on the same device as the
                network parameters.

        Returns:
            Float32 tensor of shape (B, feature_dim) — bounded latent vectors
            with values in (-1, 1) from the final tanh activation.
        """
        # Normalize pixel values to [0, 1].
        # Guard: only normalize if values appear to be in uint8 range.
        obs_f: torch.Tensor = obs.float()
        if obs_f.max() > 1.0:
            obs_f = obs_f / 255.0

        # Pass through 4 conv layers with ReLU activations.
        h: torch.Tensor = obs_f
        for conv in self.conv_layers:
            h = F.relu(conv(h))
        # h shape: (B, 32, 35, 35) for image_size=84

        # Flatten spatial dimensions: (B, 32, 35, 35) → (B, flattened_dim)
        h = h.view(h.size(0), -1)  # (B, flattened_dim)

        # Apply LayerNorm to the flattened features.
        h = self.layer_norm(h)  # (B, flattened_dim)

        # Linear projection: (B, flattened_dim) → (B, feature_dim)
        h = self.linear_proj(h)  # (B, feature_dim)

        # Tanh activation to bound the latent representation to (-1, 1).
        # This is standard in DRQ-v2 and improves training stability.
        return torch.tanh(h)  # (B, feature_dim)


# ── Sub-network: Actor MLP ────────────────────────────────────────────────────


class _ActorMLP(nn.Module):
    """Deterministic actor MLP mapping latent observations to actions.

    DRQ-v2 uses a deterministic actor (DDPG-style) rather than a stochastic
    Gaussian actor like SAC/REDQ. Exploration comes from additive Gaussian
    noise in select_action(), not from the policy distribution.

    Architecture:
        feature_dim → hidden_dim → ReLU → hidden_dim → ReLU → action_dim → tanh

    Attributes:
        fc1: First hidden layer (feature_dim → hidden_dim).
        fc2: Second hidden layer (hidden_dim → hidden_dim).
        output_layer: Output layer (hidden_dim → action_dim).
    """

    def __init__(
        self,
        feature_dim: int = 50,
        action_dim: int = 6,
        hidden_dim: int = 1024,
    ) -> None:
        """Initialises the deterministic actor MLP.

        Args:
            feature_dim: Input dimension (CNN encoder output). Corresponds to
                config.drqv2.feature_dim (default 50).
            action_dim: Output action dimension. Inferred from the environment.
            hidden_dim: Width of hidden layers. Corresponds to
                config.drqv2.hidden_dim (default 1024).
        """
        super().__init__()

        self.fc1: nn.Linear = nn.Linear(feature_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer: nn.Linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Computes deterministic actions from latent observations.

        Args:
            latent: Float32 tensor of shape (B, feature_dim) — CNN encoder
                output. Must be on the same device as the network parameters.

        Returns:
            Float32 tensor of shape (B, action_dim) with values in (-1, 1)
            from the final tanh activation.
        """
        h: torch.Tensor = F.relu(self.fc1(latent))   # (B, hidden_dim)
        h = F.relu(self.fc2(h))                        # (B, hidden_dim)
        return torch.tanh(self.output_layer(h))        # (B, action_dim)


# ── Sub-network: Critic MLP ───────────────────────────────────────────────────


class _CriticMLP(nn.Module):
    """Q-network MLP mapping (latent, action) pairs to scalar Q-values.

    Used as both Q1 and Q2 in the twin critic setup. Each critic is
    independently initialized and trained.

    Architecture:
        (feature_dim + action_dim) → hidden_dim → ReLU → hidden_dim → ReLU → 1

    Attributes:
        fc1: First hidden layer ((feature_dim + action_dim) → hidden_dim).
        fc2: Second hidden layer (hidden_dim → hidden_dim).
        output_layer: Output layer (hidden_dim → 1).
    """

    def __init__(
        self,
        feature_dim: int = 50,
        action_dim: int = 6,
        hidden_dim: int = 1024,
    ) -> None:
        """Initialises the critic MLP.

        Args:
            feature_dim: Latent observation dimension (CNN encoder output).
                Corresponds to config.drqv2.feature_dim (default 50).
            action_dim: Action dimension. Inferred from the environment.
            hidden_dim: Width of hidden layers. Corresponds to
                config.drqv2.hidden_dim (default 1024).
        """
        super().__init__()

        self.fc1: nn.Linear = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer: nn.Linear = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Computes Q-values for (latent, action) pairs.

        Args:
            latent: Float32 tensor of shape (B, feature_dim) — CNN encoder
                output. Must be on the same device as the network parameters.
            action: Float32 tensor of shape (B, action_dim) — actions taken.
                Must be on the same device as the network parameters.

        Returns:
            Float32 tensor of shape (B, 1) — scalar Q-values.
        """
        # Concatenate latent and action along the feature dimension.
        x: torch.Tensor = torch.cat([latent, action], dim=-1)  # (B, feature_dim + action_dim)
        h: torch.Tensor = F.relu(self.fc1(x))   # (B, hidden_dim)
        h = F.relu(self.fc2(h))                   # (B, hidden_dim)
        return self.output_layer(h)               # (B, 1)


# ── Main Class: DRQv2Policy ───────────────────────────────────────────────────


class DRQv2Policy:
    """DRQ-v2 policy with CNN encoder, deterministic actor, and twin Q-networks.

    Implements Data-augmented RL v2 (Yarats et al., 2021) as the RL backbone
    for pixel-based DMC experiments in PGR. The CNN encoder f_θ is shared
    between actor and critics, and its output latent vectors are used by the
    conditional diffusion model and ICM for pixel-based tasks.

    Key features:
        - Deterministic actor with Gaussian exploration noise (DDPG-style)
        - Twin Q-networks (Clipped Double Q-learning) for stable value estimates
        - Random shift augmentation applied inside update() for regularization
        - encode() method exposes latent representations for diffusion/ICM use
        - Encoder updated through critic loss only (not actor loss)

    The paper states (Section 5): "For pixel-based tasks, our policy is based
    on DRQ-V2 (Yarats et al., 2021) as in Lu et al. (2022). To maintain the
    same approach and architecture for our generative model, we follow Lu et al.
    (2024); Esser et al. (2021) and generate data in the latent space of the
    policy's CNN visual encoder."

    Attributes:
        obs_channels: Number of input image channels (frame_stack * 3).
        action_dim: Action space dimension.
        feature_dim: CNN encoder output dimension (config.drqv2.feature_dim=50).
        hidden_dim: Actor/critic MLP hidden width (config.drqv2.hidden_dim=1024).
        gamma: Discount factor (config.policy.gamma=0.99).
        tau: Target network EMA coefficient (config.policy.tau=0.005).
        device: PyTorch device string.
        encoder: CNNEncoder f_θ — shared between actor and critics.
        actor: Deterministic actor MLP.
        critic1: First Q-network (twin critic 1).
        critic2: Second Q-network (twin critic 2).
        target_encoder: Frozen EMA copy of encoder.
        target_critic1: Frozen EMA copy of critic1.
        target_critic2: Frozen EMA copy of critic2.
        encoder_optimizer: Adam optimizer for encoder (updated with critics).
        actor_optimizer: Adam optimizer for actor only.
        critic_optimizer: Adam optimizer for critics + encoder.
    """

    def __init__(
        self,
        obs_channels: int = 9,
        action_dim: int = 6,
        feature_dim: int = 50,
        hidden_dim: int = 1024,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda",
    ) -> None:
        """Initialises the DRQ-v2 policy with all sub-networks and optimizers.

        Creates the CNN encoder, actor MLP, twin critic MLPs, and their
        corresponding target networks (deep copies, frozen). Sets up three
        Adam optimizers: one for the actor, one for the critics+encoder
        (updated together during critic updates), and one for the encoder
        alone (kept for reference but encoder is updated via critic_optimizer).

        Args:
            obs_channels: Number of input image channels. For frame-stacked
                pixel observations: frame_stack * 3 (e.g. 9 for 3 stacked
                RGB frames). Corresponds to config.env.frame_stack * 3
                (default 3 * 3 = 9).
            action_dim: Action space dimension. Inferred from the environment
                wrapper at PGRTrainer init time (e.g. 6 for walker-walk).
            feature_dim: CNN encoder output dimension. Corresponds to
                config.drqv2.feature_dim (default 50).
            hidden_dim: Actor/critic MLP hidden layer width. Corresponds to
                config.drqv2.hidden_dim (default 1024).
            lr: Adam optimizer learning rate for all optimizers. Corresponds
                to config.drqv2.lr (default 1e-4).
            gamma: Discount factor for Bellman target computation. Corresponds
                to config.policy.gamma (default 0.99).
            tau: Target network EMA coefficient for soft updates. Corresponds
                to config.policy.tau (default 0.005).
            device: PyTorch device string. Corresponds to
                config.hardware.device (default "cuda").
        """
        self.obs_channels: int = obs_channels
        self.action_dim: int = action_dim
        self.feature_dim: int = feature_dim
        self.hidden_dim: int = hidden_dim
        self.gamma: float = gamma
        self.tau: float = tau
        self.device: str = device

        # ── CNN Encoder f_θ ───────────────────────────────────────────────────
        # Shared between actor and critics. Updated through critic loss only.
        self.encoder: CNNEncoder = CNNEncoder(
            obs_channels=obs_channels,
            feature_dim=feature_dim,
        ).to(device)

        # ── Deterministic Actor MLP ───────────────────────────────────────────
        # DDPG-style deterministic policy. Exploration via additive Gaussian
        # noise in select_action(), not from a stochastic distribution.
        self.actor: _ActorMLP = _ActorMLP(
            feature_dim=feature_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        # ── Twin Critic MLPs ──────────────────────────────────────────────────
        # Two independent Q-networks for Clipped Double Q-learning.
        # Independent random initialization provides diverse value estimates.
        self.critic1: _CriticMLP = _CriticMLP(
            feature_dim=feature_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        self.critic2: _CriticMLP = _CriticMLP(
            feature_dim=feature_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        # ── Target Networks (deep copies, frozen) ─────────────────────────────
        # Initialized as exact copies of the online networks.
        # Updated only via soft (Polyak) averaging — never by backpropagation.
        self.target_encoder: CNNEncoder = copy.deepcopy(self.encoder).to(device)
        self.target_critic1: _CriticMLP = copy.deepcopy(self.critic1).to(device)
        self.target_critic2: _CriticMLP = copy.deepcopy(self.critic2).to(device)

        # Freeze all target network parameters — prevents any accidental
        # gradient accumulation even if the optimizer were misconfigured.
        for target_net in [self.target_encoder, self.target_critic1, self.target_critic2]:
            for param in target_net.parameters():
                param.requires_grad = False

        # Set target networks to eval mode permanently.
        self.target_encoder.eval()
        self.target_critic1.eval()
        self.target_critic2.eval()

        # ── Optimizers ────────────────────────────────────────────────────────
        # Actor optimizer: only actor parameters.
        # Encoder is NOT updated during actor updates (latents are detached).
        self.actor_optimizer: torch.optim.Adam = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr,
        )

        # Critic optimizer: parameters from both critics AND the encoder.
        # In DRQ-v2, the encoder is updated jointly with the critics to learn
        # features that are useful for value estimation.
        self.critic_optimizer: torch.optim.Adam = torch.optim.Adam(
            list(self.critic1.parameters())
            + list(self.critic2.parameters())
            + list(self.encoder.parameters()),
            lr=lr,
        )

        # Encoder optimizer: kept as a separate reference for potential use
        # in analysis (e.g., computing encoder gradient norms). In practice,
        # the encoder is updated via critic_optimizer.
        self.encoder_optimizer: torch.optim.Adam = torch.optim.Adam(
            self.encoder.parameters(),
            lr=lr,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Selects an action for the given pixel observation.

        Encodes the pixel observation through the CNN encoder, then passes
        the latent through the deterministic actor MLP. During training
        (deterministic=False), adds small Gaussian exploration noise clipped
        to [-0.3, 0.3] — the DRQ-v2 exploration strategy.

        No random shift augmentation is applied during action selection —
        augmentation is only used during training (inside update()).

        Args:
            obs: Pixel observation as a numpy array. Expected shape:
                (obs_channels, H, W) = (9, 84, 84) for 3-frame stacked RGB.
                Values in [0, 255] (uint8 or float). Will be converted to a
                (1, obs_channels, H, W) tensor internally.
            deterministic: If True, returns the raw actor output without
                exploration noise. Used for policy evaluation by Evaluator.
                If False (default), adds Gaussian noise for exploration
                during online data collection.

        Returns:
            Float32 numpy array of shape (action_dim,) with values in [-1, 1].
            Actions are bounded by the tanh activation in the actor and
            clipped to [-1, 1] after noise addition.
        """
        # Convert numpy observation to (1, C, H, W) tensor on device.
        # Handle both (C, H, W) and (B, C, H, W) input shapes.
        obs_np: np.ndarray = np.asarray(obs, dtype=np.float32)
        if obs_np.ndim == 3:
            # (C, H, W) → (1, C, H, W)
            obs_tensor: torch.Tensor = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        else:
            # Already (B, C, H, W) — use as-is.
            obs_tensor = torch.FloatTensor(obs_np).to(self.device)

        # Set networks to eval mode for inference.
        self.encoder.eval()
        self.actor.eval()

        with torch.no_grad():
            # Encode pixel observation to latent vector.
            latent: torch.Tensor = self.encoder(obs_tensor)  # (1, feature_dim)

            # Compute deterministic action from latent.
            action: torch.Tensor = self.actor(latent)  # (1, action_dim)

            if not deterministic:
                # Add Gaussian exploration noise clipped to [-0.3, 0.3].
                # This is the DRQ-v2 exploration strategy — simpler than SAC's
                # stochastic policy but effective for continuous control tasks.
                noise: torch.Tensor = torch.randn_like(action) * 0.1
                noise = noise.clamp(-0.3, 0.3)
                action = action + noise

            # Clip final action to [-1, 1] to ensure valid range after noise.
            action = action.clamp(-1.0, 1.0)

        # Restore networks to training mode.
        self.encoder.train()
        self.actor.train()

        # Convert to numpy: squeeze batch dimension → (action_dim,)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Performs one full DRQ-v2 update cycle on a mixed batch.

        Applies random shift augmentation to pixel observations, then performs
        critic and actor updates. The encoder is updated jointly with the
        critics (not the actor). Target networks are soft-updated at the end.

        Random shift augmentation (the defining feature of DRQ-v2):
            1. Pad each observation by 4 pixels on each side (replicate padding)
            2. Randomly crop back to the original (84, 84) spatial size
            3. Applied independently to observations and next_observations
            4. Applied fresh each call (not pre-computed) for regularization

        Critic update:
            - Compute Bellman target using target encoder + target critics
            - Minimize MSE between current Q estimates and Bellman target
            - Encoder is updated jointly with critics

        Actor update:
            - Encode observations with the online encoder (detached)
            - Maximize minimum Q-value over both critics
            - Encoder is NOT updated during actor update

        The batch follows the universal transition dict format:
            'observations':      (B, obs_channels, H, W) — raw pixel tensors
            'actions':           (B, action_dim)
            'next_observations': (B, obs_channels, H, W) — raw pixel tensors
            'rewards':           (B, 1)
            'dones':             (B, 1)
        All tensors are float32 on self.device.

        Args:
            batch: Transition dict from MixedSampler.sample(). For pixel tasks,
                'observations' and 'next_observations' are raw pixel tensors
                of shape (B, obs_channels, H, W) with values in [0, 255].
                Normalization to [0, 1] happens inside CNNEncoder.forward().
                Batch size B corresponds to config.sampling.batch_size (default 256).

        Returns:
            Dict of scalar training metrics for logging:
                'critic_loss': Sum of MSE losses for both critics.
                'actor_loss':  Mean negative Q-value (actor maximizes Q).
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

        # ── Apply random shift augmentation ───────────────────────────────────
        # Augment observations and next_observations independently.
        # Each call to _random_shift produces a different random crop.
        obs_aug: torch.Tensor = self._random_shift(obs)           # (B, C, 84, 84)
        next_obs_aug: torch.Tensor = self._random_shift(next_obs)  # (B, C, 84, 84)

        # ── Step 1: Update critics (and encoder jointly) ──────────────────────
        critic_loss: float = self._update_critics(
            obs_aug, actions, next_obs_aug, rewards, dones
        )

        # ── Step 2: Update actor (encoder detached) ───────────────────────────
        # Fresh augmentation for actor update — independent from critic update.
        obs_aug_actor: torch.Tensor = self._random_shift(obs)  # (B, C, 84, 84)
        actor_loss: float = self._update_actor(obs_aug_actor)

        # ── Step 3: Soft update of target networks ────────────────────────────
        self._soft_update_targets()

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }

    def encode(self, obs: np.ndarray) -> torch.Tensor:
        """Encodes pixel observations into latent vectors using the CNN encoder.

        Called by PGRTrainer._collect_transition() to get latent representations
        before storing transitions to D_real. Also used by ICMRelevance and
        Evaluator for analysis experiments.

        The encoder is NOT frozen — it continues to be updated by critic updates
        during training. This means the latent space evolves over time, which is
        intentional and matches the paper's approach of generating in the latent
        space of the current policy encoder.

        Per the paper (Section 5): "given a visual encoder f_θ, and a transition
        (s, a, s', r) for pixel observations s, s' ∈ R^{3×h×w}, we learn to
        (conditionally) generate transitions (f_θ(s), a, f_θ(s'), r)."

        Args:
            obs: Pixel observation(s) as a numpy array or torch.Tensor.
                Accepted shapes:
                    - (obs_channels, H, W): single observation → returns (1, feature_dim)
                    - (B, obs_channels, H, W): batch → returns (B, feature_dim)
                Values in [0, 255] (uint8 or float). Normalization to [0, 1]
                happens inside CNNEncoder.forward().

        Returns:
            Float32 tensor of shape (B, feature_dim) on self.device.
            Detached from the computation graph (no gradients).
        """
        # Convert numpy to tensor if needed.
        if isinstance(obs, np.ndarray):
            obs_tensor: torch.Tensor = torch.FloatTensor(obs).to(self.device)
        else:
            obs_tensor = obs.to(device=self.device,