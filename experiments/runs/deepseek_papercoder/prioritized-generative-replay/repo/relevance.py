## relevance.py

"""
Relevance functions for Prioritized Generative Replay (PGR).

Provides an abstract base class `RelevanceFunction` and four concrete implementations:
- `RewardRelevance`: uses immediate reward as relevance.
- `ReturnRelevance`: uses Q(s, π(s)) as relevance.
- `TDErrorRelevance`: uses TD error as relevance.
- `CuriosityRelevance`: uses intrinsic curiosity (ICM) as relevance.

All relevance functions operate on batched PyTorch tensors.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim

from config import Config

if TYPE_CHECKING:
    from replay_buffer import ReplayBuffer
    from agent import PolicyAgent
else:
    ReplayBuffer = "ReplayBuffer"
    PolicyAgent = "PolicyAgent"


# ------------------------------------------------------------------------------
# Abstract base class
# ------------------------------------------------------------------------------

class RelevanceFunction(ABC):
    """Abstract relevance function. Every method expects batched tensors."""

    @abstractmethod
    def compute(self, state: Tensor, action: Tensor, next_state: Tensor,
                reward: Tensor) -> Tensor:
        """Compute scalar relevance values for a batch of transitions.

        Parameters
        ----------
        state, action, next_state, reward : Tensor
            Batched tensors with leading batch dimension.

        Returns
        -------
        Tensor of shape (B,) containing the relevance value per transition.
        """
        pass

    def update(self, buffer: ReplayBuffer) -> None:
        """Optional parameter update step (e.g., for curiosity). Called once
        per environment step for learnable relevance functions."""
        pass


# ------------------------------------------------------------------------------
# 1. Reward‑based relevance
# ------------------------------------------------------------------------------

class RewardRelevance(RelevanceFunction):
    """Relevance = immediate reward."""

    def compute(self, state: Tensor, action: Tensor, next_state: Tensor,
                reward: Tensor) -> Tensor:
        # reward may be (B,1) or (B,) – squeeze to (B,)
        return reward.squeeze(-1)


# ------------------------------------------------------------------------------
# 2. Return‑based relevance (Q‑value)
# ------------------------------------------------------------------------------

class ReturnRelevance(RelevanceFunction):
    """Relevance = Q(s, π(s)).  Uses the current policy and Q‑ensemble."""

    def __init__(self, agent: PolicyAgent) -> None:
        """
        Parameters
        ----------
        agent : PolicyAgent
            Must provide `actor` (returns mean, log_std) and a list of critics.
        """
        super().__init__()
        self.agent = agent

    def compute(self, state: Tensor, action: Tensor, next_state: Tensor,
                reward: Tensor) -> Tensor:
        with torch.no_grad():
            # Deterministic policy action
            mean, _ = self.agent.actor(state).chunk(2, dim=-1)
            # Q‑values from the ensemble
            input_tensor = torch.cat([state, mean], dim=-1)
            q_vals = torch.stack([critic(input_tensor)
                                  for critic in self.agent.critics], dim=0)  # (K, B, 1)
            q_mean = q_vals.mean(dim=0).squeeze(-1)  # (B,)
        return q_mean


# ------------------------------------------------------------------------------
# 3. TD‑error‑based relevance
# ------------------------------------------------------------------------------

class TDErrorRelevance(RelevanceFunction):
    """Relevance = r + γ Q_target(s', argmax_{a'} Q(s', a')) - Q(s, a).

    Approximates the argmax with the current policy's deterministic action.
    """

    def __init__(self, agent: PolicyAgent, gamma: float = 0.99) -> None:
        """
        Parameters
        ----------
        agent : PolicyAgent
            Must provide `actor`, a list of critics and target_critics.
        gamma : float
            Discount factor.
        """
        super().__init__()
        self.agent = agent
        self.gamma = gamma

    def compute(self, state: Tensor, action: Tensor, next_state: Tensor,
                reward: Tensor) -> Tensor:
        with torch.no_grad():
            # Q(s, a) – current estimate
            input_current = torch.cat([state, action], dim=-1)
            q_current = torch.stack(
                [critic(input_current) for critic in self.agent.critics], dim=0
            ).mean(dim=0)  # (B, 1)

            # Next action (policy's deterministic action)
            mean_next, _ = self.agent.actor(next_state).chunk(2, dim=-1)

            # Q_target(s', a')
            input_target = torch.cat([next_state, mean_next], dim=-1)
            q_target = torch.stack(
                [target_critic(input_target)
                 for target_critic in self.agent.target_critics], dim=0
            ).mean(dim=0)  # (B, 1)

            td_error = reward + self.gamma * q_target - q_current
            td_error = td_error.squeeze(-1)  # (B,)
        return td_error


# ------------------------------------------------------------------------------
# 4. Curiosity‑based relevance (Intrinsic Curiosity Module)
# ------------------------------------------------------------------------------

class CuriosityRelevance(RelevanceFunction):
    """Relevance = 0.5 * || g(h(s), a) - h(s') ||^2.

    Learns an encoder h and a forward dynamics model g to predict next‑state
    features.  The prediction error serves as the relevance condition.

    The encoder can be:
      - a simple MLP (state‑based environments)
      - a small CNN (pixel‑based environments)
      - a frozen pre‑trained feature extractor (e.g., the policy's visual encoder)
        with an optional linear projection.

    Parameters
    ----------
    state_shape : Union[int, Tuple[int, ...]]
        If int, the dimension of a flat state vector.
        If tuple, the shape of a pixel observation (C, H, W).
    action_dim : int
        Dimensionality of the continuous action space.
    config : Config
        Master configuration.  Uses sections `relevance.curiosity` and
        `replay_buffer.batch_size` (for `update`).
    feature_extractor : Optional[nn.Module]
        If provided, this module is frozen and its output is passed through
        an optional projection to produce the encoded representation.
    batch_size : int, optional
        Size of the training batch used in `update`. Defaults to 256.
    """

    def __init__(
        self,
        state_shape: Union[int, Tuple[int, ...]],
        action_dim: int,
        config: Config,
        feature_extractor: Optional[nn.Module] = None,
        batch_size: int = 256,
    ) -> None:
        super().__init__()
        self.encoded_dim = 256
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Determine if we are in a pixel-based setting
        self.is_pixel = isinstance(state_shape, tuple) and len(state_shape) == 3
        self.state_shape = state_shape  # keep for internal use

        # --- Build encoder ---
        if feature_extractor is not None:
            # Freeze the provided feature extractor
            self.feature_extractor = feature_extractor
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            self.feature_extractor.to(self.device)

            # Determine its output dimension
            dummy = torch.zeros(1, *self.state_shape, device=self.device)
            with torch.no_grad():
                out = self.feature_extractor(dummy)
            feat_dim = out.shape[-1]
            if feat_dim != self.encoded_dim:
                self.encoder = nn.Linear(feat_dim, self.encoded_dim)
            else:
                self.encoder = nn.Identity()
        else:
            # No external extractor: build our own
            if self.is_pixel:
                self.encoder = self._build_pixel_encoder()
            else:
                # state_shape is an integer (state_dim)
                self.encoder = self._build_state_encoder(state_shape, config)

        # Move own encoder to device (feature_extractor already moved)
        self.encoder.to(self.device)

        # --- Forward dynamics model ---
        self.forward_dynamics = self._build_forward_dynamics(action_dim, config)
        self.forward_dynamics.to(self.device)

        # --- Optimizer ---
        lr = config.relevance.curiosity.learning_rate
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.forward_dynamics.parameters()),
            lr=lr,
        )

        # Put everything in eval mode for `compute`
        self.eval()

    # ------------------------------------------------------------------
    # Internal helpers for building modules
    # ------------------------------------------------------------------
    def _build_state_encoder(self, state_dim: int, config: Config) -> nn.Module:
        """MLP encoder for flat state vectors."""
        hidden_sizes = config.relevance.curiosity.encoder_hidden_sizes
        layers = []
        prev_dim = state_dim
        for hdim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, self.encoded_dim))
        return nn.Sequential(*layers)

    def _build_pixel_encoder(self) -> nn.Module:
        """Simple CNN encoder for 84×84 pixel inputs."""
        # Input: (C, 84, 84)
        # After three stride‑2 convolutions → approx. (64, 10, 10) → flatten
        cnn = nn.Sequential(
            nn.Conv2d(self.state_shape[0], 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flattened size using a dummy tensor
        dummy = torch.zeros(1, *self.state_shape)
        with torch.no_grad():
            flat = cnn(dummy)
        flatten_dim = flat.shape[-1]
        return nn.Sequential(
            cnn,
            nn.Linear(flatten_dim, self.encoded_dim),
        )

    def _build_forward_dynamics(self, action_dim: int, config: Config) -> nn.Module:
        """MLP that maps concatenated [encoded_state, action] to encoded_next_state."""
        hidden_sizes = config.relevance.curiosity.forward_dynamics_hidden_sizes
        layers = []
        prev_dim = self.encoded_dim + action_dim
        for hdim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, self.encoded_dim))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Mode management helpers
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> None:
        self.encoder.train(mode)
        self.forward_dynamics.train(mode)
        if hasattr(self, 'feature_extractor'):
            # keep frozen feature extractor in eval
            self.feature_extractor.eval()
        super().train(mode)

    def eval(self) -> None:
        self.train(False)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def compute(self, state: Tensor, action: Tensor, next_state: Tensor,
                reward: Tensor) -> Tensor:
        """Compute relevance values (prediction error) without gradients."""
        # Ensure eval mode
        if self.training:
            self.eval()
        with torch.no_grad():
            if hasattr(self, 'feature_extractor'):
                feat_s = self.feature_extractor(state)
                feat_s = self.encoder(feat_s)      # possibly identity
                feat_ns = self.feature_extractor(next_state)
                feat_ns = self.encoder(feat_ns)
            else:
                feat_s = self.encoder(state)
                feat_ns = self.encoder(next_state)

            pred_ns = self.forward_dynamics(torch.cat([feat_s, action], dim=-1))
            relevance = 0.5 * ((pred_ns - feat_ns) ** 2).sum(dim=-1)  # (B,)
        return relevance

    def update(self, buffer: ReplayBuffer) -> None:
        """Perform one gradient step on a batch from the real replay buffer."""
        self.train()  # switch to train mode

        # Sample a batch (we only need state, action, next_state)
        states, actions, _, next_states, _ = buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)

        # Forward with gradients
        if hasattr(self, 'feature_extractor'):
            feat_s = self.feature_extractor(states)
            feat_s = self.encoder(feat_s)
            feat_ns = self.feature_extractor(next_states)
            feat_ns = self.encoder(feat_ns)
        else:
            feat_s = self.encoder(states)
            feat_ns = self.encoder(next_states)

        pred_ns = self.forward_dynamics(torch.cat([feat_s, actions], dim=-1))
        loss = F.mse_loss(pred_ns, feat_ns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return to eval mode for subsequent compute calls
        self.eval()


# ----------------------------------------------------------------------
__all__ = [
    "RelevanceFunction",
    "RewardRelevance",
    "ReturnRelevance",
    "TDErrorRelevance",
    "CuriosityRelevance",
]
