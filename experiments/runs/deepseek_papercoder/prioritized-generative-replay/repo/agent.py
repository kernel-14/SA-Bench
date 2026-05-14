## agent.py

"""
Reinforcement learning agents used in Prioritized Generative Replay (PGR).

Provides an abstract base `PolicyAgent` and two concrete implementations:
- `REDQAgent` for state‑based continuous control tasks.
- `DRQv2Agent` for pixel‑based visual control tasks.

Both agents can seamlessly mix real environment data and synthetically generated
(replay) data during training. The training logic follows the protocols described
in the paper and its references (REDQ, DRQ‑v2).
"""

import abc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from config import Config                     # type: ignore  (assumes config.py in package)
from replay_buffer import ReplayBuffer        # type: ignore  (assumes replay_buffer.py in package)
from utils import get_device, to_tensor       # type: ignore  (assumes utils.py in package)


# ------------------------------------------------------------------------------
# Utility for creating MLP layers
# ------------------------------------------------------------------------------
def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int],
    activation: str = "relu",
    output_activation: bool = False,
) -> nn.Sequential:
    """Build a simple MLP with the given hidden layers.

    Parameters
    ----------
    input_dim : int
    output_dim : int
    hidden_sizes : list of ints
        Sizes of hidden layers (can be empty for a linear mapping).
    activation : str
        Name of PyTorch functional activation to use (default 'relu').
    output_activation : bool
        If True, apply the activation also after the final layer.

    Returns
    -------
    nn.Sequential
    """
    act_fn = getattr(F, activation)
    layers = []
    prev_dim = input_dim
    for hdim in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hdim))
        layers.append(nn.ReLU() if activation == "relu" else nn.ELU())
        prev_dim = hdim
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation:
        layers.append(nn.ReLU() if activation == "relu" else nn.ELU())
    return nn.Sequential(*layers)


# ------------------------------------------------------------------------------
# Data augmentation for pixel‐based tasks (DRQ‑v2)
# ------------------------------------------------------------------------------
class RandomShiftAug:
    """Applies a consistent random translation (4‑px padding + random crop) to
    pairs of images (state and next_state). The same shift is applied per
    transition, which is important for correct target computation.

    Usage
    -----
    aug = RandomShiftAug()
    shifted_states, shifted_next_states = aug(states, next_states)
    """

    def __init__(self, pad: int = 4, image_size: int = 84):
        self.pad = pad
        self.image_size = image_size

    def __call__(
        self,
        states: torch.Tensor,          # (B, C, H, W)
        next_states: torch.Tensor,     # (B, C, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the same random shift to a batch of state/next_state pairs.

        Parameters
        ----------
        states : torch.Tensor
            Batch of images, shape (B, C, H, W).
        next_states : torch.Tensor
            Batch of images, shape (B, C, H, W).

        Returns
        -------
        shifted_states, shifted_next_states : torch.Tensor
            Augmented images of the same shape.
        """
        B, C, H, W = states.shape
        assert H == W == self.image_size, f"Expected {self.image_size}x{self.image_size}, got {H}x{W}"

        # Pad all images by self.pad on each side (reflect or replicate)
        # Using 'replicate' padding to avoid border artifacts
        pad = self.pad
        padded_states = F.pad(states, [pad, pad, pad, pad], mode='replicate')
        padded_next = F.pad(next_states, [pad, pad, pad, pad], mode='replicate')
        _, _, pH, pW = padded_states.shape

        # Random offsets for each sample in [0, 2*pad]
        dx = torch.randint(0, 2 * pad + 1, (B,), device=states.device)
        dy = torch.randint(0, 2 * pad + 1, (B,), device=states.device)

        # We need to crop a HxW patch from the padded image at position (dx, dy)
        # Since the images are batched, we construct a 4D tensor of the same batch.
        # This can be done efficiently with advanced indexing.
        # Create indices for each batch element
        idx_h = torch.arange(self.image_size, device=states.device)
        idx_w = torch.arange(self.image_size, device=states.device)

        # For each sample we need to add dx[i] and dy[i] to the base indices
        # We'll use list comprehension for simplicity; performance is acceptable
        # for moderate batch sizes.
        shifted_states = []
        shifted_next_states = []
        for i in range(B):
            h_start = dy[i].item()
            w_start = dx[i].item()
            crop_s = padded_states[i, :, h_start : h_start + self.image_size,
                                   w_start : w_start + self.image_size]
            crop_ns = padded_next[i, :, h_start : h_start + self.image_size,
                                  w_start : w_start + self.image_size]
            shifted_states.append(crop_s)
            shifted_next_states.append(crop_ns)

        shifted_states = torch.stack(shifted_states, dim=0)
        shifted_next_states = torch.stack(shifted_next_states, dim=0)
        return shifted_states, shifted_next_states


# ------------------------------------------------------------------------------
# CNN encoder for DRQ‑v2 (from pixels to 256‑dim latent)
# ------------------------------------------------------------------------------
class CNNEncoder(nn.Module):
    """Convolutional encoder mapping 84x84 RGB images to a 256‑dimensional
    latent vector, following the architecture used in DRQ‑v2 / SynthER.

    Layers:
        Conv2d(3,32,3,stride=2) -> ReLU
        Conv2d(32,32,3,stride=1) -> ReLU
        Conv2d(32,32,3,stride=1) -> ReLU
        Conv2d(32,32,3,stride=1) -> LayerNorm
        Flatten -> Linear(32*35*35 -> 256)  [output resolution: 84//2=42, then three stride‑1: 42-2=40, etc.]
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )
        self.ln = nn.LayerNorm([32, 42, 42])          # after the first stride‑2 conv, size halved (84->42)
        self.fc = nn.Linear(32 * 42 * 42, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x shape (B, 3, 84, 84) -> (B, 256)."""
        h = self.conv(x)           # (B, 32, 42, 42)
        h = self.ln(h)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)


# ------------------------------------------------------------------------------
# Abstract base class for all PGR‑compatible agents
# ------------------------------------------------------------------------------
class PolicyAgent(abc.ABC):
    """Abstract agent that can be trained with mixed real/synthetic replay buffers.

    Parameters
    ----------
    state_dim : int or tuple
        Dimensionality of the observation space (int for vectors, tuple for images).
    action_dim : int
        Dimensionality of the continuous action space.
    config : Config
        Master configuration object.
    """

    def __init__(
        self,
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int,
        config: Config,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = get_device()

    @abc.abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Sample an action from the current stochastic policy."""

    @abc.abstractmethod
    def train(
        self,
        real_buffer: ReplayBuffer,
        syn_buffer: ReplayBuffer,
        synthetic_ratio: float,
    ) -> Dict[str, float]:
        """Perform one policy update cycle (UTD gradient steps)."""

    @abc.abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save all model/optimiser parameters."""

    @abc.abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Restore model/optimiser parameters."""

    def _mix_batch(
        self,
        real_buffer: ReplayBuffer,
        syn_buffer: ReplayBuffer,
    ) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions mixing real and synthetic data.

        The number of real transitions is kept constant (config.real_per_batch),
        and the remainder is drawn from the synthetic buffer. The total batch
        size is config.batch_size.

        Returns
        -------
        states, actions, rewards, next_states, dones : torch.Tensor
            Tensors of shape (batch_size, ...). Device is set to self.device.
        """
        num_real = self.config.replay_buffer.real_per_batch
        num_syn = self.config.replay_buffer.batch_size - num_real

        # Sample from real buffer
        real_states, real_actions, real_rewards, real_next_states, real_dones = real_buffer.sample(num_real)
        # Sample from synthetic buffer
        syn_states, syn_actions, syn_rewards, syn_next_states, syn_dones = syn_buffer.sample(num_syn)

        # Concatenate
        states = torch.cat([real_states, syn_states], dim=0)
        actions = torch.cat([real_actions, syn_actions], dim=0)
        rewards = torch.cat([real_rewards, syn_rewards], dim=0)
        next_states = torch.cat([real_next_states, syn_next_states], dim=0)
        dones = torch.cat([real_dones, syn_dones], dim=0)

        # Move to device
        return (
            to_tensor(states, self.device),
            to_tensor(actions, self.device),
            to_tensor(rewards, self.device),
            to_tensor(next_states, self.device),
            to_tensor(dones, self.device),
        )


# ------------------------------------------------------------------------------
# REDQ agent for state‑based continuous control
# ------------------------------------------------------------------------------
class REDQAgent(PolicyAgent):
    """Randomized Ensembled Double Q‑learning agent for state‑based tasks.

    Implements an ensemble of K critics, UTD ratio, automatic entropy tuning,
    and soft target updates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Config,
    ) -> None:
        super().__init__(state_dim, action_dim, config)

        # ---- networks ----
        self.actor = build_mlp(
            state_dim,
            2 * action_dim,
            config.policy.hidden_sizes,
            activation=config.policy.activation,
        ).to(self.device)

        K = config.policy.critic_ensemble_size
        self.critics = nn.ModuleList([
            build_mlp(
                state_dim + action_dim,
                1,
                config.policy.hidden_sizes,
                activation=config.policy.activation,
            ).to(self.device) for _ in range(K)
        ])
        self.target_critics = nn.ModuleList([
            build_mlp(
                state_dim + action_dim,
                1,
                config.policy.hidden_sizes,
                activation=config.policy.activation,
            ).to(self.device) for _ in range(K)
        ])
        # initialise target nets to match
        for critic, target in zip(self.critics, self.target_critics):
            target.load_state_dict(critic.state_dict())

        # ---- entropy tuning ----
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.target_entropy = -action_dim

        # ---- optimisers ----
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=config.policy.learning_rate)
        self.critic_optim = optim.Adam(
            sum([list(c.parameters()) for c in self.critics], []),
            lr=config.policy.learning_rate,
        )
        self.alpha_optim = optim.Adam([self.log_alpha], lr=config.policy.learning_rate)

        # store hyperparameters
        self.gamma = config.policy.discount_gamma
        self.tau = config.policy.polyak_tau
        self.utd_ratio = config.policy.utd_ratio
        self.K = K

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action using the current stochastic policy."""
        with torch.no_grad():
            state_t = to_tensor(state, self.device).unsqueeze(0)
            mean, log_std = self.actor(state_t).chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()
            z = torch.randn_like(mean)
            action = torch.tanh(mean + z * std)
        return action.cpu().numpy().flatten()

    def train(
        self,
        real_buffer: ReplayBuffer,
        syn_buffer: ReplayBuffer,
        synthetic_ratio: float,
    ) -> Dict[str, float]:
        """Perform UTD gradient steps mixing real and synthetic data."""
        metrics = {
            'critic_loss': 0.0,
            'actor_loss': 0.0,
            'alpha_loss': 0.0,
            'alpha': 0.0,
            'q1_mean': 0.0,
        }

        for _ in range(self.utd_ratio):
            # Sample mixed batch
            states, actions, rewards, next_states, dones = self._mix_batch(real_buffer, syn_buffer)

            # ---- update critics ----
            # Compute next actions and log probs
            with torch.no_grad():
                next_actions, next_log_probs = self._sample_action(next_states)
                # Random subset of M=2 critics for target computation (as in REDQ)
                # For each critic, we sample a different random subset
                alpha = self.log_alpha.exp()
                idxs = torch.randperm(self.K, device=self.device)[:2]   # M=2
                # Use the minimum over the selected subset
                target_qs = torch.stack([self.target_critics[i](torch.cat([next_states, next_actions], dim=-1))
                                         for i in idxs])
                min_target_q, _ = torch.min(target_qs, dim=0)
                target = rewards + self.gamma * (1.0 - dones) * (min_target_q - alpha * next_log_probs)

            # Compute current Q estimates
            current_qs = [crit(torch.cat([states, actions], dim=-1)) for crit in self.critics]

            # Critic loss: MSE averaged over all critics
            critic_loss = sum(F.mse_loss(q, target) for q in current_qs) / self.K

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # ---- update actor ----
            new_actions, log_probs = self._sample_action(states)
            q_vals = torch.cat([crit(torch.cat([states, new_actions], dim=-1))
                                for crit in self.critics], dim=-1)
            min_q = q_vals.min(dim=-1, keepdim=True)[0]
            actor_loss = (alpha * log_probs - min_q).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # ---- update alpha ----
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            # ---- soft update target networks ----
            with torch.no_grad():
                for critic, target_critic in zip(self.critics, self.target_critics):
                    for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                        target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

            # ---- accumulate metrics ----
            metrics['critic_loss'] += critic_loss.item() / self.utd_ratio
            metrics['actor_loss'] += actor_loss.item() / self.utd_ratio
            metrics['alpha_loss'] += alpha_loss.item() / self.utd_ratio
            metrics['alpha'] += alpha.item() / self.utd_ratio
            metrics['q1_mean'] += current_qs[0].mean().item() / self.utd_ratio

        return metrics

    def _sample_action(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return log probability under the current policy."""
        mean, log_std = self.actor(states).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        z = torch.randn_like(mean)
        actions = torch.tanh(mean + z * std)
        # Log probability computation for tanh‑squashed Gaussian
        log_probs = (
            -((mean - actions.atanh()) ** 2) / (2 * std.pow(2))
            - log_std
            - np.log(2 * np.pi) / 2
        )
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        # Correction for tanh squashing
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return actions, log_probs

    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            'actor_state': self.actor.state_dict(),
            'critics_state': [c.state_dict() for c in self.critics],
            'target_critics_state': [c.state_dict() for c in self.target_critics],
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'alpha_optim': self.alpha_optim.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state'])
        for i, c in enumerate(self.critics):
            c.load_state_dict(checkpoint['critics_state'][i])
        for i, c in enumerate(self.target_critics):
            c.load_state_dict(checkpoint['target_critics_state'][i])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])
        self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])
        self.log_alpha.data = checkpoint['log_alpha'].to(self.device)


# ------------------------------------------------------------------------------
# DRQ‑v2 agent for pixel‑based continuous control
# ------------------------------------------------------------------------------
class DRQv2Agent(PolicyAgent):
    """Deep Reinforcement Learning from Pixels (DRQ‑v2) agent.

    Uses a CNN encoder to process 84x84 RGB images into a 256‑dim latent vector,
    and trains actor/critic on these latents. Supports synthetic data already in
    latent form.
    """

    def __init__(
        self,
        state_dim: Tuple[int, int, int],   # (C, H, W) = (3, 84, 84)
        action_dim: int,
        config: Config,
    ) -> None:
        super().__init__(state_dim, action_dim, config)

        self.latent_dim = 256

        # ---- networks ----
        self.encoder = CNNEncoder().to(self.device)
        # Actor head: latent -> (action_dim*2)
        self.actor = build_mlp(
            self.latent_dim,
            2 * action_dim,
            [1024, 1024],
            activation='relu',
        ).to(self.device)
        # Critic head (two separate) : latent + action -> 1
        self.critic1 = build_mlp(
            self.latent_dim + action_dim, 1, [1024, 1024], activation='relu'
        ).to(self.device)
        self.critic2 = build_mlp(
            self.latent_dim + action_dim, 1, [1024, 1024], activation='relu'
        ).to(self.device)
        self.target_critic1 = build_mlp(
            self.latent_dim + action_dim, 1, [1024, 1024], activation='relu'
        ).to(self.device)
        self.target_critic2 = build_mlp(
            self.latent_dim + action_dim, 1, [1024, 1024], activation='relu'
        ).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # ---- automatic entropy tuning ----
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.target_entropy = -action_dim

        # ---- data augmentation ----
        self.aug = RandomShiftAug(pad=4, image_size=84)

        # ---- optimisers ----
        # Encoder + actor share one optimiser (gradients from actor loss)
        self.encoder_actor_optim = optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()),
            lr=config.policy.learning_rate,
        )
        self.critic_optim = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=config.policy.learning_rate,
        )
        self.alpha_optim = optim.Adam([self.log_alpha], lr=config.policy.learning_rate)

        # ---- hyperparameters ----
        self.gamma = config.policy.discount_gamma
        self.tau = config.policy.polyak_tau
        self.utd_ratio = config.policy.utd_ratio

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Sample an action given a raw pixel observation (C,H,W or H,W,C)."""
        with torch.no_grad():
            # ensure channel‑first
            if state.shape[-1] == 3:  # H,W,C -> C,H,W
                state = np.transpose(state, (2, 0, 1))
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.0
            latent = self.encoder(state_t)
            mean, log_std = self.actor(latent).chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()
            z = torch.randn_like(mean)
            action = torch.tanh(mean + z * std)
        return action.cpu().numpy().flatten()

    def encode(self, state: np.ndarray) -> np.ndarray:
        """Encode a raw pixel observation (or batch) into a latent vector."""
        with torch.no_grad():
            if state.ndim == 3:
                # single image: ensure channel‑first
                if state.shape[-1] == 3:
                    state = np.transpose(state, (2, 0, 1))
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.0
            else:
                # batch: shape (N, C, H, W) or (N, H, W, C)
                if state.shape[-1] == 3:
                    state = np.transpose(state, (0, 3, 1, 2))
                state_t = torch.from_numpy(state).float().to(self.device) / 255.0
            latent = self.encoder(state_t)
        return latent.cpu().numpy()

    def train(
        self,
        real_buffer: ReplayBuffer,
        syn_buffer: ReplayBuffer,
        synthetic_ratio: float,
    ) -> Dict[str, float]:
        """Perform UTD gradient steps mixing real and synthetic data."""
        metrics = {
            'critic_loss': 0.0,
            'actor_loss': 0.0,
            'alpha_loss': 0.0,
            'alpha': 0.0,
            'q1_mean': 0.0,
        }

        for _ in range(self.utd_ratio):
            # Sample mixed batch (raw pixels for real, latents for synthetic)
            # _mix_batch returns tensors; real states/next_states are still raw pixels,
            # synthetic ones are already latents.
            states_raw, actions, rewards, next_states_raw, dones = self._mix_batch(real_buffer, syn_buffer)

            # Determine which samples are raw pixels (shape (B, 3, 84, 84))
            # Synthetic states are assumed to be latents of shape (B, 256)
            is_raw = (states_raw.shape[-1] != self.latent_dim)

            # Prepare latent states and latent next_states
            if is_raw.any():
                # Extract raw parts; we assume all samples in a batch come from the same buffer,
                # but due to cat they are mixed. We need to separate based on shape.
                # A simple approach: if any sample is raw, we re‑split the batch into real and synthetic parts.
                # Since we know the counts (num_real, num_syn), we can slice.
                num_real = self.config.replay_buffer.real_per_batch
                # real data are the first num_real entries, synthetic are the rest.
                real_states = states_raw[:num_real]       # (num_real, 3, 84, 84)
                real_next = next_states_raw[:num_real]
                syn_states = states_raw[num_real:]         # (num_syn, 256)
                syn_next = next_states_raw[num_real:]

                # Augment and encode real data
                if num_real > 0:
                    aug_states, aug_next = self.aug(real_states, real_next)
                    latent_real_states = self.encoder(aug_states)
                    latent_real_next = self.encoder(aug_next)
                else:
                    latent_real_states = torch.empty(0, self.latent_dim, device=self.device)
                    latent_real_next = torch.empty(0, self.latent_dim, device=self.device)

                # Synthetic latents are already encoded
                latent_syn_states = syn_states
                latent_syn_next = syn_next

                # Concatenate to form full latent batch
                states = torch.cat([latent_real_states, latent_syn_states], dim=0)
                next_states = torch.cat([latent_real_next, latent_syn_next], dim=0)
            else:
                # All data is already latent (should not happen when mixing, but handle gracefully)
                states = states_raw
                next_states = next_states_raw

            # ---- update critics ----
            with torch.no_grad():
                next_actions, next_log_probs = self._sample_action(next_states)
                # target Q: min(Q1_target, Q2_target)
                target_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=-1))
                target_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=-1))
                min_target_q = torch.min(target_q1, target_q2)
                target = rewards + self.gamma * (1.0 - dones) * (
                    min_target_q - self.log_alpha.exp() * next_log_probs
                )

            current_q1 = self.critic1(torch.cat([states, actions], dim=-1))
            current_q2 = self.critic2(torch.cat([states, actions], dim=-1))
            critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # ---- update actor (and encoder) ----
            new_actions, log_probs = self._sample_action(states.detach())   # detach to avoid critic grad flowing through encoder
            q1 = self.critic1(torch.cat([states, new_actions], dim=-1))
            q2 = self.critic2(torch.cat([states, new_actions], dim=-1))
            min_q = torch.min(q1, q2)
            actor_loss = (self.log_alpha.exp() * log_probs - min_q).mean()

            self.encoder_actor_optim.zero_grad()
            actor_loss.backward()
            self.encoder_actor_optim.step()

            # ---- update alpha ----
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            # ---- soft update target critics ----
            with torch.no_grad():
                for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
                for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

            # ---- accumulate metrics ----
            metrics['critic_loss'] += critic_loss.item() / self.utd_ratio
            metrics['actor_loss'] += actor_loss.item() / self.utd_ratio
            metrics['alpha_loss'] += alpha_loss.item() / self.utd_ratio
            metrics['alpha'] += self.log_alpha.exp().item() / self.utd_ratio
            metrics['q1_mean'] += current_q1.mean().item() / self.utd_ratio

        return metrics

    def _sample_action(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and log_prob from actor given latent states."""
        mean, log_std = self.actor(latent).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        z = torch.randn_like(mean)
        actions = torch.tanh(mean + z * std)
        # Log probability
        log_probs = (
            -((mean - actions.atanh()) ** 2) / (2 * std.pow(2))
            - log_std
            - np.log(2 * np.pi) / 2
        ).sum(dim=-1, keepdim=True)
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return actions, log_probs

    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            'encoder_state': self.encoder.state_dict(),
            'actor_state': self.actor.state_dict(),
            'critic1_state': self.critic1.state_dict(),
            'critic2_state': self.critic2.state_dict(),
            'target_critic1_state': self.target_critic1.state_dict(),
            'target_critic2_state': self.target_critic2.state_dict(),
            'encoder_actor_optim': self.encoder_actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'alpha_optim': self.alpha_optim.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.actor.load_state_dict(checkpoint['actor_state'])
        self.critic1.load_state_dict(checkpoint['critic1_state'])
        self.critic2.load_state_dict(checkpoint['critic2_state'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state'])
        self.encoder_actor_optim.load_state_dict(checkpoint['encoder_actor_optim'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])
        self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])
        self.log_alpha.data = checkpoint['log_alpha'].to(self.device)

