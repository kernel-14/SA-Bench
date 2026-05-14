import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from models.networks import GaussianActor, QNetwork, CNNEncoder, MLP, NoisyMLP


class SAC(nn.Module):
    """Soft Actor-Critic (Haarnoja et al., 2018).

    Off-policy maximum entropy deep RL with a stochastic actor.
    Used as a backbone for PGR in some experiments.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        tau: float = 0.005,
        gamma: float = 0.99,
        init_temperature: float = 0.1,
        target_entropy: Optional[float] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = GaussianActor(state_dim, action_dim, hidden_dim, n_hidden).to(device)
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim, n_hidden).to(device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim, n_hidden).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.log_alpha = nn.Parameter(torch.tensor(np.log(init_temperature), dtype=torch.float32))
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.get_action(state_t, deterministic)
        return action.cpu().numpy()[0]

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions, next_log_pi = self.actor.get_action(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_pi
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_pi = self.actor.get_action(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_pi - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def get_q_value(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        if action is None:
            action, _ = self.actor.get_action(state)
        return torch.min(self.critic1(state, action), self.critic2(state, action))


class REDQ(nn.Module):
    """Randomized Ensembled Double Q-Learning (Chen et al., 2021).

    Uses an ensemble of N Q-networks with M randomly selected for target computation.
    Supports high UTD ratios (default 20) for sample-efficient learning.
    Main backbone for PGR experiments.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        n_critics: int = 10,
        n_target_critics: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        tau: float = 0.005,
        gamma: float = 0.99,
        init_temperature: float = 0.1,
        target_entropy: Optional[float] = None,
        utd_ratio: int = 20,
        noisy: bool = False,
        use_bootstrap_mask: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.n_critics = n_critics
        self.n_target_critics = n_target_critics
        self.utd_ratio = utd_ratio
        self.use_bootstrap_mask = use_bootstrap_mask
        self.device = device

        self.actor = GaussianActor(state_dim, action_dim, hidden_dim, n_hidden, noisy=noisy).to(device)
        self.critics = nn.ModuleList([
            QNetwork(state_dim, action_dim, hidden_dim, n_hidden, noisy=noisy).to(device)
            for _ in range(n_critics)
        ])
        self.critics_target = nn.ModuleList([
            copy.deepcopy(q) for q in self.critics
        ])

        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(init_temperature), dtype=torch.float32, device=device)
        )
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            [p for q in self.critics for p in q.parameters()], lr=critic_lr
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.get_action(state_t, deterministic)
        return action.cpu().numpy()[0]

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        subset_idx = np.random.choice(self.n_critics, self.n_target_critics, replace=False)

        with torch.no_grad():
            next_actions, next_log_pi = self.actor.get_action(next_states)
            q_targets = torch.stack([
                self.critics_target[i](next_states, next_actions) for i in subset_idx
            ], dim=0)
            q_next = q_targets.min(dim=0).values - self.alpha.detach() * next_log_pi
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        if self.use_bootstrap_mask:
            mask = torch.ones(self.n_critics, device=self.device)
        else:
            mask = torch.ones(self.n_critics, device=self.device)

        critic_loss = sum(
            F.mse_loss(self.critics[i](states, actions), q_target) * mask[i]
            for i in range(self.n_critics)
        ) / self.n_critics

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_pi = self.actor.get_action(states)
        q_values = torch.stack([q(states, new_actions) for q in self.critics], dim=0)
        q_new = q_values.mean(dim=0)
        actor_loss = (self.alpha.detach() * log_pi - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for i in range(self.n_critics):
            self._soft_update(self.critics[i], self.critics_target[i])

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def get_q_value(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        if action is None:
            action, _ = self.actor.get_action(state)
        q_values = torch.stack([q(state, action) for q in self.critics], dim=0)
        return q_values.mean(dim=0)

    def get_td_error(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TD error for use as relevance function (Eq. 4 from paper)."""
        with torch.no_grad():
            next_actions, _ = self.actor.get_action(next_states)
            q_next = torch.stack([
                qt(next_states, next_actions) for qt in self.critics_target
            ], dim=0).min(dim=0).values
            td_target = rewards + self.gamma * (1.0 - dones) * q_next
            q_current = torch.stack([q(states, actions) for q in self.critics], dim=0).mean(dim=0)
            return (td_target - q_current).abs()

    def sample_noise(self):
        """Resample noise for NoisyNet layers."""
        for module in self.modules():
            if hasattr(module, "sample_noise"):
                module.sample_noise()


class DRQv2(nn.Module):
    """Data-Regularized Q-learning v2 (Yarats et al., 2021).

    Used as the policy backbone for pixel-based DMC tasks.
    Generates data in the latent space of the CNN visual encoder.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        feature_dim: int = 50,
        hidden_dim: int = 1024,
        n_hidden: int = 2,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        encoder_lr: float = 1e-4,
        tau: float = 0.01,
        gamma: float = 0.99,
        n_aug: int = 2,
        image_pad: int = 4,
        device: str = "cuda",
    ):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.n_aug = n_aug
        self.image_pad = image_pad
        self.device = device

        self.encoder = CNNEncoder(obs_shape, feature_dim).to(device)
        self.encoder_target = copy.deepcopy(self.encoder)

        self.actor = GaussianActor(feature_dim, action_dim, hidden_dim, n_hidden).to(device)
        self.critic1 = QNetwork(feature_dim, action_dim, hidden_dim, n_hidden).to(device)
        self.critic2 = QNetwork(feature_dim, action_dim, hidden_dim, n_hidden).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.critic1.parameters())
            + list(self.critic2.parameters()),
            lr=critic_lr,
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device) / 255.0
            feat = self.encoder(obs_t)
            action, _ = self.actor.get_action(feat, deterministic)
        return action.cpu().numpy()[0]

    def _random_shift_aug(self, obs: torch.Tensor) -> torch.Tensor:
        """Random shift augmentation for pixel observations."""
        b, c, h, w = obs.shape
        pad = self.image_pad
        obs_pad = F.pad(obs, [pad] * 4, mode="replicate")
        eps_h = torch.randint(0, 2 * pad + 1, (b,))
        eps_w = torch.randint(0, 2 * pad + 1, (b,))
        obs_aug = torch.stack([
            obs_pad[i, :, eps_h[i]: eps_h[i] + h, eps_w[i]: eps_w[i] + w]
            for i in range(b)
        ])
        return obs_aug

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        obs_aug = self._random_shift_aug(obs)
        next_obs_aug = self._random_shift_aug(next_obs)

        with torch.no_grad():
            next_feat = self.encoder_target(next_obs_aug)
            next_actions, next_log_pi = self.actor.get_action(next_feat)
            q1_next = self.critic1_target(next_feat, next_actions)
            q2_next = self.critic2_target(next_feat, next_actions)
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        feat = self.encoder(obs_aug)
        q1 = self.critic1(feat, actions)
        q2 = self.critic2(feat, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        feat_detached = self.encoder(obs_aug).detach()
        new_actions, log_pi = self.actor.get_action(feat_detached)
        q1_new = self.critic1(feat_detached, new_actions)
        q2_new = self.critic2(feat_detached, new_actions)
        actor_loss = -torch.min(q1_new, q2_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.encoder, self.encoder_target)
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update_from_latent(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy from latent-space transitions (f(s), a, f(s'), r).

        Used when training on synthetic data generated in latent space.
        """
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions, next_log_pi = self.actor.get_action(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_pi = self.actor.get_action(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        actor_loss = -torch.min(q1_new, q2_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }
