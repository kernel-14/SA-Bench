"""Policy architectures: SAC, REDQ, and DRQ-v2."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, TransformedDistribution


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    n_layers: int = 2,
    activation=nn.ReLU,
    noisy: bool = False,
) -> nn.Module:
    """Build MLP with optional noisy layers."""
    layers = []
    for i in range(n_layers):
        in_dim = input_dim if i == 0 else hidden_dim
        out_dim = output_dim if i == n_layers - 1 else hidden_dim
        if noisy:
            layers.append(NoisyLinear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, out_dim))
        if i < n_layers - 1:
            layers.append(activation())
    return nn.Sequential(*layers)


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration (Fortunato et al. 2018, Eq. 10)."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.empty(out_features))

        self.register_buffer("eps_w", torch.empty(out_features, in_features))
        self.register_buffer("eps_b", torch.empty(out_features))

        self.reset_parameters(sigma_init)
        self.sample_noise()

    def reset_parameters(self, sigma_init: float):
        bound = 1.0 / (self.in_features ** 0.5)
        nn.init.uniform_(self.mu_w, -bound, bound)
        nn.init.uniform_(self.mu_b, -bound, bound)
        nn.init.constant_(self.sigma_w, sigma_init / (self.in_features ** 0.5))
        nn.init.constant_(self.sigma_b, sigma_init / (self.in_features ** 0.5))

    def sample_noise(self):
        """Generate new noise parameters."""
        self.eps_w.normal_()
        # Factorized Gaussian noise
        eps_in = torch.randn(self.in_features).sign()
        eps_out = torch.randn(self.out_features).sign()
        self.eps_w.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))
        self.eps_b.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.mu_w + self.sigma_w * self.eps_w.to(x.device)
            bias = self.mu_b + self.sigma_b * self.eps_b.to(x.device)
        else:
            weight = self.mu_w
            bias = self.mu_b
        return F.linear(x, weight, bias)


class TanhGaussianPolicy(nn.Module):
    """Squashed Gaussian policy for SAC/REDQ."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        noisy: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = build_mlp(
            state_dim, hidden_dim, hidden_dim, n_layers - 1, noisy=noisy
        ) if n_layers > 1 else nn.Identity()

        trunk_out_dim = hidden_dim if n_layers > 1 else state_dim

        self.mean_layer = nn.Linear(trunk_out_dim, action_dim)
        self.log_std_layer = nn.Linear(trunk_out_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)

        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, torch.tanh(mean)


class QNetwork(nn.Module):
    """Q-value network with optional noisy layers."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        noisy: bool = False,
    ):
        super().__init__()
        self.mlp = build_mlp(
            state_dim + action_dim, 1, hidden_dim, n_layers, noisy=noisy
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.mlp(x)


class VisualEncoder(nn.Module):
    """CNN encoder for pixel-based observations (DRQ-v2)."""

    def __init__(
        self,
        in_channels: int = 3,
        features: int = 64,
        n_layers: int = 4,
        latent_dim: int = 50,
    ):
        super().__init__()
        layers = []
        cur_channels = in_channels
        cur_features = features

        for i in range(n_layers):
            layers.append(nn.Conv2d(cur_channels, cur_features, kernel_size=3, stride=2, padding=0))
            layers.append(nn.ReLU())
            cur_channels = cur_features
            cur_features = min(cur_features * 2, 512)

        self.conv = nn.Sequential(*layers)

        # Compute conv output size (84 -> with stride 2, 4 layers)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_out = self.conv(dummy)
            conv_out_size = conv_out.view(1, -1).shape[1]

        self.fc = nn.Linear(conv_out_size, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.view(h.shape[0], -1)
        h = self.fc(h)
        h = self.ln(h)
        return h


class SACPolicy(nn.Module):
    """Soft Actor-Critic policy (Haarnoja et al. 2018)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_lr: float = 3e-4,
        target_entropy: Optional[float] = None,
        noisy: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Actor
        self.actor = TanhGaussianPolicy(state_dim, action_dim, hidden_dim, n_layers, noisy=noisy)

        # Critics
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim, n_layers, noisy=noisy)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim, n_layers, noisy=noisy)

        # Target critics
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dim, n_layers, noisy=noisy)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dim, n_layers, noisy=noisy)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Temperature (log alpha)
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim

    def update_targets(self):
        """Polyak averaging of target critics."""
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            mean, _ = self.actor.forward(state)
            return torch.tanh(mean)
        else:
            action, _, _ = self.actor.sample(state)
            return action

    def critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute critic loss (MSE between Q-value and bootstrapped target)."""
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_probs
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        return critic_loss

    def actor_loss(
        self,
        states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute actor loss (maximize Q-value - entropy)."""
        actions, log_probs, _ = self.actor.sample(states)
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q_min = torch.min(q1, q2)

        actor_loss = (self.log_alpha.exp().detach() * log_probs - q_min).mean()
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        return actor_loss, alpha_loss


class REDQPolicy(SACPolicy):
    """Randomized Ensembled Double Q-Learning (Chen et al. 2021).

    Extends SAC with larger Q-ensemble and random subset for targets.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_critics: int = 10,
        n_target_critics: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_lr: float = 3e-4,
        target_entropy: Optional[float] = None,
        noisy: bool = False,
        bootstrapped: bool = False,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            gamma=gamma,
            tau=tau,
            alpha_lr=alpha_lr,
            target_entropy=target_entropy,
            noisy=noisy,
        )
        self.n_critics = n_critics
        self.n_target_critics = n_target_critics
        self.bootstrapped = bootstrapped

        # Create ensemble of Q-networks
        self.critics = nn.ModuleList([
            QNetwork(state_dim, action_dim, hidden_dim, n_layers, noisy=noisy)
            for _ in range(n_critics)
        ])
        self.critics_target = nn.ModuleList([
            QNetwork(state_dim, action_dim, hidden_dim, n_layers, noisy=noisy)
            for _ in range(n_critics)
        ])
        for critic, target_critic in zip(self.critics, self.critics_target):
            target_critic.load_state_dict(critic.state_dict())

        # Remove old single critic attributes
        del self.critic1, self.critic2, self.critic1_target, self.critic2_target

        # For bootstrapped Q: per-episode mask
        if bootstrapped:
            self.register_buffer("bootstrap_mask", torch.ones(n_critics, dtype=torch.float32))
            self._current_actor_idx = 0

    def update_targets(self):
        for critic, target_critic in zip(self.critics, self.critics_target):
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def sample_target_indices(self) -> torch.Tensor:
        """Randomly select n_target_critics from the ensemble."""
        return torch.randperm(self.n_critics)[:self.n_target_critics]

    def set_bootstrap_mask(self, mask: Optional[torch.Tensor] = None):
        """Set bootstrap mask for the episode (Bootstrapped DQN)."""
        if self.bootstrapped:
            if mask is not None:
                self.bootstrap_mask = mask.float()
            else:
                self.bootstrap_mask = torch.ones(self.n_critics).float()

    def critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)

            # Random subset of target critics
            target_idx = self.sample_target_indices()
            q_next_list = []
            for idx in target_idx:
                q_next_list.append(self.critics_target[idx](next_states, next_actions))
            q_next = torch.stack(q_next_list, dim=0).min(dim=0).values
            q_target = rewards + self.gamma * (1.0 - dones) * (q_next - self.log_alpha.exp() * next_log_probs)

        qs = [critic(states, actions) for critic in self.critics]
        critic_loss = sum(F.mse_loss(q, q_target) for q in qs)

        if self.bootstrapped:
            # Apply bootstrap mask during training: only update selected critics
            for i, (q, critic) in enumerate(zip(qs, self.critics)):
                if self.bootstrap_mask[i].item() == 0:
                    for param in critic.parameters():
                        param.requires_grad = False
                else:
                    for param in critic.parameters():
                        param.requires_grad = True

        return critic_loss

    def actor_loss(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions, log_probs, _ = self.actor.sample(states)

        # Use all critics (or subset for bootstrapped)
        qs = [critic(states, actions) for critic in self.critics]
        q_min = torch.stack(qs, dim=0).min(dim=0).values

        actor_loss = (self.log_alpha.exp().detach() * log_probs - q_min).mean()
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        return actor_loss, alpha_loss


class DRQv2Policy(nn.Module):
    """Data-regularized Q v2 for pixel-based control (Yarats et al. 2021)."""

    def __init__(
        self,
        state_latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        image_size: int = 84,
        image_channels: int = 3,
        aug: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.aug = aug

        # Visual encoder
        self.encoder = VisualEncoder(
            in_channels=image_channels,
            features=64,
            n_layers=4,
            latent_dim=state_latent_dim,
        )

        # SAC-style policy on latent states
        self.actor = TanhGaussianPolicy(state_latent_dim, action_dim, hidden_dim, n_layers)
        self.critic1 = QNetwork(state_latent_dim, action_dim, hidden_dim, n_layers)
        self.critic2 = QNetwork(state_latent_dim, action_dim, hidden_dim, n_layers)
        self.critic1_target = QNetwork(state_latent_dim, action_dim, hidden_dim, n_layers)
        self.critic2_target = QNetwork(state_latent_dim, action_dim, hidden_dim, n_layers)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.target_entropy = -action_dim

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def update_targets(self):
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        z = self.encode(obs)
        if deterministic:
            mean, _ = self.actor.forward(z)
            return torch.tanh(mean)
        else:
            action, _, _ = self.actor.sample(z)
            return action

    def critic_loss(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_latents: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_latents)
            q1_next = self.critic1_target(next_latents, next_actions)
            q2_next = self.critic2_target(next_latents, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_probs
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        q1 = self.critic1(latents, actions)
        q2 = self.critic2(latents, actions)
        return F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

    def actor_loss(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions, log_probs, _ = self.actor.sample(latents)
        q1 = self.critic1(latents, actions)
        q2 = self.critic2(latents, actions)
        q_min = torch.min(q1, q2)
        actor_loss = (self.log_alpha.exp().detach() * log_probs - q_min).mean()
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        return actor_loss, alpha_loss
