"""
Policy networks for PGR.
Supports REDQ (state-based) and DRQ-v2 (pixel-based) backbones.
Also includes SAC implementation for baseline comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# ============================================================
# MLP Actor (Policy) Network
# ============================================================

class MLPActor(nn.Module):
    """
    Gaussian policy network (SAC-style).
    Outputs mean and log_std of action distribution.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        mean = self.mean(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from the policy."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action (without log prob)."""
        if deterministic:
            mean, _ = self.forward(state)
            return torch.tanh(mean)
        else:
            action, _ = self.sample(state)
            return action


# ============================================================
# REDQ Q-Network (Ensemble of Q-functions)
# ============================================================

class REDQQNetwork(nn.Module):
    """
    Randomized Ensembled Double Q-learning (REDQ) Q-Network.
    Uses an ensemble of Q-functions with random subset updates.
    
    From Chen et al. (2021): typically uses 10 Q-networks,
    with 2 randomly selected for target computation.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        ensemble_size: int = 10,
        num_samples: int = 2,  # number of Qs to sample per update
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.num_samples = num_samples
        
        # Create ensemble of Q-networks
        self.q_networks = nn.ModuleList([
            self._build_q_network(state_dim, action_dim, hidden_dim, num_layers)
            for _ in range(ensemble_size)
        ])
    
    def _build_q_network(self, state_dim, action_dim, hidden_dim, num_layers):
        layers = []
        in_dim = state_dim + action_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute Q-values.
        If idx is None, return all Q-values.
        If idx is specified, return only that Q-network's output.
        """
        sa = torch.cat([state, action], dim=-1)
        
        if idx is not None:
            return self.q_networks[idx](sa)
        
        qs = [q(sa) for q in self.q_networks]
        return tuple(qs)
    
    def sample_qs(self, state: torch.Tensor, action: torch.Tensor):
        """Randomly sample num_samples Q-networks."""
        indices = torch.randperm(self.ensemble_size)[:self.num_samples]
        qs = []
        for idx in indices:
            qs.append(self.q_networks[idx](torch.cat([state, action], dim=-1)))
        return torch.cat(qs, dim=-1), indices


# ============================================================
# SAC-style Q-Network (for SAC baseline)
# ============================================================

class SACCritic(nn.Module):
    """Twin Q-networks for SAC."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.q1 = self._build(state_dim, action_dim, hidden_dim, num_layers)
        self.q2 = self._build(state_dim, action_dim, hidden_dim, num_layers)
    
    def _build(self, state_dim, action_dim, hidden_dim, num_layers):
        layers = []
        in_dim = state_dim + action_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# ============================================================
# Soft Actor-Critic (SAC) Agent
# ============================================================

class SACAgent:
    """
    Soft Actor-Critic agent for continuous control.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        lr: float = 3e-4,
        device: str = "cuda",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        
        # Actor
        self.actor = MLPActor(state_dim, action_dim, hidden_dim, num_layers).to(device)
        
        # Critics
        self.critic = SACCritic(state_dim, action_dim, hidden_dim, num_layers).to(device)
        self.critic_target = SACCritic(state_dim, action_dim, hidden_dim, num_layers).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Auto alpha
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            return self.actor.get_action(state, deterministic)
    
    def update(self, batch, pgr_module=None):
        """Single SAC update step."""
        states, actions, next_states, rewards, dones = batch
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + self.gamma * (1 - dones) * q_next
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
        }


# ============================================================
# REDQ Agent
# ============================================================

class REDQAgent:
    """
    Randomized Ensembled Double Q-learning Agent.
    
    From Chen et al. (2021). Uses an ensemble of Q-networks
    with random subset sampling for both target computation and updates.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        ensemble_size: int = 10,
        num_samples: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        lr: float = 3e-4,
        utd_ratio: int = 20,
        device: str = "cuda",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.utd_ratio = utd_ratio
        
        # Actor
        self.actor = MLPActor(state_dim, action_dim, hidden_dim, num_layers).to(device)
        
        # Q-networks
        self.critic = REDQQNetwork(
            state_dim, action_dim, hidden_dim, num_layers, ensemble_size, num_samples
        ).to(device)
        self.critic_target = REDQQNetwork(
            state_dim, action_dim, hidden_dim, num_layers, ensemble_size, num_samples
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Auto alpha
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            return self.actor.get_action(state, deterministic)
    
    def update(self, batch):
        """Perform UTD ratio number of updates."""
        states, actions, next_states, rewards, dones = batch
        
        info = {'critic_loss': 0.0, 'actor_loss': 0.0, 'alpha': self.alpha}
        
        for _ in range(self.utd_ratio):
            # Sample random subset of Q-networks
            q_values, indices = self.critic.sample_qs(states, actions)  # (B, num_samples)
            
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(next_states)
                
                # Target with random subset
                target_qs = []
                for idx in indices:
                    target_q = self.critic_target.q_networks[idx](
                        torch.cat([next_states, next_actions], dim=-1)
                    )
                    target_qs.append(target_q)
                target_qs = torch.cat(target_qs, dim=-1)  # (B, num_samples)
                
                target_q = torch.min(target_qs, dim=-1, keepdim=True).values
                q_target = rewards + self.gamma * (1 - dones) * (target_q - self.alpha * next_log_probs)
            
            # Critic loss
            critic_loss = F.mse_loss(q_values, q_target.expand_as(q_values))
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Actor update
            new_actions, log_probs = self.actor.sample(states)
            
            # Use full ensemble for actor update
            all_qs = []
            for q_net in self.critic.q_networks:
                all_qs.append(q_net(torch.cat([states, new_actions], dim=-1)))
            all_qs = torch.cat(all_qs, dim=-1)
            q_new = torch.min(all_qs, dim=-1, keepdim=True).values
            
            actor_loss = (self.alpha * log_probs - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update alpha
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
            
            # Polyak update target
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            info['critic_loss'] += critic_loss.item()
            info['actor_loss'] += actor_loss.item()
        
        info['critic_loss'] /= self.utd_ratio
        info['actor_loss'] /= self.utd_ratio
        
        return info


# ============================================================
# CNN Visual Encoder (for DRQ-v2 / pixel-based tasks)
# ============================================================

class VisualEncoder(nn.Module):
    """
    CNN encoder for pixel-based observations.
    Based on DRQ-v2's encoder architecture.
    
    Encodes images to a latent representation. 
    The diffusion model generates in this latent space.
    """
    def __init__(
        self,
        input_channels: int = 3,
        input_height: int = 84,
        input_width: int = 84,
        latent_dim: int = 50,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # CNN trunk (similar to DrQ-v2)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            conv_out = self.conv_layers(dummy)
            conv_out_size = conv_out.view(1, -1).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image observation to latent.
        
        Args:
            x: (B, C, H, W) image tensor
        Returns:
            (B, latent_dim) latent representation
        """
        h = self.conv_layers(x)
        h = h.view(h.shape[0], -1)
        z = self.fc(h)
        return z


# ============================================================
# DRQ-v2 Agent (for pixel-based tasks)
# ============================================================

class DRQv2Agent:
    """
    DrQ-v2: Data-regularized Q-learning v2 for visual continuous control.
    
    Uses a CNN encoder shared between actor and critic, 
    with data augmentation for regularization.
    """
    def __init__(
        self,
        state_shape: Tuple[int, int, int],  # (C, H, W)
        action_dim: int,
        latent_dim: int = 50,
        hidden_dim: int = 256,
        num_layers: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.1,
        lr: float = 1e-4,
        feature_lr: float = 1e-4,
        utd_ratio: int = 1,
        device: str = "cuda",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.utd_ratio = utd_ratio
        self.latent_dim = latent_dim
        
        C, H, W = state_shape
        
        # Visual encoder
        self.encoder = VisualEncoder(C, H, W, latent_dim, hidden_dim).to(device)
        
        # Actor (operates on latent)
        self.actor = MLPActor(latent_dim, action_dim, hidden_dim, num_layers).to(device)
        
        # Critic (operates on latent + action)
        self.critic = SACCritic(latent_dim, action_dim, hidden_dim, num_layers).to(device)
        self.critic_target = SACCritic(latent_dim, action_dim, hidden_dim, num_layers).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=feature_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
    
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode pixel observation to latent."""
        return self.encoder(obs)
    
    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            latent = self.encoder(obs)
            return self.actor.get_action(latent, deterministic)
    
    def update(self, batch):
        """DRQ-v2 update step."""
        obs, actions, next_obs, rewards, dones = batch
        
        # Encode observations
        latent = self.encoder(obs)
        with torch.no_grad():
            next_latent = self.encoder(next_obs)
        
        total_info = {'critic_loss': 0.0, 'actor_loss': 0.0}
        
        for _ in range(self.utd_ratio):
            # Update critic
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(next_latent)
                q1_next, q2_next = self.critic_target(next_latent, next_actions)
                q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
                q_target = rewards + self.gamma * (1 - dones) * q_next
            
            q1, q2 = self.critic(latent, actions)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
            
            self.critic_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.encoder_optimizer.step()
            
            # Update actor
            new_actions, log_probs = self.actor.sample(latent.detach())
            q1_new, q2_new = self.critic(latent.detach(), new_actions)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * log_probs - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            total_info['critic_loss'] += critic_loss.item()
            total_info['actor_loss'] += actor_loss.item()
        
        total_info['critic_loss'] /= self.utd_ratio
        total_info['actor_loss'] /= self.utd_ratio
        
        return total_info


# ============================================================
# Noisy Networks (for exploration bonus experiments)
# ============================================================

class NoisyLinear(nn.Module):
    """
    Noisy linear layer as in Fortunato et al. (2018).
    y = (mu_w + sigma_w * eps_w) * x + (mu_b + sigma_b * eps_b)
    """
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.mu_b = nn.Parameter(torch.Tensor(out_features))
        self.sigma_b = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters(sigma_init)
    
    def reset_parameters(self, sigma_init):
        # Initialize mu parameters
        bound = 1 / np.sqrt(self.in_features)
        self.mu_w.data.uniform_(-bound, bound)
        self.mu_b.data.uniform_(-bound, bound)
        
        # Initialize sigma parameters
        self.sigma_w.data.fill_(sigma_init / np.sqrt(self.in_features))
        self.sigma_b.data.fill_(sigma_init / np.sqrt(self.in_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample noise
        eps_w = torch.randn_like(self.mu_w)
        eps_b = torch.randn_like(self.mu_b)
        
        weight = self.mu_w + self.sigma_w * eps_w
        bias = self.mu_b + self.sigma_b * eps_b
        
        return F.linear(x, weight, bias)


# ============================================================
# Bootstrapped DQN Q-Network
# ============================================================

class BootstrappedQNetwork(nn.Module):
    """
    Q-network with bootstrapped heads for exploration (Osband et al., 2016).
    Each head sees a different subset of data via binary masks.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 10,
    ):
        super().__init__()
        self.num_heads = num_heads
        
        # Shared trunk
        layers = []
        in_dim = state_dim + action_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        
        # Multiple Q-heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_heads)
        ])
    
    def forward(self, state, action, head_idx=None):
        sa = torch.cat([state, action], dim=-1)
        h = self.trunk(sa)
        
        if head_idx is not None:
            return self.heads[head_idx](h)
        
        return torch.cat([head(h) for head in self.heads], dim=-1)
