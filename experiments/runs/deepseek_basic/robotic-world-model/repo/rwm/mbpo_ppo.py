"""
Model-Based Policy Optimization with PPO (MBPO-PPO).

Implements Algorithm 1 from the paper:
- Policy optimization on learned world models
- PPO-based policy updates with imagined rollouts
- Replay buffer management for real and imagined data

Based on: "Robotic World Model" paper, Section 3.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import copy


# ============================================================
# PPO Policy and Value Networks (Table S9)
# ============================================================
class PolicyNetwork(nn.Module):
    """
    MLP policy network.
    Architecture: (128, 128, 128) ELU activation.
    Outputs mean of Gaussian action distribution with learnable std.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128, 128),
        activation: str = 'elu',
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        layers = []
        prev_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            prev_dim = h_dim
        
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, act_dim)
        
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and std of action distribution."""
        features = self.backbone(obs)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std.clamp(-10, 2))
        return mean, std
    
    def sample(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, std = self.forward(obs)
        if deterministic:
            return mean, torch.zeros_like(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy of action."""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """
    MLP value function network.
    Architecture: (128, 128, 128) ELU activation.
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128, 128),
        activation: str = 'elu',
    ):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ============================================================
# Replay Buffer
# ============================================================
class ReplayBuffer:
    """
    Replay buffer D for environment interactions.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.obs_list = []
        self.act_list = []
        self.rew_list = []
        self.done_list = []
        self.priv_list = []
        self.ptr = 0
        self.full = False
        
    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        priv: Optional[np.ndarray] = None,
    ):
        if len(self.obs_list) < self.max_size:
            self.obs_list.append(obs)
            self.act_list.append(act)
            self.rew_list.append(rew)
            self.done_list.append(done)
            if priv is not None:
                self.priv_list.append(priv)
        else:
            idx = self.ptr % self.max_size
            self.obs_list[idx] = obs
            self.act_list[idx] = act
            self.rew_list[idx] = rew
            self.done_list[idx] = done
            if priv is not None:
                self.priv_list[idx] = priv
        self.ptr += 1
        if self.ptr >= self.max_size:
            self.full = True
    
    def sample_trajectories(
        self, num_trajectories: int = 4096, rollout_length: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample initial observations and history for imagination rollouts.
        Returns starting observations and history contexts.
        """
        n = len(self.obs_list)
        if n == 0:
            return None, None
        
        # Sample random starting points
        indices = np.random.randint(0, n, size=num_trajectories)
        
        starts = []
        for idx in indices:
            # Get observation at index
            obs = self.obs_list[idx]
            starts.append(obs)
        
        return np.stack(starts, axis=0), np.array(indices)
    
    def get_initial_observations(self, num_samples: int) -> np.ndarray:
        """Get random observations to initialize imagination agents."""
        n = len(self.obs_list)
        if n == 0:
            return None
        indices = np.random.randint(0, n, size=num_samples)
        return np.stack([self.obs_list[i] for i in indices], axis=0)
    
    def size(self) -> int:
        return len(self.obs_list)
    
    def is_full(self) -> bool:
        return self.full
    
    def get_all(self) -> Dict[str, np.ndarray]:
        """Get all data as arrays."""
        return {
            'obs': np.stack(self.obs_list, axis=0),
            'act': np.stack(self.act_list, axis=0),
            'rew': np.array(self.rew_list),
            'done': np.array(self.done_list),
        }


# ============================================================
# PPO Implementation
# ============================================================
@dataclass
class PPOConfig:
    """Configuration for PPO training (Table S11)."""
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    learning_epochs: int = 5
    mini_batches: int = 4
    kl_target: float = 0.01
    discount_factor: float = 0.99
    clip_range: float = 0.2
    entropy_coef: float = 0.005
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0


class PPO:
    """
    Proximal Policy Optimization (PPO) implementation.
    Used within MBPO for policy optimization on imagined rollouts.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: PPOConfig = None,
        device: torch.device = None,
    ):
        self.config = config or PPOConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.policy = PolicyNetwork(obs_dim, act_dim).to(self.device)
        self.value = ValueNetwork(obs_dim).to(self.device)
        
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and get value estimate."""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            action, log_prob = self.policy.sample(obs_tensor, deterministic)
            value = self.value(obs_tensor)
            
            return action.squeeze(0).cpu().numpy(), log_prob.squeeze(0).cpu().numpy(), value.squeeze(0).cpu().numpy()
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        return advantages, returns
    
    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """PPO policy and value update step."""
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.config.learning_epochs):
            # Mini-batch updates
            batch_size = obs.shape[0]
            indices = torch.randperm(batch_size)
            
            mini_batch_size = batch_size // self.config.mini_batches
            
            for i in range(self.config.mini_batches):
                start = i * mini_batch_size
                end = start + mini_batch_size
                mb_indices = indices[start:end]
                
                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Policy loss
                new_log_probs, entropy = self.policy.evaluate(mb_obs, mb_actions)
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                clip_adv = torch.clamp(
                    ratio, 
                    1.0 - self.config.clip_range, 
                    1.0 + self.config.clip_range
                ) * mb_advantages
                
                policy_loss = -torch.min(ratio * mb_advantages, clip_adv).mean()
                policy_loss = policy_loss - self.config.entropy_coef * entropy.mean()
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                
                # Value loss
                value_pred = self.value(mb_obs)
                value_loss = F.mse_loss(value_pred, mb_returns)
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), self.config.max_grad_norm)
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        n_updates = self.config.learning_epochs * self.config.mini_batches
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }


# ============================================================
# MBPO-PPO: Algorithm 1
# ============================================================
@dataclass
class MBPOPPOConfig:
    """Configuration for MBPO-PPO."""
    imagination_envs: int = 4096
    imagination_steps: int = 100
    step_time: float = 0.02
    buffer_size: int = 1000
    max_iterations: int = 2500
    ppo_config: PPOConfig = field(default_factory=PPOConfig)


class MBPOPPO:
    """
    Model-Based Policy Optimization with PPO.
    
    Implements Algorithm 1 from the paper:
    1. Collect data in D by interacting with environment using policy
    2. Update world model with autoregressive training
    3. Initialize imagination agents from D
    4. Roll out imagination trajectories using policy and world model
    5. Update policy using PPO
    """
    
    def __init__(
        self,
        world_model: nn.Module,
        obs_dim: int,
        act_dim: int,
        reward_fn: Callable,
        config: MBPOPPOConfig = None,
        device: torch.device = None,
    ):
        self.world_model = world_model
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reward_fn = reward_fn
        self.config = config or MBPOPPOConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.buffer = ReplayBuffer(max_size=self.config.buffer_size)
        self.ppo = PPO(obs_dim, act_dim, config=self.config.ppo_config, device=self.device)
        self.world_model.to(self.device)
        
        self.iteration = 0
        
    def collect_real_data(
        self,
        env_interact_fn: Callable,
        num_steps: int,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Collect data by interacting with the real environment.
        
        Args:
            env_interact_fn: Function that takes policy and returns trajectory
            num_steps: Number of environment steps to collect
            
        Returns:
            List of trajectory dictionaries
        """
        trajectories = []
        
        obs, info = env_interact_fn(reset=True)
        episode_obs = [obs]
        episode_act = []
        episode_rew = []
        episode_done = []
        episode_priv = []
        
        steps = 0
        while steps < num_steps:
            action, _, _ = self.ppo.act(obs)
            
            next_obs, reward, done, truncated, info = env_interact_fn(action)
            
            episode_obs.append(next_obs)
            episode_act.append(action)
            episode_rew.append(reward)
            episode_done.append(done or truncated)
            
            if 'priv' in info:
                episode_priv.append(info['priv'])
            
            self.buffer.add(obs, action, reward, done, info.get('priv', None))
            
            obs = next_obs
            steps += 1
            
            if done or truncated:
                traj = {
                    'obs': np.array(episode_obs[:-1]),
                    'act': np.array(episode_act),
                    'rew': np.array(episode_rew),
                }
                if episode_priv:
                    traj['priv'] = np.array(episode_priv)
                trajectories.append(traj)
                
                obs, info = env_interact_fn(reset=True)
                episode_obs = [obs]
                episode_act = []
                episode_rew = []
                episode_done = []
                episode_priv = []
        
        return trajectories
    
    def imagine_trajectories(
        self,
        num_envs: int,
        num_steps: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Roll out imagination trajectories using world model and policy.
        
        This is the core of MBPO: using the learned world model to generate
        imagined data for policy optimization.
        """
        self.world_model.eval()
        
        M = self.world_model.history_horizon
        
        # Get initial observations from replay buffer
        init_obs = self.buffer.get_initial_observations(num_envs)
        if init_obs is None:
            return None
        
        # For proper autoregressive rollout, we need history context
        # If buffer doesn't have enough history, pad with initial observation
        obs_history = np.tile(init_obs[:, np.newaxis, :], (1, M, 1))
        
        # Initialize observations
        obs_current = torch.as_tensor(init_obs, dtype=torch.float32, device=self.device)
        obs_history_tensor = torch.as_tensor(obs_history, dtype=torch.float32, device=self.device)
        
        # GRU hidden state
        h = self.world_model._get_initial_hidden(num_envs, self.device)
        
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_log_probs = []
        all_values = []
        
        for t in range(num_steps):
            # Get action from policy (conditioned on predicted observation)
            with torch.no_grad():
                action, log_prob = self.policy.policy.sample(obs_current)
                value = self.policy.value(obs_current)
            
            # Step world model forward
            with torch.no_grad():
                obs_mean, obs_std, h, priv_mean, priv_std = self.world_model._step(
                    obs_current, action, h
                )
            
            # Compute reward from predicted observation (and privileged info)
            reward = self.reward_fn(obs_mean, action, priv_mean if priv_mean is not None else None)
            
            # Check termination (from privileged info prediction)
            done = torch.zeros(num_envs, device=self.device)
            if priv_mean is not None:
                # Paper: termination predicted from privileged info (contacts)
                # If base contact predicted, terminate
                done = self._check_termination(priv_mean)
            
            all_obs.append(obs_current)
            all_actions.append(action)
            all_rewards.append(reward)
            all_dones.append(done)
            all_log_probs.append(log_prob)
            all_values.append(value)
            
            # Update current observation with prediction
            obs_current = obs_mean
            
            # Handle dones: reset hidden state for terminated envs
            if done.any():
                reset_mask = done.bool()
                obs_current[reset_mask] = torch.as_tensor(
                    init_obs, dtype=torch.float32, device=self.device
                )[reset_mask]
                h[:, reset_mask, :] = 0
        
        return {
            'obs': torch.stack(all_obs, dim=1),        # (num_envs, T, obs_dim)
            'actions': torch.stack(all_actions, dim=1),  # (num_envs, T, act_dim)
            'rewards': torch.stack(all_rewards, dim=1),  # (num_envs, T)
            'dones': torch.stack(all_dones, dim=1),      # (num_envs, T)
            'log_probs': torch.stack(all_log_probs, dim=1),  # (num_envs, T)
            'values': torch.stack(all_values, dim=1),    # (num_envs, T)
        }
    
    def _check_termination(self, priv_pred: torch.Tensor) -> torch.Tensor:
        """
        Check if episode should terminate based on privileged information.
        
        Paper (A.4.3): "We explicitly train RWM to predict such terminations 
        in its privileged information prediction head. During policy optimization, 
        MBPO-PPO treats these termination predictions as episode-ending events."
        """
        # Simplified: if any contact probability exceeds 0.5, terminate
        # In practice, this depends on how contacts are encoded
        return (priv_pred > 0.5).any(dim=-1).float()
    
    def update_policy(
        self,
        rollout_data: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Update policy using PPO on imagined rollouts.
        """
        obs = rollout_data['obs']
        actions = rollout_data['actions']
        rewards = rollout_data['rewards']
        dones = rollout_data['dones']
        old_log_probs = rollout_data['log_probs']
        old_values = rollout_data['values']
        
        # Compute advantages and returns using GAE
        advantages, returns = self.ppo.compute_gae(
            rewards, old_values, dones,
            gamma=self.config.ppo_config.discount_factor,
            lam=self.config.ppo_config.gae_lambda,
        )
        
        # Reshape for PPO update
        batch_size = obs.shape[0] * obs.shape[1]
        obs_flat = obs.reshape(batch_size, -1)
        actions_flat = actions.reshape(batch_size, -1)
        old_log_probs_flat = old_log_probs.reshape(batch_size)
        advantages_flat = advantages.reshape(batch_size)
        returns_flat = returns.reshape(batch_size)
        
        # PPO update
        metrics = self.ppo.update(
            obs_flat, actions_flat, old_log_probs_flat,
            advantages_flat, returns_flat,
        )
        
        return metrics
    
    def train_iteration(
        self,
        env_interact_fn: Callable,
        real_steps: int = 100,
    ) -> Dict[str, float]:
        """
        One iteration of MBPO-PPO training (Algorithm 1).
        """
        self.iteration += 1
        
        # Step 1: Collect real data
        trajectories = self.collect_real_data(env_interact_fn, real_steps)
        
        # Step 2: Update world model would be done externally
        # (world model training is separate from MBPO-PPO loop)
        
        # Steps 3-5: Imagination and PPO update
        rollout_data = self.imagine_trajectories(
            num_envs=self.config.imagination_envs,
            num_steps=self.config.imagination_steps,
        )
        
        if rollout_data is not None:
            ppo_metrics = self.update_policy(rollout_data)
        else:
            ppo_metrics = {}
        
        return {
            'iteration': self.iteration,
            'buffer_size': self.buffer.size(),
            **ppo_metrics,
        }
    
    def get_policy(self) -> PolicyNetwork:
        """Get the current policy."""
        return self.ppo.policy
    
    def get_value(self) -> ValueNetwork:
        """Get the current value function."""
        return self.ppo.value
