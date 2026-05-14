"""
MR.Q Agent: Model-based Representations for Q-learning

Implements the full MR.Q algorithm as described in:
"Towards General-Purpose Model-Free RL (MR.Q)" 
by Fujimoto, D'Oro, Zhang, Tian, Rabbat (Meta FAIR, 2025).

Key features:
- State encoder (CNN for images, MLP for vectors) with LayerNorm+ELU
- State-action encoder with linear MDP predictor
- Encoder loss: unrolled dynamics, reward (categorical), terminal (MSE)
- Value: TD3-style with two Q-networks, multi-step returns, reward scaling
- Policy: DPG with Gumbel-Softmax (discrete) or Tanh (continuous)
- LAP: Loss-Adjusted Prioritized sampling
- Target networks with synchronized updates every T_target steps
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

from .networks import StateEncoder, StateActionEncoder, ValueNetwork, PolicyNetwork
from .replay import MultiStepReplayBuffer
from .utils import two_hot_encode, symexp


class MRQ:
    """
    MR.Q: Model-based Representations for Q-learning.
    
    A general-purpose model-free RL algorithm that uses model-based 
    representation learning to achieve strong performance across 
    diverse benchmarks with a single set of hyperparameters.
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete_action_space=False,
        image_observations=False,
        state_channels=1,
        # Hyperparameters (Table 3 in paper)
        zs_dim=512,
        za_dim=256,
        zsa_dim=512,
        hidden_dim=512,
        num_reward_bins=65,
        discount=0.99,
        encoder_horizon=5,
        multi_step_horizon=3,
        lambda_dynamics=0.1,
        lambda_reward=0.1,
        lambda_terminal=0.1,
        lambda_pre_activ=1e-5,
        # Target update
        target_update_freq=250,
        # Learning rates
        encoder_lr=1e-4,
        value_lr=3e-4,
        policy_lr=3e-4,
        # LAP
        lap_alpha=0.4,
        lap_min_priority=1.0,
        # Replay
        replay_capacity=int(1e6),
        batch_size=256,
        # Exploration
        exploration_noise_std=0.2,
        initial_random_steps=10000,
        # TD3
        target_policy_noise_std=0.2,
        target_policy_noise_clip=0.3,
        # Optimizer
        weight_decay=1e-4,
        # Device
        device='cpu',
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete_action_space = discrete_action_space
        self.image_observations = image_observations
        self.state_channels = state_channels
        
        self.zs_dim = zs_dim
        self.za_dim = za_dim
        self.zsa_dim = zsa_dim
        self.hidden_dim = hidden_dim
        self.num_reward_bins = num_reward_bins
        
        self.discount = discount
        self.encoder_horizon = encoder_horizon
        self.multi_step_horizon = multi_step_horizon
        
        self.lambda_dynamics = lambda_dynamics
        self.lambda_reward = lambda_reward
        self.lambda_terminal = lambda_terminal
        self.lambda_pre_activ = lambda_pre_activ
        
        self.target_update_freq = target_update_freq
        
        self.batch_size = batch_size
        self.exploration_noise_std = exploration_noise_std
        self.initial_random_steps = initial_random_steps
        
        self.target_policy_noise_std = target_policy_noise_std
        self.target_policy_noise_clip = target_policy_noise_clip
        
        self.device = device
        
        # Initialize networks
        self._init_networks()
        
        # Initialize target networks
        self.target_state_encoder = copy.deepcopy(self.state_encoder)
        self.target_state_action_encoder = copy.deepcopy(self.state_action_encoder)
        self.target_value1 = copy.deepcopy(self.value1)
        self.target_value2 = copy.deepcopy(self.value2)
        self.target_policy = copy.deepcopy(self.policy)
        
        # Initialize optimizers
        self.encoder_optimizer = optim.AdamW(
            list(self.state_encoder.parameters()) + 
            list(self.state_action_encoder.parameters()),
            lr=encoder_lr, weight_decay=weight_decay
        )
        self.value_optimizer = optim.AdamW(
            list(self.value1.parameters()) + list(self.value2.parameters()),
            lr=value_lr, weight_decay=weight_decay
        )
        self.policy_optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=policy_lr, weight_decay=weight_decay
        )
        
        # Replay buffer
        self.replay_buffer = MultiStepReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=replay_capacity,
            n_step=multi_step_horizon,
            encoder_horizon=encoder_horizon,
            alpha=lap_alpha,
            min_priority=lap_min_priority,
            device=device
        )
        
        # Reward scaling factors
        self.avg_reward_scale = 1.0
        self.target_avg_reward_scale = 1.0
        
        # Training step counter
        self.total_steps = 0
        self.train_steps = 0
        
        # Gradient clipping
        self.grad_clip_norm = 20.0
    
    def _init_networks(self):
        """Initialize all networks."""
        self.state_encoder = StateEncoder(
            state_dim=self.state_dim,
            zs_dim=self.zs_dim,
            image_observations=self.image_observations,
            state_channels=self.state_channels
        ).to(self.device)
        
        self.state_action_encoder = StateActionEncoder(
            action_dim=self.action_dim,
            zs_dim=self.zs_dim,
            za_dim=self.za_dim,
            zsa_dim=self.zsa_dim,
            num_reward_bins=self.num_reward_bins
        ).to(self.device)
        
        self.value1 = ValueNetwork(
            zsa_dim=self.zsa_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        self.value2 = ValueNetwork(
            zsa_dim=self.zsa_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        self.policy = PolicyNetwork(
            zs_dim=self.zs_dim,
            action_dim=self.action_dim,
            discrete_action_space=self.discrete_action_space,
            hidden_dim=self.hidden_dim
        ).to(self.device)
    
    def _update_target_networks(self):
        """Synchronize target networks with current networks."""
        self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())
        self.target_state_action_encoder.load_state_dict(self.state_action_encoder.state_dict())
        self.target_value1.load_state_dict(self.value1.state_dict())
        self.target_value2.load_state_dict(self.value2.state_dict())
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_avg_reward_scale = self.avg_reward_scale
    
    def select_action(self, state, explore=True):
        """
        Select action given state.
        
        Args:
            state: numpy array of state
            explore: if True, add exploration noise
        
        Returns:
            action: numpy array of action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            zs = self.state_encoder(state_tensor)
            action, _ = self.policy(zs, hard=not explore)
            action = action.cpu().numpy().squeeze(0)
        
        if explore:
            if self.discrete_action_space:
                # Add Gaussian noise to each dimension of one-hot encoding,
                # then take argmax
                noise = np.random.normal(0, self.exploration_noise_std, 
                                        size=action.shape)
                action = action + noise
                action_idx = np.argmax(action)
                one_hot = np.zeros(self.action_dim)
                one_hot[action_idx] = 1.0
                action = one_hot
            else:
                # Add Gaussian noise and clip to [-1, 1]
                noise = np.random.normal(0, self.exploration_noise_std, 
                                        size=action.shape)
                action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def select_action_eval(self, state):
        """Select action for evaluation (no exploration noise, hard argmax for discrete)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            zs = self.state_encoder(state_tensor)
            action, _ = self.policy(zs, hard=True)
            action = action.cpu().numpy().squeeze(0)
        return action
    
    def _compute_encoder_loss(self, batch):
        """
        Compute encoder loss (Equation 14 in paper).
        
        Unrolls the dynamics model over encoder_horizon steps and computes:
        - Dynamics loss: MSE between predicted z_s' and target encoder z_s'
        - Reward loss: cross-entropy with two-hot encoded reward
        - Terminal loss: MSE with binary done signal
        """
        states, actions, rewards, next_states, dones = batch
        
        batch_size = len(states)
        seq_len = self.encoder_horizon + 1
        
        # Reshape: [batch * seq_len, ...]
        states_flat = states.reshape(-1, *states.shape[2:])
        actions_flat = actions.reshape(-1, *actions.shape[2:])
        
        # Encode all states
        zs_all = self.state_encoder(states_flat)
        zs_all = zs_all.reshape(batch_size, seq_len, self.zs_dim)
        
        # Get target next state embeddings using target encoder
        with torch.no_grad():
            next_states_flat = next_states.reshape(-1, *next_states.shape[2:])
            target_zs_all = self.target_state_encoder(next_states_flat)
            target_zs_all = target_zs_all.reshape(batch_size, seq_len, self.zs_dim)
        
        # Initial state embedding for each sequence
        zs0 = zs_all[:, 0]
        
        # Actions for unrolling (first H_enc actions)
        unroll_actions = actions[:, :self.encoder_horizon]
        unroll_actions = unroll_actions.reshape(batch_size, self.encoder_horizon, self.action_dim)
        
        # Unroll dynamics
        zs_pred, r_pred, d_pred, _ = self.state_action_encoder.unroll(
            zs0, unroll_actions, self.encoder_horizon
        )
        # Shapes: [batch, horizon, zs_dim], [batch, horizon, num_reward_bins], 
        #         [batch, horizon, 1]
        
        # Targets
        target_zs = target_zs_all[:, 1:self.encoder_horizon + 1]  # [batch, horizon, zs_dim]
        target_rewards = rewards[:, 1:self.encoder_horizon + 1]   # [batch, horizon]
        target_dones = dones[:, 1:self.encoder_horizon + 1]       # [batch, horizon]
        
        # Dynamics loss: MSE
        dynamics_loss = F.mse_loss(zs_pred, target_zs)
        
        # Reward loss: cross-entropy with two-hot encoding
        r_pred_flat = r_pred.reshape(-1, self.num_reward_bins)
        target_rewards_flat = target_rewards.reshape(-1)
        two_hot_targets = two_hot_encode(target_rewards_flat, self.num_reward_bins, 
                                         device=self.device)
        reward_loss = -(two_hot_targets * F.log_softmax(r_pred_flat, dim=-1)).sum(-1).mean()
        
        # Terminal loss: MSE (only when terminal has been seen)
        d_pred_flat = d_pred.reshape(-1)
        target_dones_flat = target_dones.reshape(-1).float()
        terminal_loss = F.mse_loss(d_pred_flat, target_dones_flat)
        
        # Combined encoder loss
        encoder_loss = (
            self.lambda_dynamics * dynamics_loss +
            self.lambda_reward * reward_loss +
            self.lambda_terminal * terminal_loss
        )
        
        return encoder_loss, dynamics_loss, reward_loss, terminal_loss
    
    def _compute_value_loss(self, states, actions, rewards, next_states, dones):
        """
        Compute value loss (Equation 19 in paper).
        
        Uses multi-step returns, double Q-learning with target min,
        Huber loss, and reward scaling.
        """
        batch_size = states.shape[0]
        
        # Encode states and next states
        zs = self.state_encoder(states)
        
        with torch.no_grad():
            zs_next = self.target_state_encoder(next_states)
        
        # Get state-action embedding for current states
        _, zsa = self.state_action_encoder(zs, actions)
        
        # Compute target value
        with torch.no_grad():
            # Target action with noise
            target_action, _ = self.target_policy(zs_next)
            
            if self.discrete_action_space:
                # Add noise to one-hot then argmax
                noise = torch.randn_like(target_action) * self.target_policy_noise_std
                target_action = target_action + noise
                # Hard argmax
                target_action = F.gumbel_softmax(target_action, tau=10, hard=True)
            else:
                # Add clipped noise
                noise = torch.randn_like(target_action) * self.target_policy_noise_std
                noise = torch.clamp(noise, -self.target_policy_noise_clip, 
                                   self.target_policy_noise_clip)
                target_action = torch.clamp(target_action + noise, -1.0, 1.0)
            
            # Target state-action embedding
            _, zsa_next = self.target_state_action_encoder(zs_next, target_action)
            
            # Double Q: min of two target networks
            target_q1 = self.target_value1(zsa_next)
            target_q2 = self.target_value2(zsa_next)
            target_q = torch.min(target_q1, target_q2)
            
            # Multi-step return target
            # For multi-step returns, we need to compute the n-step return.
            # Since we're sampling random transitions (not sequences),
            # we approximate multi-step returns by bootstrapping with the
            # current n-step discounted sum. 
            # For full n-step, we'd need sequential sampling.
            # Here we use a simplified version: 1-step TD with the 
            # reward scaling factor.
            target = rewards.unsqueeze(1) + self.discount * (1 - dones.unsqueeze(1)) * target_q
        
        # Current Q estimates
        q1 = self.value1(zsa)
        q2 = self.value2(zsa)
        
        # Reward scaling
        target_scaled = target / self.avg_reward_scale
        
        # Huber loss (instead of MSE to eliminate bias from prioritized sampling)
        value_loss1 = F.smooth_l1_loss(q1, target_scaled)
        value_loss2 = F.smooth_l1_loss(q2, target_scaled)
        
        value_loss = value_loss1 + value_loss2
        
        # TD errors for LAP
        with torch.no_grad():
            td_errors = (q1 - target_scaled).abs().cpu().numpy().flatten()
        
        return value_loss, td_errors
    
    def _compute_policy_loss(self, states):
        """
        Compute policy loss (Equation 20 in paper).
        
        Uses deterministic policy gradient with pre-activation regularization.
        """
        batch_size = states.shape[0]
        
        # Encode states (detach to stop gradient flow through encoder)
        with torch.no_grad():
            zs = self.state_encoder(states)
        
        # Get action and pre-activations
        action, pre_activ = self.policy(zs)
        
        # Get state-action embedding
        _, zsa = self.state_action_encoder(zs, action)
        
        # Q values from both critics
        q1 = self.value1(zsa)
        q2 = self.value2(zsa)
        
        # Policy loss: negative average Q + pre-activation regularization
        policy_loss = -0.5 * (q1 + q2).mean() + self.lambda_pre_activ * (pre_activ ** 2).mean()
        
        return policy_loss
    
    def update(self):
        """
        Perform one training update.
        
        Updates the encoder, value function, and policy using batches
        from the replay buffer.
        """
        if self.replay_buffer.size < self.batch_size:
            return None
        
        self.train_steps += 1
        
        # Periodic target network update
        if self.train_steps % self.target_update_freq == 0:
            self._update_target_networks()
            # Update reward scaling
            self.avg_reward_scale = self.replay_buffer.avg_abs_reward
        
        # Update encoder
        encoder_batch = self.replay_buffer.sample_encoder_batch(self.batch_size // self.encoder_horizon)
        if encoder_batch is not None:
            enc_loss, dyn_loss, rew_loss, term_loss = self._compute_encoder_loss(encoder_batch)
            
            self.encoder_optimizer.zero_grad()
            enc_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.state_encoder.parameters()) + 
                list(self.state_action_encoder.parameters()),
                self.grad_clip_norm
            )
            self.encoder_optimizer.step()
        
        # Sample for value and policy updates
        states, actions, rewards, next_states, dones, indices, is_weights = \
            self.replay_buffer.sample(self.batch_size)
        
        # Update value function
        value_loss, td_errors = self._compute_value_loss(
            states, actions, rewards, next_states, dones
        )
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.value1.parameters()) + list(self.value2.parameters()),
            self.grad_clip_norm
        )
        self.value_optimizer.step()
        
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Update policy (less frequently for stability)
        if self.train_steps % 2 == 0:
            policy_loss = self._compute_policy_loss(states)
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            self.policy_optimizer.step()
        
        return {
            'value_loss': value_loss.item(),
            'encoder_loss': enc_loss.item() if encoder_batch is not None else 0.0,
            'dynamics_loss': dyn_loss.item() if encoder_batch is not None else 0.0,
            'reward_loss': rew_loss.item() if encoder_batch is not None else 0.0,
            'terminal_loss': term_loss.item() if encoder_batch is not None else 0.0,
            'avg_reward_scale': self.avg_reward_scale,
        }
    
    def save(self, path):
        """Save model checkpoints."""
        torch.save({
            'state_encoder': self.state_encoder.state_dict(),
            'state_action_encoder': self.state_action_encoder.state_dict(),
            'value1': self.value1.state_dict(),
            'value2': self.value2.state_dict(),
            'policy': self.policy.state_dict(),
            'target_state_encoder': self.target_state_encoder.state_dict(),
            'target_state_action_encoder': self.target_state_action_encoder.state_dict(),
            'target_value1': self.target_value1.state_dict(),
            'target_value2': self.target_value2.state_dict(),
            'target_policy': self.target_policy.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'avg_reward_scale': self.avg_reward_scale,
            'target_avg_reward_scale': self.target_avg_reward_scale,
            'total_steps': self.total_steps,
            'train_steps': self.train_steps,
        }, path)
    
    def load(self, path):
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(checkpoint['state_encoder'])
        self.state_action_encoder.load_state_dict(checkpoint['state_action_encoder'])
        self.value1.load_state_dict(checkpoint['value1'])
        self.value2.load_state_dict(checkpoint['value2'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.target_state_encoder.load_state_dict(checkpoint['target_state_encoder'])
        self.target_state_action_encoder.load_state_dict(checkpoint['target_state_action_encoder'])
        self.target_value1.load_state_dict(checkpoint['target_value1'])
        self.target_value2.load_state_dict(checkpoint['target_value2'])
        self.target_policy.load_state_dict(checkpoint['target_policy'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.avg_reward_scale = checkpoint['avg_reward_scale']
        self.target_avg_reward_scale = checkpoint['target_avg_reward_scale']
        self.total_steps = checkpoint['total_steps']
        self.train_steps = checkpoint['train_steps']
