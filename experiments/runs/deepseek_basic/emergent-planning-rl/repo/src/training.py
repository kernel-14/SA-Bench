"""
IMPALA-based Training for DRC Agents.

Implements the training setup described in Appendix E.4:
- IMPALA (Espeholt et al., 2018) actor-critic training
- 250 million transitions on Boxoban unfiltered training set
- Discount rate γ = 0.97, V-trace λ = 0.97
- L2 penalty 1e-3 on action logits
- L2 regularization 1e-5 on policy and value heads
- Entropy penalty 1e-2 on policy
- Unroll length 20 for BPTT
- Adam optimizer, batch size 16, learning rate decaying from 4e-4 to 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import time


class IMPALATrainer:
    """
    IMPALA-style trainer for DRC agents.
    
    Args:
        agent: DRCAgent instance
        device: Torch device
        gamma: Discount factor (default 0.97)
        lambda_: V-trace lambda (default 0.97)
        entropy_weight: Entropy bonus weight (default 0.01)
        l2_weight: L2 regularization weight on heads (default 1e-5)
        logit_l2_weight: L2 penalty on action logits (default 1e-3)
        learning_rate: Initial learning rate (default 4e-4)
        batch_size: Batch size (default 16)
        unroll_length: BPTT unroll length (default 20)
    """
    def __init__(
        self,
        agent: nn.Module,
        device: str = 'cpu',
        gamma: float = 0.97,
        lambda_: float = 0.97,
        entropy_weight: float = 0.01,
        l2_weight: float = 1e-5,
        logit_l2_weight: float = 1e-3,
        learning_rate: float = 4e-4,
        batch_size: int = 16,
        unroll_length: int = 20,
    ):
        self.agent = agent.to(device)
        self.device = device
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_weight = entropy_weight
        self.l2_weight = l2_weight
        self.logit_l2_weight = logit_l2_weight
        self.batch_size = batch_size
        self.unroll_length = unroll_length
        
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
        self.initial_lr = learning_rate
        
        # Training state
        self.total_steps = 0
        self.max_steps = 250_000_000  # 250M transitions
        
    def compute_vtrace(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
        dones: torch.Tensor,
        log_probs: torch.Tensor,
        behavior_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute V-trace targets and advantages.
        
        Args:
            rewards: (T, B) rewards
            values: (T, B) value estimates
            next_value: (B,) value estimate at final state
            dones: (T, B) done flags
            log_probs: (T, B) log probabilities under current policy
            behavior_log_probs: (T, B) log probabilities under behavior policy
        
        Returns:
            vtrace_values: (T, B) V-trace value targets
            advantages: (T, B) advantage estimates
        """
        T, B = rewards.shape
        
        # Importance sampling ratios
        rho = torch.exp(log_probs - behavior_log_probs).clamp(max=1.0)  # ρ_t
        c = torch.exp(log_probs - behavior_log_probs).clamp(max=1.0)    # c_t
        
        # Compute V-trace targets backwards
        vtrace_values = torch.zeros(T + 1, B, device=self.device)
        vtrace_values[T] = next_value
        
        for t in reversed(range(T)):
            delta = rho[t] * (rewards[t] + self.gamma * (1 - dones[t].float()) * values[t + 1 if t + 1 < T else 0] - values[t])
            vtrace_values[t] = values[t] + delta + self.gamma * self.lambda_ * c[t] * (1 - dones[t].float()) * (vtrace_values[t + 1] - values[t + 1 if t + 1 < T else 0])
        
        vtrace_values = vtrace_values[:-1]  # (T, B)
        
        # Advantages
        advantages = rho * (rewards + self.gamma * (1 - dones.float()) * torch.cat([values[1:], next_value.unsqueeze(0)], dim=0) - values)
        
        return vtrace_values, advantages
    
    def train_step(
        self,
        observations: torch.Tensor,        # (T, B, C, H, W)
        actions: torch.Tensor,             # (T, B)
        rewards: torch.Tensor,             # (T, B)
        dones: torch.Tensor,               # (T, B)
        behavior_log_probs: torch.Tensor,  # (T, B)
    ) -> Dict[str, float]:
        """
        Perform a single training step over an unroll.
        
        Args:
            observations: (unroll_length, batch_size, 7, 8, 8)
            actions: (unroll_length, batch_size)
            rewards: (unroll_length, batch_size)
            dones: (unroll_length, batch_size)
            behavior_log_probs: (unroll_length, batch_size) log probs of behavior policy
        
        Returns:
            Dict of loss values
        """
        T, B = observations.shape[0], observations.shape[1]
        
        self.agent.reset_state(B, self.device)
        
        # Forward pass through unroll
        logits_list = []
        values_list = []
        log_probs_list = []
        entropies = []
        
        for t in range(T):
            obs_t = observations[t]  # (B, 7, 8, 8)
            logits, value = self.agent(obs_t)
            
            logits_list.append(logits)
            values_list.append(value.squeeze(-1))
            
            # Log probs of taken actions
            log_probs = F.log_softmax(logits, dim=-1)
            taken_log_probs = log_probs[torch.arange(B), actions[t]]
            log_probs_list.append(taken_log_probs)
            
            # Entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            entropies.append(entropy)
        
        logits = torch.stack(logits_list)    # (T, B, action_space)
        values = torch.stack(values_list)    # (T, B)
        log_probs = torch.stack(log_probs_list)  # (T, B)
        avg_entropy = torch.stack(entropies).mean()
        
        # Compute V-trace targets (need value at next state after unroll)
        # For simplicity, use zero as bootstrap if done, else use last value
        next_value = torch.zeros(B, device=self.device)
        if not dones[-1].all():
            # Would need next observation to compute next value
            # Simplified: use last value for non-done episodes
            next_value = values[-1].detach()
        
        vtrace_values, advantages = self.compute_vtrace(
            rewards, values, next_value, dones, log_probs, behavior_log_probs
        )
        
        # Policy loss
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss (MSE against V-trace targets)
        value_loss = F.mse_loss(values, vtrace_values.detach())
        
        # Entropy loss
        entropy_loss = -self.entropy_weight * avg_entropy
        
        # L2 penalties
        logit_l2 = self.logit_l2_weight * (logits ** 2).mean()
        
        # L2 on policy and value heads
        head_l2 = 0.0
        for name, param in self.agent.named_parameters():
            if 'policy_head' in name or 'value_head' in name:
                head_l2 += (param ** 2).sum()
        head_l2 *= self.l2_weight
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss + logit_l2 + head_l2
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=40.0)
        
        self.optimizer.step()
        
        # Detach states to prevent gradient flow between unrolls
        self.agent.c_t = [c.detach() if c is not None else None for c in self.agent.c_t]
        self.agent.h_t = [h.detach() if h is not None else None for h in self.agent.h_t]
        
        self.total_steps += T * B
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'avg_value': values.mean().item(),
            'total_steps': self.total_steps,
        }
    
    def update_learning_rate(self):
        """Linearly decay learning rate from initial to 0."""
        progress = min(self.total_steps / self.max_steps, 1.0)
        lr = self.initial_lr * (1.0 - progress)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def save_checkpoint(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']


def collect_trajectories(
    agent: nn.Module,
    env,
    levels: List[np.ndarray],
    unroll_length: int = 20,
    batch_size: int = 16,
    greedy: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Collect trajectories for training.
    
    Returns:
        Dict with observations, actions, rewards, dones, behavior_log_probs
    """
    observations = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    
    agent.reset_state(batch_size, device=next(agent.parameters()).device)
    
    # Initialize environments
    obs_list = []
    for b in range(batch_size):
        level = levels[np.random.randint(len(levels))]
        obs = env.reset(level)
        obs_list.append(obs)
    
    obs_tensor = torch.from_numpy(np.stack(obs_list)).permute(0, 3, 1, 2)
    obs_tensor = obs_tensor.to(next(agent.parameters()).device)
    
    for t in range(unroll_length):
        with torch.no_grad():
            logits, value = agent(obs_tensor)
        
        if greedy:
            action = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)
        
        # Log prob of taken action
        log_prob = F.log_softmax(logits, dim=-1)
        taken_log_prob = log_prob[torch.arange(batch_size), action]
        
        observations.append(obs_tensor)
        actions.append(action)
        log_probs.append(taken_log_prob)
        
        # Step environments
        next_obs_list = []
        reward_list = []
        done_list = []
        
        for b in range(batch_size):
            obs, reward, done, _ = env.step(action[b].item())
            next_obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
        
        rewards.append(torch.tensor(reward_list, device=agent.parameters().__next__().device))
        dones.append(torch.tensor(done_list, device=agent.parameters().__next__().device))
        
        obs_tensor = torch.from_numpy(np.stack(next_obs_list)).permute(0, 3, 1, 2)
        obs_tensor = obs_tensor.to(next(agent.parameters()).device)
    
    return {
        'observations': torch.stack(observations),   # (T, B, 7, 8, 8)
        'actions': torch.stack(actions),              # (T, B)
        'rewards': torch.stack(rewards),              # (T, B)
        'dones': torch.stack(dones),                  # (T, B)
        'behavior_log_probs': torch.stack(log_probs), # (T, B)
    }
