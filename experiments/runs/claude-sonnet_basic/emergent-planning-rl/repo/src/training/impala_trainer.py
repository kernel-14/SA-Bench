"""
IMPALA-based training for DRC agent.

From the paper (Appendix E.4):
- Trained on 900k levels from Boxoban unfiltered training set
- Actor-critic setting using IMPALA (Espeholt et al., 2018)
- 250 million transitions total
- Discount rate: gamma = 0.97
- V-trace target: lambda = 0.97
- L2 penalty of 1e-3 on action logits
- L2 regularization of 1e-5 on policy and value heads
- Entropy penalty of 1e-2 on policy
- Unroll length: 20
- Adam optimizer
- Batch size: 16
- Learning rate: linear decay from 4e-4 to 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
import os


class VTraceReturns:
    """
    V-trace return computation for off-policy correction.
    
    From Espeholt et al. (2018) IMPALA paper.
    """
    
    @staticmethod
    def compute_vtrace_returns(
        rewards: torch.Tensor,       # (T, B)
        values: torch.Tensor,        # (T+1, B)
        log_rhos: torch.Tensor,      # (T, B) - log importance weights
        gamma: float = 0.97,
        lambda_: float = 0.97,
        clip_rho_threshold: float = 1.0,
        clip_pg_rho_threshold: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute V-trace returns.
        
        Args:
            rewards: Rewards at each step
            values: Value estimates (including bootstrap value)
            log_rhos: Log importance sampling ratios
            gamma: Discount factor
            lambda_: V-trace lambda
            clip_rho_threshold: Clipping threshold for rho
            clip_pg_rho_threshold: Clipping threshold for policy gradient rho
            
        Returns:
            (vtrace_returns, pg_advantages)
        """
        T, B = rewards.shape
        
        rhos = torch.exp(log_rhos)
        clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        
        # Compute deltas
        deltas = clipped_rhos * (rewards + gamma * values[1:] - values[:-1])
        
        # Compute V-trace returns (backward pass)
        vtrace_returns = torch.zeros_like(rewards)
        acc = torch.zeros(B, device=rewards.device)
        
        for t in reversed(range(T)):
            acc = deltas[t] + gamma * lambda_ * clipped_rhos[t] * acc
            vtrace_returns[t] = acc + values[t]
        
        # Policy gradient advantages
        pg_advantages = clipped_pg_rhos * (rewards + gamma * vtrace_returns[1:] - values[:-1])
        
        return vtrace_returns, pg_advantages


class IMPALALoss(nn.Module):
    """
    IMPALA loss function for DRC agent training.
    
    Combines:
    - Policy gradient loss (with V-trace)
    - Value function loss
    - Entropy bonus
    - L2 regularization
    """
    
    def __init__(
        self,
        gamma: float = 0.97,
        lambda_: float = 0.97,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        logit_l2_coef: float = 1e-3,
        param_l2_coef: float = 1e-5,
    ):
        super().__init__()
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.logit_l2_coef = logit_l2_coef
        self.param_l2_coef = param_l2_coef
    
    def forward(
        self,
        logits: torch.Tensor,          # (T, B, num_actions)
        values: torch.Tensor,          # (T+1, B)
        actions: torch.Tensor,         # (T, B)
        rewards: torch.Tensor,         # (T, B)
        behavior_log_probs: torch.Tensor,  # (T, B) - log probs from behavior policy
        dones: torch.Tensor,           # (T, B)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute IMPALA loss.
        
        Args:
            logits: Policy logits from current policy
            values: Value estimates from current policy
            actions: Actions taken by behavior policy
            rewards: Rewards received
            behavior_log_probs: Log probabilities under behavior policy
            dones: Episode termination flags
            
        Returns:
            (total_loss, loss_dict)
        """
        T, B, num_actions = logits.shape
        
        # Current policy log probs
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Importance sampling ratios
        log_rhos = action_log_probs - behavior_log_probs
        
        # Mask out terminal states
        # (rewards after done should not be used)
        not_done = 1.0 - dones.float()
        
        # Compute V-trace returns
        vtrace_returns, pg_advantages = VTraceReturns.compute_vtrace_returns(
            rewards, values, log_rhos, self.gamma, self.lambda_
        )
        
        # Policy gradient loss
        pg_loss = -(pg_advantages.detach() * action_log_probs).mean()
        
        # Value function loss
        value_loss = F.mse_loss(values[:-1], vtrace_returns.detach())
        
        # Entropy bonus
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # L2 penalty on logits
        logit_l2 = (logits ** 2).mean()
        
        # Total loss
        total_loss = (
            pg_loss 
            + self.value_coef * value_loss 
            - self.entropy_coef * entropy
            + self.logit_l2_coef * logit_l2
        )
        
        loss_dict = {
            'pg_loss': pg_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'logit_l2': logit_l2.item(),
            'total_loss': total_loss.item(),
        }
        
        return total_loss, loss_dict


class DRCTrainer:
    """
    Trainer for DRC agent using IMPALA-style training.
    
    Simplified single-machine version of IMPALA.
    """
    
    def __init__(
        self,
        agent,
        optimizer: optim.Optimizer,
        loss_fn: IMPALALoss,
        device: torch.device,
        unroll_length: int = 20,
        max_grad_norm: float = 40.0,
    ):
        self.agent = agent
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.unroll_length = unroll_length
        self.max_grad_norm = max_grad_norm
    
    def train_step(
        self,
        observations: torch.Tensor,    # (T+1, B, H, W, C)
        actions: torch.Tensor,         # (T, B)
        rewards: torch.Tensor,         # (T, B)
        dones: torch.Tensor,           # (T, B)
        behavior_log_probs: torch.Tensor,  # (T, B)
        initial_hidden_states: Optional[List] = None,
    ) -> Dict:
        """
        Perform one training step.
        
        Args:
            observations: Sequence of observations
            actions: Actions taken
            rewards: Rewards received
            dones: Episode termination flags
            behavior_log_probs: Log probs under behavior policy
            initial_hidden_states: Initial hidden states
            
        Returns:
            Dict with loss values
        """
        T, B = actions.shape
        
        # Forward pass through entire sequence
        all_logits = []
        all_values = []
        
        hidden_states = initial_hidden_states
        
        for t in range(T + 1):
            obs_t = observations[t].to(self.device)
            logits, value, hidden_states, _ = self.agent(obs_t, hidden_states)
            all_logits.append(logits)
            all_values.append(value.squeeze(-1))
        
        logits = torch.stack(all_logits[:-1], dim=0)  # (T, B, num_actions)
        values = torch.stack(all_values, dim=0)         # (T+1, B)
        
        # Compute loss
        total_loss, loss_dict = self.loss_fn(
            logits, values,
            actions.to(self.device),
            rewards.to(self.device),
            behavior_log_probs.to(self.device),
            dones.to(self.device),
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        return loss_dict


def create_trainer(
    agent,
    total_transitions: int = 250_000_000,
    batch_size: int = 16,
    initial_lr: float = 4e-4,
    gamma: float = 0.97,
    lambda_: float = 0.97,
    entropy_coef: float = 0.01,
    logit_l2_coef: float = 1e-3,
    param_l2_coef: float = 1e-5,
    device: Optional[torch.device] = None,
) -> Tuple:
    """
    Create trainer with paper's hyperparameters.
    
    Args:
        agent: DRC agent
        total_transitions: Total training transitions
        batch_size: Batch size
        initial_lr: Initial learning rate (decays linearly to 0)
        gamma: Discount factor
        lambda_: V-trace lambda
        entropy_coef: Entropy bonus coefficient
        logit_l2_coef: L2 penalty on logits
        param_l2_coef: L2 regularization on parameters
        device: Device
        
    Returns:
        (trainer, scheduler)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr)
    
    # Linear learning rate decay
    total_steps = total_transitions // batch_size
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )
    
    loss_fn = IMPALALoss(
        gamma=gamma,
        lambda_=lambda_,
        entropy_coef=entropy_coef,
        logit_l2_coef=logit_l2_coef,
        param_l2_coef=param_l2_coef,
    )
    
    trainer = DRCTrainer(
        agent=agent,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    
    return trainer, scheduler
