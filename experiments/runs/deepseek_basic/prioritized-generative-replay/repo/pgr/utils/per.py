"""
Prioritized Experience Replay (PER) implementation for comparison.

From Section 5.1 / Fig. 3a: we compare PGR against standard PER
using TD-error and curiosity as priority criteria.
"""
import torch
import numpy as np
from typing import Tuple, Optional


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer (Schaul et al., 2015).
    
    Uses proportional prioritization with importance sampling weights.
    Priority can be based on TD-error (standard) or curiosity.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = 1_000_000,
        alpha: float = 0.6,  # prioritization exponent
        beta: float = 0.4,   # importance sampling exponent
        beta_increment: float = 1e-4,
        epsilon: float = 1e-6,
        device: str = "cuda",
    ):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.device = device
        
        self.ptr = 0
        self.size = 0
        
        # Storage
        self.states = torch.zeros(max_size, state_dim, device=device)
        self.actions = torch.zeros(max_size, action_dim, device=device)
        self.next_states = torch.zeros(max_size, state_dim, device=device)
        self.rewards = torch.zeros(max_size, 1, device=device)
        self.dones = torch.zeros(max_size, 1, device=device)
        
        # Priority tree (simplified as array)
        self.priorities = torch.zeros(max_size, device=device)
        self.max_priority = 1.0
    
    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        priority: Optional[torch.Tensor] = None,
    ):
        """Add transitions with priority."""
        batch_size = state.shape[0]
        
        if self.ptr + batch_size > self.max_size:
            first_part = self.max_size - self.ptr
            second_part = batch_size - first_part
            
            if first_part > 0:
                self.states[self.ptr:self.max_size] = state[:first_part]
                self.actions[self.ptr:self.max_size] = action[:first_part]
                self.next_states[self.ptr:self.max_size] = next_state[:first_part]
                self.rewards[self.ptr:self.max_size] = reward[:first_part]
                self.dones[self.ptr:self.max_size] = done[:first_part]
                if priority is not None:
                    self.priorities[self.ptr:self.max_size] = priority[:first_part].squeeze()
                else:
                    self.priorities[self.ptr:self.max_size] = self.max_priority
            
            if second_part > 0:
                self.states[0:second_part] = state[first_part:]
                self.actions[0:second_part] = action[first_part:]
                self.next_states[0:second_part] = next_state[first_part:]
                self.rewards[0:second_part] = reward[first_part:]
                self.dones[0:second_part] = done[first_part:]
                if priority is not None:
                    self.priorities[0:second_part] = priority[first_part:].squeeze()
                else:
                    self.priorities[0:second_part] = self.max_priority
            
            self.ptr = second_part
        else:
            self.states[self.ptr:self.ptr + batch_size] = state
            self.actions[self.ptr:self.ptr + batch_size] = action
            self.next_states[self.ptr:self.ptr + batch_size] = next_state
            self.rewards[self.ptr:self.ptr + batch_size] = reward
            self.dones[self.ptr:self.ptr + batch_size] = done
            
            if priority is not None:
                self.priorities[self.ptr:self.ptr + batch_size] = priority.squeeze()
            else:
                self.priorities[self.ptr:self.ptr + batch_size] = self.max_priority
            
            self.ptr += batch_size
        
        self.size = min(self.size + batch_size, self.max_size)
        
        if priority is not None:
            self.max_priority = max(self.max_priority, priority.max().item())
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with proportional prioritization."""
        # Compute sampling probabilities
        probs = self.priorities[:self.size] ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        # Compute importance sampling weights
        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # normalize
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.next_states[indices],
            self.rewards[indices],
            self.dones[indices],
            weights.unsqueeze(-1),
            indices,
        )
    
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """Update priorities for given indices."""
        self.priorities[indices] = priorities.squeeze()
        self.max_priority = max(self.max_priority, priorities.max().item())
    
    def __len__(self) -> int:
        return self.size


class PERAgent:
    """
    Wrapper that adds PER to a REDQ/SAC agent.
    Used for Fig. 3a comparison.
    """
    def __init__(
        self,
        base_agent,
        state_dim: int,
        action_dim: int,
        priority_type: str = 'td_error',  # 'td_error' or 'curiosity'
        device: str = 'cuda',
    ):
        self.base_agent = base_agent
        self.priority_type = priority_type
        self.device = device
        
        # Replace standard buffer with PER buffer
        self.buffer = PrioritizedReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )
        
        # For curiosity-based priority
        if priority_type == 'curiosity':
            from pgr.relevance.functions import ICMRelevance
            self.curiosity_module = ICMRelevance(
                state_dim=state_dim,
                action_dim=action_dim,
            ).to(device)
            self.curiosity_optimizer = torch.optim.Adam(
                self.curiosity_module.parameters(), lr=1e-3
            )
        else:
            self.curiosity_module = None
            self.curiosity_optimizer = None
    
    def compute_priority(self, batch) -> torch.Tensor:
        """Compute priority values for a batch of transitions."""
        states, actions, next_states, rewards, dones = batch
        
        if self.priority_type == 'td_error':
            # Compute TD-error priority
            with torch.no_grad():
                q1, q2 = self.base_agent.critic(states, actions)
                q_current = torch.min(q1, q2)
                
                # Target
                next_actions, _ = self.base_agent.actor.sample(next_states)
                q1_next, q2_next = self.base_agent.critic_target(next_states, next_actions)
                q_next = torch.min(q1_next, q2_next)
                q_target = rewards + self.base_agent.gamma * (1 - dones) * q_next
                
                td_error = torch.abs(q_target - q_current)
                return td_error + 1e-6  # small epsilon
        
        elif self.priority_type == 'curiosity':
            # Compute curiosity-based priority
            with torch.no_grad():
                priority = self.curiosity_module(states, actions, next_states, rewards)
                return priority + 1e-6
        
        return torch.ones(states.shape[0], 1, device=self.device)
    
    def add_to_buffer(self, states, actions, next_states, rewards, dones):
        """Add transitions with computed priority."""
        batch = (states, actions, next_states, rewards, dones)
        priorities = self.compute_priority(batch[:5] if len(batch) > 5 else batch)
        self.buffer.add(states, actions, next_states, rewards, dones, priorities)
    
    def sample_batch(self, batch_size):
        """Sample batch with priorities and importance weights."""
        states, actions, next_states, rewards, dones, weights, indices = self.buffer.sample(batch_size)
        return states, actions, next_states, rewards, dones, weights, indices
