"""
Replay buffer utilities for PGR.
Maintains separate real and synthetic replay buffers.
"""
import torch
import numpy as np
from typing import Tuple, Optional


class ReplayBuffer:
    """Standard replay buffer for storing transitions."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        max_size: int = 1_000_000,
        device: str = "cuda",
    ):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate storage
        self.states = torch.zeros(max_size, state_dim, device=device)
        self.actions = torch.zeros(max_size, action_dim, device=device)
        self.next_states = torch.zeros(max_size, state_dim, device=device)
        self.rewards = torch.zeros(max_size, 1, device=device)
        self.dones = torch.zeros(max_size, 1, device=device)
        
        # For pixel-based (latent) storage
        self.use_latent = False
        self.latent_dim = None
    
    def setup_latent(self, latent_dim: int):
        """Configure buffer for latent (pixel-based) storage."""
        self.use_latent = True
        self.latent_dim = latent_dim
        self.states = torch.zeros(self.max_size, latent_dim, device=self.device)
        self.next_states = torch.zeros(self.max_size, latent_dim, device=self.device)
    
    def add(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        next_state: torch.Tensor, 
        reward: torch.Tensor, 
        done: torch.Tensor
    ):
        """Add a batch of transitions to the buffer."""
        batch_size = state.shape[0]
        
        # If buffer is full, overwrite from beginning
        if self.ptr + batch_size > self.max_size:
            # Wrap around
            first_part = self.max_size - self.ptr
            second_part = batch_size - first_part
            
            if first_part > 0:
                self.states[self.ptr:self.max_size] = state[:first_part]
                self.actions[self.ptr:self.max_size] = action[:first_part]
                self.next_states[self.ptr:self.max_size] = next_state[:first_part]
                self.rewards[self.ptr:self.max_size] = reward[:first_part]
                self.dones[self.ptr:self.max_size] = done[:first_part]
            
            if second_part > 0:
                self.states[0:second_part] = state[first_part:]
                self.actions[0:second_part] = action[first_part:]
                self.next_states[0:second_part] = next_state[first_part:]
                self.rewards[0:second_part] = reward[first_part:]
                self.dones[0:second_part] = done[first_part:]
            
            self.ptr = second_part
        else:
            self.states[self.ptr:self.ptr + batch_size] = state
            self.actions[self.ptr:self.ptr + batch_size] = action
            self.next_states[self.ptr:self.ptr + batch_size] = next_state
            self.rewards[self.ptr:self.ptr + batch_size] = reward
            self.dones[self.ptr:self.ptr + batch_size] = done
            self.ptr += batch_size
        
        self.size = min(self.size + batch_size, self.max_size)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a random batch of transitions."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.next_states[indices],
            self.rewards[indices],
            self.dones[indices],
        )
    
    def sample_all(self) -> Tuple[torch.Tensor, ...]:
        """Return all transitions in the buffer."""
        return (
            self.states[:self.size],
            self.actions[:self.size],
            self.next_states[:self.size],
            self.rewards[:self.size],
            self.dones[:self.size],
        )
    
    def get_transitions(self, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Get transitions at specified indices."""
        return (
            self.states[indices],
            self.actions[indices],
            self.next_states[indices],
            self.rewards[indices],
            self.dones[indices],
        )
    
    def __len__(self) -> int:
        return self.size


class SyntheticReplayBuffer:
    """
    Buffer for storing synthetically generated transitions.
    Used to hold generated data between diffusion model updates.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        max_size: int = 1_000_000,
        device: str = "cuda",
    ):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.states = torch.zeros(max_size, state_dim, device=device)
        self.actions = torch.zeros(max_size, action_dim, device=device)
        self.next_states = torch.zeros(max_size, state_dim, device=device)
        self.rewards = torch.zeros(max_size, 1, device=device)
        self.dones = torch.zeros(max_size, 1, device=device)
    
    def fill(self, states, actions, next_states, rewards, dones):
        """Fill the synthetic buffer with generated transitions."""
        batch_size = states.shape[0]
        if batch_size > self.max_size:
            batch_size = self.max_size
            states = states[:batch_size]
            actions = actions[:batch_size]
            next_states = next_states[:batch_size]
            rewards = rewards[:batch_size]
            dones = dones[:batch_size]
        
        self.states[:batch_size] = states
        self.actions[:batch_size] = actions
        self.next_states[:batch_size] = next_states
        self.rewards[:batch_size] = rewards
        self.dones[:batch_size] = dones
        self.size = batch_size
        self.ptr = batch_size
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a random batch from synthetic buffer."""
        if self.size == 0:
            raise ValueError("Synthetic buffer is empty")
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.next_states[indices],
            self.rewards[indices],
            self.dones[indices],
        )
    
    def __len__(self) -> int:
        return self.size


def flatten_transitions(states, actions, next_states, rewards):
    """
    Flatten a batch of transitions into a single vector for diffusion input.
    Returns: (B, 2*state_dim + action_dim + 1) tensor.
    """
    return torch.cat([states, actions, next_states, rewards], dim=-1)


def unflatten_transitions(flat: torch.Tensor, state_dim: int, action_dim: int):
    """
    Unflatten a batch of diffusion outputs back into transitions.
    """
    s_end = state_dim
    a_end = s_end + action_dim
    ns_end = a_end + state_dim
    
    states = flat[:, :s_end]
    actions = flat[:, s_end:a_end]
    next_states = flat[:, a_end:ns_end]
    rewards = flat[:, ns_end:ns_end + 1]
    
    return states, actions, next_states, rewards
