"""
Prioritized Generative Replay (PGR) - Main Algorithm

Implements Algorithm 1 from the paper: an outer + inner loop framework
for prioritized generative replay.

The framework:
- Outer loop: interact with environment, collect real transitions, update relevance function
- Inner loop (every T steps): train generative model on real data, generate synthetic
  transitions conditioned on relevance, train policy on mixed real+synthetic data
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Optional, Tuple, Dict, Any
import copy

from pgr.models.diffusion import (
    ConditionalDiffusionModel, 
    DiffusionProcess, 
    diffusion_loss,
)
from pgr.relevance.functions import (
    RelevanceFunction,
    ICMRelevance,
    RNDRelevance,
    CTSRelevance,
    ECORelevance,
    ReturnRelevance,
    TDErrorRelevance,
    create_relevance_function,
)
from pgr.utils.replay_buffer import (
    ReplayBuffer,
    SyntheticReplayBuffer,
    flatten_transitions,
    unflatten_transitions,
)


class PrioritizedGenerativeReplay:
    """
    Prioritized Generative Replay (PGR) main class.
    
    Manages:
    - Real replay buffer D_real
    - Synthetic replay buffer D_syn
    - Conditional diffusion model G
    - Relevance function F
    - Policy (REDQ/SAC for state-based, DRQ-v2 for pixel-based)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        relevance_type: str = "curiosity",
        # Diffusion model params
        diffusion_hidden_dim: int = 1024,
        diffusion_num_residual_blocks: int = 2,
        diffusion_timesteps: int = 1000,
        # PGR params
        synthetic_data_ratio: float = 0.5,
        guidance_scale: float = 1.0,
        p_uncond: float = 0.25,
        buffer_size: int = 1_000_000,
        inner_loop_freq: int = 10_000,
        batch_size: int = 256,
        utd_ratio: int = 20,
        # Relevance params
        relevance_kwargs: Optional[Dict] = None,
        # Device
        device: str = "cuda",
        # Pixel-based
        use_latent: bool = False,
        latent_dim: int = 50,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.relevance_type = relevance_type
        self.synthetic_data_ratio = synthetic_data_ratio
        self.guidance_scale = guidance_scale
        self.p_uncond = p_uncond
        self.buffer_size = buffer_size
        self.inner_loop_freq = inner_loop_freq
        self.batch_size = batch_size
        self.utd_ratio = utd_ratio
        self.device = device
        self.use_latent = use_latent
        self.latent_dim = latent_dim
        
        # Replay buffers
        effective_state_dim = latent_dim if use_latent else state_dim
        self.D_real = ReplayBuffer(effective_state_dim, action_dim, buffer_size, device)
        self.D_syn = SyntheticReplayBuffer(effective_state_dim, action_dim, buffer_size, device)
        
        # Conditional diffusion model
        self.diffusion_model = ConditionalDiffusionModel(
            state_dim=effective_state_dim,
            action_dim=action_dim,
            hidden_dim=diffusion_hidden_dim,
            num_residual_blocks=diffusion_num_residual_blocks,
            cond_dim=1,
            use_latent=use_latent,
            latent_dim=latent_dim,
        ).to(device)
        
        self.diffusion = DiffusionProcess(
            num_timesteps=diffusion_timesteps,
            device=device,
        )
        
        self.diffusion_optimizer = optim.Adam(
            self.diffusion_model.parameters(),
            lr=3e-4,
        )
        
        # Relevance function
        self.relevance_kwargs = relevance_kwargs or {}
        if relevance_type == "curiosity" or relevance_type == "icm":
            self.F = ICMRelevance(
                state_dim=effective_state_dim,
                action_dim=action_dim,
                feature_dim=self.relevance_kwargs.get('feature_dim', 256),
                hidden_dim=self.relevance_kwargs.get('hidden_dim', 256),
                use_latent=use_latent,
                latent_dim=latent_dim,
            ).to(device)
        elif relevance_type == "rnd":
            self.F = RNDRelevance(
                state_dim=effective_state_dim,
                feature_dim=self.relevance_kwargs.get('feature_dim', 512),
                hidden_dim=self.relevance_kwargs.get('hidden_dim', 512),
                use_latent=use_latent,
                cnn_latent_dim=latent_dim,
            ).to(device)
        elif relevance_type == "cts":
            self.F = CTSRelevance(
                state_dim=effective_state_dim,
                action_dim=action_dim,
            ).to(device)
        elif relevance_type == "eco":
            self.F = ECORelevance(
                state_dim=effective_state_dim,
                embed_dim=self.relevance_kwargs.get('embed_dim', 512),
                memory_size=self.relevance_kwargs.get('memory_size', 200),
                use_latent=use_latent,
                latent_dim=latent_dim,
            ).to(device)
        elif relevance_type == "return":
            # Return-based needs Q-function and policy set later
            self.F = None
        elif relevance_type == "td_error" or relevance_type == "td":
            self.F = None
        else:
            raise ValueError(f"Unknown relevance type: {relevance_type}")
        
        # Separate optimizer for relevance function (ICM/RND)
        if hasattr(self.F, 'parameters'):
            self.F_optimizer = optim.Adam(
                [p for p in self.F.parameters() if p.requires_grad],
                lr=1e-3,
            )
        else:
            self.F_optimizer = None
        
        # Training state
        self.total_env_steps = 0
        self.generation_input_dim = self.diffusion_model.input_dim
        
        # Policy will be set externally
        self.policy = None
        self.q_function = None
        self.q_target = None
        
        # For latent (pixel-based) mode
        self.visual_encoder = None
    
    def set_policy(self, policy, q_function=None, q_target=None):
        """Set the policy and Q-networks for the agent."""
        self.policy = policy
        self.q_function = q_function
        self.q_target = q_target
        
        # Initialize return/TD-error relevance functions if needed
        if self.relevance_type == "return":
            self.F = ReturnRelevance(q_function, policy)
        elif self.relevance_type in ["td_error", "td"]:
            self.F = TDErrorRelevance(q_function, q_target)
    
    def set_visual_encoder(self, encoder):
        """Set visual encoder for pixel-based tasks."""
        self.visual_encoder = encoder
    
    def compute_relevance(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        next_states: torch.Tensor, 
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """Compute relevance values for a batch of transitions."""
        if self.F is None:
            # Default: uniform relevance
            return torch.ones(states.shape[0], 1, device=self.device)
        
        with torch.no_grad():
            return self.F(states, actions, next_states, rewards)
    
    def update_relevance_function(self, batch: Tuple[torch.Tensor, ...]):
        """
        Update the relevance function F using real data.
        For ICM: update encoder and forward dynamics model
        For RND: update predictor network
        """
        if self.F_optimizer is None:
            return 0.0
        
        states, actions, next_states, rewards, dones = batch
        
        self.F_optimizer.zero_grad()
        
        if self.relevance_type in ["curiosity", "icm"]:
            forward_loss, inverse_loss, total_loss = self.F.compute_loss(
                states, actions, next_states
            )
            total_loss.backward()
            self.F_optimizer.step()
            return total_loss.item()
        
        elif self.relevance_type == "rnd":
            loss = self.F.compute_loss(next_states)
            loss.backward()
            self.F_optimizer.step()
            return loss.item()
        
        elif self.relevance_type == "eco":
            # Update memory with new states
            self.F.update_memory(states)
            return 0.0
        
        return 0.0
    
    def train_diffusion_model(
        self, 
        num_steps: int = 1000,
    ) -> float:
        """
        Train the conditional diffusion model on real replay buffer data.
        Implements Eq. (2) with CFG conditioning dropout.
        """
        self.diffusion_model.train()
        total_loss = 0.0
        
        for step in range(num_steps):
            # Sample from real buffer
            if len(self.D_real) < self.batch_size:
                break
            
            states, actions, next_states, rewards, _ = self.D_real.sample(self.batch_size)
            
            # Compute relevance values as conditioning signal
            conditions = self.compute_relevance(states, actions, next_states, rewards)
            
            # Flatten transitions for diffusion input
            x0 = flatten_transitions(states, actions, next_states, rewards)
            
            # Compute diffusion loss
            loss = diffusion_loss(
                self.diffusion_model,
                self.diffusion,
                x0,
                condition=conditions,
                p_uncond=self.p_uncond,
            )
            
            self.diffusion_optimizer.zero_grad()
            loss.backward()
            self.diffusion_optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / max(1, num_steps)
    
    @torch.no_grad()
    def generate_synthetic_transitions(
        self,
        num_transitions: int,
        use_top_k: bool = True,
        top_k_ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Generate synthetic transitions using the conditional diffusion model.
        
        Implements the "prompting" strategy: sample conditioning values from the 
        top-k highest relevance transitions in the real buffer.
        
        Args:
            num_transitions: number of transitions to generate
            use_top_k: whether to use top-k relevance conditioning
            top_k_ratio: fraction of top transitions to sample conditions from
        
        Returns:
            states, actions, next_states, rewards, dones
        """
        if len(self.D_real) == 0:
            raise ValueError("Real buffer is empty, cannot generate")
        
        # Sample conditioning values from real buffer
        # Get all transitions in real buffer (or a large subset)
        sample_size = min(len(self.D_real), 10000)
        states, actions, next_states, rewards, _ = self.D_real.sample(sample_size)
        
        # Compute relevance for sampling
        conditions = self.compute_relevance(states, actions, next_states, rewards)
        
        if use_top_k and top_k_ratio < 1.0:
            # Select top-k% by relevance
            k = max(1, int(sample_size * top_k_ratio))
            top_k_indices = torch.topk(conditions.squeeze(), k).indices
            top_conditions = conditions[top_k_indices]
            
            # Randomly sample from top-k
            cond_indices = torch.randint(0, k, (num_transitions,), device=self.device)
            gen_conditions = top_conditions[cond_indices]
        else:
            # Randomly sample conditions
            cond_indices = torch.randint(0, sample_size, (num_transitions,), device=self.device)
            gen_conditions = conditions[cond_indices]
        
        # Generate transitions using diffusion model with CFG
        generated = self.diffusion.sample(
            self.diffusion_model,
            num_transitions,
            self.generation_input_dim,
            condition=gen_conditions,
            guidance_scale=self.guidance_scale,
        )
        
        # Unflatten into transition components
        effective_state_dim = self.latent_dim if self.use_latent else self.state_dim
        states, actions, next_states, rewards = unflatten_transitions(
            generated, effective_state_dim, self.action_dim
        )
        
        # Generate dones (all False for synthetic transitions)
        dones = torch.zeros(num_transitions, 1, device=self.device)
        
        return states, actions, next_states, rewards, dones
    
    def fill_synthetic_buffer(self, num_transitions: int = 1_000_000):
        """Generate and fill the synthetic replay buffer."""
        states, actions, next_states, rewards, dones = self.generate_synthetic_transitions(
            num_transitions
        )
        self.D_syn.fill(states, actions, next_states, rewards, dones)
    
    def sample_training_batch(
        self, 
        batch_size: int
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample a training batch mixing real and synthetic data.
        Mix ratio is determined by synthetic_data_ratio.
        """
        n_real = int(batch_size * (1 - self.synthetic_data_ratio))
        n_syn = batch_size - n_real
        
        # Sample from real buffer
        if n_real > 0 and len(self.D_real) >= n_real:
            real_batch = self.D_real.sample(n_real)
        else:
            real_batch = None
        
        # Sample from synthetic buffer
        if n_syn > 0 and len(self.D_syn) >= n_syn:
            syn_batch = self.D_syn.sample(n_syn)
        else:
            syn_batch = None
        
        # Combine batches
        if real_batch is not None and syn_batch is not None:
            combined = tuple(
                torch.cat([r, s], dim=0) 
                for r, s in zip(real_batch, syn_batch)
            )
            return combined
        elif real_batch is not None:
            return real_batch
        elif syn_batch is not None:
            return syn_batch
        else:
            raise ValueError("No data available for sampling")
    
    def should_update_generator(self) -> bool:
        """Check if we should run the inner loop (every inner_loop_freq steps)."""
        return self.total_env_steps > 0 and self.total_env_steps % self.inner_loop_freq == 0
    
    def add_real_transitions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Add real transitions to the real buffer."""
        if self.use_latent and self.visual_encoder is not None:
            # Encode pixel observations to latent space
            with torch.no_grad():
                states = self.visual_encoder(states)
                next_states = self.visual_encoder(next_states)
        
        self.D_real.add(states, actions, next_states, rewards, dones)
        self.total_env_steps += states.shape[0]
    
    def get_config(self) -> Dict[str, Any]:
        """Return the PGR configuration."""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'relevance_type': self.relevance_type,
            'synthetic_data_ratio': self.synthetic_data_ratio,
            'guidance_scale': self.guidance_scale,
            'p_uncond': self.p_uncond,
            'buffer_size': self.buffer_size,
            'inner_loop_freq': self.inner_loop_freq,
            'batch_size': self.batch_size,
            'utd_ratio': self.utd_ratio,
            'use_latent': self.use_latent,
            'latent_dim': self.latent_dim,
            'diffusion_timesteps': self.diffusion.num_timesteps,
            'generation_input_dim': self.generation_input_dim,
        }
