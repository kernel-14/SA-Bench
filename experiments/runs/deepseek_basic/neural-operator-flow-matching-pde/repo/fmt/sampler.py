"""Sampling and generation utilities for Flow Marching Transformer.

Provides:
1. Euler ODE sampler for flow marching integration
2. Autoregressive rollout with diffusion forcing
3. Ensemble generation with bridge parameter k
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import math

from .model import FlowMarchingTransformer


class FlowMarchingSampler:
    """Sampler for Flow Marching Transformer.
    
    Uses Euler ODE discretization to integrate from flow time t=0
    to t=1. Supports both deterministic prediction (k=1) and
    stochastic generation (k < 1).
    """
    
    def __init__(self, 
                 model: FlowMarchingTransformer,
                 num_steps: int = 100,
                 dt: float = 0.01):
        """
        Args:
            model: Trained FMT model
            num_steps: Number of ODE integration steps (N = 100)
            dt: Step size for Euler integration
        """
        self.model = model
        self.num_steps = num_steps
        self.dt = dt
        self.config = model.config
    
    @torch.no_grad()
    def sample_next_frame(self,
                          latent_history: List[torch.Tensor],
                          h: torch.Tensor,
                          k_values: Optional[List[float]] = None,
                          num_steps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample the next latent frame using flow marching.
        
        For deterministic prediction: k = [1, 1, 1, 1]
        For stochastic generation: k = [1, 1, 1, k3] with k3 < 1
        
        Args:
            latent_history: List of 4 previous latent states (y_0, y_1, y_2, y_3)
            h: Current GRU hidden state
            k_values: Bridge parameters for each frame, defaults to all 1s
            num_steps: Override number of integration steps
            
        Returns:
            y_next: Predicted next latent state
            h_next: Updated GRU hidden state
        """
        if k_values is None:
            k_values = [1.0, 1.0, 1.0, 1.0]  # deterministic
        
        if num_steps is None:
            num_steps = self.num_steps
        
        dt = 1.0 / num_steps
        device = latent_history[0].device
        B = latent_history[0].shape[0]
        
        # Initialize states at t=0
        # x_t^k at t=0: x_0^k = k * x_0 + (1-k) * z
        # For k=1 (deterministic): x_0^1 = x_0 (clean)
        # For k<1 (stochastic):   x_0^k = k * x_0 + (1-k) * z (noisy)
        current_states = []
        for i, (y, k) in enumerate(zip(latent_history, k_values)):
            if k >= 1.0:
                current_states.append(y.clone())
            else:
                z = torch.randn_like(y)
                k_tensor = torch.tensor(k, device=device).view(1, 1, 1, 1)
                current_states.append(k_tensor * y + (1 - k_tensor) * z)
        
        # ODE integration from t=0 to t=1
        t = 0.0
        for step in range(num_steps):
            t_tensor = torch.full((B,), t, device=device)
            
            # Get velocity field from model
            velocities, _ = self.model(current_states, t_tensor, h)
            
            # Euler step: x_{t+dt} = x_t + v(x_t, t) * dt
            for i in range(4):
                current_states[i] = current_states[i] + dt * velocities[i]
            
            t += dt
        
        # After integration, current_states[3] is the predicted y_4
        # But actually, all frames have been transported forward
        # Frame 0 transported to frame 1, ..., frame 3 transported to frame 4
        y_next = current_states[3]
        
        # Update GRU state
        # Use model inference at final state
        t_tensor = torch.full((B,), 1.0, device=device)
        _, h_next = self.model(current_states, t_tensor, h)
        
        return y_next, h_next
    
    @torch.no_grad()
    def autoregressive_rollout(self,
                                initial_frames: List[torch.Tensor],
                                num_steps: int,
                                k_prediction: float = 1.0) -> List[torch.Tensor]:
        """Perform long-term autoregressive rollout.
        
        Args:
            initial_frames: List of 4 initial latent states
            num_steps: Number of future steps to predict
            k_prediction: Bridge parameter for prediction (1.0 = deterministic)
            
        Returns:
            List of all latent states (initial + predicted)
        """
        B = initial_frames[0].shape[0]
        device = initial_frames[0].device
        
        # Initialize GRU state
        h = self.model.gru.init_state(B, device)
        
        # Warm up GRU with initial frames
        for i, y in enumerate(initial_frames):
            patches = self.model.frame_patchify[i](y)
            B_d, D, hw, ww = patches.shape
            tokens = patches.flatten(2).transpose(1, 2) + self.model.pos_embs[i]
            t_tensor = torch.zeros(B, device=device)
            h = self.model.gru(h, tokens, t_tensor)
        
        # Rolling window of 4 frames
        window = [y.clone() for y in initial_frames]
        all_states = [y.clone() for y in initial_frames]
        
        for step in range(num_steps):
            k_vals = [1.0, 1.0, 1.0, k_prediction]
            y_next, h = self.sample_next_frame(window, h, k_values=k_vals)
            
            all_states.append(y_next)
            
            # Shift window: drop oldest frame, add prediction
            window = window[1:] + [y_next]
        
        return all_states
    
    @torch.no_grad()
    def generate_ensemble(self,
                           history: List[torch.Tensor],
                           k3: float,
                           batch_size: int = 32) -> List[torch.Tensor]:
        """Generate an ensemble of possible next states.
        
        Given clean history frames (x_0, x_1, x_2, x_3),
        generates multiple possible x_4 by varying the noise.
        
        Args:
            history: 4 clean history frames [x_0, x_1, x_2, x_3]
            k3: Bridge parameter for the 4th frame (k < 1 for stochastic)
            batch_size: Number of ensemble members
            
        Returns:
            List of ensemble predictions
        """
        B_orig = history[0].shape[0]
        device = history[0].device
        
        # Expand history to batch_size
        expanded_history = []
        for y in history:
            expanded_history.append(y.expand(batch_size, -1, -1, -1))
        
        # Initialize GRU state
        h = self.model.gru.init_state(batch_size, device)
        
        # Run sampling
        k_vals = [1.0, 1.0, 1.0, k3]
        y_ensembles = []
        
        for b in range(batch_size):
            y_next, _ = self.sample_next_frame(expanded_history, h, 
                                                k_values=k_vals)
            y_ensembles.append(y_next)
        
        return y_ensembles
