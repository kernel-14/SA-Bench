"""
Consistency model trainer with Generator-Augmented Flows (GC).

Implements:
1. Standard iCT (independent coupling) training
2. iCT-OT (minibatch optimal transport coupling)
3. iCT-GC (generator-augmented coupling) with joint learning

The key algorithm (Algorithm 1 from the paper):
- At each step, sample x_star ~ p_star, z ~ p_z
- Sample timestep index i
- Sample mask m ~ Binomial(mu, batch_size)
- Compute IC intermediate points: x_{t_i} = x_star + sigma_{t_i} * z
- Predict endpoint: x_hat_{t_i} = sg(f_theta(x_{t_i}, sigma_{t_i}))
- Mix IC and GC: x_hat_{t_i} = m * x_hat_{t_i} + (1-m) * x_star
- Compute GC intermediate points: x_tilde_{t_i} = x_hat_{t_i} + sigma_{t_i} * z
- Compute consistency loss
"""

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .schedules import NoiseSchedule, TimestepSchedule, TimestepSampler


def pseudo_huber_loss(x, y, c=0.00054 * math.sqrt(3072)):
    """
    Pseudo-Huber loss (smooth L1 loss variant).
    D(x, y) = sqrt(||x - y||^2 + c^2) - c
    
    Default c is from Song & Dhariwal (2024) for CIFAR-10 (32x32x3 images).
    """
    diff = x - y
    return torch.sqrt((diff ** 2).sum(dim=[1, 2, 3]) + c ** 2) - c


def lpips_loss(x, y, lpips_fn=None):
    """LPIPS perceptual loss."""
    if lpips_fn is None:
        return F.mse_loss(x, y, reduction='none').mean(dim=[1, 2, 3])
    return lpips_fn(x, y).squeeze()


class ConsistencyTrainer:
    """
    Trainer for consistency models with generator-augmented flows.
    
    Supports three training modes:
    - 'IC': Standard independent coupling (iCT-IC)
    - 'OT': Minibatch optimal transport coupling (iCT-OT)
    - 'GC': Generator-augmented coupling with joint learning (iCT-GC)
    """
    
    def __init__(
        self,
        model,
        optimizer,
        noise_schedule,
        timestep_schedule,
        timestep_sampler,
        device,
        mu=0.5,
        ema_decay=0.9999,
        use_ema_for_gc=True,
        loss_type='pseudo_huber',
        lpips_fn=None,
        sigma_data=0.5,
    ):
        """
        Args:
            model: ConsistencyModel instance
            optimizer: PyTorch optimizer
            noise_schedule: NoiseSchedule instance
            timestep_schedule: TimestepSchedule instance
            timestep_sampler: TimestepSampler instance
            device: Training device
            mu: Joint learning factor (probability of using GC trajectories)
            ema_decay: EMA decay rate for the target model
            use_ema_for_gc: Whether to use EMA model for GC endpoint prediction
            loss_type: Type of loss function ('pseudo_huber', 'l2', 'lpips')
            lpips_fn: LPIPS function (required if loss_type='lpips')
            sigma_data: Standard deviation of data distribution
        """
        self.model = model
        self.optimizer = optimizer
        self.noise_schedule = noise_schedule
        self.timestep_schedule = timestep_schedule
        self.timestep_sampler = timestep_sampler
        self.device = device
        self.mu = mu
        self.ema_decay = ema_decay
        self.use_ema_for_gc = use_ema_for_gc
        self.loss_type = loss_type
        self.lpips_fn = lpips_fn
        self.sigma_data = sigma_data
        
        # Create EMA model for target computation
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        
        self.step = 0
        self.current_N = None

    def update_ema(self):
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def compute_loss(self, x1, x2, sigma_i, loss_weights):
        """
        Compute consistency loss between two predictions.
        
        Args:
            x1: First prediction (stop-gradient target)
            x2: Second prediction (online model)
            sigma_i: Noise level indices
            loss_weights: Loss weights lambda(sigma_i)
            
        Returns:
            Scalar loss value
        """
        weights = torch.tensor(loss_weights, device=self.device, dtype=x1.dtype)[sigma_i]
        
        if self.loss_type == 'pseudo_huber':
            c = 0.00054 * math.sqrt(x1.shape[1] * x1.shape[2] * x1.shape[3])
            diff = x1 - x2
            loss = torch.sqrt((diff ** 2).sum(dim=[1, 2, 3]) + c ** 2) - c
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x1, x2, reduction='none').mean(dim=[1, 2, 3])
        elif self.loss_type == 'lpips':
            loss = lpips_loss(x1, x2, self.lpips_fn)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return (weights * loss).mean()

    def train_step_ic(self, x_star, sigmas, loss_weights):
        """
        Standard IC consistency training step.
        
        Args:
            x_star: Real data batch of shape (B, C, H, W)
            sigmas: Array of sigma values for current N
            loss_weights: Loss weights for current N
            
        Returns:
            Loss value
        """
        batch_size = x_star.shape[0]
        
        # Sample noise
        z = torch.randn_like(x_star)
        
        # Sample timestep indices
        indices = self.timestep_sampler.sample_indices(sigmas, batch_size, device=self.device)
        
        # Compute IC intermediate points
        indices_np = indices.cpu().numpy()
        sigma_i = torch.tensor(sigmas[indices_np], device=self.device, dtype=x_star.dtype)
        sigma_i1 = torch.tensor(sigmas[indices_np + 1], device=self.device, dtype=x_star.dtype)
        
        sigma_i_4d = sigma_i.reshape(-1, 1, 1, 1)
        sigma_i1_4d = sigma_i1.reshape(-1, 1, 1, 1)
        
        x_ti = x_star + sigma_i_4d * z
        x_ti1 = x_star + sigma_i1_4d * z
        
        # Compute consistency loss
        with torch.no_grad():
            target = self.ema_model(x_ti, sigma_i)
        
        pred = self.model(x_ti1, sigma_i1)
        
        loss = self.compute_loss(target, pred, indices_np, loss_weights)
        return loss

    def train_step_gc(self, x_star, sigmas, loss_weights):
        """
        Generator-augmented coupling (GC) consistency training step.
        
        Implements Algorithm 1 from the paper with joint learning.
        
        Args:
            x_star: Real data batch of shape (B, C, H, W)
            sigmas: Array of sigma values for current N
            loss_weights: Loss weights for current N
            
        Returns:
            Loss value
        """
        batch_size = x_star.shape[0]
        
        # Sample noise
        z = torch.randn_like(x_star)
        
        # Sample timestep indices
        indices = self.timestep_sampler.sample_indices(sigmas, batch_size, device=self.device)
        
        indices_np = indices.cpu().numpy()
        sigma_i = torch.tensor(sigmas[indices_np], device=self.device, dtype=x_star.dtype)
        sigma_i1 = torch.tensor(sigmas[indices_np + 1], device=self.device, dtype=x_star.dtype)
        
        sigma_i_4d = sigma_i.reshape(-1, 1, 1, 1)
        sigma_i1_4d = sigma_i1.reshape(-1, 1, 1, 1)
        
        # Step 1: Compute IC intermediate points
        x_ti = x_star + sigma_i_4d * z
        
        # Step 2: Predict endpoint using the model (stop-gradient)
        predictor = self.ema_model if self.use_ema_for_gc else self.model
        with torch.no_grad():
            x_hat_ti = predictor(x_ti, sigma_i)
        
        # Step 3: Sample mask m ~ Binomial(mu, batch_size)
        # m=1 means use GC, m=0 means use IC (x_star)
        m = torch.bernoulli(torch.full((batch_size, 1, 1, 1), self.mu, device=self.device))
        
        # Step 4: Mix IC and GC endpoints
        # x_hat_ti = m * x_hat_ti + (1-m) * x_star
        x_hat_ti = m * x_hat_ti + (1 - m) * x_star
        
        # Step 5: Compute GC intermediate points
        x_tilde_ti = x_hat_ti + sigma_i_4d * z
        x_tilde_ti1 = x_hat_ti + sigma_i1_4d * z
        
        # Step 6: Compute consistency loss
        with torch.no_grad():
            target = self.ema_model(x_tilde_ti, sigma_i)
        
        pred = self.model(x_tilde_ti1, sigma_i1)
        
        loss = self.compute_loss(target, pred, indices_np, loss_weights)
        return loss

    def train_step_ot(self, x_star, sigmas, loss_weights):
        """
        Minibatch optimal transport (OT) coupling consistency training step.
        
        Uses Hungarian matching within a minibatch to find optimal data-noise pairs.
        
        Args:
            x_star: Real data batch of shape (B, C, H, W)
            sigmas: Array of sigma values for current N
            loss_weights: Loss weights for current N
            
        Returns:
            Loss value
        """
        from scipy.optimize import linear_sum_assignment
        
        batch_size = x_star.shape[0]
        
        # Sample noise
        z = torch.randn_like(x_star)
        
        # Compute pairwise distances for OT matching
        x_flat = x_star.reshape(batch_size, -1).float()
        z_flat = z.reshape(batch_size, -1).float()
        
        # Cost matrix: squared L2 distance
        cost = torch.cdist(x_flat, z_flat, p=2).cpu().numpy()
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost)
        z = z[col_ind]  # Reorder noise to match data
        
        # Sample timestep indices
        indices = self.timestep_sampler.sample_indices(sigmas, batch_size, device=self.device)
        
        indices_np = indices.cpu().numpy()
        sigma_i = torch.tensor(sigmas[indices_np], device=self.device, dtype=x_star.dtype)
        sigma_i1 = torch.tensor(sigmas[indices_np + 1], device=self.device, dtype=x_star.dtype)
        
        sigma_i_4d = sigma_i.reshape(-1, 1, 1, 1)
        sigma_i1_4d = sigma_i1.reshape(-1, 1, 1, 1)
        
        # Compute OT intermediate points
        x_ti = x_star + sigma_i_4d * z
        x_ti1 = x_star + sigma_i1_4d * z
        
        # Compute consistency loss
        with torch.no_grad():
            target = self.ema_model(x_ti, sigma_i)
        
        pred = self.model(x_ti1, sigma_i1)
        
        loss = self.compute_loss(target, pred, indices_np, loss_weights)
        return loss

    def train_step(self, x_star, mode='GC'):
        """
        Perform one training step.
        
        Args:
            x_star: Real data batch
            mode: Training mode ('IC', 'OT', or 'GC')
            
        Returns:
            Loss value
        """
        # Get current number of timesteps
        N = self.timestep_schedule.get_N(self.step)
        if N != self.current_N:
            self.current_N = N
        
        # Get sigmas and loss weights for current N
        sigmas = self.noise_schedule.get_sigmas(N)  # N+1 sigma values for N intervals
        loss_weights = self.noise_schedule.get_loss_weights(sigmas)
        
        # Compute loss based on mode
        if mode == 'IC':
            loss = self.train_step_ic(x_star, sigmas, loss_weights)
        elif mode == 'OT':
            loss = self.train_step_ot(x_star, sigmas, loss_weights)
        elif mode == 'GC':
            loss = self.train_step_gc(x_star, sigmas, loss_weights)
        else:
            raise ValueError(f"Unknown training mode: {mode}")
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update EMA
        self.update_ema()
        
        self.step += 1
        return loss.item()
