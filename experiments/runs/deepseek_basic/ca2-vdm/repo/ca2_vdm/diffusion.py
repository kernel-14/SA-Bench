"""
Diffusion process utilities for Ca2-VDM.

Implements:
- DDPM forward (noising) process
- Training loss computation (simple + VLB loss)
- DDPM/Improved DDPM sampling
- Partial noising for training (clean prefix + noisy target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def make_beta_schedule(
    schedule: str = "linear",
    num_timesteps: int = 1000,
    start: float = 1e-4,
    end: float = 0.02,
) -> torch.Tensor:
    """Create beta schedule for DDPM."""
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "cosine":
        steps = num_timesteps + 1
        s = 0.008
        t = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos((t / num_timesteps + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clamp(betas, max=0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return betas


class DiffusionProcess:
    """
    Manages the diffusion forward process and provides utilities
    for training and sampling.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        self.num_timesteps = num_timesteps
        
        betas = make_beta_schedule(schedule, num_timesteps, beta_start, beta_end)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register('betas', betas)
        self.register('alphas', alphas)
        self.register('alphas_cumprod', alphas_cumprod)
        self.register('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register('posterior_variance', posterior_variance)
        self.register('posterior_log_variance_clipped',
                       torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register('posterior_mean_coef1',
                       betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register('posterior_mean_coef2',
                       (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def register(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)
    
    def to(self, device):
        for key in dir(self):
            val = getattr(self, key)
            if isinstance(val, torch.Tensor):
                setattr(self, key, val.to(device))
        return self
    
    def _broadcast_shape(self, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Reshape t to broadcast with a tensor of ndim dimensions.
        t has shape (B,) and target has shape (B, ...).
        Returns shape (B, 1, ..., 1) with ndim-1 ones.
        """
        return t.reshape(-1, *([1] * (ndim - 1)))
    
    def q_sample(self, z_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(z_0)
        
        sqrt_alpha = self._broadcast_shape(
            self.sqrt_alphas_cumprod[t].to(z_0.device), z_0.dim()
        )
        sqrt_one_minus_alpha = self._broadcast_shape(
            self.sqrt_one_minus_alphas_cumprod[t].to(z_0.device), z_0.dim()
        )
        
        return sqrt_alpha * z_0 + sqrt_one_minus_alpha * noise
    
    def q_sample_partial(
        self, 
        z_0: torch.Tensor, 
        t: torch.Tensor, 
        P: int,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Partial noising: keep first P frames clean, noise remaining frames.
        
        Args:
            z_0: clean latent (B, C, L, H, W)
            t: timestep (B,)
            P: number of clean prefix frames
            noise: optional pre-sampled noise
        
        Returns:
            z_t: partially noised latent [z_0^{0:P}, z_t^{P:L}]
            noise: the noise added to target frames
            mask: loss mask (1 for target frames, 0 for prefix)
        """
        B, C, L, H, W = z_0.shape
        
        if noise is None:
            noise = torch.randn(B, C, L - P, H, W, device=z_0.device, dtype=z_0.dtype)
        
        # Clean prefix (no noise)
        z_t_prefix = z_0[:, :, :P, :, :]
        # Target frames to be noised
        z_0_target = z_0[:, :, P:, :, :]
        
        # Broadcast t to shape (B, 1, 1, 1, 1) for 5D tensors
        sqrt_alpha = self.sqrt_alphas_cumprod[t].to(z_0.device)
        sqrt_alpha = sqrt_alpha.reshape(B, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].to(z_0.device)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.reshape(B, 1, 1, 1, 1)
        
        z_t_target = sqrt_alpha * z_0_target + sqrt_one_minus_alpha * noise
        
        # Concatenate along temporal dimension
        z_t = torch.cat([z_t_prefix, z_t_target], dim=2)
        
        # Full noise tensor for loss computation (prefix part is zero)
        noise_full = torch.cat([
            torch.zeros(B, C, P, H, W, device=z_0.device, dtype=z_0.dtype),
            noise
        ], dim=2)
        
        # Loss mask: 0 for prefix, 1 for target
        mask = torch.zeros(B, C, L, H, W, device=z_0.device, dtype=z_0.dtype)
        mask[:, :, P:, :, :] = 1.0
        
        return z_t, noise_full, mask
    
    def predict_start_from_noise(
        self, z_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor
    ) -> torch.Tensor:
        sqrt_recip_alpha = self._broadcast_shape(
            self.sqrt_recip_alphas_cumprod[t].to(z_t.device), z_t.dim()
        )
        sqrt_recipm1_alpha = self._broadcast_shape(
            self.sqrt_recipm1_alphas_cumprod[t].to(z_t.device), z_t.dim()
        )
        return sqrt_recip_alpha * z_t - sqrt_recipm1_alpha * noise_pred
    
    def q_posterior_mean_variance(
        self, z_0: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape_fn = lambda x: self._broadcast_shape(x.to(z_t.device), z_t.dim())
        
        posterior_mean = (
            shape_fn(self.posterior_mean_coef1[t]) * z_0 +
            shape_fn(self.posterior_mean_coef2[t]) * z_t
        )
        posterior_variance = shape_fn(self.posterior_variance[t])
        posterior_log_variance = shape_fn(self.posterior_log_variance_clipped[t])
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance(
        self,
        model_output: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        learn_sigma: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if learn_sigma:
            noise_pred, log_var = model_output.chunk(2, dim=1)
            z_0_pred = self.predict_start_from_noise(z_t, t, noise_pred)
        else:
            noise_pred = model_output
            z_0_pred = self.predict_start_from_noise(z_t, t, noise_pred)
            log_var = None
        
        z_0_pred = torch.clamp(z_0_pred, -1.0, 1.0)
        
        posterior_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior_mean_variance(z_0_pred, z_t, t)
        
        if learn_sigma and log_var is not None:
            min_log = self._broadcast_shape(
                self.posterior_log_variance_clipped[t].to(z_t.device), z_t.dim()
            )
            max_log = self._broadcast_shape(
                torch.log(self.betas[t]).to(z_t.device), z_t.dim()
            )
            frac = (log_var + 1) / 2
            log_variance = frac * max_log + (1 - frac) * min_log
            variance = torch.exp(log_variance)
        else:
            variance = posterior_variance
            log_variance = posterior_log_variance
        
        return posterior_mean, variance, log_variance
    
    def compute_loss(
        self,
        model: nn.Module,
        z_0: torch.Tensor,
        P: int,
        t: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        cyclic_offset: int = 0,
        learn_sigma: bool = True,
        use_vlb_loss: bool = True,
        vlb_weight: float = 0.001,
    ) -> dict:
        """
        Compute training loss for Ca2-VDM.
        
        Loss = L_simple + vlb_weight * L_vlb
        
        The model receives the full sequence [z_0^{0:P}, z_t^{P:L}] with
        distinct timestep embeddings (t=0 for prefix, t for target).
        """
        # Partial noising
        z_t, noise, loss_mask = self.q_sample_partial(z_0, t, P)
        
        # Model forward
        result = model(
            z=z_t,
            t=t,
            P=P,
            text_emb=text_emb,
            cyclic_offset=cyclic_offset,
        )
        model_output = result['output']
        
        if learn_sigma:
            noise_pred, log_var = model_output.chunk(2, dim=1)
        else:
            noise_pred = model_output
            log_var = None
        
        # L_simple: MSE on noise (masked)
        mse_loss = F.mse_loss(noise_pred * loss_mask, noise * loss_mask, reduction='none')
        mse_loss = mse_loss.mean()
        
        # L_vlb
        vlb_loss = torch.tensor(0.0, device=z_0.device)
        if use_vlb_loss and learn_sigma and P < z_0.shape[2]:
            # Compute VLB on target frames only
            z_0_target = z_0[:, :, P:, :, :]
            z_t_target = z_t[:, :, P:, :, :]
            noise_pred_target = noise_pred[:, :, P:, :, :]
            model_output_target = model_output[:, :, P:, :, :]
            
            z_0_pred = self.predict_start_from_noise(z_t_target, t, noise_pred_target)
            z_0_pred = torch.clamp(z_0_pred, -1.0, 1.0)
            
            true_mean, true_var, true_log_var = self.q_posterior_mean_variance(
                z_0_target, z_t_target, t
            )
            pred_mean, pred_var, pred_log_var = self.p_mean_variance(
                model_output_target, z_t_target, t, learn_sigma
            )
            
            kl = 0.5 * (
                -1.0 + pred_log_var - true_log_var +
                (torch.exp(true_log_var) + (true_mean - pred_mean) ** 2) / torch.exp(pred_log_var)
            )
            vlb_loss = kl.mean()
        
        total_loss = mse_loss + vlb_weight * vlb_loss if use_vlb_loss else mse_loss
        
        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'vlb_loss': vlb_loss,
        }
