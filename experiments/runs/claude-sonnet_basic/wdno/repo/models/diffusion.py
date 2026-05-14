"""
DDPM/DDIM diffusion model for WDNO.

Implements:
1. DDPM forward process (noise addition)
2. DDPM reverse process (denoising)
3. DDIM sampling for accelerated inference
4. Classifier-free guidance for conditional generation
5. Classifier-based guidance for control tasks

Based on:
- Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
- Song et al. (2020) "Denoising Diffusion Implicit Models"
- Ho & Salimans (2022) "Classifier-Free Diffusion Guidance"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    """Linear beta schedule."""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    """Extract values from a at indices t, reshaping for broadcasting."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion model implementing DDPM and DDIM.
    
    This is the core diffusion model used in WDNO. It operates in the
    wavelet domain, where the denoising model predicts noise in wavelet space.
    """
    
    def __init__(
        self,
        model,
        timesteps=1000,
        beta_schedule="cosine",
        loss_type="l2",
        p2_loss_weight_gamma=0.0,
        p2_loss_weight_k=1,
    ):
        super().__init__()
        
        self.model = model
        self.timesteps = timesteps
        
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Forward process
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20))
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        
        self.loss_type = loss_type
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward process: add noise to x_start at timestep t.
        
        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def q_posterior(self, x_start, x_t, t):
        """Compute posterior mean and variance q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, t, cond=None, clip_denoised=True):
        """Compute mean and variance of p(x_{t-1} | x_t)."""
        pred_noise = self.model(x, t, cond=cond)
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        
        if clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index, cond=None):
        """Sample from p(x_{t-1} | x_t) using DDPM."""
        model_mean, _, model_log_variance, _ = self.p_mean_variance(x, t, cond=cond)
        noise = torch.randn_like(x)
        # No noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None):
        """Full DDPM sampling loop."""
        device = next(self.model.parameters()).device
        b = shape[0]
        
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                i,
                cond=cond
            )
        
        return img
    
    def compute_loss(self, x_start, t, cond=None, noise=None):
        """
        Compute training loss.
        
        L = E[||eps - eps_theta(sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps, t)||^2]
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, cond=cond)
        
        if self.loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def forward(self, x, cond=None):
        """Training forward pass."""
        b = x.shape[0]
        device = x.device
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        return self.compute_loss(x, t, cond=cond)


class DDIMSampler(nn.Module):
    """
    DDIM sampler for accelerated inference.
    
    Implements Denoising Diffusion Implicit Models (Song et al., 2020).
    Used in WDNO for both simulation and control tasks.
    """
    
    def __init__(self, diffusion_model, ddim_timesteps=50, ddim_eta=1.0):
        """
        Args:
            diffusion_model: Trained GaussianDiffusion model
            ddim_timesteps: Number of DDIM sampling steps (default: 50 for 1D, 100 for 2D)
            ddim_eta: DDIM eta parameter (default: 1.0 as per paper)
        """
        super().__init__()
        self.diffusion = diffusion_model
        self.ddim_timesteps = ddim_timesteps
        self.ddim_eta = ddim_eta
        
        # Compute DDIM timestep schedule
        c = self.diffusion.timesteps // ddim_timesteps
        self.ddim_timestep_seq = list(range(0, self.diffusion.timesteps, c))
        self.ddim_timestep_prev_seq = [-1] + self.ddim_timestep_seq[:-1]
    
    def get_noise_pred(self, x, t, cond=None, uncond_cond=None, guidance_scale=1.0):
        """
        Get noise prediction with optional classifier-free guidance.
        
        Implements: eps_theta(x, D) + w * (eps_theta(x, y) - eps_theta(x, D))
        where D is the unconditional identifier.
        """
        if uncond_cond is not None and guidance_scale != 1.0:
            # Classifier-free guidance
            noise_cond = self.diffusion.model(x, t, cond=cond)
            noise_uncond = self.diffusion.model(x, t, cond=uncond_cond)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = self.diffusion.model(x, t, cond=cond)
        
        return noise_pred
    
    @torch.no_grad()
    def sample(
        self,
        shape,
        cond=None,
        uncond_cond=None,
        guidance_scale=1.0,
        return_intermediates=False,
    ):
        """
        DDIM sampling.
        
        Args:
            shape: Output shape
            cond: Conditioning information
            uncond_cond: Unconditional conditioning for classifier-free guidance
            guidance_scale: Guidance scale (omega in paper)
            return_intermediates: Whether to return intermediate samples
        
        Returns:
            Final sample (and intermediates if requested)
        """
        device = next(self.diffusion.model.parameters()).device
        b = shape[0]
        
        x = torch.randn(shape, device=device)
        intermediates = [x]
        
        for i in reversed(range(len(self.ddim_timestep_seq))):
            t = self.ddim_timestep_seq[i]
            prev_t = self.ddim_timestep_prev_seq[i]
            
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
            
            # Get noise prediction
            noise_pred = self.get_noise_pred(x, t_tensor, cond=cond, 
                                              uncond_cond=uncond_cond,
                                              guidance_scale=guidance_scale)
            
            # Compute x_{t-1}
            alpha_bar_t = self.diffusion.alphas_cumprod[t]
            alpha_bar_prev = self.diffusion.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
            
            # Predict x_0
            x0_pred = (x - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)
            
            # Compute sigma
            sigma = self.ddim_eta * (
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            ).sqrt()
            
            # Compute direction pointing to x_t
            pred_dir = (1 - alpha_bar_prev - sigma ** 2).sqrt() * noise_pred
            
            # Compute x_{t-1}
            x_prev = alpha_bar_prev.sqrt() * x0_pred + pred_dir
            
            if i > 0:
                noise = torch.randn_like(x)
                x_prev = x_prev + sigma * noise
            
            x = x_prev
            
            if return_intermediates:
                intermediates.append(x)
        
        if return_intermediates:
            return x, intermediates
        return x
    
    def sample_with_guidance(
        self,
        shape,
        cond,
        guidance_fn,
        guidance_scale,
        uncond_cond=None,
        cfg_scale=1.0,
        guidance_scheduler="cosine",
    ):
        """
        DDIM sampling with classifier-based guidance for control tasks.
        
        Implements the control update from the paper:
        W_{f}^{(k-1)} = W_{f}^{(k)} - eta * (eps_theta(W_f^{(k)}, W_a, k) 
                         + lambda * grad_W_f I(W_hat_f^{(k)})) + xi
        
        Args:
            shape: Output shape
            cond: Conditioning information (W_a)
            guidance_fn: Function computing gradient of control objective I
            guidance_scale: Lambda in the paper
            uncond_cond: Unconditional conditioning for classifier-free guidance
            cfg_scale: Classifier-free guidance scale
            guidance_scheduler: Schedule for guidance weight ("cosine" or "constant")
        
        Returns:
            Final sample
        """
        device = next(self.diffusion.model.parameters()).device
        b = shape[0]
        
        x = torch.randn(shape, device=device)
        
        total_steps = len(self.ddim_timestep_seq)
        
        for i in reversed(range(total_steps)):
            t = self.ddim_timestep_seq[i]
            prev_t = self.ddim_timestep_prev_seq[i]
            
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
            
            # Get noise prediction (with classifier-free guidance if applicable)
            with torch.no_grad():
                noise_pred = self.get_noise_pred(x, t_tensor, cond=cond,
                                                  uncond_cond=uncond_cond,
                                                  guidance_scale=cfg_scale)
            
            # Compute x_0 estimate for guidance
            alpha_bar_t = self.diffusion.alphas_cumprod[t]
            x0_hat = (x - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
            
            # Compute guidance gradient
            if guidance_fn is not None:
                # Compute guidance weight schedule
                if guidance_scheduler == "cosine":
                    progress = i / total_steps
                    guidance_weight = guidance_scale * (1 + math.cos(math.pi * progress)) / 2
                else:
                    guidance_weight = guidance_scale
                
                x0_hat_grad = x0_hat.detach().requires_grad_(True)
                guidance_loss = guidance_fn(x0_hat_grad)
                grad = torch.autograd.grad(guidance_loss, x0_hat_grad)[0]
                
                # Add guidance to noise prediction
                # grad w.r.t. x0_hat, need to convert to grad w.r.t. x_t
                noise_pred = noise_pred + guidance_weight * grad * (1 - alpha_bar_t).sqrt() / alpha_bar_t.sqrt()
            
            # DDIM update
            alpha_bar_prev = self.diffusion.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
            
            # Recompute x0_pred with updated noise_pred
            x0_pred = (x - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)
            
            sigma = self.ddim_eta * (
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            ).sqrt()
            
            pred_dir = (1 - alpha_bar_prev - sigma ** 2).sqrt() * noise_pred
            x_prev = alpha_bar_prev.sqrt() * x0_pred + pred_dir
            
            if i > 0:
                noise = torch.randn_like(x)
                x_prev = x_prev + sigma * noise
            
            x = x_prev
        
        return x
