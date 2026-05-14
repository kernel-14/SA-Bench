import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class Diffusion:
    """DDPM diffusion process with DDIM sampling support.
    
    Following Ho et al. 2020 and Song et al. 2020.
    """
    
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule='linear'):
        self.timesteps = timesteps
        
        if schedule == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        
        # For q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        
        # For q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
    
    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1-alpha_cumprod) * noise."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def training_loss(self, denoise_fn, x_start, t, cond=None, noise=None):
        """Compute simplified training loss: ||noise - denoise_fn(x_t, cond, t)||^2."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise=noise)
        predicted_noise = denoise_fn(x_t, t, cond=cond)
        
        return (noise - predicted_noise).pow(2).mean()
    
    @torch.no_grad()
    def ddim_sample(self, denoise_fn, shape, cond=None, sampling_steps=50, eta=1.0, noise=None,
                    guidance_fn=None, guidance_weight=0.0, guidance_schedule=None):
        """DDIM sampling (Song et al. 2020).
        
        Args:
            denoise_fn: model predicting noise epsilon_theta(x_t, cond, t)
            shape: shape of the output tensor
            cond: conditioning tensor
            sampling_steps: number of DDIM sampling steps
            eta: DDIM eta parameter (0 = deterministic, 1 = DDPM)
            noise: initial noise (if None, sampled from Gaussian)
            guidance_fn: optional function for guided sampling (returns gradient)
            guidance_weight: weight for guidance gradient
            guidance_schedule: optional schedule for guidance weight
        """
        total_steps = self.timesteps
        step_ratio = total_steps // sampling_steps
        timesteps = list(reversed(range(0, total_steps, step_ratio)))
        
        if noise is None:
            img = torch.randn(shape).to(self.betas.device)
        else:
            img = noise
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((shape[0],), t, device=img.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = denoise_fn(img, t_tensor, cond=cond)
            
            # Predict x_0 from x_t and noise
            sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
            alpha_cumprod = self.alphas_cumprod[t]
            
            while sqrt_alpha_cumprod.dim() < img.dim():
                sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
                sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
                alpha_cumprod = alpha_cumprod.unsqueeze(-1)
            
            pred_x0 = (img - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha_cumprod
            
            # Apply guidance if provided (for control tasks)
            if guidance_fn is not None:
                grad = guidance_fn(pred_x0, t)
                if guidance_schedule is not None:
                    w = guidance_schedule(t)
                else:
                    w = guidance_weight
                predicted_noise = predicted_noise + w * grad
            
            # Compute next timestep parameters
            if i < len(timesteps) - 1:
                next_t = timesteps[i + 1]
            else:
                next_t = -1
            
            if next_t >= 0:
                alpha_cumprod_next = self.alphas_cumprod[next_t]
                sigma = eta * torch.sqrt((1 - alpha_cumprod_next) / (1 - alpha_cumprod) * 
                                        (1 - alpha_cumprod / alpha_cumprod_next))
            else:
                alpha_cumprod_next = torch.tensor(1.0, device=img.device)
                sigma = 0.0
            
            while alpha_cumprod_next.dim() < img.dim():
                alpha_cumprod_next = alpha_cumprod_next.unsqueeze(-1)
                sigma = sigma.unsqueeze(-1) if isinstance(sigma, torch.Tensor) else sigma
            
            # Compute x_{t-1}
            direction = torch.sqrt(1 - alpha_cumprod_next - sigma**2) * predicted_noise
            img = torch.sqrt(alpha_cumprod_next) * pred_x0 + direction
            
            if eta > 0 and next_t >= 0:
                img = img + sigma * torch.randn_like(img)
        
        return img
