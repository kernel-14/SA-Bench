"""
Diffusion model components: Gaussian Diffusion (DDPM) and DDIM sampling.

Following the setup described in the paper:
- DDPM for training with simplified variational lower-bound loss
- DDIM for accelerated inference
- Classifier-free guidance for conditional generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Callable, Tuple
import math


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract values at timestep t from a 1D tensor a into shape x_shape."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear noise schedule from DDPM."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule as in improved DDPM."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.02)


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion model (DDPM).

    Implements the forward diffusion process and training loss.
    Uses a denoising model epsilon_theta that predicts the noise.

    Args:
        model: Denoising model (U-Net) that predicts noise
        timesteps: Number of diffusion steps K
        beta_schedule: Schedule type ('linear' or 'cosine')
        loss_type: Loss type ('l1' or 'l2')
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = 'linear',
        loss_type: str = 'l2',
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type

        # Define beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Compute alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # For DDIM
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0).

        Args:
            x_start: Clean data x_0
            t: Timestep indices
            noise: Optional pre-sampled noise

        Returns:
            Noisy sample x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, conditioning: Dict[str, torch.Tensor],
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Training loss: ||epsilon - epsilon_theta(x_t, t, conditioning)||^2.

        Following Eq. 2 in the paper.

        Args:
            x_start: Clean wavelet coefficients W_u^{(0)}
            t: Timestep
            conditioning: Dictionary of conditioning tensors (e.g., W_a)
            noise: Optional pre-sampled noise

        Returns:
            Loss value
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_t, t, **conditioning)

        if self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(predicted_noise, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    def forward(self, x_start: torch.Tensor, conditioning: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute training loss for a batch.

        Args:
            x_start: Clean wavelet coefficients
            conditioning: Dictionary of conditioning tensors

        Returns:
            Training loss
        """
        b = x_start.shape[0]
        t = torch.randint(1, self.timesteps, (b,), device=x_start.device)
        return self.p_losses(x_start, t, conditioning)


class DDIMSampler:
    """
    Denoising Diffusion Implicit Model (DDIM) sampler.

    Provides accelerated sampling as described in the paper (Song et al., 2020).
    Used during inference to speed up the sampling process.

    Args:
        diffusion: GaussianDiffusion model
        ddim_timesteps: Number of DDIM sampling steps (typically 50-100)
        ddim_eta: DDIM eta parameter (0 for deterministic, 1 for DDPM)
    """

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        ddim_timesteps: int = 50,
        ddim_eta: float = 1.0,
    ):
        self.diffusion = diffusion
        self.ddim_timesteps = ddim_timesteps
        self.ddim_eta = ddim_eta

        # Compute DDIM timestep sequence
        self.register_ddim_timesteps()

    def register_ddim_timesteps(self):
        """Compute DDIM timestep indices."""
        total_steps = self.diffusion.timesteps
        c = total_steps // self.ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, total_steps, c)))
        ddim_timestep_seq = ddim_timestep_seq + 1  # 1-indexed
        ddim_timestep_prev_seq = np.append(np.asarray([1]), ddim_timestep_seq[:-1])

        self.ddim_timesteps_tensor = torch.from_numpy(ddim_timestep_seq).long()
        self.ddim_timesteps_prev_tensor = torch.from_numpy(ddim_timestep_prev_seq).long()

    def ddim_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single DDIM step.

        Args:
            x: Current noisy sample x_t
            t: Current timestep index
            t_prev: Previous timestep index
            noise_pred: Predicted noise epsilon_theta(x_t, t)
            noise: Random noise (zero for last step)

        Returns:
            Denoised sample x_{t-1}
        """
        alpha_cumprod_t = extract(self.diffusion.alphas_cumprod, t, x.shape)
        alpha_cumprod_t_prev = extract(self.diffusion.alphas_cumprod, t_prev, x.shape)

        # Predict x_0
        pred_x0 = (x - torch.sqrt(1.0 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)

        # Direction pointing to x_t
        pred_dir = torch.sqrt(1.0 - alpha_cumprod_t_prev - self.ddim_eta ** 2 *
                              (1.0 - alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t) *
                              (1.0 - alpha_cumprod_t / alpha_cumprod_t_prev)) * noise_pred

        # Random noise component
        if self.ddim_eta > 0:
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir + self.ddim_eta * torch.sqrt(
                (1.0 - alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t) *
                (1.0 - alpha_cumprod_t / alpha_cumprod_t_prev)
            ) * noise
        else:
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir

        return x_prev

    def sample(
        self,
        shape: Tuple[int, ...],
        conditioning: Dict[str, torch.Tensor],
        device: torch.device,
        progress: bool = False,
        guidance_fn: Optional[Callable] = None,
        guidance_weight: float = 0.0,
    ) -> torch.Tensor:
        """
        Sample using DDIM.

        Args:
            shape: Shape of the output sample (B, C, H, W)
            conditioning: Dictionary of conditioning tensors
            device: Torch device
            progress: Whether to show progress bar
            guidance_fn: Optional guidance function for control tasks
            guidance_weight: Weight lambda for guidance

        Returns:
            Generated sample W_u^{(0)}
        """
        b = shape[0]
        model = self.diffusion.model

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Iterate over DDIM timesteps in reverse
        for i in range(self.ddim_timesteps - 1, -1, -1):
            t = torch.full((b,), self.ddim_timesteps_tensor[i].item(), device=device, dtype=torch.long)
            t_prev = torch.full((b,), self.ddim_timesteps_prev_tensor[i].item(), device=device, dtype=torch.long)

            # Predict noise
            noise_pred = model(x, t, **conditioning)

            # Apply classifier-free guidance if conditioning has a null variant
            if 'null_cond' in conditioning:
                null_pred = model(x, t, **conditioning['null_cond'])
                noise_pred = null_pred + guidance_weight * (noise_pred - null_pred)

            # Apply control guidance if provided
            if guidance_fn is not None:
                with torch.enable_grad():
                    x_grad = x.detach().requires_grad_(True)
                    noise_pred_grad = model(x_grad, t, **conditioning)

                    # Predict clean x_0 for guidance computation
                    alpha_cumprod_t = extract(self.diffusion.alphas_cumprod, t, x_grad.shape)
                    pred_x0 = (x_grad - torch.sqrt(1.0 - alpha_cumprod_t) * noise_pred_grad) / torch.sqrt(alpha_cumprod_t)

                    guidance_loss = guidance_fn(pred_x0)
                    grad = torch.autograd.grad(guidance_loss, x_grad)[0]

                # Update: x = x - eta * (epsilon_theta + lambda * grad_I) + xi
                # The scaling eta is implicitly handled through the DDIM step
                noise_pred = noise_pred + guidance_weight * grad

            # Sample noise for this step
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)

            # DDIM step
            x = self.ddim_step(x, t, t_prev, noise_pred, noise)

        return x
