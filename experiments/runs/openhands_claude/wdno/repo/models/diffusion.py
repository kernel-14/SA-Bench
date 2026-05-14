"""
DDPM and DDIM diffusion model for WDNO.

Implements:
  - DDPM forward process (noise schedule, noisy sample generation)
  - DDPM training loss (simplified ELBO, noise prediction)
  - DDIM reverse process (accelerated sampling with η parameter)
  - Classifier-free guidance (CFG) for simulation conditioning
  - Classifier-based guidance for control tasks

Key equations from the paper:
  Training loss:
    L = E[||ε - ε_θ(√ᾱ_k x_0 + √(1-ᾱ_k) ε, k)||²]

  DDIM sampling (simulation):
    W^(k-1) = W^(k) - η ε_θ(W^(k), W_a, k) + ξ

  DDIM sampling (control):
    W^(k-1) = W^(k) - η (ε_θ(W^(k), W_a, k) + λ ∇_W I(Ŵ^(k))) + ξ

  Denoised estimate:
    Ŵ^(k) = (W^(k) - √(1-ᾱ_k) ε_θ(W^(k), W_a, k)) / √ᾱ_k
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """Extract values from a 1D tensor at indices t, reshape to broadcast with x_shape."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()).to(t.device)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# ---------------------------------------------------------------------------
# Guidance schedule (cosine)
# ---------------------------------------------------------------------------

def cosine_guidance_schedule(k: int, K: int, lambda_max: float) -> float:
    """
    Cosine schedule for guidance weight λ over denoising steps.
    λ is larger at early (high-noise) steps and smaller at later steps.
    """
    progress = k / K  # 1 at start (k=K), 0 at end (k=0)
    return lambda_max * 0.5 * (1 + math.cos(math.pi * (1 - progress)))


# ---------------------------------------------------------------------------
# DDPM / DDIM
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    """
    DDPM with DDIM sampling.

    Args:
        model: noise prediction network ε_θ
        timesteps: number of diffusion steps K (default 1000)
        schedule: noise schedule type ('linear' or 'cosine')
        ddim_sampling_eta: η for DDIM (1.0 = DDPM-like, 0.0 = deterministic)
        p_uncond: probability of dropping condition during training (classifier-free guidance)
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        schedule: str = "linear",
        ddim_sampling_eta: float = 1.0,
        p_uncond: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.p_uncond = p_uncond

        if schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior variance q(x_{k-1} | x_k, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample x_k from q(x_k | x_0) = N(√ᾱ_k x_0, (1-ᾱ_k) I)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute simplified ELBO training loss.

        L = E[||ε - ε_θ(√ᾱ_k x_0 + √(1-ᾱ_k) ε, k)||²]

        Classifier-free guidance: randomly drop condition with probability p_uncond.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)

        # Classifier-free guidance: randomly drop condition
        if cond is not None and self.training:
            mask = torch.rand(x_start.shape[0], device=x_start.device) < self.p_uncond
            # Zero out condition for dropped samples
            cond_input = cond.clone()
            cond_input[mask] = 0.0
        else:
            cond_input = cond

        predicted_noise = self.model(x_noisy, t, cond_input)
        return F.mse_loss(noise, predicted_noise)

    def forward(
        self,
        x_start: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample random timestep and compute training loss."""
        b = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x_start.device).long()
        return self.p_losses(x_start, t, cond)

    # ------------------------------------------------------------------
    # Denoised estimate x̂_0 from x_k
    # ------------------------------------------------------------------

    def predict_start_from_noise(self, x_k: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Ŵ^(k) = (W^(k) - √(1-ᾱ_k) ε_θ) / √ᾱ_k"""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_k.shape) * x_k
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_k.shape) * noise
        )

    # ------------------------------------------------------------------
    # DDIM sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: Tuple,
        cond: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        guidance_weight: float = 0.0,
        guidance_schedule: Optional[Callable] = None,
        cfg_weight: float = 1.0,
        ddim_steps: int = 50,
        eta: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        DDIM sampling with optional classifier-free and classifier-based guidance.

        Args:
            shape: output shape [batch, C, ...]
            cond: conditioning signal (wavelet coefficients of condition)
            guidance_fn: callable(x_hat_0) → scalar loss I for control guidance
            guidance_weight: λ for control guidance
            guidance_schedule: callable(k, K) → λ_k (cosine schedule)
            cfg_weight: ω for classifier-free guidance (1.0 = no extra boost)
            ddim_steps: number of DDIM denoising steps
            eta: DDIM η parameter (None uses self.ddim_sampling_eta)
            device: target device
        Returns:
            x_0: denoised sample [batch, C, ...]
        """
        if eta is None:
            eta = self.ddim_sampling_eta
        if device is None:
            device = next(self.model.parameters()).device

        # Build DDIM timestep sequence (evenly spaced)
        times = torch.linspace(-1, self.timesteps - 1, steps=ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x = torch.randn(shape, device=device)

        for time, time_next in time_pairs:
            t_batch = torch.full((shape[0],), time, device=device, dtype=torch.long)

            # Predict noise (with classifier-free guidance)
            if cond is not None and cfg_weight != 1.0:
                # Combined CFG: ε_uncond + ω*(ε_cond - ε_uncond)
                eps_uncond = self.model(x, t_batch, None)
                eps_cond = self.model(x, t_batch, cond)
                eps = eps_uncond + cfg_weight * (eps_cond - eps_uncond)
            else:
                eps = self.model(x, t_batch, cond)

            # Classifier-based guidance for control
            if guidance_fn is not None and guidance_weight > 0:
                # Compute denoised estimate Ŵ^(k)
                x_hat = self.predict_start_from_noise(x, t_batch, eps)

                # Get guidance weight for this step
                if guidance_schedule is not None:
                    lam = guidance_schedule(time, self.timesteps)
                else:
                    lam = guidance_weight

                if lam > 0:
                    x_hat_grad = x_hat.detach().requires_grad_(True)
                    loss_val = guidance_fn(x_hat_grad)
                    grad = torch.autograd.grad(loss_val, x_hat_grad)[0]
                    eps = eps + lam * grad.detach()

            # DDIM update
            alpha_bar = self.alphas_cumprod[time]
            alpha_bar_next = self.alphas_cumprod[time_next] if time_next >= 0 else torch.tensor(1.0)

            # Predicted x_0
            x_0_pred = (x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
            x_0_pred = x_0_pred.clamp(-1, 1)

            # Direction pointing to x_t
            dir_xt = (1 - alpha_bar_next - eta ** 2 * (1 - alpha_bar_next) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_next)).sqrt() * eps

            # Noise term
            noise = torch.randn_like(x) if eta > 0 and time_next >= 0 else torch.zeros_like(x)
            sigma = eta * ((1 - alpha_bar_next) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_next)).sqrt()

            x = alpha_bar_next.sqrt() * x_0_pred + dir_xt + sigma * noise

        return x

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple,
        cond: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        guidance_weight: float = 0.0,
        guidance_schedule: Optional[Callable] = None,
        cfg_weight: float = 1.0,
        ddim_steps: int = 50,
        eta: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Wrapper around ddim_sample."""
        return self.ddim_sample(
            shape=shape,
            cond=cond,
            guidance_fn=guidance_fn,
            guidance_weight=guidance_weight,
            guidance_schedule=guidance_schedule,
            cfg_weight=cfg_weight,
            ddim_steps=ddim_steps,
            eta=eta,
            device=device,
        )
