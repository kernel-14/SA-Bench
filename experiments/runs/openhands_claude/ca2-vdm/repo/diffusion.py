from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise Schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule (Ho et al., 2020)."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule (Nichol & Dhariwal, 2021).
    Improved DDPM uses this schedule at inference.
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def get_named_beta_schedule(schedule_name: str, num_timesteps: int, **kwargs) -> torch.Tensor:
    if schedule_name == "linear":
        return linear_beta_schedule(num_timesteps, **kwargs)
    elif schedule_name == "cosine":
        return cosine_beta_schedule(num_timesteps, **kwargs)
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")


# ---------------------------------------------------------------------------
# Gaussian Diffusion (DDPM + Improved DDPM)
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    """
    Gaussian diffusion process implementing DDPM (Ho et al., 2020) and
    Improved DDPM (Nichol & Dhariwal, 2021).

    Training schedule: linear with T=1000, beta_1=1e-4, beta_T=0.02
    Inference schedule: improved DDPM cosine with 100 steps
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
        predict_xstart: bool = False,
        learn_variance: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.predict_xstart = predict_xstart
        self.learn_variance = learn_variance

        # Compute noise schedule
        betas = get_named_beta_schedule(schedule, num_train_timesteps,
                                         beta_start=beta_start, beta_end=beta_end)
        self.register_schedule(betas)

    def register_schedule(self, betas: torch.Tensor) -> None:
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.alphas_cumprod_next = F.pad(self.alphas_cumprod[1:], (0, 1), value=0.0)

        # Forward process quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape: Tuple) -> torch.Tensor:
        """Extract values from arr at timesteps t and reshape to match x."""
        device = t.device
        arr = arr.to(device)
        out = arr[t]
        while out.dim() < len(shape):
            out = out.unsqueeze(-1)
        return out.expand(shape)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior q(x_{t-1} | x_t, x_0) mean and variance."""
        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, variance, log_variance

    def predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Recover x_0 from predicted noise eps."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_mean_variance(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute p_theta(x_{t-1} | x_t) mean and variance.

        model_output: (B, 2*C, ...) — predicted noise and log-variance (if learn_variance)
        """
        C = x_t.shape[1]
        if self.learn_variance:
            eps_pred, log_var_pred = model_output[:, :C], model_output[:, C:]
        else:
            eps_pred = model_output
            log_var_pred = None

        x_start_pred = self.predict_xstart_from_eps(x_t, t, eps_pred)
        x_start_pred = torch.clamp(x_start_pred, -1.0, 1.0)

        mean, _, posterior_log_var = self.q_posterior_mean_variance(x_start_pred, x_t, t)

        if self.learn_variance:
            # Interpolate between lower and upper bounds of variance
            min_log = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
            max_log = self._extract(torch.log(self.betas), t, x_t.shape)
            frac = (log_var_pred + 1) / 2  # map from [-1, 1] to [0, 1]
            model_log_var = frac * max_log + (1 - frac) * min_log
        else:
            model_log_var = posterior_log_var

        return {
            "mean": mean,
            "log_variance": model_log_var,
            "x_start_pred": x_start_pred,
            "eps_pred": eps_pred,
        }

    def p_sample(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """One step of reverse diffusion: sample x_{t-1} ~ p_theta(x_{t-1} | x_t)."""
        out = self.p_mean_variance(model_output, x_t, t)
        noise = torch.randn_like(x_t)
        # No noise at t=0
        nonzero_mask = (t != 0).float().reshape(-1, *([1] * (x_t.dim() - 1)))
        return out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

    def training_losses(
        self,
        model_output: torch.Tensor,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined training loss: L_simple + L_vlb (Eq. 2 in paper).

        Args:
            model_output: (B, L, 2*C, H, W) — model predictions
            x_start: (B, L, C, H, W) — clean latents
            x_t: (B, L, C, H, W) — noisy latents
            t: (B,) — timestep
            noise: (B, L, C, H, W) — ground truth noise
            loss_mask: (B, L, 1, 1, 1) — 1 for denoising target, 0 for prefix
        Returns:
            losses: dict with "simple", "vlb", "total"
        """
        B, L, C, H, W = x_start.shape
        C_out = model_output.shape[2]

        eps_pred = model_output[:, :, :C]      # (B, L, C, H, W)
        log_var_pred = model_output[:, :, C:]  # (B, L, C, H, W)

        # L_simple: MSE between predicted and true noise (Eq. 2)
        simple_loss = F.mse_loss(eps_pred, noise, reduction="none")  # (B, L, C, H, W)

        # L_vlb: KL divergence term
        # Compute per-frame VLB loss
        vlb_loss = self._vlb_loss(
            x_start=x_start.reshape(B * L, C, H, W),
            x_t=x_t.reshape(B * L, C, H, W),
            t=t.unsqueeze(1).expand(-1, L).reshape(B * L),
            eps_pred=eps_pred.reshape(B * L, C, H, W),
            log_var_pred=log_var_pred.reshape(B * L, C, H, W),
        ).reshape(B, L, C, H, W)

        # Apply loss mask (exclude clean prefix)
        if loss_mask is not None:
            simple_loss = simple_loss * loss_mask
            vlb_loss = vlb_loss * loss_mask

        return {
            "simple": simple_loss.mean(),
            "vlb": vlb_loss.mean(),
            "total": simple_loss.mean() + vlb_loss.mean(),
        }

    def _vlb_loss(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps_pred: torch.Tensor,
        log_var_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute VLB loss term for learned variance (Nichol & Dhariwal, 2021).
        Uses the KL divergence between q(x_{t-1}|x_t,x_0) and p_theta(x_{t-1}|x_t).
        """
        # True posterior
        x_start_pred = self.predict_xstart_from_eps(x_t, t, eps_pred)
        x_start_pred = torch.clamp(x_start_pred, -1.0, 1.0)
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_start_pred, x_t, t)

        # Model posterior
        min_log = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        max_log = self._extract(torch.log(self.betas), t, x_t.shape)
        frac = (log_var_pred + 1) / 2
        model_log_var = frac * max_log + (1 - frac) * min_log
        model_mean = true_mean  # mean is determined by eps_pred

        # KL divergence between two Gaussians
        kl = 0.5 * (
            -1.0
            + model_log_var - true_log_var
            + torch.exp(true_log_var - model_log_var)
            + ((true_mean - model_mean) ** 2) * torch.exp(-model_log_var)
        )
        return kl / math.log(2.0)  # convert to bits


# ---------------------------------------------------------------------------
# DDIM Sampler (for faster inference)
# ---------------------------------------------------------------------------

class DDIMSampler:
    """
    DDIM sampler (Song et al., 2021a) for deterministic inference.
    Used as an alternative to DDPM sampling.
    """

    def __init__(self, diffusion: GaussianDiffusion, num_inference_steps: int = 100):
        self.diffusion = diffusion
        self.num_inference_steps = num_inference_steps
        self._set_timesteps()

    def _set_timesteps(self) -> None:
        T = self.diffusion.num_train_timesteps
        step_ratio = T // self.num_inference_steps
        self.timesteps = torch.arange(0, T, step_ratio).flip(0)

    def step(
        self,
        model_output: torch.Tensor,
        t: int,
        x_t: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        One DDIM step.

        Args:
            model_output: (B, C, ...) — predicted noise
            t: current timestep
            x_t: (B, C, ...) — current noisy sample
            eta: stochasticity (0 = deterministic DDIM)
        Returns:
            x_{t-1}: (B, C, ...)
        """
        t_tensor = torch.tensor([t], device=x_t.device)
        alpha_bar_t = self.diffusion.alphas_cumprod[t]
        alpha_bar_prev = self.diffusion.alphas_cumprod_prev[t]

        # Predict x_0
        x_start = self.diffusion.predict_xstart_from_eps(x_t, t_tensor.expand(x_t.shape[0]), model_output)
        x_start = torch.clamp(x_start, -1.0, 1.0)

        # Direction pointing to x_t
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)

        x_prev = (
            torch.sqrt(alpha_bar_prev) * x_start
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * model_output
            + sigma * noise
        )
        return x_prev


# ---------------------------------------------------------------------------
# Improved DDPM Sampler (used in paper: 100 inference steps)
# ---------------------------------------------------------------------------

class ImprovedDDPMSampler:
    """
    Improved DDPM sampler (Nichol & Dhariwal, 2021) with cosine schedule
    and 100 inference steps. This is the sampler used in the paper.
    """

    def __init__(self, num_train_timesteps: int = 1000, num_inference_steps: int = 100):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # Cosine schedule for inference
        betas = cosine_beta_schedule(num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.betas = betas

        # Posterior variance
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self._set_timesteps()

    def _set_timesteps(self) -> None:
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        self.timesteps = list(range(0, self.num_train_timesteps, step_ratio))[::-1]

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape: Tuple) -> torch.Tensor:
        device = t.device
        arr = arr.to(device)
        out = arr[t]
        while out.dim() < len(shape):
            out = out.unsqueeze(-1)
        return out.expand(shape)

    def predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def step(
        self,
        model_output: torch.Tensor,
        t: int,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        One improved DDPM step.

        Args:
            model_output: (B, 2*C, ...) — predicted noise and log-variance
            t: current timestep (integer)
            x_t: (B, C, ...) — current noisy sample
        Returns:
            x_{t-1}: (B, C, ...)
        """
        B = x_t.shape[0]
        C = x_t.shape[1]
        t_tensor = torch.tensor([t] * B, device=x_t.device)

        eps_pred = model_output[:, :C]
        log_var_pred = model_output[:, C:]

        x_start = self.predict_xstart_from_eps(x_t, t_tensor, eps_pred)
        x_start = torch.clamp(x_start, -1.0, 1.0)

        # Posterior mean
        mean = (
            self._extract(self.posterior_mean_coef1, t_tensor, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t_tensor, x_t.shape) * x_t
        )

        # Learned variance
        min_log = self._extract(self.posterior_log_variance_clipped, t_tensor, x_t.shape)
        max_log = self._extract(torch.log(self.betas), t_tensor, x_t.shape)
        frac = (log_var_pred + 1) / 2
        model_log_var = frac * max_log + (1 - frac) * min_log

        noise = torch.randn_like(x_t)
        nonzero_mask = (t_tensor != 0).float().reshape(-1, *([1] * (x_t.dim() - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * model_log_var) * noise

    def add_noise(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self._extract(torch.sqrt(self.alphas_cumprod), t, x_start.shape)
        sqrt_one_minus = self._extract(torch.sqrt(1.0 - self.alphas_cumprod), t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise
