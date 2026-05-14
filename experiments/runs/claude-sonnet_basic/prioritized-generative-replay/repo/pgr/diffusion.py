"""
Conditional Diffusion Model for Prioritized Generative Replay (PGR).

Implements a conditional DDPM with classifier-free guidance (CFG) for
generating RL transitions. Architecture follows SynthER (Lu et al., 2024)
with an additional conditioning pathway for the relevance function F(tau).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, time_emb_dim: int, cond_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim))
        self.cond_proj = (
            nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim)) if cond_dim > 0 else None
        )

    def forward(self, x, time_emb, cond_emb=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.fc1(h)
        h = h + self.time_proj(time_emb)
        if cond_emb is not None and self.cond_proj is not None:
            h = h + self.cond_proj(cond_emb)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class TransitionDenoiser(nn.Module):
    """
    Residual MLP denoising network for transition tuples (s, a, s', r).
    Accepts an optional scalar condition c = F(tau) for classifier-free guidance.
    """

    def __init__(
        self,
        transition_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        time_emb_dim: int = 128,
        cond_emb_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.transition_dim = transition_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(1, cond_emb_dim),
            nn.SiLU(),
            nn.Linear(cond_emb_dim, cond_emb_dim),
        )
        self.null_cond_emb = nn.Parameter(torch.zeros(cond_emb_dim))

        self.input_proj = nn.Linear(transition_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_emb_dim, cond_emb_dim, dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, transition_dim),
        )

    def forward(self, x, t, cond=None, use_null_cond=False):
        time_emb = self.time_mlp(t)
        if use_null_cond or cond is None:
            cond_emb = self.null_cond_emb.unsqueeze(0).expand(x.shape[0], -1)
        else:
            cond_emb = self.cond_mlp(cond)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, time_emb, cond_emb)
        return self.output_proj(h)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


class ConditionalDiffusion(nn.Module):
    """
    Conditional DDPM with classifier-free guidance for RL transition generation.

    Training objective (Eq. 2 in paper):
        E_{x0~D, eps~N(0,I), y, n~Unif(1,N), p~Bernoulli(p_uncond)}
            || eps_theta(x^n, n, (1-p)*y + p*null) ||_2^2

    Sampling uses CFG:
        eps = omega * eps_theta(x^n, n, y) + (1-omega) * eps_theta(x^n, n, null)
    """

    def __init__(
        self,
        transition_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_timesteps: int = 100,
        time_emb_dim: int = 128,
        cond_emb_dim: int = 64,
        p_uncond: float = 0.25,
        guidance_scale: float = 1.2,
        beta_schedule: str = "cosine",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.transition_dim = transition_dim
        self.n_timesteps = n_timesteps
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale

        self.model = TransitionDenoiser(
            transition_dim=transition_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            time_emb_dim=time_emb_dim,
            cond_emb_dim=cond_emb_dim,
            dropout=dropout,
        )

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        else:
            betas = linear_beta_schedule(n_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def compute_loss(self, x, cond=None):
        """
        Training loss with classifier-free guidance dropout (Eq. 2).
        x: [B, transition_dim], cond: [B, 1] or None
        """
        B = x.shape[0]
        t = torch.randint(0, self.n_timesteps, (B,), device=x.device)
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)

        if cond is not None:
            drop_mask = torch.bernoulli(
                torch.full((B,), self.p_uncond, device=x.device)
            ).bool()
            pred_noise = self._forward_with_dropout(x_noisy, t, cond, drop_mask)
        else:
            pred_noise = self.model(x_noisy, t, cond=None, use_null_cond=True)

        return F.mse_loss(pred_noise, noise)

    def _forward_with_dropout(self, x, t, cond, drop_mask):
        B = x.shape[0]
        pred = torch.zeros(B, self.transition_dim, device=x.device)
        cond_idx = (~drop_mask).nonzero(as_tuple=True)[0]
        uncond_idx = drop_mask.nonzero(as_tuple=True)[0]
        if len(cond_idx) > 0:
            pred[cond_idx] = self.model(x[cond_idx], t[cond_idx], cond=cond[cond_idx])
        if len(uncond_idx) > 0:
            pred[uncond_idx] = self.model(x[uncond_idx], t[uncond_idx], use_null_cond=True)
        return pred

    @torch.no_grad()
    def p_mean_variance(self, x, t, cond=None, guidance_scale=None):
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        if cond is not None and guidance_scale != 1.0:
            eps_cond = self.model(x, t, cond=cond)
            eps_uncond = self.model(x, t, use_null_cond=True)
            eps = guidance_scale * eps_cond + (1 - guidance_scale) * eps_uncond
        elif cond is not None:
            eps = self.model(x, t, cond=cond)
        else:
            eps = self.model(x, t, use_null_cond=True)

        sqrt_recip = self.sqrt_recip_alphas_cumprod[t][:, None]
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t][:, None]
        x_recon = (sqrt_recip * x - sqrt_recipm1 * eps).clamp(-1.0, 1.0)

        coef1 = self.posterior_mean_coef1[t][:, None]
        coef2 = self.posterior_mean_coef2[t][:, None]
        model_mean = coef1 * x_recon + coef2 * x
        model_log_variance = self.posterior_log_variance_clipped[t][:, None]
        return model_mean, model_log_variance

    @torch.no_grad()
    def p_sample(self, x, t_int, cond=None, guidance_scale=None):
        B = x.shape[0]
        t = torch.full((B,), t_int, device=x.device, dtype=torch.long)
        model_mean, model_log_variance = self.p_mean_variance(x, t, cond=cond, guidance_scale=guidance_scale)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float()[:, None]
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def sample(self, n_samples, cond=None, guidance_scale=None, device=None):
        """
        Generate n_samples transitions using reverse diffusion with CFG.
        cond: [n_samples, 1] relevance conditioning values, or None for unconditional.
        """
        if device is None:
            device = next(self.parameters()).device
        x = torch.randn(n_samples, self.transition_dim, device=device)
        for t in reversed(range(self.n_timesteps)):
            x = self.p_sample(x, t, cond=cond, guidance_scale=guidance_scale)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
