"""
Hi-MAR modules:
  - ScaleAwareTransformerBlock  (Figure 2c)
  - MLPDiffusionHead            (Figure 2d, phase-1 head)
  - DiffusionTransformerHead    (Figure 2e, phase-2 head)
  - GaussianDiffusion           (noise schedule + loss helpers)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import (
    AdaLNZero,
    FeedForward,
    MultiHeadSelfAttention,
    RMSNorm,
    sinusoidal_embedding,
)


# ---------------------------------------------------------------------------
# Scale-aware Transformer Block  (Eq. 2 in the paper)
# ---------------------------------------------------------------------------

class ScaleAwareTransformerBlock(nn.Module):
    """
    Single Transformer block conditioned on a scale vector via AdaLN-Zero.

    The scale vector v is produced externally (sinusoidal embedding → MLP) and
    passed in.  The block applies a learned affine transform to v before
    splitting into (α1, β1, γ1, α2, β2, γ2).

    Equations (paper §3.2):
        ṽ = a·v + b
        α1,β1,γ1,α2,β2,γ2 = split(ṽ)
        z_a = z + γ1 · Attention(α1 · LN(z) + β1)
        z'  = z_a + γ2 · FFN(α2 · LN(z_a) + β2)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        scale_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, mlp_ratio, dropout)

        # a·v + b  (learnable linear on the scale vector)
        self.scale_proj = nn.Linear(scale_dim, scale_dim, bias=True)
        # split into 6 modulation parameters
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(scale_dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        z: torch.Tensor,
        scale_vec: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        v_tilde = self.scale_proj(scale_vec)
        mods = self.adaLN(v_tilde)
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = mods.chunk(6, dim=-1)

        # Attention sub-layer
        z_normed = self.norm1(z)
        z_a = z + gamma1 * self.attn(alpha1 * z_normed + beta1, attn_mask)

        # FFN sub-layer
        z_normed2 = self.norm2(z_a)
        z_out = z_a + gamma2 * self.ffn(alpha2 * z_normed2 + beta2)
        return z_out


# ---------------------------------------------------------------------------
# MLP-based Diffusion Head  (Figure 2d, used in phase 1)
# ---------------------------------------------------------------------------

class MLPDiffusionBlock(nn.Module):
    """Single block of the MLP diffusion head (adaLN + LN + FFN)."""

    def __init__(self, dim: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = FeedForward(dim, mlp_ratio)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        alpha, beta, gamma = self.adaLN(cond).chunk(3, dim=-1)
        return x + gamma * self.ffn(alpha * self.norm(x) + beta)


class MLPDiffusionHead(nn.Module):
    """
    Per-token MLP diffusion head (phase 1).

    For each masked token position i, takes the noisy token x_i^t and the
    conditional token z_i from the backbone, and predicts the noise ε.

    The conditioning is: c_i = time_emb(t) + proj(z_i)
    """

    def __init__(
        self,
        token_dim: int,
        cond_dim: int,
        hidden_dim: int,
        num_layers: int,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.input_proj = nn.Linear(token_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [MLPDiffusionBlock(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.out_proj = nn.Linear(hidden_dim, token_dim)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_noisy: (B, N, token_dim) noisy tokens
            t:       (B,) integer timesteps
            cond:    (B, N, cond_dim) conditional tokens from backbone
        Returns:
            noise_pred: (B, N, token_dim)
        """
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)  # (B, time_emb_dim)
        t_emb = self.time_mlp(t_emb)                        # (B, hidden_dim)

        c = t_emb.unsqueeze(1) + self.cond_proj(cond)       # (B, N, hidden_dim)
        h = self.input_proj(x_noisy)                         # (B, N, hidden_dim)

        for block in self.blocks:
            h = block(h, c)

        return self.out_proj(self.norm_out(h))


# ---------------------------------------------------------------------------
# Diffusion Transformer Head  (Figure 2e, used in phase 2)
# ---------------------------------------------------------------------------

class DiffusionTransformerBlock(nn.Module):
    """
    Single block of the Diffusion Transformer head (adaLN + LN + Attention + FFN).

    Equations (paper §3.3):
        α1,β1,γ1,α2,β2,γ2 = split(c)
        y_a = y + γ1 · Attention(α1 · LN(y) + β1)
        y'  = y_a + γ2 · FFN(α2 · LN(y_a) + β2)

    where c = time_step_embedding + conditional_tokens (summed).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, mlp_ratio, dropout)

        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (B, N, dim) token features
            c: (B, N, cond_dim) per-token context = time_emb + cond_token
        """
        mods = self.adaLN(c)
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = mods.chunk(6, dim=-1)

        y_normed = self.norm1(y)
        y_a = y + gamma1 * self.attn(alpha1 * y_normed + beta1)

        y_normed2 = self.norm2(y_a)
        y_out = y_a + gamma2 * self.ffn(alpha2 * y_normed2 + beta2)
        return y_out


class DiffusionTransformerHead(nn.Module):
    """
    Diffusion Transformer head (phase 2).

    Processes ALL tokens (masked + unmasked) jointly via self-attention.
    Context c per token = time_emb(t) + cond_token_i.

    The loss is computed only on masked positions.
    """

    def __init__(
        self,
        token_dim: int,
        cond_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        time_emb_dim: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.input_proj = nn.Linear(token_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_dim, num_heads, hidden_dim, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.out_proj = nn.Linear(hidden_dim, token_dim)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_noisy: (B, N, token_dim) noisy tokens for all positions
            t:       (B,) integer timesteps
            cond:    (B, N, cond_dim) conditional tokens from backbone
        Returns:
            noise_pred: (B, N, token_dim)
        """
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)  # (B, time_emb_dim)
        t_emb = self.time_mlp(t_emb)                        # (B, hidden_dim)

        # c_i = time_emb + cond_i  (broadcast time_emb over sequence)
        c = t_emb.unsqueeze(1) + self.cond_proj(cond)       # (B, N, hidden_dim)
        h = self.input_proj(x_noisy)                         # (B, N, hidden_dim)

        for block in self.blocks:
            h = block(h, c)

        return self.out_proj(self.norm_out(h))


# ---------------------------------------------------------------------------
# Gaussian Diffusion (noise schedule + forward/reverse helpers)
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    """
    DDPM-style Gaussian diffusion with cosine noise schedule.

    Used by both the MLP and Transformer diffusion heads.
    """

    def __init__(self, num_timesteps: int = 1000, beta_schedule: str = "cosine"):
        super().__init__()
        self.num_timesteps = num_timesteps

        if beta_schedule == "cosine":
            betas = self._cosine_betas(num_timesteps)
        elif beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("sqrt_recip_alphas_cumprod", (1.0 / alphas_cumprod).sqrt())
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", (1.0 / alphas_cumprod - 1.0).sqrt()
        )

        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod),
        )

    @staticmethod
    def _cosine_betas(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps) / num_timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return betas.clamp(0, 0.999)

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def diffusion_loss(
        self,
        noise_pred: torch.Tensor,
        x_start: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MSE loss between predicted and actual noise, optionally masked.

        Args:
            noise_pred: (B, N, D) predicted noise
            x_start:    (B, N, D) clean tokens
            t:          (B,) timesteps
            mask:       (B, N) bool, True = masked (compute loss here)
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        loss = F.mse_loss(noise_pred, noise, reduction="none")  # (B, N, D)
        if mask is not None:
            loss = loss[mask]
        return loss.mean()

    def p_mean_variance(
        self,
        noise_pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior mean and variance for one denoising step."""
        x_recon = self._predict_xstart_from_eps(x_t, t, noise_pred)
        x_recon = x_recon.clamp(-10, 10)

        model_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_recon
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        model_log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return model_mean, model_log_var

    def _predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    @torch.no_grad()
    def ddim_sample(
        self,
        model_fn,
        shape: Tuple,
        cond: torch.Tensor,
        num_steps: int = 100,
        eta: float = 0.0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        DDIM sampling for a diffusion head.

        Args:
            model_fn: callable(x_noisy, t, cond) -> noise_pred
            shape:    output shape (B, N, D)
            cond:     conditioning tensor passed to model_fn
            num_steps: number of DDIM steps
            eta:      stochasticity (0 = deterministic DDIM)
        """
        if device is None:
            device = cond.device

        # Evenly spaced timesteps
        step_size = self.num_timesteps // num_steps
        timesteps = list(reversed(range(0, self.num_timesteps, step_size)))[:num_steps]

        x = torch.randn(shape, device=device)

        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            noise_pred = model_fn(x, t_batch, cond)

            alpha_bar = self.alphas_cumprod[t_val]
            alpha_bar_prev = (
                self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)
            )

            x0_pred = (x - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
            x0_pred = x0_pred.clamp(-10, 10)

            sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)).sqrt()
            dir_xt = (1 - alpha_bar_prev - sigma ** 2).clamp(min=0).sqrt() * noise_pred
            noise = torch.randn_like(x) if eta > 0 else 0.0

            x = alpha_bar_prev.sqrt() * x0_pred + dir_xt + sigma * noise

        return x

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, shape: Tuple) -> torch.Tensor:
        """Extract values from a 1-D tensor at indices t and reshape to match shape."""
        out = a.gather(0, t)
        while out.ndim < len(shape):
            out = out.unsqueeze(-1)
        return out.expand(shape)
