## diffusion.py
"""
Denoising Diffusion Probabilistic Model (DDPM) and fast DDIM sampling
component for the Wavelet Diffusion Neural Operator (WDNO).

Implements:
    - Cosine and linear noise schedules
    - Forward diffusion step ``add_noise``
    - Conditional noise prediction with classifier‑free guidance ``denoise``
    - DDIM step (private)
    - A complete DDIM sampling loop ``sample_ddim``

All methods operate on wavelet coefficient tensors provided by
``wavelet_utils.WaveletTransform``.  The module is agnostic to whether the
denoiser is a 2D or 3D U‑Net.

References:
    Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
    Song et al. (2020) "Denoising Diffusion Implicit Models"
    Nichol & Dhariwal (2021) "Improved Denoising Diffusion Probabilistic Models"
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule as proposed in "Improved DDPM".

    Args:
        timesteps: total number of diffusion steps.
        s:          small offset to prevent divide‑by‑zero at the start.

    Returns:
        tensor of shape (timesteps,) with beta values.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos((t / timesteps + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """
    Classic linear noise schedule from the original DDPM paper.

    Args:
        timesteps:   total number of diffusion steps.
        beta_start:  small noise level at t=1.
        beta_end:    noise level at t=T.

    Returns:
        tensor of shape (timesteps,) with beta values.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


class DDPM:
    """
    DDPM/DDIM diffusion process operating entirely in the wavelet domain.

    Example usage:
        model = UNet2D(...)
        ddpm = DDPM(model, n_timesteps=1000, schedule='cosine', device='cuda')

        # Training step
        xt = ddpm.add_noise(x0, noise, t)
        noise_pred = ddpm.denoise(xt, t, cond=W_cond, guidance_w=0.0)
        loss = F.mse_loss(noise_pred, noise)

        # Conditional generation (simulation)
        x_generated = ddpm.sample_ddim(cond=W_cond, guidance_w=2.0)
    """

    def __init__(
        self,
        denoiser: Union["UNet2D", "UNet3D"],
        n_timesteps: int = 1000,
        schedule: str = "cosine",
        device: str = "cpu",
    ) -> None:
        """
        Args:
            denoiser:      trained or untrained U‑Net that predicts noise.
                           Must accept (x, t, cond) where cond may be ``None``.
            n_timesteps:   total diffusion steps (from config `diffusion.num_timesteps`).
            schedule:      noise schedule type, either ``"cosine"`` or ``"linear"``.
            device:        PyTorch device on which to allocate internal buffers.
        """
        if schedule not in ("cosine", "linear"):
            raise ValueError(f"Unknown schedule '{schedule}'. Choose 'cosine' or 'linear'.")

        self.denoiser = denoiser
        self.n_timesteps = n_timesteps
        self.device = torch.device(device)

        # Infer number of output channels (number of noisy wavelet sub‑bands)
        # from the denoiser.  Both UNet2D and UNet3D have an `out_channels` attribute.
        if not hasattr(denoiser, "out_channels"):
            raise AttributeError("The denoiser must expose an 'out_channels' attribute.")
        self.out_channels = denoiser.out_channels

        # ---- generate noise schedule ----
        if schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        else:
            betas = linear_beta_schedule(n_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Register everything as buffers so they are moved to self.device automatically
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("one_over_sqrt_alphas", 1.0 / torch.sqrt(alphas))
        # For DDIM: precomputed values for the coarsest possible lookup
        # Not strictly needed but stored for completeness
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        self.to(self.device)

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """Helper to register a buffer and immediately move it to the selected device."""
        setattr(self, name, tensor.to(self.device))

    def to(self, device: torch.device) -> "DDPM":
        """Move all buffers to the given device."""
        self.device = device
        for attr in dir(self):
            if not attr.startswith("_") and isinstance(getattr(self, attr), torch.Tensor):
                setattr(self, attr, getattr(self, attr).to(device))
        return self

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion step: q(x_t | x_0).

        Args:
            x0:    clean wavelet coefficients, shape (B, C, ...).
            noise: Gaussian noise, same shape as x0.
            t:     time indices, shape (B,), values in [0, n_timesteps-1].

        Returns:
            Noisy sample x_t, same shape as x0.
        """
        sqrt_alpha_cumprod = _extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_cumprod = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

    def denoise(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        guidance_w: float = 0.0,
    ) -> torch.Tensor:
        """
        Predict the noise that was added, optionally applying classifier‑free guidance.

        Args:
            x_noisy:     current noisy sample x_t, shape (B, C, ...).
            t:           time indices, shape (B,).
            cond:        conditioning tensor (e.g., wavelet‑transformed equation
                         parameters).  Must be ``None`` for unconditional training.
            guidance_w:  classifier‑free guidance weight ω.  A value of 0.0
                         disables guidance.

        Returns:
            Predicted noise tensor of the same shape as x_noisy.
        """
        # Conditional prediction (if cond is provided)
        if cond is not None:
            noise_cond = self.denoiser(x_noisy, t, cond)
        else:
            noise_cond = None

        # If guidance is requested, we need the unconditional prediction as well
        if guidance_w > 0.0 and cond is not None:
            # Create a null condition (all zeros) matching the shape of cond
            null_cond = torch.zeros_like(cond)
            noise_uncond = self.denoiser(x_noisy, t, null_cond)
            noise = noise_uncond + guidance_w * (noise_cond - noise_uncond)
            return noise

        # Fallback: either guidance_w == 0 or cond is None
        if noise_cond is not None:
            return noise_cond
        else:
            # Fully unconditional sampling
            return self.denoiser(x_noisy, t, None)

    def sample_ddim(
        self,
        cond: torch.Tensor,
        guidance_w: float = 0.0,
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling loop (deterministic or stochastic).

        Args:
            cond:         conditioning tensor, shape (B, C_cond, *spatial).
                          The spatial dimensions are used to define the shape of the
                          generated wavelet coefficients.
            guidance_w:   classifier‑free guidance weight ω.
            ddim_steps:   number of DDIM steps (must be ≤ n_timesteps).
            ddim_eta:     stochasticity parameter η; 0.0 = deterministic,
                          1.0 = full stochasticity.

        Returns:
            Generated clean wavelet coefficients, shape (B, out_channels, *spatial).
        """
        batch_size = cond.shape[0]
        spatial = cond.shape[2:]   # spatial dimensions (e.g., H, W) or (T, H, W)

        # Construct the sequence of time steps to visit (descending)
        times = torch.linspace(
            self.n_timesteps - 1, 0, ddim_steps, device=self.device
        ).round().long()

        # Initial random noise in wavelet domain
        x_cur = torch.randn(batch_size, self.out_channels, *spatial, device=self.device)

        for i in range(len(times) - 1):
            t_cur = times[i]
            t_next = times[i + 1]

            # Noise prediction with optional guidance
            noise_pred = self.denoise(x_cur, t_cur.expand(batch_size), cond, guidance_w)

            # DDIM step
            x_cur, _ = self._ddim_step(x_cur, noise_pred, t_cur, t_next, ddim_eta)

        return x_cur

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ddim_step(
        self,
        x: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
        next_t: torch.Tensor,
        ddim_eta: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single DDIM reverse step.

        Args:
            x:          current sample x_t, shape (B, C, ...).
            noise_pred: predicted noise ε_θ(x_t, t).
            t:          current timestep index (scalar or per‑element).
            next_t:     next timestep index (scalar or per‑element).
            ddim_eta:   stochasticity coefficient.

        Returns:
            - next sample x_{next_t}
            - denoised estimate x̂_0
        """
        # Current and next cumulative products
        alpha_bar_t     = _extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_next  = _extract(self.alphas_cumprod, next_t, x.shape)

        # Equation (6) of DDIM paper: estimate x0
        x0_pred = (x - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()

        if next_t.min() < 0:   # next_t may be -1 when the loop reaches the end; we treat as alpha_bar=1.0
            # When next_t == -1 (the final step), ᾱ = 1.0 (no noise) and we just return x0_pred.
            alpha_bar_next = torch.ones_like(alpha_bar_next)
            sigma = torch.zeros_like(alpha_bar_next)
        else:
            # Sigma computation
            sigma = ddim_eta * (
                (1 - alpha_bar_next) / (1 - alpha_bar_t)
            ).sqrt() * (
                1 - alpha_bar_t / alpha_bar_next
            ).sqrt()

        # Direction pointing back to the original noisy sample
        direction = (1 - alpha_bar_next - sigma**2).sqrt() * noise_pred

        # Random noise (only when eta > 0)
        if ddim_eta > 0.0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        # Next sample
        next_x = alpha_bar_next.sqrt() * x0_pred + direction + sigma * z

        # Clip to avoid numerical issues (optional, keeping the original range)
        # Not needed in practice for wavelet coefficients.

        return next_x, x0_pred


# ------------------------------------------------------------------
# Utility to extract appropriate values for a batch of indices
# ------------------------------------------------------------------

def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Index a 1‑D tensor `a` by `t` and reshape it to broadcast with `x_shape`.

    Args:
        a:      (T,) tensor of precomputed values.
        t:      (B,) integer indices.
        x_shape: shape of the noise/image tensor (B, C, ...).

    Returns:
        Tensor of shape (B, 1, ...) where extra dims are singleton.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    # Reshape to (batch_size, 1, *spatial_dims) by adding as many dimensions as necessary
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
