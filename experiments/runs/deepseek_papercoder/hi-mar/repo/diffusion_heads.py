"""
diffusion_heads.py

Module implementing the two diffusion heads used in Hi‑MAR:
  - MLPDiffusionHead   : MLP‑based head for low‑resolution tokens (Phase 1)
  - DiffusionTransformerHead : Transformer‑based head for high‑resolution tokens (Phase 2)

Both heads operate in the continuous latent space (dimension 16) and share
a common DDPM noise schedule (linear beta schedule, T=1000).  They support
training via ``forward`` (noise prediction) and inference via ``sample``
(DDIM denoising).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Project imports
from config import ModelConfig
from utils import timestep_embedding
from utils import chunk_adaLN_parameters  # used for 6‑parameter split


# ---------------------------------------------------------------------------
#  Common – register a linear DDPM noise schedule as module buffers
# ---------------------------------------------------------------------------

def _register_diffusion_schedule(module: nn.Module, timesteps: int = 1000) -> None:
    """
    Add ``betas``, ``alphas``, ``alphas_cumprod``, ``sqrt_alphas_cumprod``,
    and ``sqrt_one_minus_alphas_cumprod`` as registered buffers on *module*.

    The schedule uses a linear beta schedule from 1e‑4 to 0.02,
    which is standard for image generation tasks.
    """
    betas = torch.linspace(1e-4, 0.02, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    module.register_buffer("betas", betas)
    module.register_buffer("alphas", alphas)
    module.register_buffer("alphas_cumprod", alphas_cumprod)
    module.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
    module.register_buffer(
        "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
    )


# ---------------------------------------------------------------------------
#  Building blocks for the heads
# ---------------------------------------------------------------------------

class MlpAdaLNBlock(nn.Module):
    """
    MLP block with adaptive layer norm (adaLN) modulation.

    Used inside ``MLPDiffusionHead``.  The block receives per‑token context ``c``,
    which is the sum of the time embedding and the projected conditional token.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.modulation = nn.Linear(hidden_size, 3 * hidden_size)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, hidden_size) – token features.
            c: (B, N, hidden_size) – per‑token context (time + conditional).
        Returns:
            (B, N, hidden_size) – updated token features.
        """
        scale, shift, gate = self.modulation(c).chunk(3, dim=-1)
        h = F.layer_norm(x, (x.shape[-1],))
        h = h * scale + shift
        h = self.mlp(h)
        return x + gate * h


class DiffusionTransformerBlock(nn.Module):
    """
    Transformer block with adaLN modulation (6 parameters) used in
    ``DiffusionTransformerHead``.  It employs multi‑head self‑attention
    followed by a feed‑forward network, both gated by context‑derived parameters.

    This block corresponds to Figure 2(e) of the paper.
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Linear layer to produce the 6 modulation parameters
        self.modulation = nn.Linear(hidden_size, 6 * hidden_size)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

        # Self‑attention (batch_first for (B, N, C))
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )

        # Feed‑forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, hidden_size) – token features.
            c: (B, N, hidden_size) – per‑token context (time + conditional).
        Returns:
            (B, N, hidden_size) – updated token features.
        """
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.modulation(c).chunk(6, dim=-1)

        # Attention sub‑layer
        h = F.layer_norm(x, (self.hidden_size,))
        h = h * alpha1 + beta1
        h, _ = self.attn(h, h, h)
        x = x + gamma1 * h

        # Feed‑forward sub‑layer
        h = F.layer_norm(x, (self.hidden_size,))
        h = h * alpha2 + beta2
        h = self.ffn(h)
        x = x + gamma2 * h

        return x


# =========================================================================
#  Phase 1 – MLP‑based Diffusion Head (for low‑resolution tokens)
# =========================================================================

class MLPDiffusionHead(nn.Module):
    """
    MLP‑based diffusion head for low‑resolution (Phase 1) token prediction.

    This head processes each token **independently** and is conditioned on
    the Hi‑MAR Transformer’s output ``z_cond`` plus a diffusion timestep.
    It follows the design of MAR’s diffusion head (Li et al. 2024).

    Architecture:
        - Time embedding MLP
        - Projection of conditional tokens to head dimension
        - Input projection (latent → head dim)
        - Stack of ``MlpAdaLNBlock``s
        - Output projection → latent dim (noise prediction)
    """

    def __init__(self, model_config: ModelConfig, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Retrieve configuration for Phase 1 head from the global model config
        dh1_cfg = model_config.diffusion_head_1
        self.hidden_size = dh1_cfg.hidden_size
        self.num_layers = dh1_cfg.num_layers
        self.transformer_hidden_size = model_config.hidden_size

        # ────────────────  Time embedding  ────────────────
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # ────────────────  Conditioning projection  ────────────────
        self.cond_proj = nn.Linear(self.transformer_hidden_size, self.hidden_size)

        # ────────────────  Latent → head projection  ────────────────
        self.input_proj = nn.Linear(latent_dim, self.hidden_size)

        # ────────────────  Core blocks  ────────────────
        self.blocks = nn.ModuleList(
            [MlpAdaLNBlock(self.hidden_size) for _ in range(self.num_layers)]
        )

        # ────────────────  Output projection  ────────────────
        self.output_proj = nn.Linear(self.hidden_size, latent_dim)

        # ────────────────  Diffusion schedule  ────────────────
        _register_diffusion_schedule(self)

    # ------------------------------------------------------------------
    #  Forward (noise prediction) – training
    # ------------------------------------------------------------------
    def forward(
        self,
        z_cond: Tensor,
        x_noisy: Tensor,
        t: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict the noise that was added to ``x_noisy``.

        Args:
            z_cond:  ``(B, N, transformer_hidden_size)`` – conditional tokens
                     from the Hi‑MAR Transformer.
            x_noisy: ``(B, N, latent_dim)`` – noise‑corrupted latent tokens.
            t:       ``(B,)`` float tensor in [0, 1] indicating the diffusion timestep.
            mask:    Optional boolean mask (unused; kept for interface compatibility).

        Returns:
            Noise prediction tensor of shape ``(B, N, latent_dim)``.
        """
        B, N, _ = x_noisy.shape

        # 1. Time embedding and combination with conditional tokens
        t_scaled = t * 1000.0  # scale to [0, 1000] for sinusoidal embedding
        t_emb = self.time_embed(timestep_embedding(t_scaled, self.hidden_size))  # (B, H)
        c = self.cond_proj(z_cond) + t_emb.unsqueeze(1)              # (B, N, H)

        # 2. Project noisy latent to head dimension
        x = self.input_proj(x_noisy)                                  # (B, N, H)

        # 3. Apply sequential blocks
        for blk in self.blocks:
            x = blk(x, c)

        # 4. Map back to latent space
        noise = self.output_proj(x)                                   # (B, N, D)
        return noise

    # ------------------------------------------------------------------
    #  DDIM sampling (inference)
    # ------------------------------------------------------------------
    def sample(
        self,
        z_cond: Tensor,
        mask: Tensor,
        steps: int,
        x: Optional[Tensor] = None,
        known_latents: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Denoise the masked positions using DDIM sampling.

        Args:
            z_cond:  ``(B, N, transformer_hidden_size)`` – conditional tokens.
            mask:    ``(B, N)`` boolean mask indicating positions to predict.
            steps:   Number of DDIM reverse steps (e.g., 5).
            x:       ``(B, N, latent_dim)`` – current latent sequence.  Masked
                     positions should contain random noise, unmasked positions
                     contain already decoded clean latents.
            known_latents: Optionally supply the clean latents for unmasked
                           positions; defaults to the current value of ``x``.
        Returns:
            Denoised latent tensor of shape ``(B, N, latent_dim)``, where masked
            positions have been predicted and unmasked positions are unchanged.
        """
        B, N, D = z_cond.shape[0], z_cond.shape[1], self.latent_dim

        if x is None:
            x = torch.randn(B, N, D, device=z_cond.device)

        if known_latents is None:
            known_latents = x.clone()

        # Determine the DDIM timestep sequence (from 999 down to 0)
        times = torch.linspace(999, 0, steps + 1, device=z_cond.device).long()
        for i in range(steps):
            t = times[i].item()
            t_next = times[i + 1].item()

            # Normalise timestep for the model
            t_norm = torch.full((B,), t / 999.0, device=z_cond.device)
            noise_pred = self.forward(z_cond, x, t_norm)

            # Compute predicted x0
            sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            x0_pred = (x - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t

            # Take a step towards x0_pred
            if t_next >= 0:
                alpha_cumprod_next = self.alphas_cumprod[t_next]
                sqrt_alpha_next = torch.sqrt(alpha_cumprod_next)
                sqrt_one_minus_alpha_next = torch.sqrt(1.0 - alpha_cumprod_next)
                x = sqrt_alpha_next * x0_pred + sqrt_one_minus_alpha_next * noise_pred
            else:
                x = x0_pred

            # Restore unmasked positions to their known clean latents
            x = torch.where(mask.unsqueeze(-1), x, known_latents)

        return x


# =========================================================================
#  Phase 2 – Diffusion Transformer Head (for high‑resolution tokens)
# =========================================================================

class DiffusionTransformerHead(nn.Module):
    """
    Diffusion Transformer head for high‑resolution (Phase 2) token prediction.

    This head processes **all** token positions jointly through a stack of
    Transformer blocks with self‑attention, exploiting inter‑token dependencies.
    Conditioning is provided by the Hi‑MAR Transformer’s output ``z_cond`` and
    a diffusion timestep.  Architecture corresponds to Figure 2(e) of the paper.

    Architecture:
        - Time embedding MLP
        - Projection of conditional tokens
        - Input projection (latent → head dim)
        - Stack of ``DiffusionTransformerBlock``s
        - Output projection → latent dim (noise prediction)
    """

    def __init__(self, model_config: ModelConfig, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Retrieve configuration for Phase 2 head
        dh2_cfg = model_config.diffusion_head_2
        self.hidden_size = dh2_cfg.hidden_size
        self.num_layers = dh2_cfg.num_layers
        self.num_heads = dh2_cfg.num_heads
        self.transformer_hidden_size = model_config.hidden_size

        # ────────────────  Time embedding  ────────────────
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # ────────────────  Conditioning projection  ────────────────
        self.cond_proj = nn.Linear(self.transformer_hidden_size, self.hidden_size)

        # ────────────────  Latent → head projection  ────────────────
        self.input_proj = nn.Linear(latent_dim, self.hidden_size)

        # ────────────────  Core blocks  ────────────────
        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(self.hidden_size, self.num_heads)
                for _ in range(self.num_layers)
            ]
        )

        # ────────────────  Output projection  ────────────────
        self.output_proj = nn.Linear(self.hidden_size, latent_dim)

        # ────────────────  Diffusion schedule  ────────────────
        _register_diffusion_schedule(self)

    # ------------------------------------------------------------------
    #  Forward (noise prediction) – training
    # ------------------------------------------------------------------
    def forward(
        self,
        z_cond_all: Tensor,
        x_noisy_all: Tensor,
        t: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict the noise that was added to ``x_noisy_all``.

        Args:
            z_cond_all:   ``(B, N, transformer_hidden_size)`` – conditional tokens
                          for every high‑res position.
            x_noisy_all:  ``(B, N, latent_dim)`` – noise‑corrupted latent tokens.
            t:            ``(B,)`` float tensor in [0, 1] – diffusion timestep.
            mask:         Optional, unused.

        Returns:
            Noise prediction of shape ``(B, N, latent_dim)``.
        """
        B, N, _ = x_noisy_all.shape

        # Time embedding
        t_scaled = t * 1000.0
        t_emb = self.time_embed(timestep_embedding(t_scaled, self.hidden_size))   # (B, H)
        c = self.cond_proj(z_cond_all) + t_emb.unsqueeze(1)                      # (B, N, H)

        # Project noisy latent
        x = self.input_proj(x_noisy_all)                                           # (B, N, H)

        for blk in self.blocks:
            x = blk(x, c)

        noise = self.output_proj(x)                                               # (B, N, D)
        return noise

    # ------------------------------------------------------------------
    #  DDIM sampling (inference)
    # ------------------------------------------------------------------
    def sample(
        self,
        z_cond_all: Tensor,
        mask: Tensor,
        steps: int,
        x: Optional[Tensor] = None,
        known_latents: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Denoise the masked positions using DDIM sampling with full
        self‑attention across tokens.

        Args:
            z_cond_all: ``(B, N, transformer_hidden_size)`` – conditional tokens.
            mask:       ``(B, N)`` boolean mask indicating positions to predict.
            steps:      Number of DDIM reverse steps (e.g., 5).
            x:          ``(B, N, latent_dim)`` – current latent sequence.
            known_latents: Optional clean latents for unmasked positions.
        Returns:
            Denoised latent tensor of shape ``(B, N, latent_dim)``.
        """
        B, N, D = z_cond_all.shape[0], z_cond_all.shape[1], self.latent_dim

        if x is None:
            x = torch.randn(B, N, D, device=z_cond_all.device)

        if known_latents is None:
            known_latents = x.clone()

        times = torch.linspace(999, 0, steps + 1, device=z_cond_all.device).long()
        for i in range(steps):
            t = times[i].item()
            t_next = times[i + 1].item()

            t_norm = torch.full((B,), t / 999.0, device=z_cond_all.device)
            noise_pred = self.forward(z_cond_all, x, t_norm)

            sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
            x0_pred = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

            if t_next >= 0:
                alpha_next = self.alphas_cumprod[t_next]
                sqrt_alpha_next = torch.sqrt(alpha_next)
                sqrt_one_minus_alpha_next = torch.sqrt(1.0 - alpha_next)
                x = sqrt_alpha_next * x0_pred + sqrt_one_minus_alpha_next * noise_pred
            else:
                x = x0_pred

            x = torch.where(mask.unsqueeze(-1), x, known_latents)

        return x
