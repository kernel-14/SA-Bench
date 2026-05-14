"""Conditional diffusion model for PGR. Residual MLP architecture matching SynthER."""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return timestep_embedding(t, self.dim)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, time_emb_dim: int, cond_emb_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim * 2),
        )
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_emb_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: torch.Tensor,
    ) -> torch.Tensor:
        # time conditioning
        time_out = self.time_mlp(time_emb)
        scale, shift = time_out.chunk(2, dim=-1)

        # condition conditioning
        cond_out = self.cond_mlp(cond_emb)

        h = self.norm1(x)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        h = h + cond_out.unsqueeze(1)
        h = self.linear1(F.silu(h))

        h = self.norm2(h)
        h = self.linear2(F.silu(h))
        return x + h


class ConditionalDiffusionModel(nn.Module):
    """Conditional diffusion model for transition generation.

    Generates transitions (s, a, s', r) conditioned on relevance value c = F(tau).
    Uses classifier-free guidance (CFG) during sampling.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_timesteps: int = 1000,
        model_dims: int = 512,
        n_residual_blocks: int = 6,
        block_dims: int = 512,
        time_emb_dims: int = 256,
        cond_emb_dims: int = 128,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        p_uncond: float = 0.25,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.transition_dim = 2 * state_dim + action_dim + 1  # s, a, s', r
        self.n_timesteps = n_timesteps
        self.p_uncond = p_uncond

        # Noise schedule: linear beta
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, n_timesteps))
        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod)
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_emb_dims),
            nn.Linear(time_emb_dims, time_emb_dims),
            nn.SiLU(),
            nn.Linear(time_emb_dims, time_emb_dims),
        )

        # Condition embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(1, cond_emb_dims),
            nn.SiLU(),
            nn.Linear(cond_emb_dims, cond_emb_dims),
        )

        # Null condition embedding (for CFG)
        self.null_cond = nn.Parameter(torch.randn(1, 1, cond_emb_dims) * 0.02)

        # Input projection
        self.input_proj = nn.Linear(self.transition_dim, block_dims)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(block_dims, time_emb_dims, cond_emb_dims)
            for _ in range(n_residual_blocks)
        ])

        # Output projection
        self.output_proj = nn.Linear(block_dims, self.transition_dim)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cond_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise given noisy input, timestep, and optional condition.

        Args:
            x: (B, transition_dim) noisy transition
            time: (B,) integer timesteps
            cond: (B, 1) relevance values. If cond_drop_mask is provided,
                  dropped positions use the learned null embedding.
            cond_drop_mask: (B, 1) boolean mask, True = use null (drop condition)

        Returns:
            Predicted noise epsilon of shape (B, transition_dim)
        """
        b = x.shape[0]

        # Time embedding
        time_emb = self.time_embed(time)  # (B, time_emb_dims)

        # Condition embedding
        if cond is None:
            cond_emb = self.null_cond.expand(b, -1, -1).squeeze(1)  # (B, cond_emb_dims)
        else:
            cond_emb = self.cond_embed(cond)  # (B, cond_emb_dims)
            if cond_drop_mask is not None and cond_drop_mask.any():
                null_emb = self.null_cond.expand(b, -1, -1).squeeze(1)
                cond_emb = torch.where(cond_drop_mask, null_emb, cond_emb)

        # Project input
        h = self.input_proj(x)  # (B, block_dims)

        # Pass through residual blocks
        for block in self.residual_blocks:
            h = block(h, time_emb, cond_emb)

        # Project to output
        eps_pred = self.output_proj(h)
        return eps_pred

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: add noise to clean data."""
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.5,
    ) -> torch.Tensor:
        """Reverse process: single denoising step with CFG.

        Uses classifier-free guidance: epsilon = omega * eps_cond + (1 - omega) * eps_uncond
        """
        beta_t = self.betas[t].view(-1, 1)
        alpha_t = 1.0 - beta_t
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1)

        # Predict noise with and without condition
        eps_uncond = self.forward(x, t, cond=None)
        if cond is not None and guidance_scale != 1.0:
            eps_cond = self.forward(x, t, cond=cond)
            eps = guidance_scale * eps_cond + (1.0 - guidance_scale) * eps_uncond
        else:
            eps = eps_uncond

        # Compute mean of p_theta(x_{t-1} | x_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps
        )

        # Add noise (DDPM sampling)
        if t.min() > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            return mean + sigma * noise
        else:
            return mean

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        cond: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.5,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Generate transitions via iterative denoising.

        Args:
            batch_size: number of transitions to generate
            cond: (batch_size, 1) relevance values
            guidance_scale: omega for CFG
            device: torch device

        Returns:
            Generated transitions of shape (batch_size, transition_dim)
        """
        x = torch.randn(batch_size, self.transition_dim, device=device)

        for t in reversed(range(self.n_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, cond=cond, guidance_scale=guidance_scale)

        return x

    def training_loss(
        self,
        x_start: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute diffusion training loss with optional CFG training.

        Follows Eq. (2) from the paper.
        """
        b = x_start.shape[0]
        device = x_start.device

        # Sample random timestep
        t = torch.randint(0, self.n_timesteps, (b,), device=device)

        # Sample noise
        noise = torch.randn_like(x_start)

        # Add noise
        x_noisy = self.q_sample(x_start, t, noise)

        # CFG training: randomly drop condition (Eq. 2)
        if cond is not None and self.p_uncond > 0:
            drop_mask = torch.rand(b, 1, device=device) < self.p_uncond
            eps_pred = self.forward(x_noisy, t, cond=cond, cond_drop_mask=drop_mask)
        else:
            eps_pred = self.forward(x_noisy, t, cond=cond)

        # MSE loss
        loss = F.mse_loss(eps_pred, noise)
        return loss


class DiffusionBuffer:
    """Manages the parametric synthetic replay buffer (D_syn)."""

    def __init__(self, capacity: int, transition_dim: int):
        self.capacity = capacity
        self.transition_dim = transition_dim
        self.data = torch.zeros(capacity, transition_dim)
        self.size = 0
        self.ptr = 0

    def add(self, transitions: torch.Tensor):
        """Add generated transitions to the buffer (FIFO)."""
        n = transitions.shape[0]
        if n > self.capacity:
            transitions = transitions[-self.capacity:]
            n = self.capacity

        end_idx = min(self.ptr + n, self.capacity)
        self.data[self.ptr:end_idx] = transitions[:end_idx - self.ptr]
        remainder = n - (end_idx - self.ptr)
        if remainder > 0:
            self.data[:remainder] = transitions[end_idx - self.ptr:]
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> torch.Tensor:
        """Randomly sample transitions from the buffer."""
        indices = torch.randint(0, self.size, (batch_size,))
        return self.data[indices]
