"""
Diffusion loss for continuous-valued token prediction.
Based on MAR (Li et al., 2024): Autoregressive Image Generation without Vector Quantization.
"""

import torch
import torch.nn as nn
import numpy as np
from functools import partial


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class MLPDiffusionHead(nn.Module):
    """
    MLP-based diffusion head for per-token probability modeling.
    Used in the first phase of Hi-MAR and as baseline in MAR.
    Each token is processed independently.
    """

    def __init__(self, in_channels, hidden_size, depth, token_dim):
        """
        Args:
            in_channels: dimension of input noise tokens
            hidden_size: hidden dimension of the MLP
            depth: number of MLP blocks
            token_dim: dimension of the conditional tokens from the Transformer
        """
        super().__init__()
        self.in_channels = in_channels
        self.token_dim = token_dim

        self.time_embed = TimestepEmbedder(hidden_size)

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_size)

        # Condition projection (from Transformer output)
        self.cond_proj = nn.Linear(token_dim, hidden_size)

        # MLP blocks with AdaLN
        self.blocks = nn.ModuleList([
            MLPBlock(hidden_size) for _ in range(depth)
        ])

        # Output projection
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(hidden_size, in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize output projection to zero
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, x, t, cond):
        """
        Args:
            x: noisy tokens [B, N, C] or [B*N, C]
            t: timesteps [B] or [B*N]
            cond: conditional tokens from Transformer [B, N, token_dim] or [B*N, token_dim]
        Returns:
            predicted noise [same shape as x]
        """
        # Handle batched inputs
        orig_shape = x.shape
        if x.dim() == 3:
            B, N, C = x.shape
            x = x.reshape(B * N, C)
            if cond.dim() == 3:
                cond = cond.reshape(B * N, -1)
            if t.dim() == 1 and t.shape[0] == B:
                t = t.unsqueeze(1).expand(B, N).reshape(B * N)

        t_emb = self.time_embed(t)
        x = self.input_proj(x)
        c = t_emb + self.cond_proj(cond)

        for block in self.blocks:
            x = block(x, c)

        x = self.norm_out(x)
        x = self.out_proj(x)

        if len(orig_shape) == 3:
            x = x.reshape(orig_shape)
        return x


class MLPBlock(nn.Module):
    """Single MLP block with AdaLN conditioning."""

    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # Initialize adaLN to zero
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        x = x + gate * self.ff(modulate(self.norm(x), shift, scale))
        return x


class DiffusionTransformerHead(nn.Module):
    """
    Diffusion Transformer head that uses self-attention to model
    interdependency among all tokens (masked and unmasked).
    Used in the second phase of Hi-MAR.

    Unlike MLP-based head, this processes all tokens jointly via self-attention,
    enabling global context propagation during denoising.
    """

    def __init__(self, in_channels, hidden_size, depth, token_dim, num_heads=8):
        """
        Args:
            in_channels: dimension of input noise tokens
            hidden_size: hidden dimension of the Transformer
            depth: number of Transformer blocks
            token_dim: dimension of the conditional tokens from the Transformer
            num_heads: number of attention heads
        """
        super().__init__()
        self.in_channels = in_channels
        self.token_dim = token_dim
        self.hidden_size = hidden_size

        self.time_embed = TimestepEmbedder(hidden_size)

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_size)

        # Condition projection (from Transformer output)
        self.cond_proj = nn.Linear(token_dim, hidden_size)

        # Transformer blocks with AdaLN
        self.blocks = nn.ModuleList([
            DiffTransformerBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # Output
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.out_proj = nn.Linear(hidden_size, in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t, cond):
        """
        Args:
            x: noisy tokens [B, N, C]
            t: timesteps [B]
            cond: conditional tokens from Transformer [B, N, token_dim]
        Returns:
            predicted noise [B, N, C]
        """
        B, N, C = x.shape

        t_emb = self.time_embed(t)  # [B, hidden_size]

        # Project inputs
        x = self.input_proj(x)  # [B, N, hidden_size]
        cond_proj = self.cond_proj(cond)  # [B, N, hidden_size]

        # Context vector: sum of time embedding and per-token condition
        # t_emb is [B, hidden_size], broadcast to [B, N, hidden_size]
        c = t_emb.unsqueeze(1) + cond_proj  # [B, N, hidden_size]

        for block in self.blocks:
            x = block(x, c)

        # Final output modulation
        # Use mean of context as global condition for final norm
        c_global = c.mean(dim=1)  # [B, hidden_size]
        shift, scale = self.adaLN_modulation(c_global).chunk(2, dim=-1)
        x = modulate(self.norm_out(x), shift, scale)
        x = self.out_proj(x)
        return x


class DiffTransformerBlock(nn.Module):
    """
    Transformer block for the Diffusion Transformer head.
    Uses AdaLN conditioning with per-token context vectors.
    """

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # AdaLN modulation: produces alpha1, beta1, gamma1, alpha2, beta2, gamma2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Initialize adaLN to zero
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        """
        Args:
            x: input tokens [B, N, hidden_size]
            c: context vectors [B, N, hidden_size] (per-token conditioning)
        """
        # Use mean of context for AdaLN (global conditioning)
        c_mean = c.mean(dim=1)  # [B, hidden_size]
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.adaLN_modulation(c_mean).chunk(6, dim=-1)

        # Self-attention with AdaLN
        x_norm = alpha1.unsqueeze(1) * self.norm1(x) + beta1.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gamma1.unsqueeze(1) * attn_out

        # FFN with AdaLN
        x_norm = alpha2.unsqueeze(1) * self.norm2(x) + beta2.unsqueeze(1)
        x = x + gamma2.unsqueeze(1) * self.ff(x_norm)

        return x


class DiffusionLoss(nn.Module):
    """
    Diffusion loss for continuous-valued token prediction.
    Wraps either MLPDiffusionHead or DiffusionTransformerHead.
    """

    def __init__(
        self,
        target_channels,
        hidden_size,
        depth,
        token_dim,
        num_heads=8,
        use_transformer_head=False,
        num_sampling_steps=100,
        grad_checkpointing=False,
    ):
        super().__init__()
        self.target_channels = target_channels
        self.num_sampling_steps = num_sampling_steps
        self.grad_checkpointing = grad_checkpointing

        if use_transformer_head:
            self.net = DiffusionTransformerHead(
                in_channels=target_channels,
                hidden_size=hidden_size,
                depth=depth,
                token_dim=token_dim,
                num_heads=num_heads,
            )
        else:
            self.net = MLPDiffusionHead(
                in_channels=target_channels,
                hidden_size=hidden_size,
                depth=depth,
                token_dim=token_dim,
            )

        self.use_transformer_head = use_transformer_head

    def forward(self, target, z, mask=None):
        """
        Compute diffusion loss.
        Args:
            target: ground truth tokens [B, N, C]
            z: conditional tokens from Transformer [B, N, token_dim]
            mask: binary mask indicating which tokens to compute loss on [B, N]
        Returns:
            loss scalar
        """
        B, N, C = target.shape

        # Sample random timesteps
        t = torch.randint(0, self.num_sampling_steps, (B,), device=target.device)

        # Compute noise schedule (linear schedule)
        alpha_bar = self._get_alpha_bar(t, target.device)  # [B]

        # Add noise to target
        noise = torch.randn_like(target)
        alpha_bar_expanded = alpha_bar[:, None, None]
        noisy_target = torch.sqrt(alpha_bar_expanded) * target + torch.sqrt(1 - alpha_bar_expanded) * noise

        # Predict noise
        if self.use_transformer_head:
            pred_noise = self.net(noisy_target, t, z)
        else:
            pred_noise = self.net(noisy_target, t, z)

        # Compute loss
        loss = (pred_noise - noise) ** 2

        if mask is not None:
            # Only compute loss on masked tokens
            loss = loss * mask.unsqueeze(-1).float()
            loss = loss.sum() / (mask.sum() * C + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def _get_alpha_bar(self, t, device):
        """Linear noise schedule."""
        T = self.num_sampling_steps
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod[t]

    @torch.no_grad()
    def sample(self, z, temperature=1.0, num_steps=None):
        """
        Sample tokens using DDPM sampling.
        Args:
            z: conditional tokens [B, N, token_dim]
            temperature: sampling temperature
            num_steps: number of denoising steps (defaults to num_sampling_steps)
        Returns:
            sampled tokens [B, N, C]
        """
        if num_steps is None:
            num_steps = self.num_sampling_steps

        B, N, _ = z.shape
        device = z.device

        # Start from pure noise
        x = torch.randn(B, N, self.target_channels, device=device) * temperature

        # Precompute noise schedule
        T = self.num_sampling_steps
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])

        # Select timesteps for sampling
        step_size = T // num_steps
        timesteps = list(range(T - 1, -1, -step_size))[:num_steps]

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Predict noise
            pred_noise = self.net(x, t, z)

            # DDPM update
            alpha_bar_t = alphas_cumprod[t_val]
            alpha_bar_prev = alphas_cumprod_prev[t_val]
            beta_t = betas[t_val]

            # Compute x0 prediction
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-10, 10)

            if t_val > 0:
                # Compute posterior mean
                posterior_mean = (
                    torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t) * x0_pred
                    + torch.sqrt(alphas[t_val]) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x
                )
                posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                noise = torch.randn_like(x) * temperature
                x = posterior_mean + torch.sqrt(posterior_var) * noise
            else:
                x = x0_pred

        return x
