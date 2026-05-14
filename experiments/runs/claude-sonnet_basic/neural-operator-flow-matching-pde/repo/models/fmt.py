"""
FMT: Flow Marching Transformer

Implements the Flow Marching Transformer that bridges deterministic neural operators
and stochastic flow matching through a bridge parameter k.

Architecture:
- SiT (Scalable Interpolant Transformer) backbone with AdaLN-Zero conditioning
- RMSNorm and SwiGLU (Llama-2 style)
- Multi-head self-attention with head_dim=64
- Latent temporal pyramids for efficiency
- Diffusion forcing RNN for conditional generation

Three variants:
- FMT-S (Small): embed_dim=256, ~6M params
- FMT-B (Base): embed_dim=512, ~42M params
- FMT-L (Large): embed_dim=768, ~138M params
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .diffusion_forcing import DiffusionForcingRNN


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama-2 style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function (Llama-2 style)."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, bias: bool = False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * dim / 3)
            # Round to multiple of 256 for efficiency
            hidden_dim = 256 * ((hidden_dim + 255) // 256)

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with head_dim=64.
    Uses standard PyTorch scaled_dot_product_attention for efficiency.
    """

    def __init__(self, embed_dim: int, head_dim: int = 64, bias: bool = False):
        super().__init__()
        assert embed_dim % head_dim == 0, f"embed_dim {embed_dim} must be divisible by head_dim {head_dim}"
        self.num_heads = embed_dim // head_dim
        self.head_dim = head_dim
        self.embed_dim = embed_dim

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Use PyTorch's efficient attention (equivalent to FlashAttention when available)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class AdaLNZeroBlock(nn.Module):
    """
    Transformer block with AdaLN-Zero conditioning (DiT/SiT style).

    Conditions on a conditioning vector c (from time embedding + diffusion forcing state).
    Uses RMSNorm and SwiGLU (Llama-2 style).
    """

    def __init__(self, embed_dim: int, head_dim: int = 64, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, head_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.mlp = SwiGLU(embed_dim)

        # AdaLN-Zero: 6 parameters (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim, bias=True),
        )
        # Initialize to zero for stable training (AdaLN-Zero)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim) token sequence
            c: (B, embed_dim) conditioning vector

        Returns:
            x: (B, N, embed_dim) updated token sequence
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Self-attention with AdaLN
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        # MLP with AdaLN
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """Final layer with AdaLN-Zero for output projection."""

    def __init__(self, embed_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, embed_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class PatchEmbed(nn.Module):
    """Patch embedding for latent fields."""

    def __init__(self, latent_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.num_patches = (latent_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) latent field

        Returns:
            tokens: (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """
    Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: embedding dimension
        grid_size: grid size (H = W = grid_size)

    Returns:
        pos_embed: (grid_size*grid_size, embed_dim)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)  # (2, grid_size, grid_size)
    grid = grid.reshape(2, 1, grid_size, grid_size)

    # Sinusoidal embedding for each axis
    half_dim = embed_dim // 2
    omega = torch.arange(half_dim // 2, dtype=torch.float32) / (half_dim // 2)
    omega = 1.0 / (10000.0 ** omega)

    # H axis
    out_h = grid[0].reshape(-1)[:, None] * omega[None, :]  # (N, half_dim//2)
    out_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=-1)  # (N, half_dim)

    # W axis
    out_w = grid[1].reshape(-1)[:, None] * omega[None, :]
    out_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=-1)

    pos_embed = torch.cat([out_h, out_w], dim=-1)  # (N, embed_dim)
    return pos_embed


class FlowMarchingTransformer(nn.Module):
    """
    Flow Marching Transformer (FMT).

    Implements the conditional flow marching algorithm with:
    1. Latent temporal pyramids for efficiency
    2. Diffusion forcing for conditional generation
    3. AdaLN-Zero conditioning (SiT-style)
    4. RMSNorm + SwiGLU (Llama-2 style)

    The model takes 4 consecutive noisy states as input:
    - x0_noisy: downsampled by 8 (2x2 tokens)
    - x1_noisy: downsampled by 4 (4x4 tokens)
    - x2_noisy: downsampled by 2 (8x8 tokens)
    - x3_noisy: full resolution (16x16 tokens = 256 tokens)

    Total tokens: 4 + 16 + 64 + 256 = 340 tokens
    vs. naive 4*256 = 1024 tokens -> ~15x efficiency gain in attention

    The model predicts the flow marching velocity for x3 (the target state).

    The diffusion forcing RNN processes all frames sequentially to build
    the conditioning state h:
        h_s ~ p_phi(h_s | h_{s-1}, x_{s,t_s}^{k_s}, t_s)

    Training objective (conditional flow marching):
        L_CFM = 0.5 * E[||(1-t) * g(x_t^k, t, h) - (x1 - x_t^k)||^2]

    where x_t^k = x0 + t*(x1-x0) - (1-t)*(1-k)*(x0 - z), z ~ N(0,I)
    """

    def __init__(
        self,
        latent_channels: int = 16,
        latent_size: int = 16,
        patch_size: int = 1,
        embed_dim: int = 512,
        depth: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        num_frames: int = 4,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_frames = num_frames

        # Pyramid downsampling factors for each frame
        # Frame 0: 8x downsample (2x2), Frame 1: 4x (4x4), Frame 2: 2x (8x8), Frame 3: 1x (16x16)
        self.pyramid_factors = [8, 4, 2, 1]

        # Patch embeddings for each pyramid level
        self.patch_embeds = nn.ModuleList()
        for factor in self.pyramid_factors:
            # After downsampling by factor, the spatial size is latent_size // factor
            # We use patch_size=1 for all levels (each pixel is a token)
            downsampled_size = latent_size // factor
            self.patch_embeds.append(
                PatchEmbed(
                    latent_size=downsampled_size,
                    patch_size=patch_size,
                    in_channels=latent_channels,
                    embed_dim=embed_dim,
                )
            )

        # Positional embeddings for each pyramid level
        self.pos_embeds = nn.ParameterList()
        for factor in self.pyramid_factors:
            downsampled_size = latent_size // factor
            pos_embed = get_2d_sincos_pos_embed(embed_dim, downsampled_size // patch_size)
            self.pos_embeds.append(nn.Parameter(pos_embed.unsqueeze(0), requires_grad=False))

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(embed_dim)

        # Diffusion forcing RNN
        # Processes all frames sequentially to build conditioning state
        self.df_rnn = DiffusionForcingRNN(
            embed_dim=embed_dim,
            latent_channels=latent_channels,
            latent_size=latent_size,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(embed_dim, head_dim, mlp_ratio)
            for _ in range(depth)
        ])

        # Final layer for velocity prediction (only for the target frame tokens)
        self.final_layer = FinalLayer(embed_dim, patch_size, latent_channels)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following DiT/SiT practice."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # Initialize patch embeddings
        for pe in self.patch_embeds:
            w = pe.proj.weight.data
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))

        # Initialize timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def downsample_latent(self, y: torch.Tensor, factor: int) -> torch.Tensor:
        """
        Downsample latent by given factor using average pooling.

        Args:
            y: (B, C, H, W) latent
            factor: downsampling factor

        Returns:
            y_down: (B, C, H/factor, W/factor) downsampled latent
        """
        if factor == 1:
            return y
        return F.avg_pool2d(y, kernel_size=factor, stride=factor)

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Reconstruct spatial field from patch tokens.

        Args:
            x: (B, N, patch_size*patch_size*C) patch tokens
            h: height in patches
            w: width in patches

        Returns:
            field: (B, C, h*patch_size, w*patch_size)
        """
        p = self.patch_size
        c = self.latent_channels
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        x = x.reshape(x.shape[0], c, h * p, w * p)
        return x

    def forward(
        self,
        frames: List[torch.Tensor],
        t: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        t_all: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of FMT.

        The diffusion forcing RNN processes all frames sequentially to build
        the conditioning state. The transformer then predicts the velocity
        for the last frame.

        Args:
            frames: list of 4 tensors, each (B, latent_channels, latent_size, latent_size)
                    These are the noisy latent states [y0_t^k0, y1_t^k1, y2_t^k2, y3_t^k3]
            t: (B,) time values for the target frame (t3)
            h_prev: (B, embed_dim) previous diffusion forcing state from prior autoregressive step
            t_all: (B, 4) time values for all frames (for sequential RNN update)
                   If None, uses t for all frames

        Returns:
            velocity: (B, latent_channels, latent_size, latent_size) predicted velocity for frame 3
            h_new: (B, embed_dim) updated diffusion forcing state (after processing all frames)
        """
        assert len(frames) == self.num_frames, f"Expected {self.num_frames} frames, got {len(frames)}"
        B = frames[0].shape[0]

        # Build pyramid tokens for all frames
        all_tokens = []
        for i, (frame, factor) in enumerate(zip(frames, self.pyramid_factors)):
            # Downsample frame
            frame_down = self.downsample_latent(frame, factor)

            # Patch embed
            tokens = self.patch_embeds[i](frame_down)  # (B, num_patches_i, embed_dim)

            # Add positional embedding
            tokens = tokens + self.pos_embeds[i]

            all_tokens.append(tokens)

        # Concatenate all pyramid tokens
        # [frame0_tokens (4), frame1_tokens (16), frame2_tokens (64), frame3_tokens (256)]
        x = torch.cat(all_tokens, dim=1)  # (B, total_tokens, embed_dim)

        # Sequential diffusion forcing: process all frames to build conditioning state
        # h_s ~ p_phi(h_s | h_{s-1}, x_{s,t_s}^{k_s}, t_s)
        h_state = h_prev
        for s in range(self.num_frames):
            if t_all is not None:
                t_s = t_all[:, s]
            else:
                t_s = t  # Use same t for all frames if not specified
            h_state = self.df_rnn(h_state, frames[s], t_s)

        h_new = h_state

        # Conditioning: time embedding + diffusion forcing state
        t_emb = self.t_embedder(t)  # (B, embed_dim)
        c = t_emb + h_new  # (B, embed_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Extract target frame tokens (last 256 tokens)
        target_size = self.latent_size // self.pyramid_factors[-1]
        num_target_tokens = (target_size // self.patch_size) ** 2
        target_tokens = x[:, -num_target_tokens:, :]  # (B, 256, embed_dim)

        # Final layer: predict velocity
        velocity_tokens = self.final_layer(target_tokens, c)  # (B, 256, p*p*C)

        # Unpatchify
        h_patches = w_patches = target_size // self.patch_size
        velocity = self.unpatchify(velocity_tokens, h_patches, w_patches)  # (B, C, H, W)

        return velocity, h_new


def FMTSmall(**kwargs) -> FlowMarchingTransformer:
    """FMT-S: ~6M parameters (embed_dim=256, depth=12)."""
    return FlowMarchingTransformer(embed_dim=256, depth=12, **kwargs)


def FMTBase(**kwargs) -> FlowMarchingTransformer:
    """FMT-B: ~42M parameters (embed_dim=512, depth=12)."""
    return FlowMarchingTransformer(embed_dim=512, depth=12, **kwargs)


def FMTLarge(**kwargs) -> FlowMarchingTransformer:
    """FMT-L: ~138M parameters (embed_dim=768, depth=24)."""
    return FlowMarchingTransformer(embed_dim=768, depth=24, **kwargs)
