"""
Visual Encoder for NaViL.

The visual encoder V_{d,w}(I) consists of d transformer layers with hidden dimension w.
It uses bidirectional attention and 2D-RoPE to capture global spatial relationships.
When d=0, it degenerates to a simple patch embedding layer.

Architecture follows the paper:
  V_{d,w}(I) = C ⊙ F_d^w ⊙ ... ⊙ F_1^w ⊙ P(I)

where:
  - P is the Patch Embedding Layer (stride 16)
  - F_i^w is the i-th transformer layer with hidden dim w
  - C is the connector (pixel shuffle downsampling + MLP projection)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .norm import get_rms_norm as RMSNorm

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisualEncoderConfig:
    """Configuration for the visual encoder."""
    # Encoder architecture
    depth: int = 24          # d: number of transformer layers
    width: int = 1472        # w: hidden dimension
    num_heads: int = 16      # number of attention heads
    mlp_ratio: float = 4.0  # FFN expansion ratio
    patch_size: int = 16     # patch size for patch embedding
    image_size: int = 448    # default image size (can be dynamic)
    in_channels: int = 3     # input image channels

    # Connector
    pixel_shuffle_factor: int = 2   # downsampling factor for pixel shuffle
    llm_hidden_size: int = 2048     # LLM hidden size for projection

    # Positional encoding
    use_2d_rope: bool = True        # use 2D-RoPE for spatial relationships

    # Normalization
    norm_eps: float = 1e-6

    # Dropout
    dropout: float = 0.0
    attn_dropout: float = 0.0

    @property
    def num_patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.num_patches_per_side ** 2

    @property
    def head_dim(self) -> int:
        return self.width // self.num_heads

    @classmethod
    def from_param_budget(cls, param_budget_m: float, llm_hidden_size: int = 2048) -> "VisualEncoderConfig":
        """
        Create a visual encoder config from a parameter budget (in millions).
        Uses the formula N ≈ 12 * d * w^2 to determine d and w.
        
        Based on paper's observation that a wide range of d/w combinations work,
        we use a moderate depth-to-width ratio.
        """
        # Target parameter count
        N = param_budget_m * 1e6

        # Use a moderate depth: roughly d ~ w/64 (similar to ViT-B/L ratios)
        # N ≈ 12 * d * w^2, solve for w given d = w/64
        # N ≈ 12 * (w/64) * w^2 = 12/64 * w^3
        # w^3 = N * 64 / 12
        # w = (N * 64 / 12)^(1/3)
        w = int((N * 64 / 12) ** (1/3))
        # Round to nearest multiple of 64
        w = max(64, round(w / 64) * 64)
        d = max(1, round(w / 64))

        # Adjust num_heads to be divisible
        num_heads = max(1, w // 64)

        return cls(
            depth=d,
            width=w,
            num_heads=num_heads,
            llm_hidden_size=llm_hidden_size,
        )


class PatchEmbedding(nn.Module):
    """
    Patch Embedding Layer P(I).
    Converts image I ∈ R^{H×W×3} to patch tokens.
    Stride is set to patch_size (16 by default).
    """

    def __init__(self, config: VisualEncoderConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.width = config.width

        self.proj = nn.Conv2d(
            config.in_channels,
            config.width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.norm = nn.LayerNorm(config.width, eps=config.norm_eps)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: Image tensor of shape (B, C, H, W)
        Returns:
            patches: (B, num_patches, width)
            grid_size: (H//patch_size, W//patch_size)
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Image size ({H}, {W}) must be divisible by patch size {self.patch_size}"

        x = self.proj(x)  # (B, width, H/p, W/p)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, width)
        x = self.norm(x)
        return x, (h, w)


class RoPE2D(nn.Module):
    """
    2D Rotary Position Embedding for visual encoder.
    Applies separate RoPE to height and width dimensions.
    """

    def __init__(self, head_dim: int, max_size: int = 64):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D-RoPE"
        self.head_dim = head_dim
        self.half_dim = head_dim // 2  # half for each spatial dimension

        # Precompute frequencies
        dim = self.half_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_size = max_size

    def _get_sincos(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """positions: (N,) -> sin/cos: (N, half_dim)"""
        freqs = torch.outer(positions.float(), self.inv_freq)  # (N, dim/4)
        emb = torch.cat([freqs, freqs], dim=-1)  # (N, dim/2)
        return emb.sin(), emb.cos()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D-RoPE to queries and keys.
        
        Args:
            q: (B, num_heads, N, head_dim)
            k: (B, num_heads, N, head_dim)
            grid_size: (H, W) grid dimensions
        Returns:
            q, k with 2D-RoPE applied
        """
        H, W = grid_size
        device = q.device

        # Create 2D position grid
        h_pos = torch.arange(H, device=device)
        w_pos = torch.arange(W, device=device)

        sin_h, cos_h = self._get_sincos(h_pos)  # (H, half_dim/2)
        sin_w, cos_w = self._get_sincos(w_pos)  # (W, half_dim/2)

        # Expand to grid
        sin_h = sin_h.unsqueeze(1).expand(H, W, -1)  # (H, W, half_dim/2)
        cos_h = cos_h.unsqueeze(1).expand(H, W, -1)
        sin_w = sin_w.unsqueeze(0).expand(H, W, -1)  # (H, W, half_dim/2)
        cos_w = cos_w.unsqueeze(0).expand(H, W, -1)

        # Concatenate h and w embeddings
        sin_2d = torch.cat([sin_h, sin_w], dim=-1).reshape(H * W, self.half_dim)  # (N, half_dim)
        cos_2d = torch.cat([cos_h, cos_w], dim=-1).reshape(H * W, self.half_dim)

        # Apply to first half of head_dim (second half uses 1D or is zero)
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        # Apply 2D-RoPE to first half of head_dim
        q_2d = q[..., : self.half_dim]
        k_2d = k[..., : self.half_dim]

        sin_2d = sin_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, N, half_dim)
        cos_2d = cos_2d.unsqueeze(0).unsqueeze(0)

        q_2d = q_2d * cos_2d + rotate_half(q_2d) * sin_2d
        k_2d = k_2d * cos_2d + rotate_half(k_2d) * sin_2d

        q = torch.cat([q_2d, q[..., self.half_dim :]], dim=-1)
        k = torch.cat([k_2d, k[..., self.half_dim :]], dim=-1)

        return q, k


class VisualAttention(nn.Module):
    """
    Bidirectional multi-head self-attention for visual encoder.
    Uses 2D-RoPE for positional encoding.
    """

    def __init__(self, config: VisualEncoderConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.width = config.width
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.width, config.width, bias=False)
        self.k_proj = nn.Linear(config.width, config.width, bias=False)
        self.v_proj = nn.Linear(config.width, config.width, bias=False)
        self.o_proj = nn.Linear(config.width, config.width, bias=False)

        self.attn_dropout = nn.Dropout(config.attn_dropout)

        if config.use_2d_rope:
            self.rope = RoPE2D(self.head_dim)
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, width)
            grid_size: (H, W) for 2D-RoPE
        Returns:
            (B, N, width)
        """
        B, N, _ = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None and grid_size is not None:
            q, k = self.rope(q, k, grid_size)

        # Bidirectional attention (no causal mask)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, self.width)
        out = self.o_proj(out)
        return out


class VisualFFN(nn.Module):
    """Feed-forward network for visual encoder (SwiGLU activation)."""

    def __init__(self, config: VisualEncoderConfig):
        super().__init__()
        hidden_dim = int(config.width * config.mlp_ratio)
        # SwiGLU uses 2/3 of the hidden dim to keep parameter count similar
        hidden_dim = int(2 * hidden_dim / 3)
        # Round to multiple of 64
        hidden_dim = (hidden_dim + 63) // 64 * 64

        self.gate_proj = nn.Linear(config.width, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.width, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VisualEncoderLayer(nn.Module):
    """Single transformer layer F_i^w for visual encoder."""

    def __init__(self, config: VisualEncoderConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.width, eps=config.norm_eps)
        self.attn = VisualAttention(config)
        self.norm2 = RMSNorm(config.width, eps=config.norm_eps)
        self.ffn = VisualFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), grid_size=grid_size)
        x = x + self.ffn(self.norm2(x))
        return x


class PixelShuffleConnector(nn.Module):
    """
    Connector C that downsamples encoded image embeddings through pixel shuffle
    and projects them to the LLM's feature space via MLP.
    
    Pixel shuffle (inverse) reduces spatial resolution by factor r,
    increasing channel dimension by r^2.
    """

    def __init__(self, config: VisualEncoderConfig):
        super().__init__()
        self.factor = config.pixel_shuffle_factor
        in_dim = config.width * (config.pixel_shuffle_factor ** 2)

        # MLP projector: two-layer with GELU
        self.proj = nn.Sequential(
            nn.Linear(in_dim, config.llm_hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(config.llm_hidden_size, config.llm_hidden_size, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: (B, H*W, width) visual tokens
            grid_size: (H, W)
        Returns:
            projected: (B, H'*W', llm_hidden_size)
            new_grid_size: (H', W')
        """
        B, N, C = x.shape
        H, W = grid_size
        assert H * W == N

        r = self.factor
        # Reshape to spatial grid
        x = x.reshape(B, H, W, C)

        # Pixel unshuffle: merge r×r spatial blocks into channels
        # New spatial: (H/r, W/r), new channels: C*r^2
        assert H % r == 0 and W % r == 0, \
            f"Grid size ({H}, {W}) must be divisible by pixel shuffle factor {r}"

        x = x.reshape(B, H // r, r, W // r, r, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H/r, W/r, r, r, C)
        x = x.reshape(B, H // r, W // r, r * r * C)

        new_H, new_W = H // r, W // r
        x = x.reshape(B, new_H * new_W, r * r * C)

        # Project to LLM hidden size
        x = self.proj(x)

        return x, (new_H, new_W)


class VisualEncoder(nn.Module):
    """
    Visual Encoder V_{d,w}(I) for NaViL.
    
    Consists of:
    1. Patch Embedding Layer P
    2. d transformer layers F_i^w with bidirectional attention and 2D-RoPE
    3. Connector C (pixel shuffle + MLP projection)
    
    When d=0, degenerates to patch embedding + connector only.
    
    Parameter count: N ≈ 12 * d * w^2
    """

    def __init__(self, config: VisualEncoderConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(config)

        # Transformer layers (d layers)
        self.layers = nn.ModuleList([
            VisualEncoderLayer(config) for _ in range(config.depth)
        ])

        # Final norm
        if config.depth > 0:
            self.norm = RMSNorm(config.width, eps=config.norm_eps)
        else:
            self.norm = None

        # Connector
        self.connector = PixelShuffleConnector(config)

    def forward(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            images: (B, C, H, W) - input images, H and W must be multiples of patch_size
        Returns:
            visual_tokens: (B, N', llm_hidden_size) - projected visual tokens
            grid_size: (H', W') - spatial grid size after connector
        """
        # Patch embedding
        x, grid_size = self.patch_embed(images)  # (B, H*W, width)

        # Transformer layers with bidirectional attention
        for layer in self.layers:
            x = layer(x, grid_size=grid_size)

        if self.norm is not None:
            x = self.norm(x)

        # Connector: pixel shuffle + MLP projection
        x, new_grid_size = self.connector(x, grid_size)

        return x, new_grid_size

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_param_budget(
        cls,
        param_budget_m: float,
        llm_hidden_size: int = 2048,
        depth: Optional[int] = None,
        width: Optional[int] = None,
    ) -> "VisualEncoder":
        """
        Create a visual encoder from a parameter budget in millions.
        
        The paper explores depth d ∈ {3, 6, 12, 24, 48} and
        width w ∈ {4096, 2880, 2048, 1472, 1024} for 600M budget.
        """
        if depth is not None and width is not None:
            config = VisualEncoderConfig(
                depth=depth,
                width=width,
                num_heads=max(1, width // 64),
                llm_hidden_size=llm_hidden_size,
            )
        else:
            config = VisualEncoderConfig.from_param_budget(param_budget_m, llm_hidden_size)
        return cls(config)
