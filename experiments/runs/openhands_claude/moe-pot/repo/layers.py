import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvExpert(nn.Module):
    """Single CNN expert network used in the MoE layer.

    Each expert is a two-layer convolutional network that maps spatial features
    of shape (B, C, H, W) to the same shape.
    """

    def __init__(self, channels: int, hidden_channels: Optional[int] = None) -> None:
        super().__init__()
        if hidden_channels is None:
            hidden_channels = channels
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RouterGating(nn.Module):
    """CNN-based router-gating network.

    Takes spatial features (B, C, H, W) and produces per-sample routing logits
    of shape (B, num_routed_experts) via global average pooling.
    """

    def __init__(self, channels: int, num_routed_experts: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_routed_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits: (B, num_routed_experts)
        """
        h = self.conv(x)
        h = self.pool(h).flatten(1)   # (B, C)
        return self.fc(h)             # (B, num_routed_experts)


class FourierHead(nn.Module):
    """Single head of the multi-head Fourier layer.

    Implements the frequency-domain MLP:
        z_0i(x) = F^{-1}[ W2 * sigma(W1 * F[zi] + b1) + b2 ](x)

    where W1, W2 ∈ R^{d/h × d/h} are applied pointwise in frequency space.
    """

    def __init__(self, head_dim: int, num_modes_h: int, num_modes_w: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_modes_h = num_modes_h
        self.num_modes_w = num_modes_w

        # Frequency-domain weights (complex-valued, stored as real pairs)
        # W1, W2 operate on the truncated frequency tensor of shape
        # (num_modes_h, num_modes_w//2+1, head_dim)
        scale = 1.0 / math.sqrt(head_dim)
        self.W1_real = nn.Parameter(
            scale * torch.randn(num_modes_h, num_modes_w // 2 + 1, head_dim, head_dim)
        )
        self.W1_imag = nn.Parameter(
            scale * torch.randn(num_modes_h, num_modes_w // 2 + 1, head_dim, head_dim)
        )
        self.b1_real = nn.Parameter(torch.zeros(num_modes_h, num_modes_w // 2 + 1, head_dim))
        self.b1_imag = nn.Parameter(torch.zeros(num_modes_h, num_modes_w // 2 + 1, head_dim))

        self.W2_real = nn.Parameter(
            scale * torch.randn(num_modes_h, num_modes_w // 2 + 1, head_dim, head_dim)
        )
        self.W2_imag = nn.Parameter(
            scale * torch.randn(num_modes_h, num_modes_w // 2 + 1, head_dim, head_dim)
        )
        self.b2_real = nn.Parameter(torch.zeros(num_modes_h, num_modes_w // 2 + 1, head_dim))
        self.b2_imag = nn.Parameter(torch.zeros(num_modes_h, num_modes_w // 2 + 1, head_dim))

    def _complex_mul(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pointwise complex matrix-vector multiply per frequency bin.

        x: (B, mh, mw, C_in)
        w: (mh, mw, C_in, C_out)
        out: (B, mh, mw, C_out)
        """
        out_real = (
            torch.einsum("bmnc,mncd->bmnd", x_real, w_real)
            - torch.einsum("bmnc,mncd->bmnd", x_imag, w_imag)
        )
        out_imag = (
            torch.einsum("bmnc,mncd->bmnd", x_real, w_imag)
            + torch.einsum("bmnc,mncd->bmnd", x_imag, w_real)
        )
        return out_real, out_imag

    def _complex_gelu(
        self, x_real: torch.Tensor, x_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply GELU independently to real and imaginary parts."""
        return F.gelu(x_real), F.gelu(x_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, head_dim, H, W) — spatial features for this head
        Returns:
            out: (B, head_dim, H, W)
        """
        B, C, H, W = x.shape

        # 2D real FFT: (B, C, H, W//2+1) complex
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Truncate to num_modes
        mh = min(self.num_modes_h, H)
        mw = min(self.num_modes_w // 2 + 1, W // 2 + 1)

        # Extract truncated spectrum: (B, mh, mw, C)
        x_ft_trunc = x_ft[:, :, :mh, :mw].permute(0, 2, 3, 1)
        xr = x_ft_trunc.real   # (B, mh, mw, C)
        xi = x_ft_trunc.imag

        # Layer 1: W1 * x + b1
        h1r, h1i = self._complex_mul(xr, xi, self.W1_real[:mh, :mw], self.W1_imag[:mh, :mw])
        h1r = h1r + self.b1_real[:mh, :mw]
        h1i = h1i + self.b1_imag[:mh, :mw]

        # Activation
        h1r, h1i = self._complex_gelu(h1r, h1i)

        # Layer 2: W2 * h1 + b2
        h2r, h2i = self._complex_mul(h1r, h1i, self.W2_real[:mh, :mw], self.W2_imag[:mh, :mw])
        h2r = h2r + self.b2_real[:mh, :mw]
        h2i = h2i + self.b2_imag[:mh, :mw]

        # Reconstruct full spectrum (zero-pad high frequencies)
        out_ft = torch.zeros(B, H, W // 2 + 1, C, dtype=torch.cfloat, device=x.device)
        out_ft[:, :mh, :mw, :] = torch.complex(h2r, h2i)

        # Inverse FFT: (B, C, H, W)
        out_ft = out_ft.permute(0, 3, 1, 2)
        out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return out


class PatchEmbedding(nn.Module):
    """Patchification layer with learnable positional encodings.

    Applies a Conv2D with kernel=patch_size, stride=patch_size to map
    (B, C_in, H, W) → (B, embed_dim, H/P, W/P).

    Positional encoding p_{i,j}^t = W_p(x_i, y_j, t) is implemented as a
    learnable linear projection of (x, y, t) coordinates.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        num_timesteps: int,
        spatial_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps

        # Convolutional patch projection P
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable positional encoding W_p ∈ R^{embed_dim × 3}
        # Maps (x_i, y_j, t) → embed_dim
        self.pos_proj = nn.Linear(3, embed_dim)

        # Pre-compute normalized grid coordinates
        num_patches = spatial_size // patch_size
        xs = torch.linspace(0, 1, num_patches)
        ys = torch.linspace(0, 1, num_patches)
        grid_y, grid_x = torch.meshgrid(xs, ys, indexing="ij")
        # (H/P, W/P, 2)
        self.register_buffer("grid_xy", torch.stack([grid_x, grid_y], dim=-1))

    def forward(self, u_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Args:
            u_t: (B, C, H, W) — spatial field at timestep t
            t:   integer timestep index (0-based)
        Returns:
            Z_p_t: (B, embed_dim, H/P, W/P)
        """
        B = u_t.shape[0]
        H_p = u_t.shape[2] // self.patch_size
        W_p = u_t.shape[3] // self.patch_size

        # Positional encoding: (H/P, W/P, 3) → (H/P, W/P, embed_dim)
        t_norm = t / max(self.num_timesteps - 1, 1)
        t_coord = torch.full(
            (H_p, W_p, 1), t_norm, device=u_t.device, dtype=u_t.dtype
        )
        coords = torch.cat([self.grid_xy.to(u_t.dtype), t_coord], dim=-1)  # (H/P, W/P, 3)
        pos_enc = self.pos_proj(coords)  # (H/P, W/P, embed_dim)
        pos_enc = pos_enc.permute(2, 0, 1).unsqueeze(0)  # (1, embed_dim, H/P, W/P)

        # Patch projection
        Z = self.proj(u_t) + pos_enc  # (B, embed_dim, H/P, W/P)
        return Z


class TemporalAggregation(nn.Module):
    """Temporal aggregation layer.

    Aggregates T patch-embedded frames into a single feature map:
        z_agg = Σ_t W_t(z_p^t) * e^{-i*γ*t}

    Implemented as a learnable weighted sum with Fourier features.
    W_t is a per-timestep linear projection; γ ∈ R^C is a learnable frequency.
    The output is the real part of the complex sum.
    """

    def __init__(self, embed_dim: int, num_timesteps: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps

        # Per-timestep linear projections W_t: (embed_dim → embed_dim)
        self.W_t = nn.ModuleList(
            [nn.Conv2d(embed_dim, embed_dim, kernel_size=1) for _ in range(num_timesteps)]
        )

        # Learnable Fourier frequency γ ∈ R^{embed_dim}
        self.gamma = nn.Parameter(torch.randn(embed_dim) * 0.1)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_features: list) -> torch.Tensor:
        """
        Args:
            patch_features: list of T tensors, each (B, embed_dim, H/P, W/P)
        Returns:
            z_agg: (B, embed_dim, H/P, W/P)
        """
        T = len(patch_features)
        B, C, Hp, Wp = patch_features[0].shape

        agg_real = torch.zeros(B, C, Hp, Wp, device=patch_features[0].device,
                               dtype=patch_features[0].dtype)
        agg_imag = torch.zeros_like(agg_real)

        for t, z_t in enumerate(patch_features):
            Wz = self.W_t[t](z_t)  # (B, C, Hp, Wp)
            # e^{-i*γ*t}: γ shape (C,) → broadcast over (B, C, Hp, Wp)
            angle = self.gamma * t  # (C,)
            cos_t = torch.cos(angle).view(1, C, 1, 1)
            sin_t = torch.sin(angle).view(1, C, 1, 1)
            agg_real = agg_real + Wz * cos_t
            agg_imag = agg_imag - Wz * sin_t  # e^{-iγt} = cos - i*sin

        # Take real part and apply layer norm
        out = agg_real  # (B, C, Hp, Wp)
        # LayerNorm over channel dim
        out = out.permute(0, 2, 3, 1)  # (B, Hp, Wp, C)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # (B, C, Hp, Wp)
        return out
