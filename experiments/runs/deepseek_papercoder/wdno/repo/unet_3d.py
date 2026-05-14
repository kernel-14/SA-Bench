## unet_3d.py
"""
3D conditional U‑Net denoiser for the Wavelet Diffusion Neural Operator (WDNO).

Implements the architecture described in the paper’s Table 20 and config.yaml,
using 3D convolutions, time‑step conditioning, residual blocks, and multi‑head
self‑attention. The network operates on spatio‑temporal wavelet coefficient
tensors obtained from a 3D discrete wavelet transform (typically 8 sub‑bands).

Design:
  - Spatial‑only down‑/up‑sampling (temporal dimension kept unchanged).
  - Time embedding injection through scale‑and‑shift in residual blocks.
  - Attention at the coarsest spatial resolutions.
  - Condition information (e.g., initial density, control forces) is
    concatenated along the channel dimension before the first convolution.

The class accepts the hyper‑parameter dictionary from the `model_3d` section
of config.yaml and an optional time‑embedding dimension.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def adjust_shape_3d(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Pad or crop the spatial (last two) dimensions of `x` so that they exactly
    match the spatial dimensions of `target`.  The temporal dimension and
    batch/channel dimensions are left unchanged.

    Args:
        x:      Tensor of shape (B, C, T, H, W).
        target: Tensor of shape (B, C_s, T_t, H_t, W_t).

    Returns:
        Tensor of shape (B, C, T, H_t, W_t) if T == T_t.
    """
    _, _, T, H, W = x.shape
    _, _, T_t, H_t, W_t = target.shape

    # Temporal adjustment (should not be needed because we keep T the same)
    if T != T_t:
        raise ValueError(f"Temporal dimension mismatch: x has T={T}, target has T_t={T_t}")

    # Height
    if H_t > H:
        pad_h = (H_t - H) // 2
        pad_h_end = H_t - H - pad_h
        x = F.pad(x, (0, 0, pad_h, pad_h_end))  # padding order: last dim first, so (left, right, top, bottom)
    elif H_t < H:
        start_h = (H - H_t) // 2
        x = x[:, :, :, start_h : start_h + H_t, :]

    # Width
    if W_t > W:
        pad_w = (W_t - W) // 2
        pad_w_end = W_t - W - pad_w
        x = F.pad(x, (pad_w, pad_w_end, 0, 0))
    elif W_t < W:
        start_w = (W - W_t) // 2
        x = x[:, :, :, :, start_w : start_w + W_t]

    return x


# ----------------------------------------------------------------------
# Time embedding
# ----------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional encoding for diffusion time steps.
    Projects integer time step to a vector of length `dim`.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: (B,) tensor of integer time steps.

        Returns:
            (B, dim) sinusoidal embedding.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ----------------------------------------------------------------------
# Residual block with time conditioning
# ----------------------------------------------------------------------

class ResidualBlock3D(nn.Module):
    """
    3D residual block with time‑step conditioning.
    Two convolutions with GroupNorm, SiLU, and a time‑dependent scale‑shift.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int = 8,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        padding: Tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        groups1 = min(num_groups, in_channels)
        groups2 = min(num_groups, out_channels)

        self.norm1 = nn.GroupNorm(groups1, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
        )

        # Time embedding projection: produces scale and shift scalars
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * out_channels),
        )

        self.norm2 = nn.GroupNorm(groups2, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C_in, T, H, W) input features.
            t_emb: (B, time_emb_dim) time embedding.

        Returns:
            (B, C_out, T, H, W) output features.
        """
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # Scale and shift
        scale_shift = self.time_proj(t_emb)[:, :, None, None, None]  # (B, 2*C_out, 1, 1, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale) + shift

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


# ----------------------------------------------------------------------
# Multi‑head self‑attention block (3D version)
# ----------------------------------------------------------------------

class AttentionBlock3D(nn.Module):
    """
    Self‑attention block for 3D feature maps.
    Projects the channels to a lower embedding dimension, applies
    multi‑head scaled dot‑product attention, and projects back.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        head_dim: int = 32,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.embed_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.norm = nn.GroupNorm(1, channels)          # group norm with 1 group = layer norm
        self.qkv = nn.Conv3d(channels, self.embed_dim * 3, kernel_size=1)
        self.out_proj = nn.Conv3d(self.embed_dim, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) input feature map.

        Returns:
            (B, C, T, H, W) attended feature map.
        """
        B, C, T, H, W = x.shape

        h = self.norm(x)  # (B, C, T, H, W)

        # Project to Q, K, V and reshape for attention
        qkv = self.qkv(h)  # (B, 3*E, T, H, W)
        qkv = qkv.reshape(B, 3, self.embed_dim, T * H * W)  # (B, 3, E, N)
        qkv = qkv.permute(1, 0, 2, 3)                       # (3, B, E, N)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Reshape for multi‑head attention: (B, H, N, d)
        q = q.view(B, self.num_heads, self.head_dim, T * H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, T * H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, T * H * W).permute(0, 1, 3, 2)

        # Scaled dot‑product attention (PyTorch ≥ 2.0)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, num_heads, N, d)

        # Merge heads and project back to original channel
        out = out.permute(0, 1, 3, 2).reshape(B, self.embed_dim, T * H * W)
        out = out.view(B, self.embed_dim, T, H, W)
        return self.out_proj(out)  # (B, C, T, H, W)


# ----------------------------------------------------------------------
# Encoder block (down‑sampling)
# ----------------------------------------------------------------------

class DownBlock(nn.Module):
    """
    Down‑sampling stage consisting of two residual blocks, optional attention,
    and a stride‑2 spatial convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        num_heads: int,
        head_dim: int,
        use_attention: bool,
        time_emb_dim: int,
        kernel_size: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        downsample_kernel: Tuple[int, int, int],
        downsample_padding: Tuple[int, int, int],
        downsample_stride: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.resblock1 = ResidualBlock3D(
            in_channels, out_channels, time_emb_dim, num_groups, kernel_size, padding
        )
        self.resblock2 = ResidualBlock3D(
            out_channels, out_channels, time_emb_dim, num_groups, kernel_size, padding
        )
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionBlock3D(out_channels, num_heads, head_dim)
        self.downsample = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=downsample_kernel,
            stride=downsample_stride,
            padding=downsample_padding,
        )

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            skip:  feature map before down‑sampling (B, C_out, T, H, W).
            h:     down‑sampled feature map       (B, C_out, T, H/2, W/2).
        """
        h = self.resblock1(x, t_emb)
        h = self.resblock2(h, t_emb)
        if self.use_attention:
            h = self.attn(h)
        skip = h
        h = self.downsample(h)
        return skip, h


# ----------------------------------------------------------------------
# Decoder block (up‑sampling)
# ----------------------------------------------------------------------

class UpBlock(nn.Module):
    """
    Up‑sampling stage consisting of a nearest‑neighbour up‑sampling followed
    by a convolution, concatenation with the skip connection, two residual
    blocks, and optional attention.
    """

    def __init__(
        self,
        in_channels: int,          # channels from previous decoder layer
        out_channels: int,         # desired channels after this block
        skip_channels: int,        # channels of the skip connection
        num_groups: int,
        num_heads: int,
        head_dim: int,
        use_attention: bool,
        time_emb_dim: int,
        kernel_size: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        upsample_kernel: Tuple[int, int, int],
        upsample_padding: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        # Upsample spatially (2×) using nearest neighbour + conv
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.conv_up = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=upsample_kernel,
            padding=upsample_padding,
            stride=1,
        )
        # After concatenation, input channels = in_channels + skip_channels
        total_in = in_channels + skip_channels
        self.resblock1 = ResidualBlock3D(
            total_in, out_channels, time_emb_dim, num_groups, kernel_size, padding
        )
        self.resblock2 = ResidualBlock3D(
            out_channels, out_channels, time_emb_dim, num_groups, kernel_size, padding
        )
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionBlock3D(out_channels, num_heads, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, C_in, T_in, H_in, W_in) from previous block.
            skip: (B, C_skip, T_skip, H_skip, W_skip) from encoder.
            t_emb: time embedding.

        Returns:
            (B, C_out, T_skip, H_skip, W_skip) up‑sampled output.
        """
        x = self.upsample(x)                # spatial size roughly doubled, T unchanged
        x = adjust_shape_3d(x, skip)        # ensure exact spatial match
        x = self.conv_up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.resblock1(x, t_emb)
        x = self.resblock2(x, t_emb)
        if self.use_attention:
            x = self.attn(x)
        return x


# ----------------------------------------------------------------------
# Bottleneck block (middle of U‑Net)
# ----------------------------------------------------------------------

class BottleneckBlock(nn.Module):
    """
    Central block operating at the coarsest spatial resolution.
    Two residual blocks with an attention block in between.
    """

    def __init__(
        self,
        channels: int,
        num_groups: int,
        num_heads: int,
        head_dim: int,
        time_emb_dim: int,
        kernel_size: Tuple[int, int, int],
        padding: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.resblock1 = ResidualBlock3D(
            channels, channels, time_emb_dim, num_groups, kernel_size, padding
        )
        self.attn = AttentionBlock3D(channels, num_heads, head_dim)
        self.resblock2 = ResidualBlock3D(
            channels, channels, time_emb_dim, num_groups, kernel_size, padding
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.resblock1(x, t_emb)
        x = self.attn(x)
        x = self.resblock2(x, t_emb)
        return x


# ----------------------------------------------------------------------
# Full 3D U‑Net
# ----------------------------------------------------------------------

class UNet3D(nn.Module):
    """
    3D U‑Net denoising model for WDNO (2D incompressible fluid, ERA5).

    Construction parameters are read from a dictionary that corresponds to the
    ``model_3d`` section of config.yaml.  The total number of input channels
    (noisy wavelet + condition) is obtained from that config as well.

    Attributes:
        in_channels (int): Total input channels (noise + condition).
        out_channels (int): Number of noisy wavelet channels (typically 8).
        initial_dim (int): Base feature dimension.
        dim_mults (List[int]): Channel multipliers per resolution level.
        time_emb_dim (int): Dimension of the time embedding vector.
        # Derived
        cond_channels (int): in_channels - out_channels, i.e. number of condition channels.

    Methods:
        forward(x, t, cond=None) -> Tensor:
            x:    (B, out_channels, T, H, W) noisy wavelet.
            t:    (B,) integer time steps.
            cond: optional (B, cond_channels, T, H, W) condition, concatenated.
            Returns: (B, out_channels, T, H, W) predicted noise.
    """

    def __init__(
        self,
        params: dict,
        time_emb_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            params: Dictionary containing the model_3d configuration.
                    Expected keys:
                        - in_channels (int)
                        - out_channels (int)
                        - dim (int)                  -> initial_dim
                        - dim_mults (List[int])
                        - kernel_size (List[int])
                        - padding (List[int])
                        - resnet_groups (int)
                        - attention_heads (int)
                        - attention_dim (int)
                        - downsample_kernel (List[int])
                        - downsample_padding (List[int])
                        - downsample_stride (List[int])
                        - upsample_kernel (List[int])
                        - upsample_padding (List[int])
            time_emb_dim: Optional dimension for the time embedding. If None,
                          it defaults to 4 * initial_dim.
        """
        super().__init__()

        # Unpack required parameters from config dictionary
        try:
            in_channels  = params["in_channels"]
            out_channels = params["out_channels"]
            initial_dim  = params["dim"]
            dim_mults    = params["dim_mults"]
            kernel_size  = tuple(params["kernel_size"])
            padding      = tuple(params["padding"])
            num_groups   = params["resnet_groups"]
            num_heads    = params["attention_heads"]
            head_dim     = params["attention_dim"]
            ds_kernel    = tuple(params["downsample_kernel"])
            ds_padding   = tuple(params["downsample_padding"])
            ds_stride    = tuple(params["downsample_stride"])
            us_kernel    = tuple(params["upsample_kernel"])
            us_padding   = tuple(params["upsample_padding"])
        except KeyError as e:
            raise KeyError(f"Missing key in model_3d configuration: {e}") from e

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_dim = initial_dim
        self.dim_mults = dim_mults
        if time_emb_dim is None:
            time_emb_dim = initial_dim * 4
        self.time_emb_dim = time_emb_dim
        self.cond_channels = in_channels - out_channels

        # ---------- Time embedding ----------
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ---------- Initial convolution ----------
        self.init_conv = nn.Conv3d(
            in_channels, initial_dim,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
        )

        # ---------- Encoder (down‑sampling path) ----------
        self.encoder = nn.ModuleList()
        in_ch = initial_dim
        for i, mult in enumerate(dim_mults):
            out_ch = initial_dim * mult
            use_attn = mult >= 4  # attention at coarsest spatial resolutions
            self.encoder.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    num_groups=num_groups,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    use_attention=use_attn,
                    time_emb_dim=time_emb_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    downsample_kernel=ds_kernel,
                    downsample_padding=ds_padding,
                    downsample_stride=ds_stride,
                )
            )
            in_ch = out_ch

        # ---------- Bottleneck ----------
        mid_ch = initial_dim * dim_mults[-1]
        self.bottleneck = BottleneckBlock(
            channels=mid_ch,
            num_groups=num_groups,
            num_heads=num_heads,
            head_dim=head_dim,
            time_emb_dim=time_emb_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

        # ---------- Decoder (up‑sampling path) ----------
        # Channel sizes for skip connections (from last encoder to first)
        skip_channels = [initial_dim * m for m in dim_mults[::-1]]
        # Output mults: drop the highest, repeat the smallest for the last stage
        out_multipliers = dim_mults[-2::-1] + [dim_mults[0]]

        self.decoder = nn.ModuleList()
        previous_out_ch = mid_ch
        for skip_ch, out_mult in zip(skip_channels, out_multipliers):
            out_ch = initial_dim * out_mult
            use_attn = out_mult >= 4
            self.decoder.append(
                UpBlock(
                    in_channels=previous_out_ch,
                    out_channels=out_ch,
                    skip_channels=skip_ch,
                    num_groups=num_groups,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    use_attention=use_attn,
                    time_emb_dim=time_emb_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    upsample_kernel=us_kernel,
                    upsample_padding=us_padding,
                )
            )
            previous_out_ch = out_ch

        # ---------- Final output layer ----------
        final_groups = min(8, initial_dim)
        self.final_norm = nn.GroupNorm(final_groups, initial_dim)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv3d(
            initial_dim, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Denoise the input wavelet coefficients.

        Args:
            x:    (B, out_channels, T, H, W) noisy wavelet coefficients.
            t:    (B,) integer time steps.
            cond: (B, cond_channels, T, H, W) optional condition tensor.
                  If provided, it is concatenated along the channel dimension
                  with `x`. The total channels must match `self.in_channels`.

        Returns:
            (B, out_channels, T, H, W) predicted noise.
        """
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        elif self.cond_channels > 0:
            raise ValueError(
                f"Condition channels expected ({self.cond_channels}) but cond is None. "
                "Provide a zero tensor for unconditional generation."
            )

        t_emb = self.time_mlp(t)

        h = self.init_conv(x)

        skips = []
        for down in self.encoder:
            skip, h = down(h, t_emb)
            skips.append(skip)

        h = self.bottleneck(h, t_emb)

        skips = reversed(skips)
        for up in self.decoder:
            skip = next(skips)
            h = up(h, skip, t_emb)

        h = self.final_norm(h)
        h = self.final_act(h)
        h = self.final_conv(h)
        return h


# ----------------------------------------------------------------------
# Quick sanity check (can be run independently)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate a config dict similar to config.yaml model_3d
    test_config = {
        "in_channels": 44,   # e.g., noise (32) + condition (12) for 2D fluid sim
        "out_channels": 32,  # noise channels (8 subbands × 4 physical variables)
        "dim": 64,
        "dim_mults": [1, 2, 4],
        "kernel_size": [3, 3, 3],
        "padding": [1, 1, 1],
        "resnet_groups": 8,
        "attention_heads": 4,
        "attention_dim": 32,
        "downsample_kernel": [1, 4, 4],
        "downsample_padding": [0, 1, 1],
        "downsample_stride": [1, 2, 2],
        "upsample_kernel": [3, 3, 3],
        "upsample_padding": [1, 1, 1],
    }

    model = UNet3D(test_config, time_emb_dim=256).to(device)

    batch_size = 2
    T, H, W = 18, 34, 34
    x_noisy = torch.randn(batch_size, 32, T, H, W, device=device)
    cond = torch.randn(batch_size, 12, T, H, W, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)

    out = model(x_noisy, t, cond)
    print(f"Output shape: {out.shape}")  # should be (2, 32, 18, 34, 34)

    # Test missing condition (should raise ValueError)
    try:
        model(x_noisy, t, None)
    except ValueError as e:
        print(f"Expected error: {e}")

    print("UNet3D sanity check passed.")
