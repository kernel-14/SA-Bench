## unet_2d.py
"""
2D conditional U‑Net denoiser for the Wavelet Diffusion Neural Operator (WDNO).

Implements the architecture described in the paper’s Table 18, using
GroupNorm, SiLU activations, residual blocks, multi‑head self‑attention
(at the two coarsest resolutions), and time‑step conditioning.

The network expects the noisy wavelet coefficients concatenated with the
(optional) conditioning tensor as a single input `x`. During training,
classifier‑free guidance is achieved by replacing the conditioning channels
with zeros; the caller must handle this concatenation externally.

Configuration parameters are read from the project’s `config.yaml` via the
`Config` object, but this module is independent: it simply takes the
hyper‑parameters as constructor arguments.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Helper function
# ----------------------------------------------------------------------

def adjust_shape(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Pad or crop the spatial dimensions of `x` so that they exactly match
    the spatial dimensions of `target`.  Used to align up‑sampled feature
    maps with skip connections when the spatial size is odd.

    Args:
        x:      Tensor of shape (B, C, H, W).
        target: Tensor of shape (B, C_s, H_t, W_t).

    Returns:
        Tensor of shape (B, C, H_t, W_t).
    """
    _, _, H, W = x.shape
    _, _, H_t, W_t = target.shape

    # Height
    if H_t > H:
        pad_h = (H_t - H) // 2
        pad_h_end = H_t - H - pad_h
        x = F.pad(x, (0, 0, pad_h, pad_h_end))
    elif H_t < H:
        start_h = (H - H_t) // 2
        x = x[:, :, start_h : start_h + H_t, :]

    # Width
    if W_t > W:
        pad_w = (W_t - W) // 2
        pad_w_end = W_t - W - pad_w
        x = F.pad(x, (pad_w, pad_w_end, 0, 0))
    elif W_t < W:
        start_w = (W - W_t) // 2
        x = x[:, :, :, start_w : start_w + W_t]

    return x


# ----------------------------------------------------------------------
# Time embedding
# ----------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional encoding adapted from "Attention Is All You Need".
    Projects each time step index to a vector of length `dim`.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: (batch,) tensor of integer time steps.

        Returns:
            (batch, dim) sinusoidal embedding.
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
# Residual block
# ----------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Two‑convolutions residual block with time embedding injection.
    The time embedding is projected and added as a bias after the first conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int = 8,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        groups1 = min(num_groups, in_channels)
        groups2 = min(num_groups, out_channels)

        self.norm1 = nn.GroupNorm(groups1, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(groups2, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C_in, H, W) input features.
            t_emb: (B, time_emb_dim) time embedding.

        Returns:
            (B, C_out, H, W) output features.
        """
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # Add time embedding as channel‑wise bias
        h = h + self.time_proj(t_emb)[:, :, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


# ----------------------------------------------------------------------
# Multi‑head self‑attention block
# ----------------------------------------------------------------------

class AttentionBlock(nn.Module):
    """
    Self‑attention block for 2D feature maps.  Projects the channels to an
    internal embedding dimension, applies multi‑head scaled‑dot‑product
    attention, and projects back.
    """

    def __init__(self, channels: int, num_heads: int = 4, head_dim: int = 32) -> None:
        super().__init__()
        self.channels = channels
        self.embed_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Normalisation first
        self.norm = nn.GroupNorm(1, channels)

        # 1×1 convs to project to/from the embedding space
        self.qkv = nn.Conv2d(channels, self.embed_dim * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(self.embed_dim, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input feature map.

        Returns:
            (B, C, H, W) attended feature map.
        """
        B, C, H, W = x.shape

        h = self.norm(x)

        # Project to Q, K, V and reshape for attention
        qkv = self.qkv(h).reshape(B, 3, self.embed_dim, H * W)  # (B,3,E,N)
        qkv = qkv.permute(1, 0, 2, 3)                            # (3,B,E,N)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Reshape for multi‑head attention: (B, H, N, d)
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        # Scaled dot‑product attention (PyTorch >= 2.0)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, H, N, d)

        # Merge heads and project back
        out = out.permute(0, 1, 3, 2).reshape(B, self.embed_dim, H * W)
        out = out.view(B, self.embed_dim, H, W)
        return self.out_proj(out)


# ----------------------------------------------------------------------
# Down‑sampling block
# ----------------------------------------------------------------------

class DownBlock(nn.Module):
    """
    Down‑sampling stage: residual block → (optional attention) → stride‑2 conv.
    Returns the features before down‑sampling (skip connection) and the
    down‑sampled output.
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
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.resblock = ResidualBlock(
            in_channels, out_channels, time_emb_dim, num_groups, kernel_size
        )
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionBlock(out_channels, num_heads, head_dim)
        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Returns:
            skip:  features before down‑sampling (B, C_out, H, W).
            h:     features after down‑sampling   (B, C_out, H/2, W/2).
        """
        h = self.resblock(x, t_emb)
        if self.use_attention:
            h = self.attn(h)
        skip = h
        h = self.downsample(h)
        return skip, h


# ----------------------------------------------------------------------
# Up‑sampling block
# ----------------------------------------------------------------------

class UpBlock(nn.Module):
    """
    Up‑sampling stage: nearest up‑sample → conv → concatenate skip → residual
    block → (optional attention).  The spatial size is adjusted to match the
    skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        num_groups: int,
        num_heads: int,
        head_dim: int,
        use_attention: bool,
        time_emb_dim: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_up = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.resblock = ResidualBlock(
            in_channels + skip_channels,
            out_channels,
            time_emb_dim,
            num_groups,
            kernel_size,
        )
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionBlock(out_channels, num_heads, head_dim)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, C_in, H_in, W_in) from previous block.
            skip: (B, C_skip, H_skip, W_skip) from encoder.
            t_emb: time embedding.

        Returns:
            (B, C_out, H_skip, W_skip) up‑sampled output.
        """
        x = self.upsample(x)
        x = adjust_shape(x, skip)
        x = self.conv_up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.resblock(x, t_emb)
        if self.use_attention:
            x = self.attn(x)
        return x


# ----------------------------------------------------------------------
# Full U‑Net 2D
# ----------------------------------------------------------------------

class UNet2D(nn.Module):
    """
    2D U‑Net denoising model for WDNO.

    Parameters are designed to match Table 18 of the paper and the
    `config.yaml` model_base section.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        initial_dim: int = 128,
        dim_mults: Optional[List[int]] = None,
        resnet_groups: int = 8,
        attention_heads: int = 4,
        attention_dim: int = 32,
        time_emb_dim: Optional[int] = None,
        kernel_size: int = 3,
    ) -> None:
        """
        Args:
            in_channels:     Total number of input channels (noisy wavelet
                             plus concatenated condition).
            out_channels:    Number of output channels (equal to wavelet
                             subbands, usually 4).
            initial_dim:     Base feature dimension (128).
            dim_mults:       Channel multipliers per resolution level
                             (default [1, 2, 4, 8]).
            resnet_groups:   Group norm groups.
            attention_heads: Number of attention heads.
            attention_dim:   Per‑head dimension (so embed_dim = heads*dim).
            time_emb_dim:    If None, defaults to initial_dim * 4.
            kernel_size:     Convolution kernel size.
        """
        super().__init__()

        if dim_mults is None:
            dim_mults = [1, 2, 4, 8]
        if time_emb_dim is None:
            time_emb_dim = initial_dim * 4

        self.time_emb_dim = time_emb_dim
        self.initial_dim = initial_dim
        self.out_channels = out_channels
        self.cond_channels = in_channels - out_channels  # deduced

        # ---------- Time embedding ----------
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ---------- Initial convolution ----------
        self.init_conv = nn.Conv2d(
            in_channels, initial_dim, kernel_size=kernel_size, padding=kernel_size // 2
        )

        # ---------- Down‑sampling path ----------
        self.downs = nn.ModuleList()
        in_ch = initial_dim
        for mult in dim_mults:
            out_ch = initial_dim * mult
            use_attn = mult >= 4  # attention at the two coarsest levels
            self.downs.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    resnet_groups,
                    attention_heads,
                    attention_dim,
                    use_attn,
                    time_emb_dim,
                    kernel_size,
                )
            )
            in_ch = out_ch

        # ---------- Middle block ----------
        mid_ch = initial_dim * dim_mults[-1]
        self.mid_block1 = ResidualBlock(
            mid_ch, mid_ch, time_emb_dim, resnet_groups, kernel_size
        )
        self.mid_attn = AttentionBlock(mid_ch, attention_heads, attention_dim)
        self.mid_block2 = ResidualBlock(
            mid_ch, mid_ch, time_emb_dim, resnet_groups, kernel_size
        )

        # ---------- Up‑sampling path ----------
        # Skip channels from down blocks (reverse order)
        skip_channels = [initial_dim * m for m in dim_mults[::-1]]
        # Output multipliers: drop the highest, then repeat the smallest
        out_multipliers = dim_mults[-2::-1] + [dim_mults[0]]

        self.ups = nn.ModuleList()
        previous_out_ch = mid_ch
        for skip_ch, out_mult in zip(skip_channels, out_multipliers):
            out_ch = initial_dim * out_mult
            use_attn = out_mult >= 4
            self.ups.append(
                UpBlock(
                    previous_out_ch,
                    out_ch,
                    skip_ch,
                    resnet_groups,
                    attention_heads,
                    attention_dim,
                    use_attn,
                    time_emb_dim,
                    kernel_size,
                )
            )
            previous_out_ch = out_ch

        # ---------- Final output ----------
        final_groups = min(8, initial_dim)
        self.final_norm = nn.GroupNorm(final_groups, initial_dim)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(
            initial_dim, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, C_noisy, H, W) noisy wavelet coefficients.
            t:    (B,) integer time steps.
            cond: (B, C_cond, H, W) conditioning coefficients.
                  If provided, it is concatenated to `x`.  If ``None``
                  and ``self.cond_channels > 0``, a ``ValueError`` is raised.

        Returns:
            (B, out_channels, H, W) predicted noise.
        """
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        elif self.cond_channels > 0:
            raise ValueError(
                "Condition expected (cond_channels > 0) but cond is None. "
                "Provide a zero tensor for unconditional generation."
            )

        t_emb = self.time_mlp(t)

        h = self.init_conv(x)

        skips = []
        for down in self.downs:
            skip, h = down(h, t_emb)
            skips.append(skip)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip, t_emb)

        h = self.final_norm(h)
        h = self.final_act(h)
        h = self.final_conv(h)
        return h


# ----------------------------------------------------------------------
# Quick sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage mimicking 1D Burgers simulation (10 input channels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet2D(
        in_channels=10,
        out_channels=4,
        initial_dim=128,
        dim_mults=[1, 2, 4, 8],
        resnet_groups=8,
        attention_heads=4,
        attention_dim=32,
        kernel_size=3,
    ).to(device)

    batch_size = 2
    H, W = 41, 60  # typical wavelet coefficient spatial size for 1D Burgers
    x_noisy = torch.randn(batch_size, 4, H, W, device=device)
    cond = torch.randn(batch_size, 6, H, W, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)

    out = model(x_noisy, t, cond)
    print(f"Output shape: {out.shape}")  # should be (2, 4, 41, 60)

    # Test unconditional (cond=None -> error unless cond_channels=0)
    try:
        model(x_noisy, t, None)
    except ValueError as e:
        print(f"Expected error: {e}")

    print("UNet2D sanity check passed.")
