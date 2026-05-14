"""
model.py – Consistency model with EDM‑style SongUNet backbone.

This module defines the ConsistencyModel class, which wraps a SongUNet 
and implements the skip‑ / output‑scaling parametrization described in the paper:
    f_θ(x, σ) = c_skip(σ) * x + c_out(σ) * F_θ(x, σ)

The architecture follows Karras et al. (2022) and is identical to the one used 
in Song & Dhariwal (2024). Configuration parameters are taken from config.yaml.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Helper modules for the U‑Net
# ---------------------------------------------------------------------------

class FourierEmbedding(nn.Module):
    """
    Maps a noise level σ ∈ ℝ⁺ to an embedding vector using random Fourier
    features followed by a small MLP, as in EDM.

    Args:
        embed_dim:   dimension of the output embedding vector.
        log_sigma:   if True, apply log to the input σ before the Fourier 
                     features. In the paper, sigma is used directly, but 
                     logarithm is a common stabilisation (optional).
        num_frequencies: half the number of random frequencies; the total 
                     Fourier feature dimension is 2 * num_frequencies.
    """
    def __init__(
        self,
        embed_dim: int = 512,
        log_sigma: bool = True,
        num_frequencies: int = 128,
    ) -> None:
        super().__init__()
        self.log_sigma = log_sigma
        # Random frequencies (fixed, not learned)
        self.frequencies = nn.Parameter(
            torch.randn(1, num_frequencies) * 2.0,
            requires_grad=False,
        )
        # MLP: Fourier cat -> hidden -> embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_frequencies, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, sigma: Tensor) -> Tensor:
        """
        Args:
            sigma: (B,) tensor of noise levels.

        Returns:
            emb: (B, embed_dim) tensor.
        """
        if self.log_sigma:
            sigma = torch.log(sigma + 1e-5)             # avoid log(0)

        # Compute Fourier features
        sigma = sigma.view(-1, 1)                       # (B, 1)
        freq = self.frequencies.to(sigma.dtype)         # (1, F)
        x = 2.0 * torch.pi * sigma @ freq               # (B, F)
        fourier = torch.cat([torch.sin(x), torch.cos(x)], dim=1)  # (B, 2F)

        emb = self.mlp(fourier)                         # (B, embed_dim)
        return emb


class ResBlock(nn.Module):
    """
    Residual block with conditioning on a global embedding vector.
    Used in both down‑ and up‑sampling stages of SongUNet.

    Structure (EDM style):
        GroupNorm(32, in_ch) → add emb → SiLU → conv1 →
        GroupNorm(32, out_ch) → SiLU → dropout → conv2
        (+ skip connection (1×1 conv) if in_ch != out_ch)

    Args:
        in_ch:    number of input channels.
        out_ch:   number of output channels.
        emb_dim:  dimension of the conditioning embedding.
        dropout:  dropout rate applied before the second convolution.
    """
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(32, in_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch, bias=False) if out_ch > 0 else None
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Skip connection (1×1 conv) when channel count differs
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Args:
            x:   input feature map (B, in_ch, H, W).
            emb: conditioning embedding (B, emb_dim).

        Returns:
            output: (B, out_ch, H, W).
        """
        # First block
        h = self.norm1(x)
        if self.emb_proj is not None:
            # Project embedding to bias (broadcasted to H, W)
            emb_bias = self.emb_proj(F.silu(emb))[:, :, None, None]  # (B,out_ch,1,1)
            h = h + emb_bias
        h = F.silu(h)
        h = self.conv1(h)

        # Second block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Self‑attention block for 2D feature maps.

    Input is (B, C, H, W). The block applies GroupNorm(32), then computes
    Q, K, V projections and performs scaled dot‑product attention.

    Args:
        ch: number of feature channels.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Linear(ch, 3 * ch)
        self.proj = nn.Linear(ch, ch)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        # Normalise
        h = self.norm(x)
        # Flatten spatial dims → (B, C, H*W) → transpose to (B, N, C)
        h = h.view(B, C, -1).transpose(1, 2)   # (B, N, C)
        # QKV projection
        qkv = self.qkv(h)                      # (B, N, 3C)
        q, k, v = qkv.chunk(3, dim=-1)        # each (B, N, C)

        # Scaled dot‑product attention
        scale = C ** -0.5
        attn = torch.bmm(q.softmax(dim=-1) * scale, k.transpose(1, 2))  # (B, N, N)
        h = torch.bmm(attn, v)                 # (B, N, C)
        h = self.proj(h)
        # Reshape back to (B, C, H, W)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return x + h


# ---------------------------------------------------------------------------
# SongUNet (internal class)
# ---------------------------------------------------------------------------

class SongUNet(nn.Module):
    """
    The full U‑Net backbone used by EDM / consistency models.

    Architecture overview:
        - conv_in (3 -> model_channels)
        - embedding of σ
        - alternating down‑sampling stages (residual blocks + optional attention)
        - middle stage (residual blocks + attention)
        - up‑sampling stages with skip connections
        - final GroupNorm + SiLU + conv_out

    Args:
        img_channels:        number of input image channels (3).
        model_channels:      base channel width (e.g. 128).
        num_blocks:          number of residual blocks per resolution stage.
                             Can be an int (same for all) or a list.
        channel_mult:        multiplicative factors for successive stages.
        attn_resolutions:    list of spatial sizes where self‑attention is inserted.
        dropout:             dropout rate(s). Can be a float or a list of
                             per‑resolution values.
        emb_dim:             dimension of the sigma embedding (used by ResBlocks).
        use_attention:       whether to insert attention layers (default: True).
    """
    def __init__(
        self,
        img_channels: int = 3,
        model_channels: int = 128,
        num_blocks: Union[int, List[int]] = 3,
        channel_mult: List[int] = [1, 2, 2],
        attn_resolutions: List[int] = [],
        dropout: Union[float, List[float]] = 0.0,
        emb_dim: int = 512,
        use_attention: bool = True,
    ) -> None:
        super().__init__()

        # ----------------------------------------------------------------
        # Normalise num_blocks and dropout to per‑resolution lists
        # ----------------------------------------------------------------
        n_stages = len(channel_mult)
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks] * n_stages
        else:
            assert len(num_blocks) == n_stages, \
                f"len(num_blocks)={len(num_blocks)} != n_stages={n_stages}"

        if isinstance(dropout, (int, float)):
            dropout = [float(dropout)] * n_stages
        else:
            assert len(dropout) == n_stages, \
                f"len(dropout)={len(dropout)} != n_stages={n_stages}"

        # Channel dimension at each resolution level
        stage_ch = [model_channels * m for m in channel_mult]  # e.g. [128, 256, 256]

        # ----------------------------------------------------------------
        # Input convolution
        # ----------------------------------------------------------------
        self.conv_in = nn.Conv2d(img_channels, model_channels, 3, padding=1)

        # ----------------------------------------------------------------
        # Sigma embedding
        # ----------------------------------------------------------------
        self.sigma_embed = FourierEmbedding(embed_dim=emb_dim)

        # ----------------------------------------------------------------
        # Down‑sampling stages
        # ----------------------------------------------------------------
        self.down_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()   # for storing (optionally attention) + downsample
        in_ch = model_channels
        current_res = None  # will be set in forward pass

        for i, out_ch in enumerate(stage_ch):
            blocks = nn.ModuleList()
            for _ in range(num_blocks[i]):
                blocks.append(ResBlock(in_ch, out_ch, emb_dim, dropout[i]))
                in_ch = out_ch
            # Attention after the last block of this stage if resolution matches
            if use_attention and any(r == current_res for r in attn_resolutions):
                blocks.append(AttentionBlock(out_ch))
            self.downs.append(blocks)

            # Down‑sampling (except after the last down stage)
            if i < n_stages - 1:
                self.downs.append(nn.Conv2d(out_ch, out_ch, 2, stride=2, padding=0))

        # ----------------------------------------------------------------
        # Middle stage
        # ----------------------------------------------------------------
        mid_ch = stage_ch[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, emb_dim, dropout[-1])
        if use_attention:
            self.mid_attn = AttentionBlock(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, emb_dim, dropout[-1])

        # ----------------------------------------------------------------
        # Up‑sampling stages
        # ----------------------------------------------------------------
        self.up_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(n_stages - 1, -1, -1):
            out_ch = stage_ch[i]
            # If there is a down stage above, the skip connection will double channels
            skip_ch = stage_ch[i] if i > 0 else model_channels

            # Upsampling
            if i < n_stages - 1:
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    )
                )
                in_ch = out_ch + skip_ch
            # Residual blocks
            blocks = nn.ModuleList()
            for j in range(num_blocks[i]):
                # First block reduces channels if j==0 and skip is present
                block_out = out_ch if (j == num_blocks[i] - 1) else out_ch
                blocks.append(ResBlock(in_ch, block_out, emb_dim, dropout[i]))
                in_ch = block_out
            # Attention after last block if needed
            if use_attention and any(r == current_res for r in attn_resolutions):
                blocks.append(AttentionBlock(out_ch))
            self.up_blocks.append(blocks)

        # ----------------------------------------------------------------
        # Output convolution (final layer before consistency parametrization)
        # ----------------------------------------------------------------
        self.norm_out = nn.GroupNorm(32, out_ch)
        self.conv_out = nn.Conv2d(out_ch, img_channels, 3, padding=1)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """
        Args:
            x:     input images (B, C, H, W), range around [-1, 1].
            sigma: noise levels (B,).

        Returns:
            F_out: base UNet output (B, C, H, W).
        """
        # Embedding
        emb = self.sigma_embed(sigma)                  # (B, emb_dim)
        # Input convolution
        h = self.conv_in(x)                            # (B, model_channels, H, W)

        # Down‑sampling path with skip connections
        skips = []
        down_idx = 0
        for module in self.downs:
            if isinstance(module, nn.ModuleList):      # blocks + optional attention
                for block in module:
                    if isinstance(block, ResBlock):
                        h = block(h, emb)
                    else:                             # AttentionBlock
                        h = block(h)
                if down_idx < len(stage_ch) - 1:      # only save skip if not last down stage
                    skips.append(h)
                down_idx += 1
            else:                                     # downsample conv
                h = module(h)

        # Middle
        h = self.mid_block1(h, emb)
        if hasattr(self, 'mid_attn'):
            h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        # Up‑sampling path
        up_idx = 0
        skip_idx = len(skips) - 1
        for module in self.up_blocks:
            # Upsample (if not the first up block, i.e., we are after the middle)
            if up_idx < len(self.ups):
                h = self.ups[up_idx](h)
                up_idx += 1
            # Concatenate skip connection
            if skip_idx >= 0:
                h = torch.cat([h, skips[skip_idx]], dim=1)
                skip_idx -= 1
            # Blocks
            for block in module:
                if isinstance(block, ResBlock):
                    h = block(h, emb)
                else:
                    h = block(h)

        # Final output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


# ---------------------------------------------------------------------------
# Consistency Model
# ---------------------------------------------------------------------------

class ConsistencyModel(nn.Module):
    """
    Consistency model as proposed by Song et al. (2023).

    Given a noisy point x_t and noise level σ_t, the model approximates the
    output of the probability flow ODE at time 0, i.e., the clean data point.

    The output is computed as:
        f_θ(x, σ) = c_skip(σ) * x + c_out(σ) * F_θ(x, σ)
    where F_θ is the SongUNet backbone.

    Args:
        img_channels:     number of image channels (3).
        model_channels:   base channels of the UNet (e.g., 128).
        num_blocks:       number of residual blocks per stage (int or list).
        channel_mult:     channel multipliers for each stage.
        attn_resolutions: list of spatial sizes where self‑attention is used.
        dropout:          dropout rate(s).
        sigma_data:       σ_data constant for skip/out scaling (default 0.5).
        sigma_min:        σ_min (σ_0) used in scaling (default 0.002).
    """
    def __init__(
        self,
        img_channels: int = 3,
        model_channels: int = 128,
        num_blocks: Union[int, List[int]] = 3,
        channel_mult: List[int] = [1, 2, 2],
        attn_resolutions: List[int] = [],
        dropout: Union[float, List[float]] = 0.0,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min

        # Base UNet
        self.net = SongUNet(
            img_channels=img_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            channel_mult=channel_mult,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
        )

    # -------------------------------------------------------------------
    # Scaling functions (static for serialisation friendliness)
    # -------------------------------------------------------------------
    @staticmethod
    def c_skip(sigma_data: float, sigma_min: float, sigma: Tensor) -> Tensor:
        """
        c_skip(σ) = σ_data² / (σ_data² + (σ - σ_min)²)
        """
        sigma = sigma.view(-1, 1, 1, 1)
        return sigma_data ** 2 / (sigma_data ** 2 + (sigma - sigma_min) ** 2)

    @staticmethod
    def c_out(sigma_data: float, sigma_min: float, sigma: Tensor) -> Tensor:
        """
        c_out(σ) = σ_data * (σ - σ_min) / sqrt(σ_data² + σ²)
        """
        sigma = sigma.view(-1, 1, 1, 1)
        return sigma_data * (sigma - sigma_min) / torch.sqrt(sigma_data ** 2 + sigma ** 2)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """
        Forward pass of the consistency model.

        Args:
            x:     tensor of shape (B, C, H, W).
            sigma: tensor of shape (B,) containing the noise level corresponding
                   to each input.

        Returns:
            Tensor of shape (B, C, H, W), the predicted clean data point.
        """
        # Base UNet output
        F_out = self.net(x, sigma)                       # (B, C, H, W)

        # Compute scaling factors
        c_skip = self.c_skip(self.sigma_data, self.sigma_min, sigma)
        c_out  = self.c_out(self.sigma_data, self.sigma_min, sigma)

        return c_skip * x + c_out * F_out
