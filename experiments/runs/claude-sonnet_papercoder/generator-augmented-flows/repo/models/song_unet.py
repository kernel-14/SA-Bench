```python
## models/song_unet.py
"""SongUNet (NCSNpp) backbone for the iCT-GC consistency model.

This module implements the U-Net architecture from Song et al. (2021) /
Karras et al. (2022) (EDM), which serves as the neural network backbone
F_θ inside ``ConsistencyModel``. The consistency model wraps this network
with EDM-style preconditioning (c_skip, c_out, c_in, c_noise).

``SongUNet.forward(x, sigma)`` receives:
  - ``x``: pre-conditioned input ``c_in(σ) * x_noisy``, shape ``(B, C, H, W)``
  - ``sigma``: raw noise level σ, shape ``(B,)`` or broadcastable

and returns the raw network output ``F_θ`` before skip-connection scaling.

Architecture follows https://github.com/NVlabs/edm (training/networks.py).

Supported configs (from config.yaml):
  - CIFAR-10:      img_resolution=32, channel_mult=[1,2,2], num_blocks=3,
                   attn_resolutions=[], model_channels=128
  - ImageNet-32:   img_resolution=32, channel_mult=[1,1,2], num_blocks=[3,5,7],
                   attn_resolutions=[16], model_channels=128
  - CelebA/LSUN:  img_resolution=64, channel_mult=[1,2,2,2],
                   num_blocks=[3,3,4,5], attn_resolutions=[], model_channels=128
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time conditioning embeddings
# ---------------------------------------------------------------------------


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for noise level conditioning.

    Maps a scalar noise level σ to a vector of sinusoidal features using
    the EDM formula:
        w = log(σ / 0.25) * 4 / ln(10)
    then applies standard sinusoidal encoding over a log-spaced frequency
    grid.

    This is the ``embedding_type='positional'`` variant used in all paper
    experiments (Tables 4, 5, 6).

    Args:
        num_channels: Output embedding dimensionality. Must be even.
            Typically ``model_channels`` (e.g. 128).
        max_positions: Controls the frequency range. Default 10000 matches
            the original transformer positional encoding convention.
        endpoint: If True, include the endpoint in the frequency grid.
    """

    def __init__(
        self,
        num_channels: int = 128,
        max_positions: int = 10000,
        endpoint: bool = False,
    ) -> None:
        super().__init__()
        self.num_channels: int = num_channels
        self.max_positions: int = max_positions
        self.endpoint: bool = endpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embedding for noise levels.

        Args:
            x: Noise level tensor of shape ``(B,)`` or scalar. Values are
               raw σ in ``[sigma_min, sigma_max]`` (e.g. ``[0.002, 80]``).

        Returns:
            Embedding tensor of shape ``(B, num_channels)``.
        """
        # Flatten to 1-D batch
        x = x.reshape(-1)

        # EDM mapping: w = log(σ / 0.25) * 4 / ln(10)
        # This maps σ ∈ [0.002, 80] to roughly w ∈ [-8.3, 10.1]
        freqs = torch.arange(
            start=0,
            end=self.num_channels // 2,
            dtype=torch.float32,
            device=x.device,
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1.0 / self.max_positions) ** freqs  # shape: (num_channels/2,)

        # x: (B,), freqs: (num_channels/2,) → outer product → (B, num_channels/2)
        x_outer = x.float().outer(freqs)

        # Concatenate sin and cos features → (B, num_channels)
        emb = torch.cat([x_outer.sin(), x_outer.cos()], dim=-1)
        return emb


class FourierEmbedding(nn.Module):
    """Random Fourier feature embedding for noise level conditioning.

    An alternative to ``PositionalEmbedding`` using random Fourier features
    sampled from N(0, scale²). The frequencies are fixed (not learned).

    Not used in the paper's main experiments but included for completeness.

    Args:
        num_channels: Output embedding dimensionality. Must be even.
        scale: Standard deviation of the random frequency distribution.
            Default 16.0 follows EDM convention.
    """

    def __init__(
        self,
        num_channels: int = 128,
        scale: float = 16.0,
    ) -> None:
        super().__init__()
        self.num_channels: int = num_channels
        # Fixed random frequencies — registered as buffer so they are saved
        # in state_dict and moved with .to(device)
        self.register_buffer(
            "freqs",
            torch.randn(num_channels // 2) * scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute random Fourier embedding for noise levels.

        Args:
            x: Noise level tensor of shape ``(B,)`` or scalar.

        Returns:
            Embedding tensor of shape ``(B, num_channels)``.
        """
        x = x.reshape(-1).float()
        # x: (B,), freqs: (num_channels/2,) → (B, num_channels/2)
        x_outer = x.outer(self.freqs * 2.0 * math.pi)
        emb = torch.cat([x_outer.cos(), x_outer.sin()], dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------


class ResnetBlock(nn.Module):
    """Residual block with adaptive group normalization for time conditioning.

    Implements the AdaGN (Adaptive Group Normalization) pattern:
        1. GroupNorm + SiLU on input
        2. Conv2d
        3. Time-conditional affine transform: x = x * (1 + scale) + shift
        4. GroupNorm + SiLU + Dropout
        5. Conv2d
        6. Skip connection (with optional channel projection)

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        emb_dim: Dimensionality of the time embedding vector.
        dropout: Dropout probability applied before the second conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels

        # --- First normalisation + conv ---
        self.norm1 = nn.GroupNorm(
            num_groups=min(32, in_channels),
            num_channels=in_channels,
            eps=1e-5,
            affine=True,
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        # --- Time embedding projection → scale + shift ---
        # Output dim is 2 * out_channels: first half is scale, second is shift
        self.emb_proj = nn.Linear(emb_dim, 2 * out_channels)

        # --- Second normalisation + dropout + conv ---
        self.norm2 = nn.GroupNorm(
            num_groups=min(32, out_channels),
            num_channels=out_channels,
            eps=1e-5,
            affine=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        # --- Skip connection ---
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1
            )
        else:
            self.skip_conv = nn.Identity()

        # Weight initialisation following EDM convention
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights following EDM conventions."""
        # Conv layers: kaiming uniform
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)

        # Second conv: zero-init for stable training start
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        # Embedding projection: kaiming uniform
        nn.init.kaiming_uniform_(self.emb_proj.weight, a=math.sqrt(5))
        if self.emb_proj.bias is not None:
            nn.init.zeros_(self.emb_proj.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.

        Args:
            x: Input feature map of shape ``(B, in_channels, H, W)``.
            emb: Time embedding of shape ``(B, emb_dim)``.

        Returns:
            Output feature map of shape ``(B, out_channels, H, W)``.
        """
        # Skip connection branch
        skip = self.skip_conv(x)

        # Main branch: norm → activation → conv
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Time conditioning: project embedding to scale + shift
        # emb_out: (B, 2 * out_channels) → split into (B, out_channels) each
        emb_out = self.emb_proj(F.silu(emb))
        # Reshape for broadcasting: (B, out_channels, 1, 1)
        scale, shift = emb_out.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        # Adaptive group norm: apply scale and shift after second norm
        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + skip


class AttentionBlock(nn.Module):
    """Multi-head self-attention block with residual connection.

    Applied at resolutions listed in ``attn_resolutions`` (e.g. ``[16]``
    for ImageNet-32). Always applied at the bottleneck (middle) of the U-Net.

    Uses ``GroupNorm`` pre-normalisation and a residual connection.
    The number of attention heads is ``max(1, channels // 64)``.

    Args:
        channels: Number of feature channels. Must be divisible by
            ``num_heads``.
        num_heads: Number of attention heads. If ``None``, computed as
            ``max(1, channels // 64)``.
    """

    def __init__(
        self,
        channels: int,
        num_heads: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.channels: int = channels
        self.num_heads: int = (
            num_heads if num_heads is not None else max(1, channels // 64)
        )

        assert channels % self.num_heads == 0, (
            f"channels ({channels}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )

        self.head_dim: int = channels // self.num_heads

        self.norm = nn.GroupNorm(
            num_groups=min(32, channels),
            num_channels=channels,
            eps=1e-5,
            affine=True,
        )

        # Q, K, V projections as a single fused conv for efficiency
        self.qkv_proj = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # Zero-init output projection for stable training start
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention with residual connection.

        Args:
            x: Feature map of shape ``(B, C, H, W)``.

        Returns:
            Feature map of the same shape ``(B, C, H, W)``.
        """
        B, C, H, W = x.shape

        # Pre-normalisation
        h = self.norm(x)

        # Compute Q, K, V: (B, 3*C, H, W) → split → each (B, C, H, W)
        qkv = self.qkv_proj(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention:
        # (B, C, H, W) → (B, num_heads, head_dim, H*W)
        def reshape_for_attn(t: torch.Tensor) -> torch.Tensor:
            t = t.reshape(B, self.num_heads, self.head_dim, H * W)
            return t

        q = reshape_for_attn(q)  # (B, nh, hd, N)
        k = reshape_for_attn(k)  # (B, nh, hd, N)
        v = reshape_for_attn(v)  # (B, nh, hd, N)

        # Scaled dot-product attention
        # q, k: (B, nh, hd, N) → attention: (B, nh, N, N)
        scale = self.head_dim ** -0.5
        # Transpose q for matmul: (B, nh, N, hd) @ (B, nh, hd, N) → (B, nh, N, N)
        attn = torch.einsum("bnhd,bnhd->bnh", q.permute(0, 1, 3, 2), k.permute(0, 1, 3, 2))

        # Recompute properly:
        # q: (B, nh, hd, N) → (B, nh, N, hd)
        q = q.permute(0, 1, 3, 2)  # (B, nh, N, hd)
        k = k.permute(0, 1, 3, 2)  # (B, nh, N, hd)
        v = v.permute(0, 1, 3, 2)  # (B, nh, N, hd)

        # Attention weights: (B, nh, N, N)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum of values: (B, nh, N, hd)
        out = torch.matmul(attn_weights, v)

        # Reshape back: (B, nh, N, hd) → (B, C, H, W)
        out = out.permute(0, 1, 3, 2)  # (B, nh, hd, N)
        out = out.reshape(B, C, H, W)

        # Output projection
        out = self.out_proj(out)

        # Residual connection
        return x + out


class Downsample(nn.Module):
    """Strided convolution downsampling (halves spatial resolution).

    Uses a 3×3 conv with stride 2 rather than pooling, following EDM.

    Args:
        channels: Number of input and output channels (unchanged).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample by factor 2.

        Args:
            x: Feature map of shape ``(B, C, H, W)``.

        Returns:
            Feature map of shape ``(B, C, H//2, W//2)``.
        """
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour upsampling followed by conv (doubles spatial resolution).

    Uses ``nn.Upsample(scale_factor=2, mode='nearest')`` + ``Conv2d(3×3)``,
    following EDM convention (avoids checkerboard artefacts from transposed conv).

    Args:
        channels: Number of input and output channels (unchanged).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample by factor 2.

        Args:
            x: Feature map of shape ``(B, C, H, W)``.

        Returns:
            Feature map of shape ``(B, C, H*2, W*2)``.
        """
        x = self.upsample(x)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Main SongUNet
# ---------------------------------------------------------------------------


class SongUNet(nn.Module):
    """NCSNpp / SongUNet backbone for consistency models.

    Implements the U-Net architecture from Song et al. (2021) and
    Karras et al. (2022) (EDM). This is the raw neural network F_θ that
    ``ConsistencyModel`` wraps with EDM preconditioning.

    The network takes a pre-conditioned image ``c_in(σ) * x`` and the raw
    noise level ``σ`` as inputs, and returns the raw network output before
    the ``c_skip``/``c_out`` scaling applied by ``ConsistencyModel``.

    Architecture:
        - Input conv: ``Conv2d(in_channels, model_channels, 3)``
        - Encoder: ``len(channel_mult)`` resolution levels, each with
          ``num_blocks[l]`` ResNet blocks + optional attention + downsampling
        - Middle: 2 ResNet blocks + 1 attention block (always)
        - Decoder: mirror of encoder with skip connections + upsampling
        - Output: ``GroupNorm → SiLU → Conv2d(model_channels, out_channels, 3)``

    Args:
        img_resolution: Spatial resolution of input images (e.g. 32, 64).
        in_channels: Number of input image channels (3 for RGB).
        out_channels: Number of output channels (3 for RGB).
        model_channels: Base channel count. Actual channels at level ``l``
            are ``model_channels * channel_mult[l]``.
        channel_mult: Per-level channel multipliers. Length determines the
            number of resolution levels. E.g. ``[1, 2, 2]`` for 3 levels.
        num_blocks: Number of ResNet blocks per level. Can be a single
            ``int`` (same for all levels) or a ``list`` of length
            ``len(channel_mult)``.
        attn_resolutions: List of spatial resolutions at which to apply
            self-attention. E.g. ``[16]`` applies attention at 16×16.
            Use ``[]`` for no attention (CIFAR-10, CelebA, LSUN).
        dropout: Dropout probability. Can be a single ``float`` or a
            ``list`` of per-level floats matching ``channel_mult``.
        embedding_type: Time embedding type. ``'positional'`` (default,
            used in all paper experiments) or ``'fourier'``.

    Attributes:
        encoder_blocks: ``nn.ModuleList`` of encoder modules (ResnetBlocks,
            AttentionBlocks, Downsamplers) in forward-pass order.
        decoder_blocks: ``nn.ModuleList`` of decoder modules (Upsamplers,
            ResnetBlocks, AttentionBlocks) in forward-pass order.
        time_embed: ``nn.Sequential`` MLP mapping raw embedding to
            ``emb_dim = model_channels * 4`` dimensional vector.
    """

    def __init__(
        self,
        img_resolution: int = 32,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        channel_mult: Optional[List[int]] = None,
        num_blocks: Union[int, List[int]] = 3,
        attn_resolutions: Optional[List[int]] = None,
        dropout: Union[float, List[float]] = 0.0,
        embedding_type: str = "positional",
    ) -> None:
        super().__init__()

        # --- Defaults ---
        if channel_mult is None:
            channel_mult = [1, 2, 2]
        if attn_resolutions is None:
            attn_resolutions = []

        self.img_resolution: int = img_resolution
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.model_channels: int = model_channels
        self.channel_mult: List[int] = list(channel_mult)
        self.attn_resolutions: List[int] = list(attn_resolutions)
        self.embedding_type: str = embedding_type

        num_levels: int = len(channel_mult)

        # Normalise num_blocks to a list of length num_levels
        if isinstance(num_blocks, int):
            self._num_blocks: List[int] = [num_blocks] * num_levels
        else:
            self._num_blocks = list(num_blocks)
            assert len(self._num_blocks) == num_levels, (
                f"len(num_blocks)={len(self._num_blocks)} must equal "
                f"len(channel_mult)={num_levels}"
            )

        # Normalise dropout to a list of length num_levels
        if isinstance(dropout, float):
            self._dropout: List[float] = [dropout] * num_levels
        else:
            self._dropout = list(dropout)
            assert len(self._dropout) == num_levels, (
                f"len(dropout)={len(self._dropout)} must equal "
                f"len(channel_mult)={num_levels}"
            )

        # Embedding dimension: model_channels * 4 (EDM convention)
        emb_channels: int = model_channels  # raw embedding output dim
        self.emb_dim: int = model_channels * 4

        # --- Time embedding ---
        if embedding_type == "positional":
            self._raw_embedding = PositionalEmbedding(
                num_channels=emb_channels,
                max_positions=10000,
                endpoint=False,
            )
        elif embedding_type == "fourier":
            self._raw_embedding = FourierEmbedding(
                num_channels=emb_channels,
                scale=16.0,
            )
        else:
            raise ValueError(
                f"Unknown embedding_type '{embedding_type}'. "
                "Expected 'positional' or 'fourier'."
            )

        # Two-layer MLP: emb_channels → emb_dim → emb_dim
        self.time_embed = nn.Sequential(
            nn.Linear(emb_channels, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

        # --- Input conv ---
        self.input_conv = nn.Conv2d(
            in_channels, model_channels, kernel_size=3, padding=1
        )

        # --- Build encoder and decoder ---
        # _build_encoder and _build_decoder also populate helper lists
        # used by forward() to manage skip connections.
        self.encoder_blocks: nn.ModuleList
        self.decoder_blocks: nn.ModuleList
        self._encoder_skip_channels: List[int] = []  # channel counts at each skip
        self._encoder_block_types: List[str] = []    # 'resnet'|'attn'|'down'
        self._decoder_block_types: List[str] = []    # 'up'|'resnet'|'attn'

        self.encoder_blocks = self._build_encoder()
        self.middle_blocks = self._build_middle()
        self.decoder_blocks = self._build_decoder()

        # --- Output block ---
        final_channels: int = model_channels  # decoder ends at model_channels
        self.output_norm = nn.GroupNorm(
            num_groups=min(32, final_channels),
            num_channels=final_channels,
            eps=1e-5,
            affine=True,
        )
        self.output_conv = nn.Conv2d(
            final_channels, out_channels, kernel_size=3, padding=1
        )

        # Zero-init output conv for stable training start
        nn.init.zeros_(self.output_conv.weight)
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias)

        # Init time_embed MLP
        self._init_time_embed()

    def _init_time_embed(self) -> None:
        """Initialise time embedding MLP weights."""
        for module in self.time_embed.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _build_encoder(self) -> nn.ModuleList:
        """Build the encoder (downsampling) path.

        Constructs a flat ``nn.ModuleList`` of encoder modules in forward-
        pass order. Also populates ``self._encoder_skip_channels`` and
        ``self._encoder_block_types`` for use in ``forward()``.

        Returns:
            ``nn.ModuleList`` containing all encoder modules.
        """
        modules: List[nn.Module] = []
        block_types: List[str] = []
        skip_channels: List[int] = []

        # Track current resolution and channel count
        current_res: int = self.img_resolution
        prev_channels: int = self.model_channels

        # The input_conv output is the first skip connection
        skip_channels.append(prev_channels)

        for level_idx in range(len(self.channel_mult)):
            out_channels: int = self.model_channels * self.channel_mult[level_idx]
            n_blocks: int = self._num_blocks[level_idx]
            drop: float = self._dropout[level_idx]

            for block_idx in range(n_blocks):
                in_ch: int = prev_channels if block_idx == 0 else out_channels
                modules.append(
                    ResnetBlock(
                        in_channels=in_ch,
                        out_channels=out_channels,
                        emb_dim=self.emb_dim,
                        dropout=drop,
                    )
                )
                block_types.append("resnet")
                skip_channels.append(out_channels)

                # Attention at this resolution?
                if current_res in self.attn_resolutions:
                    modules.append(AttentionBlock(channels=out_channels))
                    block_types.append("attn")
                    # Attention does not change channels; no new skip entry
                    # (attention output is added to the same skip slot)
                    # We track attention separately — skip_channels only
                    # records ResNet outputs.

            prev_channels = out_channels

            # Downsample (except at the last level)
            if level_idx < len(self.channel_mult) - 1:
                modules.append(Downsample(channels=out_channels))
                block_types.append("down")
                current_res = current_res // 2

        self._encoder_skip_channels = skip_channels
        self._encoder_block_types = block_types

        return nn.ModuleList(modules)

    def _build_middle(self) -> nn.ModuleList:
        """Build the bottleneck (middle) blocks.

        The middle always consists of:
            ResnetBlock → AttentionBlock → ResnetBlock

        Returns:
            ``nn.ModuleList`` with 3 modules.
        """
        # Channel count at the bottleneck = last encoder level channels
        bottleneck_channels: int = (
            self.model_channels * self.channel_mult[-1]
        )
        drop: float = self._dropout[-1]

        return nn.ModuleList([
            ResnetBlock(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                emb_dim=self.emb_dim,
                dropout=drop,
            ),
            AttentionBlock(channels=bottleneck_channels),
            ResnetBlock(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                emb_dim=self.emb_dim,
                dropout=drop,
            ),
        ])

    def _build_decoder(self) -> nn.ModuleList:
        """Build the decoder (upsampling) path.

        Mirrors the encoder with skip connections. Each decoder level has
        ``num_blocks[l] + 1`` ResNet blocks (the extra block handles the
        concatenated skip connection from the encoder).

        Also populates ``self._decoder_block_types`` for use in ``forward()``.

        Returns:
            ``nn.ModuleList`` containing all decoder modules.
        """
        modules: List[nn.Module] = []
        block_types: List[str] = []

        # We iterate levels in reverse order
        # Skip channel stack: we'll pop from the end of _encoder_skip_channels
        # Make a copy to pop from
        skip_ch_stack: List[int] = list(self._encoder_skip_channels)

        # Current channel count starts at bottleneck
        prev_channels: int = self.model_channels * self.channel_mult[-1]

        # Track current resolution (starts at bottleneck resolution)
        current_res: int = self.img_resolution // (
            2 ** (len(self.channel_mult) - 1)
        )

        for level_idx in reversed(range(len(self.channel_mult))):
            out_channels: int = self.model_channels * self.channel_mult[level_idx]
            n_blocks: int = self._num_blocks[level_idx] + 1  # +1 for skip
            drop: float = self._dropout[level_idx]

            for block_idx in range(n_blocks):
                # Pop skip connection channel count
                skip_ch: int = skip_ch_stack.pop()
                in_ch: int = prev_channels + skip_ch

                modules.append(
                    ResnetBlock(
                        in_channels=in_