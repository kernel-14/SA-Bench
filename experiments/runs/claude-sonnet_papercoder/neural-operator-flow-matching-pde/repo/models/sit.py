```python
## models/sit.py
"""SiT (Scalable Interpolant Transformer) backbone for FMT.

Implements the Transformer architecture described in the paper (Section 4.1),
combining three design lineages:
  - DiT (Peebles & Xie 2023): AdaLN-Zero conditioning mechanism
  - Llama-2 (Touvron et al. 2023): RMSNorm + SwiGLU FFN
  - FlashAttention v2 (Dao 2023): efficient multi-head self-attention

This file is a leaf module with no internal project dependencies.
All components are consumed by models/fmt.py.

Key design constraints from the paper:
  - k-free conditioning: SiT blocks conditioned only on t, not k
  - Bidirectional (non-causal) attention across all pyramid tokens
  - head_dim=64 for all model sizes (paper Section 4.1)
  - AdaLN-Zero and FinalLayer linear layers are zero-initialized
  - Positional embeddings are added in fmt.py, not here
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# FlashAttention v2 import with graceful fallback
# ---------------------------------------------------------------------------
try:
    from flash_attn import flash_attn_func  # type: ignore[import]
    FLASH_ATTN_AVAILABLE: bool = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization from Llama-2.

    Normalizes the input by its RMS (no mean subtraction), then scales
    by a learnable weight. Used in all normalization positions within
    SiTBlock and FinalLayer.

    Math:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    The normalization is computed in float32 for numerical stability when
    training with AMP (float16/bfloat16), then cast back to the input dtype.

    Attributes:
        weight: Learnable scale parameter of shape (dim,), initialized to ones.
        eps: Small constant for numerical stability.
        dim: Normalized dimension size.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm.

        Args:
            dim: Size of the last dimension to normalize. Corresponds to
                embed_dim (256, 512, or 768 depending on FMT variant).
            eps: Small constant added to the denominator for numerical
                stability. Default 1e-6 matches Llama-2.
        """
        super().__init__()
        self.dim: int = dim
        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., dim). Typically (B, N, dim)
                where N is the sequence length (up to 340 tokens).

        Returns:
            Normalized tensor of the same shape and dtype as x.
        """
        # Compute in float32 for numerical stability.
        x_float: Tensor = x.float()

        # RMS = sqrt(mean(x^2) + eps), computed over the last dimension.
        rms: Tensor = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Normalize and scale.
        normed: Tensor = x_float * rms * self.weight.float()

        # Cast back to original dtype (float16 or float32).
        return normed.to(x.dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network from Llama-2.

    Replaces the standard GELU MLP with a gated variant that uses SiLU
    (Swish) activation. Three linear projections are used: w1 and w3
    project up (gate and value paths), w2 projects back down.

    Math:
        gate = SiLU(w1(x))
        value = w3(x)
        output = w2(gate * value)

    No bias terms, consistent with Llama-2.

    Attributes:
        w1: Gate projection Linear(dim, hidden_dim, bias=False).
        w2: Down projection Linear(hidden_dim, dim, bias=False).
        w3: Value projection Linear(dim, hidden_dim, bias=False).
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """Initialize SwiGLUFFN.

        Args:
            dim: Input and output dimension (embed_dim).
            hidden_dim: Hidden dimension for the up-projections. In FMT,
                this is computed as int(dim * mlp_ratio) in the calling
                code (fmt.py), where mlp_ratio=4.0 from config.yaml.
        """
        super().__init__()
        self.w1: nn.Linear = nn.Linear(dim, hidden_dim, bias=False)
        self.w2: nn.Linear = nn.Linear(hidden_dim, dim, bias=False)
        self.w3: nn.Linear = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU feed-forward transformation.

        Args:
            x: Input tensor of shape (B, N, dim).

        Returns:
            Output tensor of shape (B, N, dim).
        """
        # Gate path: SiLU(w1(x)), shape (B, N, hidden_dim)
        gate: Tensor = F.silu(self.w1(x))

        # Value path: w3(x), shape (B, N, hidden_dim)
        value: Tensor = self.w3(x)

        # Element-wise product then down-project.
        return self.w2(gate * value)


# ---------------------------------------------------------------------------
# Timestep Embedder
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps t ∈ [0, 1] into dense vectors.

    Two-stage process:
      1. Sinusoidal embedding: t → (B, freq_embed_size) using cosine/sine
         frequencies, analogous to positional encoding for continuous scalars.
      2. MLP projection: (B, freq_embed_size) → (B, hidden_size) via a
         two-layer MLP with SiLU activation.

    The output conditioning vector c of shape (B, hidden_size) is fed into
    AdaLNZero in every SiTBlock. In FMT, this is combined with the GRU
    hidden state before being passed to the blocks.

    Attributes:
        hidden_size: Output embedding dimension (embed_dim: 256/512/768).
        freq_embed_size: Sinusoidal frequency embedding size (default 256).
        mlp: Two-layer MLP projecting from freq_embed_size to hidden_size.
    """

    def __init__(
        self,
        hidden_size: int,
        freq_embed_size: int = 256,
    ) -> None:
        """Initialize TimestepEmbedder.

        Args:
            hidden_size: Output embedding dimension. Matches embed_dim of
                the FMT variant (256 for FMT-S, 512 for FMT-B, 768 for FMT-L).
            freq_embed_size: Dimension of the sinusoidal frequency embedding.
                Default 256 is standard for DiT/SiT models.
        """
        super().__init__()
        self.hidden_size: int = hidden_size
        self.freq_embed_size: int = freq_embed_size

        # Two-layer MLP: freq_embed_size → hidden_size → hidden_size
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int) -> Tensor:
        """Create sinusoidal embeddings for continuous scalar timesteps.

        Adapted from the standard positional encoding for continuous values.
        For a batch of scalars t ∈ [0, 1] and embedding dimension D:

            half = D // 2
            freqs = exp(-log(10000) * arange(half) / half)
            args = t[:, None] * freqs[None, :]
            embedding = cat([cos(args), sin(args)], dim=-1)

        If D is odd, the last dimension is padded with a zero column.

        Args:
            t: Timestep tensor of shape (B,). Values in [0, 1] during
                training (sampled from Uniform) and [0, 1] during inference
                (Euler steps with dt=0.01).
            dim: Embedding dimension (freq_embed_size, default 256).

        Returns:
            Sinusoidal embedding tensor of shape (B, dim).
        """
        assert t.ndim == 1, f"Expected 1D timestep tensor, got shape {t.shape}"

        half: int = dim // 2

        # Frequency bands: exp(-log(10000) * i / half) for i in [0, half).
        # Shape: (half,)
        freqs: Tensor = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / float(half)
        )

        # Outer product: t[:, None] * freqs[None, :] → (B, half)
        args: Tensor = t.float()[:, None] * freqs[None, :]

        # Concatenate cosine and sine components: (B, 2*half)
        embedding: Tensor = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # Handle odd dim: pad with a zero column to reach exactly dim.
        if dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1), mode="constant", value=0.0)

        return embedding  # (B, dim)

    def forward(self, t: Tensor) -> Tensor:
        """Embed a batch of timesteps into dense conditioning vectors.

        Args:
            t: Timestep tensor of shape (B,). Each value is a scalar in
                [0, 1] representing the flow matching time parameter.

        Returns:
            Conditioning vector of shape (B, hidden_size). This is the
            primary conditioning signal c fed into AdaLNZero in each
            SiTBlock.
        """
        # Stage 1: sinusoidal embedding → (B, freq_embed_size)
        freq_embed: Tensor = self.timestep_embedding(t, self.freq_embed_size)

        # Stage 2: MLP projection → (B, hidden_size)
        return self.mlp(freq_embed)


# ---------------------------------------------------------------------------
# AdaLN-Zero
# ---------------------------------------------------------------------------


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with zero initialization from DiT.

    Produces 6 modulation parameters (scale1, shift1, gate1, scale2, shift2,
    gate2) from a conditioning vector c. These modulate the attention and
    FFN sub-layers in SiTBlock via:

        Attention path:
            x_mod = RMSNorm(x) * (1 + scale1) + shift1
            x = x + gate1 * Attention(x_mod)

        FFN path:
            x_mod = RMSNorm(x) * (1 + scale2) + shift2
            x = x + gate2 * FFN(x_mod)

    The single linear layer is zero-initialized so that at the start of
    training all modulations are identity (scale=0 → effective scale=1,
    shift=0, gate=0 → residual path only). This is the DiT training
    stability trick.

    Attributes:
        hidden_size: Conditioning and modulation vector dimension.
        adaLN_modulation: Sequential(SiLU, Linear(hidden_size, 6*hidden_size))
            with zero-initialized weights and biases.
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialize AdaLNZero.

        Args:
            hidden_size: Dimension of the conditioning vector c and of each
                modulation parameter. Matches embed_dim of the FMT variant.
        """
        super().__init__()
        self.hidden_size: int = hidden_size

        # Single linear layer projecting c → 6 modulation vectors.
        # SiLU activation before the linear layer (DiT convention).
        self.adaLN_modulation: nn.Sequential = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        # Zero-initialize the linear layer for training stability.
        # At init: all modulations are zero → identity transformation.
        linear_layer: nn.Linear = self.adaLN_modulation[1]  # type: ignore[index]
        nn.init.zeros_(linear_layer.weight)
        nn.init.zeros_(linear_layer.bias)

    def forward(
        self, x: Tensor, c: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute 6 modulation parameters from conditioning vector.

        Args:
            x: Token sequence of shape (B, N, hidden_size). Not used in
                the computation but included for API consistency with the
                design spec. The modulation depends only on c.
            c: Conditioning vector of shape (B, hidden_size). Typically
                the sum of timestep embedding and GRU hidden state.

        Returns:
            Tuple of 6 tensors, each of shape (B, 1, hidden_size):
                (scale1, shift1, gate1, scale2, shift2, gate2)
            The shape (B, 1, hidden_size) enables broadcasting over the
            sequence dimension N in SiTBlock.
        """
        # Project c → (B, 6 * hidden_size), then split into 6 chunks.
        modulations: Tensor = self.adaLN_modulation(c)  # (B, 6 * hidden_size)

        # Split along last dim into 6 tensors of shape (B, hidden_size).
        chunks: List[Tensor] = modulations.chunk(6, dim=-1)

        # Unsqueeze sequence dimension for broadcasting: (B, 1, hidden_size).
        scale1, shift1, gate1, scale2, shift2, gate2 = [
            chunk.unsqueeze(1) for chunk in chunks
        ]

        return scale1, shift1, gate1, scale2, shift2, gate2


# ---------------------------------------------------------------------------
# SiTBlock
# ---------------------------------------------------------------------------


class SiTBlock(nn.Module):
    """Single Transformer block for the Scalable Interpolant Transformer.

    Combines AdaLN-Zero conditioned self-attention (via FlashAttention v2)
    and SwiGLU FFN. This is the core repeating unit of FMT.

    Architecture:
        Input x: (B, N, hidden_size), conditioning c: (B, hidden_size)

        1. Compute 6 modulation params from AdaLNZero(x, c)
        2. Attention sub-layer:
               norm1_mod = RMSNorm(x) * (1 + scale1) + shift1
               attn_out  = MultiHeadSelfAttention(norm1_mod)
               x = x + gate1 * attn_out
        3. FFN sub-layer:
               norm2_mod = RMSNorm(x) * (1 + scale2) + shift2
               ffn_out   = SwiGLUFFN(norm2_mod)
               x = x + gate2 * ffn_out

    Attention uses FlashAttention v2 when available, with a fallback to
    F.scaled_dot_product_attention for CPU/non-CUDA environments.

    Attributes:
        hidden_size: Token embedding dimension.
        num_heads: Number of attention heads (hidden_size // head_dim).
        head_dim: Per-head dimension (64 from paper Section 4.1).
        norm1: RMSNorm applied before attention.
        norm2: RMSNorm applied before FFN.
        qkv: Fused QKV projection Linear(hidden_size, 3*hidden_size, bias=False).
        proj: Output projection Linear(hidden_size, hidden_size, bias=False).
        ffn: SwiGLUFFN feed-forward network.
        adaLN: AdaLNZero modulation module.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
    ) -> None:
        """Initialize SiTBlock.

        Args:
            hidden_size: Token embedding dimension. From config:
                fmt.variants.fmt_s.embed_dim = 256,
                fmt.variants.fmt_b.embed_dim = 512,
                fmt.variants.fmt_l.embed_dim = 768.
            num_heads: Number of attention heads. Computed as
                hidden_size // head_dim. From config:
                fmt.variants.fmt_s.num_heads = 4,
                fmt.variants.fmt_b.num_heads = 8,
                fmt.variants.fmt_l.num_heads = 12.
            head_dim: Per-head dimension. From config:
                fmt.head_dim = 64 (paper Section 4.1).
            mlp_ratio: Ratio of FFN hidden dim to hidden_size. From config:
                fmt.mlp_ratio = 4.0.

        Raises:
            ValueError: If hidden_size is not divisible by head_dim, or if
                num_heads does not equal hidden_size // head_dim.
        """
        super().__init__()

        if hidden_size % head_dim != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by "
                f"head_dim={head_dim}."
            )
        expected_heads: int = hidden_size // head_dim
        if num_heads != expected_heads:
            raise ValueError(
                f"num_heads={num_heads} does not match "
                f"hidden_size // head_dim = {expected_heads}."
            )

        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim

        # Normalization layers (RMSNorm from Llama-2).
        self.norm1: RMSNorm = RMSNorm(hidden_size)
        self.norm2: RMSNorm = RMSNorm(hidden_size)

        # Fused QKV projection for efficiency.
        self.qkv: nn.Linear = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        # Output projection.
        self.proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=False)

        # SwiGLU FFN with hidden_dim = hidden_size * mlp_ratio.
        hidden_dim: int = int(hidden_size * mlp_ratio)
        self.ffn: SwiGLUFFN = SwiGLUFFN(dim=hidden_size, hidden_dim=hidden_dim)

        # AdaLN-Zero conditioning module.
        self.adaLN: AdaLNZero = AdaLNZero(hidden_size)

        # Softmax scale for attention: 1 / sqrt(head_dim).
        self._attn_scale: float = float(head_dim) ** -0.5

    def _attention(self, x: Tensor) -> Tensor:
        """Compute multi-head self-attention.

        Uses FlashAttention v2 when available (CUDA required), otherwise
        falls back to F.scaled_dot_product_attention (PyTorch >= 2.0).

        Args:
            x: Input tensor of shape (B, N, hidden_size).

        Returns:
            Attention output of shape (B, N, hidden_size).
        """
        b, n, _ = x.shape

        # Fused QKV projection: (B, N, 3 * hidden_size)
        qkv: Tensor = self.qkv(x)

        # Split into Q, K, V: each (B, N, hidden_size)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, N, num_heads, head_dim) for attention.
        q = q.reshape(b, n, self.num_heads, self.head_dim)
        k = k.reshape(b, n, self.num_heads, self.head_dim)
        v = v.reshape(b, n, self.num_heads, self.head_dim)

        if FLASH_ATTN_AVAILABLE and x.is_cuda:
            # FlashAttention v2 interface:
            # flash_attn_func(q, k, v, dropout_p, softmax_scale, causal)
            # q, k, v: (B, N, num_heads, head_dim)
            # Returns: (B, N, num_heads, head_dim)
            attn_out: Tensor = flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=self._attn_scale,
                causal=False,
            )
        else:
            # Fallback: F.scaled_dot_product_attention (PyTorch >= 2.0).
            # Requires (B, num_heads, N, head_dim) layout.
            q_t: Tensor = q.transpose(1, 2)  # (B, num_heads, N, head_dim)
            k_t: Tensor = k.transpose(1, 2)
            v_t: Tensor = v.transpose(1, 2)

            attn_out_t: Tensor = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self._attn_scale,
            )
            # Back to (B, N, num_heads, head_dim)
            attn_out = attn_out_t.transpose(1, 2)

        # Merge heads: (B, N, hidden_size)
        attn_out = attn_out.reshape(b, n, self.hidden_size)

        # Output projection.
        return self.proj(attn_out)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Apply one SiT Transformer block.

        Args:
            x: Token sequence of shape (B, N, hidden_size). N is the total
                number of tokens (up to 340 for the full temporal pyramid).
            c: Conditioning vector of shape (B, hidden_size). Typically
                the sum of timestep embedding and GRU hidden state, computed
                in FMT.forward before calling this block.

        Returns:
            Updated token sequence of shape (B, N, hidden_size).
        """
        # Get 6 AdaLN-Zero modulation parameters.
        scale1, shift1, gate1, scale2, shift2, gate2 = self.adaLN(x, c)

        # --- Attention sub-layer ---
        # Apply AdaLN-Zero modulation to normalized input.
        norm1_out: Tensor = self.norm1(x)
        norm1_mod: Tensor = norm1_out * (1.0 + scale1) + shift1

        # Multi-head self-attention.
        attn_out: Tensor = self._attention(norm1_mod)

        # Gated residual connection.
        x = x + gate1 * attn_out

        # --- FFN sub-layer ---
        # Apply AdaLN-Zero modulation to normalized input.
        norm2_out: Tensor = self.norm2(x)
        norm2_mod: Tensor = norm2_out * (1.0 + scale2) + shift2

        # SwiGLU feed-forward network.
        ffn_out: Tensor = self.ffn(norm2_mod)

        # Gated residual connection.
        x = x + gate2 * ffn_out

        return x


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Converts spatial latent tensors to token sequences for the Transformer.

    With patch_size=1 (from config: fmt.patch_size = 1), each spatial
    location in the latent grid becomes one token. This is equivalent to
    a per-pixel linear projection.

    For the temporal pyramid, PatchEmbed is applied to each pyramid level
    separately (in fmt.py), producing token sequences of lengths 4, 16, 64,
    and 256 for the four pyramid levels. These are concatenated in fmt.py
    to form the full 340-token sequence.

    Positional embeddings are NOT added here — that is handled in FMT.__init__
    after concatenating all pyramid tokens.

    Attributes:
        patch_size: Spatial patch size (1 from config).
        in_channels: Number of input channels (16 = latent_channels from config).
        embed_dim: Output embedding dimension (256/512/768 from config).
        proj: Conv2d projection (kernel_size=patch_size, stride=patch_size).
    """

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 16,
        embed_dim: int = 512,
    ) -> None:
        """Initialize PatchEmbed.

        Args:
            patch_size: Spatial patch size. From config: fmt.patch_size = 1.
                With patch_size=1, each spatial location is one token.
            in_channels: Number of input channels. From config:
                p2vae.latent_channels = 16 (the latent space channel count).
            embed_dim: Output token embedding dimension. From config:
                fmt.variants.fmt_b.embed_dim = 512 (or 256/768 for S/L).
        """
        super().__init__()
        self.patch_size: int = patch_size
        self.in_channels: int = in_channels
        self.embed_dim: int = embed_dim

        # Conv2d with kernel_size=patch_size, stride=patch_size.
        # For patch_size=1: equivalent to a per-pixel linear projection.
        self.proj: nn.Conv2d = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project spatial latent to token sequence.

        Args:
            x: Latent tensor of shape (B, in_channels, H, W). H and W
                depend on the pyramid level: 2, 4, 8, or 16 for the four
                levels (config: fmt.temporal_pyramid.token_counts = [4,16,64,256]).

        Returns:
            Token sequence of shape (B, H*W // patch_size^2, embed_dim).
            With patch_size=1: (B, H*W, embed_dim).
        """
        b, c, h, w = x.shape

        # Apply patch projection: (B, embed_dim, H/patch_size, W/patch_size)
        x_proj: Tensor = self.proj(x)

        # Flatten spatial dims and transpose to (B, N, embed_dim).
        # N = (H/patch_size) * (W/patch_size)
        x_flat: Tensor = x_proj.flatten(2)  # (B, embed_dim, N)
        x_tokens: Tensor = x_flat.transpose(1, 2)  # (B, N, embed_dim)

        return x_tokens

    def num_tokens(self, h: int, w: int) -> int:
        """Compute the number of tokens for a given spatial resolution.

        Args:
            h: Spatial height of the input latent.
            w: Spatial width of the input latent.

        Returns:
            Number of tokens: (h // patch_size) * (w // patch_size).
        """
        return (h // self.patch_size) * (w // self.patch_size)


# ---------------------------------------------------------------------------
# FinalLayer
# ---------------------------------------------------------------------------


class FinalLayer(nn.Module):
    """Output head for the SiT with AdaLN-Zero conditioning.

    Applied after all Transformer blocks to produce the velocity prediction
    in latent space. Uses a simplified AdaLN-Zero with only 2 modulation
    parameters (scale + shift, no gate) — the standard DiT final layer.

    Architecture:
        scale, shift = adaLN_modulation(c).chunk(2, dim=-1)
        x = norm_final(x) * (1 + scale) + shift
        x = linear(x)

    The linear layer is zero-initialized for training stability.

    Attributes:
        hidden_size: Input token embedding dimension.
        out_channels: Output channels per token (latent_channels = 16).
        norm_final: RMSNorm applied before the output projection.
        adaLN_modulation: Sequential(SiLU, Linear(hidden_size, 2*hidden_size))
            with zero-initialized weights and biases.
        linear: Output projection Linear(hidden_size, out_channels)
            with zero-initialized weights and biases.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
    ) -> None:
        """Initialize FinalLayer.

        Args:
            hidden_size: Input token embedding dimension (embed_dim).
            out_channels: Number of output channels per token. For patch_size=1
                and latent_channels=16: out_channels = patch_size^2 * latent_channels
                = 1 * 16 = 16. This is the velocity prediction dimension in
                latent space.
        """
        super().__init__()
        self.hidden_size: int = hidden_size
        self.out_channels: int = out_channels

        # Final normalization.
        self.norm_final: RMSNorm = RMSNorm(hidden_size)

        # AdaLN-Zero modulation: 2 params (scale + shift, no gate).
        self.adaLN_modulation: nn.Sequential = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

        # Zero-initialize the modulation linear layer.
        mod_linear: nn.Linear = self.adaLN_modulation[1]  # type: ignore[index]
        nn.init.zeros_(mod_linear.weight)
        nn.init.zeros_(mod_linear.bias)

        # Output projection: hidden_size → out_channels.
        self.linear: nn.Linear = nn.Linear(hidden_size, out_channels, bias=True)

        # Zero-initialize the output linear layer for training stability.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Apply the final AdaLN-Zero conditioned