"""
Ca2-VDM Transformer blocks and model.

Implements the spatial-temporal Transformer architecture for Ca2-VDM,
based on Open-Sora / PixArt-alpha style DiT (Diffusion Transformer).

Key components:
  - Ca2VDMBlock: A single transformer block with:
      * Causal temporal attention
      * Prefix-enhanced spatial attention
      * Visual-text cross attention (for T2V)
      * Feed-forward network
      * AdaLN-Zero conditioning on timestep and text
  - Ca2VDMTransformer: Full transformer model stacking multiple blocks.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalTemporalAttention, PrefixEnhancedSpatialAttention
from .positional_embedding import SpatialPositionalEmbedding, TemporalPositionalEmbedding


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding followed by MLP projection.

    Args:
        dim: Output embedding dimension.
        freq_dim: Sinusoidal frequency dimension (default 256).
    """

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> (B, freq_dim)"""
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, freq_dim)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor of shape (B,).
        Returns:
            Embedding of shape (B, dim).
        """
        emb = self._sinusoidal_embedding(t)
        return self.mlp(emb)


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization Zero (AdaLN-Zero) from DiT.

    Modulates layer norm with scale and shift from conditioning signal,
    and applies a gate to the residual connection.

    Args:
        dim: Feature dimension.
        cond_dim: Conditioning dimension.
        num_params: Number of (scale, shift, gate) triplets (default 2 for attn + ffn).
    """

    def __init__(self, dim: int, cond_dim: int, num_params: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # Project conditioning to 3 * num_params * dim (scale, shift, gate for each)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * num_params * dim, bias=True),
        )
        # Initialize to zero for stable training (DiT-style)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.num_params = num_params
        self.dim = dim

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: Input of shape (B, N, dim).
            cond: Conditioning of shape (B, cond_dim).

        Returns:
            Tuple of (shift_1, scale_1, gate_1, shift_2, scale_2, gate_2, ...)
            each of shape (B, 1, dim).
        """
        params = self.adaLN_modulation(cond)  # (B, 3*num_params*dim)
        params = params.chunk(3 * self.num_params, dim=-1)  # list of (B, dim)
        # Reshape for broadcasting: (B, 1, dim)
        params = [p.unsqueeze(1) for p in params]
        return tuple(params)

    def modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Apply AdaLN modulation: norm(x) * (1 + scale) + shift."""
        return self.norm(x) * (1 + scale) + shift


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.

    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden dimension (default 4*dim).
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    """
    Cross-attention for visual-text conditioning.

    Args:
        dim: Query dimension.
        context_dim: Key/value dimension (text encoder output dim).
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(context_dim, dim, bias=True)
        self.v_proj = nn.Linear(context_dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Query of shape (B, N, dim).
            context: Key/value of shape (B, M, context_dim).
            context_mask: Optional mask of shape (B, M), True for valid positions.

        Returns:
            Output of shape (B, N, dim).
        """
        B, N, _ = x.shape
        M = context.shape[1]

        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, N, M)

        if context_mask is not None:
            # context_mask: (B, M), True = valid
            mask = (~context_mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, M)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)  # (B, H, N, Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, self.num_heads * self.head_dim)
        return self.out_proj(out)


class Ca2VDMBlock(nn.Module):
    """
    A single Ca2-VDM transformer block (Figure 3(c) in the paper).

    Contains:
      1. Causal temporal attention (with AdaLN-Zero)
      2. Prefix-enhanced spatial attention (with AdaLN-Zero)
      3. Visual-text cross attention (optional, for T2V)
      4. Feed-forward network (with AdaLN-Zero)

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        context_dim: Text encoder dimension (for cross-attention). None = no cross-attn.
        prefix_len: P', number of prefix frames for spatial attention enhancement.
        ffn_hidden_dim: Hidden dimension for FFN.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        context_dim: Optional[int] = None,
        prefix_len: int = 3,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.use_cross_attn = context_dim is not None

        # Causal temporal attention
        self.temporal_attn = CausalTemporalAttention(dim, num_heads, dropout)
        self.norm_temporal = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Prefix-enhanced spatial attention
        self.spatial_attn = PrefixEnhancedSpatialAttention(dim, num_heads, prefix_len, dropout)
        self.norm_spatial = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Visual-text cross attention (optional)
        if self.use_cross_attn:
            self.cross_attn = CrossAttention(dim, context_dim, num_heads, dropout)
            self.norm_cross = nn.LayerNorm(dim, eps=1e-6)

        # Feed-forward network
        self.ffn = FeedForward(dim, ffn_hidden_dim, dropout)
        self.norm_ffn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # AdaLN-Zero for timestep conditioning
        # We need modulation for: temporal_attn, spatial_attn, ffn (+ cross_attn if used)
        num_ada_params = 3  # temporal_attn, spatial_attn, ffn
        self.adaLN = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True))
            for _ in range(num_ada_params)
        ])
        # Initialize to zero
        for ada in self.adaLN:
            nn.init.zeros_(ada[-1].weight)
            nn.init.zeros_(ada[-1].bias)

    def _modulate(self, norm: nn.Module, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Apply AdaLN: norm(x) * (1 + scale) + shift."""
        return norm(x) * (1 + scale) + shift

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        temporal_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        spatial_prefix: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        return_kv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass for a single Ca2-VDM block.

        Args:
            x: Input of shape (B, L, H, W, C) or (B, L*H*W, C).
               We use (B, L, H*W, C) internally.
            t_emb: Timestep embedding of shape (B, dim).
            temporal_kv_cache: Optional (K, V) for temporal attention cache.
            spatial_prefix: Optional prefix frames for spatial attention enhancement,
                            shape (P', H*W, C).
            context: Optional text context for cross-attention, shape (B, M, context_dim).
            context_mask: Optional text mask of shape (B, M).
            return_kv: If True, return KV pairs for caching.

        Returns:
            out: Output of shape (B, L, H*W, C).
            kv_dict: Optional dict with 'temporal_kv' and 'spatial_kv' if return_kv.
        """
        # x: (B, L, HW, C)
        B, L, HW, C = x.shape

        # --- Causal Temporal Attention ---
        # Reshape: treat HW as batch dimension -> (B*HW, L, C)
        x_temp = x.permute(0, 2, 1, 3).reshape(B * HW, L, C)

        # AdaLN modulation for temporal attention
        # t_emb: (B, C) -> need to expand for HW
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, HW, -1).reshape(B * HW, C)
        # Simple scale/shift (we use a simplified AdaLN here)
        shift_t, scale_t, gate_t = self.adaLN[0](t_emb_expanded).chunk(3, dim=-1)
        shift_t = shift_t.unsqueeze(1)  # (B*HW, 1, C)
        scale_t = scale_t.unsqueeze(1)
        gate_t = gate_t.unsqueeze(1)

        x_temp_normed = self._modulate(self.norm_temporal, x_temp, shift_t, scale_t)
        attn_out, temporal_kv = self.temporal_attn(
            x_temp_normed,
            kv_cache=temporal_kv_cache,
            return_kv=return_kv,
        )
        x_temp = x_temp + gate_t.tanh() * attn_out
        # Reshape back: (B*HW, L, C) -> (B, L, HW, C)
        x = x_temp.reshape(B, HW, L, C).permute(0, 2, 1, 3)

        # --- Prefix-Enhanced Spatial Attention ---
        # Reshape: treat L as batch dimension -> (B*L, HW, C)
        x_spat = x.reshape(B * L, HW, C)

        t_emb_spat = t_emb.unsqueeze(1).expand(-1, L, -1).reshape(B * L, C)
        shift_s, scale_s, gate_s = self.adaLN[1](t_emb_spat).chunk(3, dim=-1)
        shift_s = shift_s.unsqueeze(1)
        scale_s = scale_s.unsqueeze(1)
        gate_s = gate_s.unsqueeze(1)

        x_spat_normed = self._modulate(self.norm_spatial, x_spat, shift_s, scale_s)

        # For spatial attention, we need to handle prefix per frame
        # spatial_prefix: (P', HW, C) - same for all frames in the batch
        spatial_kv = None
        if spatial_prefix is not None:
            # Apply prefix enhancement for denoising target frames
            spat_out, spatial_kv = self.spatial_attn(
                x_spat_normed,
                prefix_frames=spatial_prefix,
                return_kv=return_kv,
            )
        else:
            spat_out, spatial_kv = self.spatial_attn(
                x_spat_normed,
                prefix_frames=None,
                return_kv=return_kv,
            )

        x_spat = x_spat + gate_s.tanh() * spat_out
        x = x_spat.reshape(B, L, HW, C)

        # --- Visual-Text Cross Attention (optional) ---
        if self.use_cross_attn and context is not None:
            # Reshape to (B, L*HW, C) for cross-attention
            x_flat = x.reshape(B, L * HW, C)
            x_flat = x_flat + self.cross_attn(
                self.norm_cross(x_flat), context, context_mask
            )
            x = x_flat.reshape(B, L, HW, C)

        # --- Feed-Forward Network ---
        x_flat = x.reshape(B * L, HW, C)
        t_emb_ffn = t_emb.unsqueeze(1).expand(-1, L, -1).reshape(B * L, C)
        shift_f, scale_f, gate_f = self.adaLN[2](t_emb_ffn).chunk(3, dim=-1)
        shift_f = shift_f.unsqueeze(1)
        scale_f = scale_f.unsqueeze(1)
        gate_f = gate_f.unsqueeze(1)

        x_flat_normed = self._modulate(self.norm_ffn, x_flat, shift_f, scale_f)
        ffn_out = self.ffn(x_flat_normed)
        x_flat = x_flat + gate_f.tanh() * ffn_out
        x = x_flat.reshape(B, L, HW, C)

        kv_dict = None
        if return_kv:
            kv_dict = {
                "temporal_kv": temporal_kv,
                "spatial_kv": spatial_kv,
            }

        return x, kv_dict


class Ca2VDMTransformer(nn.Module):
    """
    Full Ca2-VDM Transformer model.

    Stacks multiple Ca2VDMBlocks with shared positional embeddings.
    Handles the full forward pass for both training and inference.

    Architecture follows Open-Sora / PixArt-alpha style DiT with:
      - Patch embedding for spatial tokens
      - Sinusoidal SPE + TPE (with Cyclic-TPE support)
      - Multiple Ca2VDMBlocks
      - Final layer norm + linear projection to output

    Args:
        in_channels: Number of input channels (VAE latent channels, typically 4).
        out_channels: Number of output channels (2*in_channels for learned variance).
        patch_size: Spatial patch size.
        hidden_size: Transformer hidden dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        context_dim: Text encoder dimension. None = no text conditioning.
        prefix_len: P', number of prefix frames for spatial attention.
        max_seq_len: Maximum training sequence length (L_train = P_max + l).
        max_height: Maximum height in patches.
        max_width: Maximum width in patches.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 8,
        patch_size: int = 2,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        context_dim: Optional[int] = 4096,  # T5-XXL output dim
        prefix_len: int = 3,
        max_seq_len: int = 65,
        max_height: int = 32,
        max_width: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.prefix_len = prefix_len
        self.max_seq_len = max_seq_len

        # Patch embedding: (B, L, H, W, C) -> (B, L, H/p, W/p, hidden_size)
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size, bias=True
        )

        # Positional embeddings
        self.spe = SpatialPositionalEmbedding(hidden_size, max_height, max_width)
        self.tpe = TemporalPositionalEmbedding(hidden_size, max_seq_len)

        # Timestep embedding
        self.t_embedder = TimestepEmbedding(hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Ca2VDMBlock(
                dim=hidden_size,
                num_heads=num_heads,
                context_dim=context_dim,
                prefix_len=prefix_len,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # Final layer: AdaLN + linear projection
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.final_proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following DiT."""
        # Patch embedding
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

        # Timestep embedding MLP
        for module in self.t_embedder.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Convert spatial frames to patch tokens.

        Args:
            x: (B*L, C, H, W)

        Returns:
            patches: (B*L, H/p * W/p, hidden_size)
            h_patches: H/p
            w_patches: W/p
        """
        BL, C, H, W = x.shape
        patches = self.patch_embed(x)  # (BL, hidden_size, H/p, W/p)
        h_p = patches.shape[2]
        w_p = patches.shape[3]
        patches = patches.flatten(2).transpose(1, 2)  # (BL, H/p*W/p, hidden_size)
        return patches, h_p, w_p

    def unpatchify(self, x: torch.Tensor, h_p: int, w_p: int) -> torch.Tensor:
        """
        Convert patch tokens back to spatial frames.

        Args:
            x: (B*L, H/p*W/p, patch_size*patch_size*out_channels)
            h_p: H/p
            w_p: W/p

        Returns:
            (B*L, out_channels, H, W)
        """
        BL, HW, _ = x.shape
        p = self.patch_size
        x = x.view(BL, h_p, w_p, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(BL, self.out_channels, h_p * p, w_p * p)
        return x

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        prefix_len: int,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        temporal_kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        spatial_prefix_cache: Optional[torch.Tensor] = None,
        cyclic_tpe_offset: int = 0,
        return_kv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass for Ca2-VDM Transformer.

        Args:
            z: Input latent of shape (B, L, C, H, W).
               L = P + l (prefix + denoising target) during training.
               L = l (denoising target only) during inference denoising stage.
               L = l (denoised chunk) during inference cache writing stage.
            t: Timestep vector of shape (B, L) or (B,).
               During training: t[i] = 0 for prefix frames, t[i] = t for denoising target.
               During inference: scalar t for all denoising target frames.
            prefix_len: P, number of clean prefix frames in z.
            context: Optional text embeddings of shape (B, M, context_dim).
            context_mask: Optional text mask of shape (B, M).
            temporal_kv_caches: Optional list of KV caches per layer for temporal attention.
            spatial_prefix_cache: Optional spatial prefix for prefix-enhanced attention,
                                  shape (P', H*W, hidden_size).
            cyclic_tpe_offset: Cyclic offset for TPE (used during training).
            return_kv: If True, return KV pairs for cache writing.

        Returns:
            out: Predicted noise of shape (B, L-prefix_len, C, H, W) (denoising target only).
                 Or (B, L, C, H, W) if prefix_len=0.
            kv_dict: Optional dict with per-layer KV pairs if return_kv.
        """
        B, L, C, H, W = z.shape
        device = z.device

        # Patchify all frames
        z_flat = z.reshape(B * L, C, H, W)
        patches, h_p, w_p = self.patchify(z_flat)  # (B*L, HW, hidden_size)
        HW = h_p * w_p
        patches = patches.reshape(B, L, HW, self.hidden_size)

        # Add spatial positional embeddings
        spe = self.spe(h_p, w_p)  # (HW, hidden_size)
        patches = patches + spe.unsqueeze(0).unsqueeze(0)  # broadcast over B, L

        # Add temporal positional embeddings
        # During training: cyclically shifted TPE
        # During inference: handled by caller via cyclic_tpe_offset
        frame_indices = torch.arange(L, device=device) + cyclic_tpe_offset
        tpe = self.tpe(frame_indices)  # (L, hidden_size)
        patches = patches + tpe.unsqueeze(0).unsqueeze(2)  # (B, L, 1, hidden_size) broadcast

        # Compute timestep embeddings
        # t can be (B,) scalar or (B, L) per-frame
        # During training: t is (B, L) with t_i=0 for prefix, t_i=t for denoising target
        # During inference: t is (B,) scalar for all denoising target frames
        if t.dim() == 1:
            # Scalar timestep: same for all frames (inference)
            t_emb = self.t_embedder(t)  # (B, hidden_size)
        else:
            # Per-frame timestep (training): use the denoising target timestep
            # The denoising target frames all have the same t (last non-zero value)
            # We use the last frame's timestep as the representative timestep
            # This is the actual diffusion timestep t for the denoising target
            t_emb = self.t_embedder(t[:, -1])  # (B, hidden_size)

        # Forward through transformer blocks
        all_kv_dicts = [] if return_kv else None

        for i, block in enumerate(self.blocks):
            # Get temporal KV cache for this layer
            layer_temporal_cache = None
            if temporal_kv_caches is not None and i < len(temporal_kv_caches):
                layer_temporal_cache = temporal_kv_caches[i]

            patches, kv_dict = block(
                patches,
                t_emb,
                temporal_kv_cache=layer_temporal_cache,
                spatial_prefix=spatial_prefix_cache,
                context=context,
                context_mask=context_mask,
                return_kv=return_kv,
            )

            if return_kv and kv_dict is not None:
                all_kv_dicts.append(kv_dict)

        # Final layer norm and projection
        shift_f, scale_f = self.final_adaLN(t_emb).chunk(2, dim=-1)
        shift_f = shift_f.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, hidden_size)
        scale_f = scale_f.unsqueeze(1).unsqueeze(1)

        patches = self.final_norm(patches) * (1 + scale_f) + shift_f
        patches = patches.reshape(B * L, HW, self.hidden_size)
        out = self.final_proj(patches)  # (B*L, HW, p*p*out_channels)
        out = self.unpatchify(out, h_p, w_p)  # (B*L, out_channels, H, W)
        out = out.reshape(B, L, self.out_channels, H, W)

        # Return only the denoising target frames (exclude prefix)
        if prefix_len > 0:
            out = out[:, prefix_len:]  # (B, l, out_channels, H, W)

        return out, all_kv_dicts
