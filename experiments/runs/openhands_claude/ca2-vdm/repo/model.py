from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from layers import (
    FinalLayer,
    PatchEmbed,
    SpatialPositionalEmbedding,
    TemporalPositionalEmbedding,
    TimestepEmbedding,
)
from modules import (
    BidirectionalGenerationBlock,
    CausalGenerationBlock,
    SpatialKVCache,
    TemporalKVCacheQueue,
)


# ---------------------------------------------------------------------------
# Cyclic Temporal Positional Embedding Assignment
# ---------------------------------------------------------------------------

def assign_cyclic_tpe(
    frame_indices: torch.Tensor,
    max_train_len: int,
    cyclic_offset: int = 0,
) -> torch.Tensor:
    """
    Assign temporal positional embedding indices with cyclic shift.

    During training: each sample gets a random cyclic offset.
    During inference: when P_k >= P_max, denoising target wraps around.

    Args:
        frame_indices: (L,) absolute frame indices
        max_train_len: L_train = P_max + l
        cyclic_offset: random shift applied during training
    Returns:
        tpe_indices: (L,) indices into the TPE table (in [0, max_train_len))
    """
    return (frame_indices + cyclic_offset) % max_train_len


# ---------------------------------------------------------------------------
# Ca2-VDM: Main Model
# ---------------------------------------------------------------------------

class Ca2VDM(nn.Module):
    """
    Ca2-VDM: Efficient Autoregressive Video Diffusion Model with
    Causal Generation and Cache Sharing.

    Architecture based on spatial-temporal Transformer (Open-Sora v1.0 style),
    initialized from Open-Sora v1.0 weights.

    Key modifications over standard bidirectional VDM:
    1. Causal temporal attention (lower-triangular mask)
    2. Prefix-enhanced spatial attention (P' prefix frames concatenated spatially)
    3. Separate timestep embeddings for prefix (t=0) and denoising target (t)
    4. Cyclic-TPEs for generation beyond training length
    """

    def __init__(
        self,
        # Architecture
        in_channels: int = 4,           # VAE latent channels
        patch_size: int = 2,
        hidden_dim: int = 1152,
        num_layers: int = 28,
        num_heads: int = 16,
        context_dim: Optional[int] = 4096,  # T5 embedding dim
        ff_mult: int = 4,
        dropout: float = 0.0,
        # Positional embeddings
        max_spatial_h: int = 64,
        max_spatial_w: int = 64,
        max_temporal_len: int = 512,
        # Prefix enhancement
        prefix_len: int = 3,            # P'
        # Training config
        chunk_len: int = 16,            # l
        p_max: int = 49,                # P_max = 1 + 3*l
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.chunk_len = chunk_len
        self.p_max = p_max
        self.prefix_len = prefix_len

        # Patch embedding
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_dim)

        # Positional embeddings (SPE and TPE, ViT-style sinusoidal)
        self.spe = SpatialPositionalEmbedding(hidden_dim, max_spatial_h, max_spatial_w)
        self.tpe = TemporalPositionalEmbedding(hidden_dim, max_temporal_len)

        # Timestep embedding (shared MLP, indexed by t)
        self.time_embed = TimestepEmbedding(hidden_dim, hidden_dim * 4)
        cond_dim = hidden_dim * 4

        # Transformer blocks (causal generation blocks)
        self.blocks = nn.ModuleList([
            CausalGenerationBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                context_dim=context_dim,
                cond_dim=cond_dim,
                prefix_len=prefix_len,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer: predict noise (and optionally learned variance)
        # Output: 2 * in_channels for (noise, log_variance)
        self.final_layer = FinalLayer(hidden_dim, patch_size, in_channels * 2, cond_dim)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

    def _patchify(self, z: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            z: (B, L, C, H, W)
        Returns:
            patches: (B, L, HW, hidden_dim)
            h_patches: number of patch rows
            w_patches: number of patch cols
        """
        B, L, C, H, W = z.shape
        z_flat = rearrange(z, "b l c h w -> (b l) c h w")
        patches = self.patch_embed(z_flat)  # (B*L, HW, hidden_dim)
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        patches = rearrange(patches, "(b l) hw d -> b l hw d", b=B, l=L)
        return patches, h_patches, w_patches

    def _unpatchify(self, x: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        """
        Args:
            x: (B, L, HW, patch_size^2 * out_channels)
        Returns:
            out: (B, L, out_channels, H, W)
        """
        B, L, HW, _ = x.shape
        p = self.patch_size
        c = self.in_channels * 2  # noise + log_variance
        x = rearrange(x, "b l (h w) (p1 p2 c) -> b l c (h p1) (w p2)",
                      h=h_patches, w=w_patches, p1=p, p2=p, c=c)
        return x

    def _add_positional_embeddings(
        self,
        x: torch.Tensor,
        h_patches: int,
        w_patches: int,
        tpe_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add SPE and TPE to patch embeddings.

        Args:
            x: (B, L, HW, dim)
            h_patches, w_patches: spatial grid dimensions
            tpe_indices: (L,) temporal position indices (possibly cyclic)
        Returns:
            x: (B, L, HW, dim) with positional embeddings added
        """
        B, L, HW, dim = x.shape
        device = x.device

        # Spatial positional embedding: (HW, dim)
        spe = self.spe(h_patches, w_patches, device)  # (HW, dim)
        x = x + spe.unsqueeze(0).unsqueeze(0)  # broadcast over B, L

        # Temporal positional embedding: (L, dim)
        tpe = self.tpe(tpe_indices)  # (L, dim)
        x = x + tpe.unsqueeze(0).unsqueeze(2)  # broadcast over B, HW

        return x

    def forward(
        self,
        z: torch.Tensor,
        t_vec: torch.Tensor,
        prefix_frames: int,
        tpe_indices: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temporal_kv_cache: Optional[TemporalKVCacheQueue] = None,
        spatial_kv_cache: Optional[SpatialKVCache] = None,
        cache_write_mode: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass of Ca2-VDM.

        Args:
            z: (B, L, C, H, W) — input latents
                - z[:, :prefix_frames] are clean (t=0)
                - z[:, prefix_frames:] are noisy (t=t)
            t_vec: (B, L) — per-frame timestep values
                - t_vec[:, :prefix_frames] = 0
                - t_vec[:, prefix_frames:] = t
            prefix_frames: P — number of clean prefix frames
            tpe_indices: (L,) — temporal position indices (cyclic during inference)
            context: (B, S, context_dim) — text embeddings (optional)
            temporal_kv_cache: TemporalKVCacheQueue — for inference
            spatial_kv_cache: SpatialKVCache — for inference
            cache_write_mode: if True, compute and return KV-caches for writing
        Returns:
            output: (B, L, 2*C, H, W) — predicted noise and log-variance
                    (only for denoising target frames; prefix frames are masked in loss)
            new_caches: dict with new temporal/spatial KV for cache writing (if cache_write_mode)
        """
        B, L, C, H, W = z.shape

        # Patchify
        x, h_patches, w_patches = self._patchify(z)  # (B, L, HW, dim)

        # Add positional embeddings
        x = self._add_positional_embeddings(x, h_patches, w_patches, tpe_indices)

        # Compute per-frame timestep embeddings
        # t_vec: (B, L) — use tEmb(0) for prefix, tEmb(t) for denoising target
        t_flat = t_vec.reshape(B * L)
        t_emb_flat = self.time_embed(t_flat)  # (B*L, cond_dim)
        t_emb = t_emb_flat.reshape(B, L, -1)  # (B, L, cond_dim)

        # For conditioning, use the denoising target's timestep embedding
        # (representative of the current noise level)
        # We use the mean over denoising target frames as the block condition
        if prefix_frames < L:
            cond = t_emb[:, prefix_frames:].mean(dim=1)  # (B, cond_dim)
        else:
            cond = t_emb[:, -1]  # fallback

        # Transformer blocks
        new_temp_ks: List[torch.Tensor] = []
        new_temp_vs: List[torch.Tensor] = []
        new_spatial_ks: List[Optional[torch.Tensor]] = []
        new_spatial_vs: List[Optional[torch.Tensor]] = []

        for i, block in enumerate(self.blocks):
            temp_k_cache, temp_v_cache = None, None
            spat_k_cache, spat_v_cache = None, None

            if temporal_kv_cache is not None:
                temp_k_cache, temp_v_cache = temporal_kv_cache.get(i)
            if spatial_kv_cache is not None:
                spat_k_cache, spat_v_cache = spatial_kv_cache.get(i)

            x, new_tk, new_tv, new_sk, new_sv = block(
                x=x,
                cond=cond,
                prefix_frames=prefix_frames,
                context=context,
                temporal_cached_k=temp_k_cache,
                temporal_cached_v=temp_v_cache,
                spatial_cached_k=spat_k_cache,
                spatial_cached_v=spat_v_cache,
            )
            new_temp_ks.append(new_tk)
            new_temp_vs.append(new_tv)
            new_spatial_ks.append(new_sk)
            new_spatial_vs.append(new_sv)

        # Final layer: apply per-frame conditioning
        # Use per-frame t_emb for final layer
        x_out_frames = []
        for frame_idx in range(L):
            x_frame = x[:, frame_idx]  # (B, HW, dim)
            frame_cond = t_emb[:, frame_idx]  # (B, cond_dim)
            x_frame_out = self.final_layer(x_frame, frame_cond)  # (B, HW, p^2 * 2C)
            x_out_frames.append(x_frame_out)
        x_out = torch.stack(x_out_frames, dim=1)  # (B, L, HW, p^2 * 2C)

        # Unpatchify
        output = self._unpatchify(x_out, h_patches, w_patches)  # (B, L, 2C, H, W)

        new_caches = None
        if cache_write_mode:
            new_caches = {
                "temporal_k": new_temp_ks,
                "temporal_v": new_temp_vs,
                "spatial_k": new_spatial_ks,
                "spatial_v": new_spatial_vs,
            }

        return output, new_caches


# ---------------------------------------------------------------------------
# OS-Fix Baseline: Bidirectional, Fixed-Length Condition
# ---------------------------------------------------------------------------

class OSFix(nn.Module):
    """
    OS-Fix baseline: bidirectional temporal attention with fixed-length
    conditional frames (P = L_train / 2).

    Based on Open-Sora v1.0 with standard bidirectional attention.
    """

    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 2,
        hidden_dim: int = 1152,
        num_layers: int = 28,
        num_heads: int = 16,
        context_dim: Optional[int] = 4096,
        ff_mult: int = 4,
        dropout: float = 0.0,
        max_spatial_h: int = 64,
        max_spatial_w: int = 64,
        max_temporal_len: int = 512,
        chunk_len: int = 16,
        fixed_prefix: int = 16,  # P = L_train / 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.chunk_len = chunk_len
        self.fixed_prefix = fixed_prefix

        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_dim)
        self.spe = SpatialPositionalEmbedding(hidden_dim, max_spatial_h, max_spatial_w)
        self.tpe = TemporalPositionalEmbedding(hidden_dim, max_temporal_len)
        self.time_embed = TimestepEmbedding(hidden_dim, hidden_dim * 4)
        cond_dim = hidden_dim * 4

        self.blocks = nn.ModuleList([
            BidirectionalGenerationBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                context_dim=context_dim,
                cond_dim=cond_dim,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_layer = FinalLayer(hidden_dim, patch_size, in_channels * 2, cond_dim)
        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

    def _patchify(self, z: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, L, C, H, W = z.shape
        z_flat = rearrange(z, "b l c h w -> (b l) c h w")
        patches = self.patch_embed(z_flat)
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        patches = rearrange(patches, "(b l) hw d -> b l hw d", b=B, l=L)
        return patches, h_patches, w_patches

    def _unpatchify(self, x: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        B, L, HW, _ = x.shape
        p = self.patch_size
        c = self.in_channels * 2
        x = rearrange(x, "b l (h w) (p1 p2 c) -> b l c (h p1) (w p2)",
                      h=h_patches, w=w_patches, p1=p, p2=p, c=c)
        return x

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        prefix_frames: int,
        tpe_indices: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z: (B, L, C, H, W)
            t: (B,) — single timestep for all frames (unified embedding)
            prefix_frames: P
            tpe_indices: (L,)
            context: (B, S, context_dim)
        Returns:
            output: (B, L, 2*C, H, W)
        """
        B, L, C, H, W = z.shape

        x, h_patches, w_patches = self._patchify(z)

        # SPE
        spe = self.spe(h_patches, w_patches, z.device)
        x = x + spe.unsqueeze(0).unsqueeze(0)

        # TPE
        tpe = self.tpe(tpe_indices)
        x = x + tpe.unsqueeze(0).unsqueeze(2)

        # Timestep embedding (unified for all frames)
        cond = self.time_embed(t)  # (B, cond_dim)

        for block in self.blocks:
            x = block(x, cond, context)

        # Final layer
        x_out_frames = []
        for frame_idx in range(L):
            x_frame = x[:, frame_idx]
            x_frame_out = self.final_layer(x_frame, cond)
            x_out_frames.append(x_frame_out)
        x_out = torch.stack(x_out_frames, dim=1)

        return self._unpatchify(x_out, h_patches, w_patches)


# ---------------------------------------------------------------------------
# OS-Ext Baseline: Bidirectional, Extendable Condition
# ---------------------------------------------------------------------------

class OSExt(nn.Module):
    """
    OS-Ext baseline: bidirectional temporal attention with autoregressively
    extendable conditional frames (same training config as Ca2-VDM but
    bidirectional attention).
    """

    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 2,
        hidden_dim: int = 1152,
        num_layers: int = 28,
        num_heads: int = 16,
        context_dim: Optional[int] = 4096,
        ff_mult: int = 4,
        dropout: float = 0.0,
        max_spatial_h: int = 64,
        max_spatial_w: int = 64,
        max_temporal_len: int = 512,
        chunk_len: int = 16,
        p_max: int = 49,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.chunk_len = chunk_len
        self.p_max = p_max

        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_dim)
        self.spe = SpatialPositionalEmbedding(hidden_dim, max_spatial_h, max_spatial_w)
        self.tpe = TemporalPositionalEmbedding(hidden_dim, max_temporal_len)
        self.time_embed = TimestepEmbedding(hidden_dim, hidden_dim * 4)
        cond_dim = hidden_dim * 4

        self.blocks = nn.ModuleList([
            BidirectionalGenerationBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                context_dim=context_dim,
                cond_dim=cond_dim,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_layer = FinalLayer(hidden_dim, patch_size, in_channels * 2, cond_dim)
        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

    def _patchify(self, z: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, L, C, H, W = z.shape
        z_flat = rearrange(z, "b l c h w -> (b l) c h w")
        patches = self.patch_embed(z_flat)
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        patches = rearrange(patches, "(b l) hw d -> b l hw d", b=B, l=L)
        return patches, h_patches, w_patches

    def _unpatchify(self, x: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        B, L, HW, _ = x.shape
        p = self.patch_size
        c = self.in_channels * 2
        x = rearrange(x, "b l (h w) (p1 p2 c) -> b l c (h p1) (w p2)",
                      h=h_patches, w=w_patches, p1=p, p2=p, c=c)
        return x

    def forward(
        self,
        z: torch.Tensor,
        t_vec: torch.Tensor,
        prefix_frames: int,
        tpe_indices: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z: (B, L, C, H, W)
            t_vec: (B, L) — per-frame timesteps (0 for prefix, t for target)
            prefix_frames: P
            tpe_indices: (L,)
            context: (B, S, context_dim)
        Returns:
            output: (B, L, 2*C, H, W)
        """
        B, L, C, H, W = z.shape

        x, h_patches, w_patches = self._patchify(z)

        spe = self.spe(h_patches, w_patches, z.device)
        x = x + spe.unsqueeze(0).unsqueeze(0)

        tpe = self.tpe(tpe_indices)
        x = x + tpe.unsqueeze(0).unsqueeze(2)

        # Use denoising target timestep as condition
        if prefix_frames < L:
            t_cond = t_vec[:, prefix_frames]  # (B,)
        else:
            t_cond = t_vec[:, -1]
        cond = self.time_embed(t_cond)  # (B, cond_dim)

        for block in self.blocks:
            x = block(x, cond, context)

        x_out_frames = []
        for frame_idx in range(L):
            x_frame = x[:, frame_idx]
            x_frame_out = self.final_layer(x_frame, cond)
            x_out_frames.append(x_frame_out)
        x_out = torch.stack(x_out_frames, dim=1)

        return self._unpatchify(x_out, h_patches, w_patches)


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_model(model_type: str, config) -> nn.Module:
    """
    Build model from config.

    Args:
        model_type: "ca2vdm", "osfix", or "osext"
        config: model configuration object
    Returns:
        model: nn.Module
    """
    common_kwargs = dict(
        in_channels=config.in_channels,
        patch_size=config.patch_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        context_dim=config.context_dim if config.use_text else None,
        ff_mult=config.ff_mult,
        dropout=config.dropout,
        max_spatial_h=config.max_spatial_h,
        max_spatial_w=config.max_spatial_w,
        max_temporal_len=config.max_temporal_len,
        chunk_len=config.chunk_len,
    )

    if model_type == "ca2vdm":
        return Ca2VDM(
            **common_kwargs,
            p_max=config.p_max,
            prefix_len=config.prefix_len,
        )
    elif model_type == "osfix":
        return OSFix(
            **common_kwargs,
            fixed_prefix=config.fixed_prefix,
        )
    elif model_type == "osext":
        return OSExt(
            **common_kwargs,
            p_max=config.p_max,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
