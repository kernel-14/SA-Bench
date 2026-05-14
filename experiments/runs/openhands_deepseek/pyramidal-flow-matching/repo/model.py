"""Pyramidal Flow Matching MM-DiT Model.

Based on SD3 Medium architecture with:
- 24 transformer layers, 2B parameters
- Joint text-image conditioning (T5 + CLIP)
- Blockwise causal attention for autoregressive video generation
- Sinusoidal spatial position encoding + RoPE temporal encoding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
import einops

from layers import (
    MMDiTBlock,
    PatchEmbed,
    PatchUnembed,
    TimestepEmbedding,
    SinusoidalPositionEncoding,
)


class MMDiT(nn.Module):
    """Multi-Modal Diffusion Transformer for pyramidal flow matching."""

    def __init__(
        self,
        num_layers: int = 24,
        hidden_size: int = 2048,
        num_heads: int = 32,
        head_dim: int = 64,
        ff_mult: float = 4.0,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: int = 16,
        pooled_text_dim: int = 2048,
        context_dim: int = 4096,
        clip_dim: int = 768,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.patch_unembed = PatchUnembed(patch_size, out_channels, hidden_size)

        self.spatial_pos_enc = SinusoidalPositionEncoding(hidden_size)

        self.time_embed = TimestepEmbedding(hidden_size)
        self.pooled_text_proj = nn.Linear(pooled_text_dim, hidden_size)

        self.blocks = nn.ModuleList([
            MMDiTBlock(
                dim=hidden_size,
                context_dim=context_dim + clip_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                ff_mult=ff_mult,
                dropout=dropout,
                qk_norm=qk_norm,
                rope_theta=rope_theta,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _build_causal_mask(
        self,
        seq_len: int,
        frame_boundaries: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Build blockwise causal attention mask.

        Each token cannot attend to subsequent frames (blockwise causal).
        Within a frame, full attention is allowed.
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        for i in range(len(frame_boundaries) - 1):
            start_i = frame_boundaries[i]
            end_i = frame_boundaries[i + 1]
            for j in range(i + 1):
                start_j = frame_boundaries[j]
                end_j = frame_boundaries[j + 1]
                mask[start_i:end_i, start_j:end_j] = 0.0
        return mask

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        pooled_text: torch.Tensor,
        clip_context: Optional[torch.Tensor] = None,
        spatial_h: Optional[int] = None,
        spatial_w: Optional[int] = None,
        temporal_pos_ids: Optional[torch.Tensor] = None,
        frame_boundaries: Optional[List[int]] = None,
        resolution_level: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            latent: Noisy latent of shape (B, C, H, W) for image or
                    (B, T, C, H, W) for video, or flattened (B, N, C).
            timestep: Timestep tensor (B,) in [0, 1].
            context: T5 text context (B, L_t5, context_dim).
            pooled_text: Pooled T5 text embedding (B, pooled_text_dim).
            clip_context: CLIP text context (B, L_clip, clip_dim) or None.
            spatial_h, spatial_w: Spatial dimensions of the latent grid.
            temporal_pos_ids: Frame indices for temporal RoPE.
            frame_boundaries: Start/end token indices for each frame.
            resolution_level: Current pyramid resolution level (0 = full).
        """
        B = latent.shape[0]
        T = 1

        if latent.dim() == 4:
            B, C, H, W = latent.shape
            latent = self.patch_embed(latent)
            h_patches, w_patches = H // self.patch_size, W // self.patch_size
            latent = latent.view(B, h_patches * w_patches, self.hidden_size)
        elif latent.dim() == 5:
            B, T, C, H, W = latent.shape
            latent = einops.rearrange(latent, "b t c h w -> (b t) c h w")
            latent = self.patch_embed(latent)
            _, N, _ = latent.shape
            h_patches, w_patches = H // self.patch_size, W // self.patch_size
            latent = latent.view(B, T * N, self.hidden_size)
        else:
            B, N, _ = latent.shape
            h_patches = spatial_h // self.patch_size
            w_patches = spatial_w // self.patch_size
            T = N // (h_patches * w_patches) if h_patches * w_patches > 0 else 1

        if spatial_h is None:
            spatial_h = h_patches * self.patch_size
            spatial_w = w_patches * self.patch_size

        seq_len = latent.shape[1]

        pos_enc = self.spatial_pos_enc(h_patches, w_patches, device=latent.device)
        pos_enc = pos_enc.view(h_patches * w_patches, self.hidden_size).unsqueeze(0)

        if T > 1:
            num_frames = seq_len // (h_patches * w_patches)
            pos_enc = pos_enc.repeat(1, num_frames, 1)

        latent = latent + pos_enc

        time_emb = self.time_embed(timestep)
        pooled = self.pooled_text_proj(pooled_text)
        adaln = time_emb + pooled

        if clip_context is not None:
            full_context = torch.cat([context, clip_context], dim=-1)
        else:
            full_context = context

        causal_mask = None
        if frame_boundaries is not None and len(frame_boundaries) > 1:
            full_seq = sum(b[1] - b[0] for b in zip(frame_boundaries[:-1], frame_boundaries[1:]))
            causal_mask = self._build_causal_mask(
                seq_len, frame_boundaries, device=latent.device
            )

        adaln = adaln.unsqueeze(1).expand(-1, seq_len, -1)

        x = latent
        for block in self.blocks:
            x = block(
                x,
                full_context,
                causal_mask=causal_mask,
                temporal_pos_ids=temporal_pos_ids,
            )
            x = x + adaln

        x = self.final_norm(x)
        out = self.patch_unembed(x, h_patches, w_patches)
        return out
