"""MM-DiT architecture for pyramidal flow matching.

Based on SD3 Medium (Esser et al., 2024) with:
- 24 transformer layers
- 2B parameters
- Sinusoidal position encoding for spatial dimensions
- 1D RoPE for temporal dimension
- Blockwise causal attention for autoregressive video generation
- Joint text-visual attention (MM-DiT style)
- T5 + CLIP text conditioning
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.attention import JointAttention, build_causal_mask
from model.layers import (
    AdaLayerNorm,
    FeedForward,
    RMSNorm,
    SinusoidalPositionEmbedding,
    TimestepEmbedding,
)


class MMDiTBlock(nn.Module):
    """Single MM-DiT transformer block with joint text-visual attention.

    Implements the SD3-style multimodal DiT block where text and visual
    tokens are processed jointly with separate norms and projections.
    """

    def __init__(
        self,
        hidden_size: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Visual stream
        self.vis_norm1 = AdaLayerNorm(hidden_size, hidden_size)
        self.vis_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.vis_ff = FeedForward(hidden_size, mlp_ratio, dropout)
        self.vis_ff_gate = nn.Linear(hidden_size, hidden_size)

        # Text stream
        self.txt_norm1 = AdaLayerNorm(context_dim, hidden_size)
        self.txt_norm2 = nn.LayerNorm(context_dim, elementwise_affine=False, eps=1e-6)
        self.txt_ff = FeedForward(context_dim, mlp_ratio, dropout)
        self.txt_ff_gate = nn.Linear(hidden_size, context_dim)

        # Joint attention
        self.attn = JointAttention(
            dim=hidden_size,
            context_dim=context_dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            dropout=dropout,
        )

    def forward(
        self,
        vis_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        condition: torch.Tensor,
        vis_mask: Optional[torch.Tensor] = None,
        rope_freqs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Visual pre-norm with AdaLN
        vis_normed, vis_gate_msa, vis_shift_mlp, vis_scale_mlp, vis_gate_mlp = (
            self.vis_norm1(vis_tokens, condition)
        )
        # Text pre-norm with AdaLN
        txt_normed, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = (
            self.txt_norm1(txt_tokens, condition)
        )

        # Joint attention
        vis_attn_out, txt_attn_out = self.attn(
            vis_normed, txt_normed, vis_mask=vis_mask, rope_freqs=rope_freqs
        )

        # Visual residual with gating
        vis_tokens = vis_tokens + vis_gate_msa.unsqueeze(1).tanh() * vis_attn_out
        txt_tokens = txt_tokens + txt_gate_msa.unsqueeze(1).tanh() * txt_attn_out

        # Visual FFN with AdaLN
        vis_ff_in = self.vis_norm2(vis_tokens) * (1 + vis_scale_mlp.unsqueeze(1)) + vis_shift_mlp.unsqueeze(1)
        vis_tokens = vis_tokens + vis_gate_mlp.unsqueeze(1).tanh() * self.vis_ff(vis_ff_in)

        # Text FFN with AdaLN
        txt_ff_in = self.txt_norm2(txt_tokens) * (1 + txt_scale_mlp.unsqueeze(1)) + txt_shift_mlp.unsqueeze(1)
        txt_tokens = txt_tokens + txt_gate_mlp.unsqueeze(1).tanh() * self.txt_ff(txt_ff_in)

        return vis_tokens, txt_tokens


class FinalLayer(nn.Module):
    """Final layer of MM-DiT that projects to output patch space."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class MMDiT(nn.Module):
    """Multimodal Diffusion Transformer for pyramidal flow matching.

    Architecture based on SD3 Medium with 24 layers and 2B parameters.
    Supports:
    - Text-to-image generation
    - Text-to-video generation (autoregressive with causal attention)
    - Image-to-video generation (first frame as condition)
    - Variable resolution via position encoding extrapolation/interpolation
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        num_layers: int = 24,
        num_heads: int = 24,
        mlp_ratio: float = 4.0,
        in_channels: int = 16,
        patch_size: int = 2,
        context_dim: int = 4096,
        clip_dim: int = 768,
        pooled_dim: int = 2048,
        qk_norm: bool = True,
        dropout: float = 0.0,
        use_causal_attention: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.use_causal_attention = use_causal_attention

        # Patch embedding
        self.patch_embed = nn.Linear(
            in_channels * patch_size * patch_size, hidden_size, bias=True
        )

        # Position embeddings
        self.spatial_pos_embed = SinusoidalPositionEmbedding(hidden_size)
        # Temporal RoPE: applied within attention to temporal dimension
        # head_dim for RoPE
        head_dim = hidden_size // num_heads
        self.temporal_rope_dim = head_dim // 2  # half of head_dim for temporal RoPE

        # Timestep embedding
        self.time_embed = TimestepEmbedding(256, hidden_size)

        # Text conditioning: T5 (4096-dim) + CLIP (768-dim pooled)
        # Project T5 features to context_dim
        self.t5_proj = nn.Linear(4096, context_dim)
        # Project CLIP pooled features to hidden_size for conditioning
        self.clip_proj = nn.Linear(pooled_dim, hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            MMDiTBlock(
                hidden_size=hidden_size,
                context_dim=context_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qk_norm=qk_norm,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final output layer
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        # Zero-init output projection
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def patchify(
        self, x: torch.Tensor, num_frames: int
    ) -> Tuple[torch.Tensor, int, int]:
        """Convert latent to patch tokens.

        Args:
            x: (B, C, H, W) or (B*T, C, H, W) spatial latent
            num_frames: number of frames

        Returns:
            tokens: (B, T*H'*W', hidden_size)
            h_patches: number of patches in height
            w_patches: number of patches in width
        """
        B_T, C, H, W = x.shape
        B = B_T // num_frames
        p = self.patch_size

        assert H % p == 0 and W % p == 0, f"H={H}, W={W} must be divisible by patch_size={p}"
        h_patches = H // p
        w_patches = W // p

        # Reshape to patches
        x = rearrange(x, "(b t) c (h p1) (w p2) -> b (t h w) (c p1 p2)",
                      b=B, t=num_frames, p1=p, p2=p)
        tokens = self.patch_embed(x)
        return tokens, h_patches, w_patches

    def unpatchify(
        self,
        tokens: torch.Tensor,
        num_frames: int,
        h_patches: int,
        w_patches: int,
    ) -> torch.Tensor:
        """Convert patch tokens back to spatial latent.

        Returns:
            x: (B*T, C, H, W)
        """
        p = self.patch_size
        B = tokens.shape[0]
        x = rearrange(
            tokens,
            "b (t h w) (c p1 p2) -> (b t) c (h p1) (w p2)",
            t=num_frames, h=h_patches, w=w_patches, p1=p, p2=p,
        )
        return x

    def get_spatial_pos_embed(
        self,
        h_patches: int,
        w_patches: int,
        device: torch.device,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Get spatial position embeddings with optional scaling for extrapolation.

        For spatial pyramid: extrapolate (scale > 1) for higher resolution stages.
        For temporal pyramid: interpolate (scale < 1) for lower resolution history.
        """
        pos = self.spatial_pos_embed(h_patches, w_patches, device)
        return pos

    def get_temporal_rope_freqs(
        self,
        num_frames: int,
        tokens_per_frame: int,
        device: torch.device,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute 1D RoPE frequencies for temporal dimension.

        Args:
            num_frames: total number of frames
            tokens_per_frame: spatial tokens per frame
            device: target device
            frame_indices: optional explicit frame indices for interpolation

        Returns:
            freqs: (T*tokens_per_frame, head_dim//2, 2) [cos, sin]
        """
        dim = self.temporal_rope_dim
        if frame_indices is None:
            frame_indices = torch.arange(num_frames, device=device).float()

        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        freqs = torch.outer(frame_indices, theta)  # (T, dim//2)
        freqs = torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # (T, dim//2, 2)

        # Repeat for each spatial token in the frame
        freqs = freqs.unsqueeze(1).expand(-1, tokens_per_frame, -1, -1)
        freqs = freqs.reshape(num_frames * tokens_per_frame, dim // 2, 2)
        return freqs

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        t5_embeds: torch.Tensor,
        clip_pooled: torch.Tensor,
        num_frames: int,
        history_tokens: Optional[torch.Tensor] = None,
        history_frame_indices: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        spatial_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass of MM-DiT.

        Args:
            x: (B*T, C, H, W) noisy latent patches for current generation
            timestep: (B,) timestep values in [0, 1]
            t5_embeds: (B, L, 4096) T5 text embeddings
            clip_pooled: (B, 2048) CLIP pooled text embeddings
            num_frames: number of frames being generated
            history_tokens: (B, T_hist*H'*W', hidden_size) pre-embedded history tokens
            history_frame_indices: frame indices for history (for RoPE interpolation)
            causal_mask: precomputed causal attention mask
            spatial_scale: scale factor for position encoding (>1 for extrapolation)

        Returns:
            velocity: (B*T, C, H, W) predicted velocity field
        """
        B = x.shape[0] // num_frames

        # Patchify current frames
        vis_tokens, h_patches, w_patches = self.patchify(x, num_frames)
        tokens_per_frame = h_patches * w_patches

        # Spatial position embeddings (sinusoidal, extrapolated for higher res)
        spatial_pos = self.get_spatial_pos_embed(
            h_patches, w_patches, x.device, scale=spatial_scale
        )  # (H'*W', hidden_size)
        spatial_pos = spatial_pos.unsqueeze(0).unsqueeze(0)  # (1, 1, H'*W', hidden_size)
        spatial_pos = spatial_pos.expand(B, num_frames, -1, -1)
        spatial_pos = spatial_pos.reshape(B, num_frames * tokens_per_frame, self.hidden_size)
        vis_tokens = vis_tokens + spatial_pos

        # Prepend history tokens if provided
        if history_tokens is not None:
            vis_tokens = torch.cat([history_tokens, vis_tokens], dim=1)
            total_frames = history_tokens.shape[1] // tokens_per_frame + num_frames
        else:
            total_frames = num_frames

        # Temporal RoPE frequencies
        rope_freqs = self.get_temporal_rope_freqs(
            total_frames, tokens_per_frame, x.device,
            frame_indices=history_frame_indices,
        )
        rope_freqs = rope_freqs.unsqueeze(0).expand(B, -1, -1, -1)

        # Timestep + CLIP conditioning
        t_emb = self.time_embed(timestep)  # (B, hidden_size)
        clip_emb = self.clip_proj(clip_pooled)  # (B, hidden_size)
        condition = t_emb + clip_emb  # (B, hidden_size)

        # T5 text tokens
        txt_tokens = self.t5_proj(t5_embeds)  # (B, L, context_dim)

        # Transformer blocks
        for block in self.blocks:
            vis_tokens, txt_tokens = block(
                vis_tokens,
                txt_tokens,
                condition,
                vis_mask=causal_mask,
                rope_freqs=rope_freqs,
            )

        # Extract only the current frame tokens (not history)
        if history_tokens is not None:
            n_hist = history_tokens.shape[1]
            vis_tokens = vis_tokens[:, n_hist:]

        # Final layer
        vis_tokens = self.final_layer(vis_tokens, condition)

        # Unpatchify
        velocity = self.unpatchify(vis_tokens, num_frames, h_patches, w_patches)
        return velocity

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        t5_embeds: torch.Tensor,
        clip_pooled: torch.Tensor,
        null_t5_embeds: torch.Tensor,
        null_clip_pooled: torch.Tensor,
        cfg_scale: float,
        num_frames: int,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with classifier-free guidance."""
        # Concatenate conditional and unconditional inputs
        x_double = torch.cat([x, x], dim=0)
        t_double = torch.cat([timestep, timestep], dim=0)
        t5_double = torch.cat([t5_embeds, null_t5_embeds], dim=0)
        clip_double = torch.cat([clip_pooled, null_clip_pooled], dim=0)

        velocity = self.forward(
            x_double, t_double, t5_double, clip_double, num_frames, **kwargs
        )
        v_cond, v_uncond = velocity.chunk(2, dim=0)
        return v_uncond + cfg_scale * (v_cond - v_uncond)
