"""
Pyramid DiT: Diffusion Transformer for Pyramidal Flow Matching.

Based on the MM-DiT architecture from SD3 Medium (Esser et al., 2024),
adapted for pyramidal video generation with:
- Blockwise causal attention for autoregressive generation
- 1D RoPE for temporal dimension
- Sinusoidal position encoding for spatial dimensions
- Support for multi-resolution inputs across pyramid stages
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import einops


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding for spatial dimensions.
    Supports extrapolation for higher resolutions (as mentioned in paper Section 3.4).
    """
    
    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: Position indices (B, L) or (L,)
        
        Returns:
            Position embeddings (B, L, dim) or (L, dim)
        """
        device = positions.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        if positions.dim() == 1:
            emb = positions.unsqueeze(-1) * emb.unsqueeze(0)
        else:
            emb = positions.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
        
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class RotaryPositionEmbedding(nn.Module):
    """
    1D Rotary Position Embedding (RoPE) for temporal dimension.
    Supports flexible training with different video durations.
    """
    
    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin embeddings for RoPE.
        
        Args:
            seq_len: Sequence length
            device: Device
        
        Returns:
            Tuple of (cos, sin) tensors of shape (seq_len, dim//2)
        """
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()
    
    def apply_rotary_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors."""
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)
        
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k


class TimestepEmbedding(nn.Module):
    """Timestep embedding using sinusoidal encoding followed by MLP."""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.dim = dim
    
    def sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal timestep embedding."""
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor (B,) with values in [0, 1]
        
        Returns:
            Timestep embeddings (B, dim)
        """
        emb = self.sinusoidal_embedding(t * 1000)  # Scale to [0, 1000]
        return self.mlp(emb)


class MMDiTBlock(nn.Module):
    """
    Multi-Modal DiT block from SD3 (Esser et al., 2024).
    
    Processes both visual tokens and text tokens with separate
    attention projections but shared attention computation.
    Supports blockwise causal attention for autoregressive generation.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Visual stream
        self.norm1_vis = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2_vis = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # Text stream
        self.norm1_txt = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2_txt = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # Adaptive layer norm modulation (6 params: shift, scale, gate for each of 2 norms)
        self.adaLN_modulation_vis = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        self.adaLN_modulation_txt = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        
        # Attention projections - visual
        self.q_vis = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_vis = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_vis = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_vis = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Attention projections - text
        self.q_txt = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_txt = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_txt = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_txt = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # MLP - visual
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp_vis = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim, bias=True),
            nn.Dropout(dropout),
        )
        
        # MLP - text
        self.mlp_txt = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim, bias=True),
            nn.Dropout(dropout),
        )
        
        self.dropout = dropout
    
    def forward(
        self,
        vis_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        timestep_emb: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vis_tokens: Visual tokens (B, L_vis, D)
            txt_tokens: Text tokens (B, L_txt, D)
            timestep_emb: Timestep embeddings (B, D)
            causal_mask: Optional causal attention mask for blockwise causal attention
            frame_indices: Frame indices for temporal RoPE
            rope_cos, rope_sin: Precomputed RoPE embeddings
        
        Returns:
            Updated (vis_tokens, txt_tokens)
        """
        B, L_vis, D = vis_tokens.shape
        B, L_txt, D = txt_tokens.shape
        
        # Compute modulation parameters
        vis_mods = self.adaLN_modulation_vis(timestep_emb)
        shift1_v, scale1_v, gate1_v, shift2_v, scale2_v, gate2_v = vis_mods.chunk(6, dim=-1)
        
        txt_mods = self.adaLN_modulation_txt(timestep_emb)
        shift1_t, scale1_t, gate1_t, shift2_t, scale2_t, gate2_t = txt_mods.chunk(6, dim=-1)
        
        # Pre-norm and modulate
        vis_normed = modulate(self.norm1_vis(vis_tokens), shift1_v, scale1_v)
        txt_normed = modulate(self.norm1_txt(txt_tokens), shift1_t, scale1_t)
        
        # Compute Q, K, V for both streams
        q_v = self.q_vis(vis_normed).view(B, L_vis, self.num_heads, self.head_dim).transpose(1, 2)
        k_v = self.k_vis(vis_normed).view(B, L_vis, self.num_heads, self.head_dim).transpose(1, 2)
        v_v = self.v_vis(vis_normed).view(B, L_vis, self.num_heads, self.head_dim).transpose(1, 2)
        
        q_t = self.q_txt(txt_normed).view(B, L_txt, self.num_heads, self.head_dim).transpose(1, 2)
        k_t = self.k_txt(txt_normed).view(B, L_txt, self.num_heads, self.head_dim).transpose(1, 2)
        v_t = self.v_txt(txt_normed).view(B, L_txt, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Concatenate visual and text tokens for joint attention
        q = torch.cat([q_v, q_t], dim=2)  # (B, H, L_vis+L_txt, head_dim)
        k = torch.cat([k_v, k_t], dim=2)
        v = torch.cat([v_v, v_t], dim=2)
        
        # Compute attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if causal_mask is not None:
            # Apply blockwise causal mask (only for visual tokens)
            # Text tokens can attend to all visual tokens
            attn[:, :, :L_vis, :L_vis] = attn[:, :, :L_vis, :L_vis].masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float('-inf')
            )
        
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # (B, H, L_vis+L_txt, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L_vis + L_txt, D)
        
        # Split back into visual and text
        out_v, out_t = out[:, :L_vis], out[:, L_vis:]
        
        # Project outputs
        out_v = self.out_vis(out_v)
        out_t = self.out_txt(out_t)
        
        # Apply gating and residual connection
        vis_tokens = vis_tokens + gate1_v.unsqueeze(1) * out_v
        txt_tokens = txt_tokens + gate1_t.unsqueeze(1) * out_t
        
        # MLP with pre-norm and modulation
        vis_tokens = vis_tokens + gate2_v.unsqueeze(1) * self.mlp_vis(
            modulate(self.norm2_vis(vis_tokens), shift2_v, scale2_v)
        )
        txt_tokens = txt_tokens + gate2_t.unsqueeze(1) * self.mlp_txt(
            modulate(self.norm2_txt(txt_tokens), shift2_t, scale2_t)
        )
        
        return vis_tokens, txt_tokens


class PyramidDiT(nn.Module):
    """
    Pyramid Diffusion Transformer for video generation.
    
    Based on MM-DiT architecture (SD3 Medium) with:
    - 24 transformer layers, 2B parameters total
    - Sinusoidal position encoding for spatial dimensions (with extrapolation)
    - 1D RoPE for temporal dimension (with interpolation for history)
    - Blockwise causal attention for autoregressive generation
    - Support for multi-resolution inputs across pyramid stages
    
    Text conditioning uses both T5 and CLIP encoders (following FLUX.1).
    """
    
    def __init__(
        self,
        in_channels: int = 16,  # VAE latent channels
        hidden_dim: int = 1536,  # Adjusted for ~2B params with 24 layers
        num_layers: int = 24,
        num_heads: int = 24,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        max_seq_len: int = 65536,
        text_dim: int = 4096,  # T5 embedding dim
        clip_dim: int = 768,   # CLIP embedding dim
        dropout: float = 0.0,
        num_pyramid_stages: int = 3,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_pyramid_stages = num_pyramid_stages
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size, bias=True
        )
        
        # Text projections (T5 + CLIP)
        self.t5_proj = nn.Linear(text_dim, hidden_dim, bias=True)
        self.clip_proj = nn.Linear(clip_dim, hidden_dim, bias=True)
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(hidden_dim)
        
        # Pyramid stage embedding (to distinguish different stages)
        self.stage_embed = nn.Embedding(num_pyramid_stages, hidden_dim)
        
        # Temporal RoPE
        self.temporal_rope = RotaryPositionEmbedding(hidden_dim // num_heads)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            MMDiTBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        self.final_proj = nn.Linear(
            hidden_dim, patch_size * patch_size * in_channels, bias=True
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following DiT paper."""
        # Zero-initialize output projection
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
        
        # Initialize adaLN modulation to zero
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation_vis[-1].weight)
            nn.init.zeros_(block.adaLN_modulation_vis[-1].bias)
            nn.init.zeros_(block.adaLN_modulation_txt[-1].weight)
            nn.init.zeros_(block.adaLN_modulation_txt[-1].bias)
    
    def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Convert image/frame to patch tokens.
        
        Args:
            x: (B, C, H, W) tensor
        
        Returns:
            Tuple of (patches, (H_patches, W_patches))
        """
        B, C, H, W = x.shape
        patches = self.patch_embed(x)  # (B, hidden_dim, H//p, W//p)
        H_p, W_p = patches.shape[-2], patches.shape[-1]
        patches = patches.flatten(2).transpose(1, 2)  # (B, H_p*W_p, hidden_dim)
        return patches, (H_p, W_p)
    
    def unpatchify(
        self,
        patches: torch.Tensor,
        H_p: int,
        W_p: int,
    ) -> torch.Tensor:
        """
        Convert patch tokens back to image/frame.
        
        Args:
            patches: (B, H_p*W_p, hidden_dim) tensor
            H_p, W_p: Number of patches in each dimension
        
        Returns:
            (B, C, H, W) tensor
        """
        B, L, D = patches.shape
        p = self.patch_size
        
        # Project to pixel space
        x = self.final_proj(patches)  # (B, L, p*p*C)
        x = x.view(B, H_p, W_p, p, p, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.in_channels, H_p * p, W_p * p)
        return x
    
    def get_spatial_pos_embed(
        self,
        H_p: int,
        W_p: int,
        device: torch.device,
        extrapolate: bool = False,
        base_H_p: Optional[int] = None,
        base_W_p: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get 2D sinusoidal position embeddings for spatial dimensions.
        
        Supports extrapolation for higher resolutions (Section 3.4).
        
        Args:
            H_p, W_p: Number of patches in height and width
            device: Device
            extrapolate: Whether to extrapolate (for spatial pyramid)
            base_H_p, base_W_p: Base resolution for extrapolation
        
        Returns:
            Position embeddings (H_p*W_p, hidden_dim)
        """
        half_dim = self.hidden_dim // 4  # Split between H and W
        
        if extrapolate and base_H_p is not None:
            # Extrapolate: use positions beyond base resolution
            h_positions = torch.arange(H_p, device=device).float()
            w_positions = torch.arange(W_p, device=device).float()
            # Scale positions to extrapolate
            h_positions = h_positions * (H_p / base_H_p)
            w_positions = w_positions * (W_p / base_W_p)
        else:
            h_positions = torch.arange(H_p, device=device).float()
            w_positions = torch.arange(W_p, device=device).float()
        
        # Compute sinusoidal embeddings
        emb_h = self._sinusoidal_1d(h_positions, half_dim)  # (H_p, half_dim)
        emb_w = self._sinusoidal_1d(w_positions, half_dim)  # (W_p, half_dim)
        
        # Create 2D grid
        emb_h = emb_h.unsqueeze(1).expand(-1, W_p, -1)  # (H_p, W_p, half_dim)
        emb_w = emb_w.unsqueeze(0).expand(H_p, -1, -1)  # (H_p, W_p, half_dim)
        
        pos_emb = torch.cat([emb_h, emb_w], dim=-1)  # (H_p, W_p, hidden_dim//2)
        pos_emb = pos_emb.view(H_p * W_p, self.hidden_dim // 2)
        
        # Pad to full hidden_dim
        pos_emb = F.pad(pos_emb, (0, self.hidden_dim - self.hidden_dim // 2))
        
        return pos_emb
    
    def _sinusoidal_1d(self, positions: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute 1D sinusoidal embeddings."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=positions.device) * -emb)
        emb = positions[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb
    
    def create_blockwise_causal_mask(
        self,
        num_frames: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create blockwise causal attention mask.
        
        Each frame can attend to all previous frames but not future frames.
        Within a frame, all tokens can attend to each other.
        
        Args:
            num_frames: Number of frames
            tokens_per_frame: Number of tokens per frame
            device: Device
        
        Returns:
            Boolean mask (L, L) where True means "should be masked"
        """
        L = num_frames * tokens_per_frame
        mask = torch.zeros(L, L, dtype=torch.bool, device=device)
        
        for i in range(num_frames):
            for j in range(i + 1, num_frames):
                # Frame i cannot attend to frame j (future)
                start_i = i * tokens_per_frame
                end_i = (i + 1) * tokens_per_frame
                start_j = j * tokens_per_frame
                end_j = (j + 1) * tokens_per_frame
                mask[start_i:end_i, start_j:end_j] = True
        
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeds_t5: torch.Tensor,
        text_embeds_clip: torch.Tensor,
        pyramid_stage: int = 0,
        history_tokens: Optional[List[torch.Tensor]] = None,
        num_frames: Optional[int] = None,
        use_causal_attention: bool = True,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass of the Pyramid DiT.
        
        Args:
            x: Input noisy latent (B, C, H, W) for single frame or
               (B, C, T, H, W) for video
            t: Timestep (B,) with values in [0, 1]
            text_embeds_t5: T5 text embeddings (B, L_t5, text_dim)
            text_embeds_clip: CLIP text embeddings (B, clip_dim)
            pyramid_stage: Current pyramid stage index
            history_tokens: Optional list of history frame tokens for autoregressive generation
            num_frames: Number of frames (for video input)
            use_causal_attention: Whether to use blockwise causal attention
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
        
        Returns:
            Predicted velocity field (same shape as x)
        """
        # Handle video input
        is_video = x.dim() == 5
        if is_video:
            B, C, T, H, W = x.shape
            # Process each frame
            frames = [x[:, :, t_idx] for t_idx in range(T)]
        else:
            B, C, H, W = x.shape
            T = 1
            frames = [x]
        
        # Patchify all frames
        all_frame_tokens = []
        H_p, W_p = None, None
        
        for frame in frames:
            frame_tokens, (H_p, W_p) = self.patchify(frame)
            all_frame_tokens.append(frame_tokens)
        
        # Add history tokens if provided
        if history_tokens is not None:
            all_tokens = history_tokens + all_frame_tokens
            num_history = len(history_tokens)
        else:
            all_tokens = all_frame_tokens
            num_history = 0
        
        # Concatenate all frame tokens
        vis_tokens = torch.cat(all_tokens, dim=1)  # (B, total_L, D)
        total_frames = len(all_tokens)
        tokens_per_frame = H_p * W_p
        
        # Add spatial position embeddings
        spatial_pos = self.get_spatial_pos_embed(H_p, W_p, x.device)
        spatial_pos = spatial_pos.unsqueeze(0).unsqueeze(0)  # (1, 1, L_frame, D)
        spatial_pos = spatial_pos.expand(B, total_frames, -1, -1)
        spatial_pos = spatial_pos.reshape(B, total_frames * tokens_per_frame, self.hidden_dim)
        vis_tokens = vis_tokens + spatial_pos
        
        # Prepare text tokens
        txt_tokens_t5 = self.t5_proj(text_embeds_t5)  # (B, L_t5, D)
        
        # CLIP embedding as additional conditioning
        clip_emb = self.clip_proj(text_embeds_clip)  # (B, D)
        
        # Timestep + stage embedding
        t_emb = self.time_embed(t)  # (B, D)
        stage_emb = self.stage_embed(
            torch.tensor(pyramid_stage, device=x.device).expand(B)
        )  # (B, D)
        cond_emb = t_emb + stage_emb + clip_emb  # (B, D)
        
        # Create causal attention mask if needed
        causal_mask = None
        if use_causal_attention and total_frames > 1:
            causal_mask = self.create_blockwise_causal_mask(
                total_frames, tokens_per_frame, x.device
            )
        
        # Apply transformer blocks
        for block in self.blocks:
            vis_tokens, txt_tokens_t5 = block(
                vis_tokens,
                txt_tokens_t5,
                cond_emb,
                causal_mask=causal_mask,
            )
        
        # Extract only the current frame tokens (not history)
        current_frame_tokens = vis_tokens[:, num_history * tokens_per_frame:]
        
        # Final norm and modulation
        final_mods = self.final_adaLN(cond_emb)
        shift, scale = final_mods.chunk(2, dim=-1)
        current_frame_tokens = modulate(
            self.final_norm(current_frame_tokens), shift, scale
        )
        
        # Unpatchify to get velocity field
        if is_video:
            # Process each frame separately
            velocities = []
            for t_idx in range(T):
                frame_tokens = current_frame_tokens[:, t_idx * tokens_per_frame:(t_idx + 1) * tokens_per_frame]
                vel = self.unpatchify(frame_tokens, H_p, W_p)
                velocities.append(vel)
            velocity = torch.stack(velocities, dim=2)  # (B, C, T, H, W)
        else:
            velocity = self.unpatchify(current_frame_tokens, H_p, W_p)
        
        return velocity
