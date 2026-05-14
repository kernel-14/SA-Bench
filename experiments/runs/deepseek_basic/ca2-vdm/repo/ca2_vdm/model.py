"""
Ca2-VDM: Main Model Implementation.

Spatial-temporal Transformer with:
1. Causal temporal attention (lower triangular mask)
2. Prefix-enhanced spatial attention (spatial concatenation with prefix)
3. Visual-text cross attention (optional, for T2V)
4. Distinct timestep embeddings for clean prefix (t=0) and denoising target (t)
5. KV-cache sharing mechanism

The model follows the spatial-temporal Transformer structure from:
- PixArt-alpha (Chen et al., 2024) / Latte (Ma et al., 2025) / Open-Sora (Zheng et al., 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from .attention import CausalTemporalAttention, PrefixEnhancedSpatialAttention
from .tpe import PositionalEmbeddings
from .cache import KVCacheManager


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: timestep values, shape (B,) or (B, L)
        Returns:
            emb: shape (B, L, 1, dim) or (B, 1, 1, dim) if t is 1D
        """
        half_dim = self.dim // 2
        emb_factor = math.log(self.max_period) / (half_dim - 1)
        emb_factor = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_factor)
        
        if t.dim() == 1:
            t = t.unsqueeze(1)  # (B, 1)
        emb = t.float().unsqueeze(-1) * emb_factor.unsqueeze(0).unsqueeze(0)  # (B, L, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, L, dim)
        
        emb = self.mlp(emb)
        return emb.unsqueeze(2)  # (B, L, 1, dim) for broadcasting


class CausalGenerationBlock(nn.Module):
    """
    A single Transformer block with:
    1. Causal temporal attention (with KV-cache support)
    2. Prefix-enhanced spatial attention (with spatial KV-cache support)
    3. Cross-attention for text conditioning (optional)
    4. Feed-forward network
    
    Order follows Figure 3(c):
    Temporal attention -> Spatial attention -> Cross attention -> FFN
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        prefix_len: int = 3,
        dropout: float = 0.0,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_cross_attn = use_cross_attn
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6) if use_cross_attn else None
        self.norm4 = nn.LayerNorm(dim, eps=1e-6)
        
        # AdaLN modulation (conditioned on timestep)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),  # 6 for shift, scale, gate x 2 (temporal, spatial) + gate for cross + gate_mlp
        )
        
        # Attention layers
        self.temporal_attn = CausalTemporalAttention(dim, num_heads, dropout)
        self.spatial_attn = PrefixEnhancedSpatialAttention(dim, num_heads, prefix_len, dropout)
        
        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, batch_first=True
            )
        
        # Feed-forward network
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        temporal_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        spatial_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        P: int = 0,
        return_kv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[Tuple]]:
        """
        Args:
            x: (B, L, S, dim) where S = H*W
            c: (B, L, dim) timestep conditioning (from adaLN)
            text_emb: (B, T, dim) text embeddings for cross-attention
            temporal_cache: KV cache for temporal attention
            spatial_cache: KV cache for spatial attention
            P: number of clean prefix frames
            return_kv: if True, return KV for caching
        
        Returns:
            out: (B, L, S, dim)
            temporal_kv: (K, V) if return_kv else None
            spatial_kv: (K, V) if return_kv else None
        """
        B, L, S, D = x.shape
        
        # Compute modulation parameters
        modulation = self.adaLN_modulation(c)  # (B, L, 1, 6*dim)
        
        # Reshape c to (B, L, 1, D) if needed
        if c.shape[-1] != D:
            c = c.unsqueeze(2).expand(-1, -1, S, -1)
        
        # Split modulation into shifts, scales, gates
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        
        # --- Temporal attention ---
        norm_x = self.norm1(x)
        # Apply AdaLN: norm_x * (1 + scale) + shift
        norm_x_mod = norm_x * (1 + scale_msa) + shift_msa
        
        attn_out, temporal_kv = self.temporal_attn(norm_x_mod, temporal_cache)
        x = x + gate_msa * attn_out
        
        # --- Spatial attention (prefix-enhanced) ---
        norm_x = self.norm2(x)
        norm_x_mod = norm_x * (1 + scale_msa) + shift_msa  # reuse same modulation? 
        # Actually we have 6*dim = 2 sets of (shift, scale, gate) for temporal and spatial
        # Let me recompute with correct split
        # 6 = (shift_temporal, scale_temporal, gate_temporal, shift_spatial, scale_spatial, gate_spatial)
        # Wait, that's only 6 params for 2 attn + 1 cross + 1 ffn -> need 12 for 4 modules
        # Rethinking: paper says temporal -> spatial -> cross -> ffn
        # 4 modules need 4 * 3 = 12 dims. But paper might simplify.
        # Let's use 12-dim modulation: 4 sets of (shift, scale, gate)
        pass  # We'll fix in the block revision

        return x, None, None  # placeholder


class Ca2VDM(nn.Module):
    """
    Ca2-VDM: Efficient Autoregressive Video Diffusion Model.
    
    Architecture: Spatial-temporal Transformer with:
    - Causal temporal attention with KV-cache
    - Prefix-enhanced spatial attention
    - Visual-text cross attention (for T2V)
    - Distinct timestep embeddings for clean prefix vs denoising target
    - Cyclic-TPEs for long-term autoregression
    """
    
    def __init__(
        self,
        # Input dimensions
        in_channels: int = 4,  # VAE latent channels
        H: int = 32,           # spatial height after VAE
        W: int = 32,           # spatial width after VAE
        
        # Model dimensions
        dim: int = 1152,
        num_heads: int = 16,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        
        # Video generation
        l: int = 16,           # chunk length
        P_max: int = 49,       # max conditional frames
        L_train: int = 65,     # max training length = P_max + l
        
        # Prefix enhancement
        prefix_len: int = 3,   # P'
        
        # Text conditioning
        use_text_cond: bool = True,
        text_dim: int = 4096,  # T5 output dim
        
        # Other
        dropout: float = 0.0,
        learn_sigma: bool = True,  # learnable covariance
        use_vb_loss: bool = True,  # use variational bound loss
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.H = H
        self.W = W
        self.S = H * W
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.l = l
        self.P_max = P_max
        self.L_train = L_train
        self.prefix_len = prefix_len
        self.use_text_cond = use_text_cond
        self.learn_sigma = learn_sigma
        self.use_vb_loss = use_vb_loss
        
        # Patch embedding: input is already VAE-encoded latent
        # Flatten spatial dims and project to dim
        self.input_proj = nn.Linear(in_channels, dim)
        
        # Positional embeddings (SPE + CyclicTPE)
        self.pos_emb = PositionalEmbeddings(
            dim=dim,
            H=H,
            W=W,
            L_train=L_train,
            P_max=P_max,
            l=l,
            use_learned_spe=True,
        )
        
        # Timestep embedding (for noisy frames)
        self.t_embed = TimestepEmbedding(dim)
        
        # Zero timestep embedding (for clean prefix, always t=0)
        self.t_zero = nn.Parameter(torch.zeros(1, 1, 1, dim))
        
        # Text embedding projection
        if use_text_cond:
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                prefix_len=prefix_len,
                dropout=dropout,
                use_cross_attn=use_text_cond,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.output_proj = nn.Linear(dim, in_channels * (2 if learn_sigma else 1))
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
        
        # Initialize output projection to zero for better training start
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def _get_timestep_embedding(
        self,
        t: torch.Tensor,
        P: int,
        L: int,
    ) -> torch.Tensor:
        """
        Get distinct timestep embeddings:
        - Clean prefix (first P frames): t=0 embedding
        - Denoising target (remaining L-P frames): t embedding
        
        Args:
            t: timestep value(s), shape (B,)
            P: number of clean prefix frames
            L: total number of frames
        
        Returns:
            emb: (B, L, 1, dim)
        """
        B = t.shape[0]
        
        # Expand t to per-frame: first P frames get t=0, rest get t
        t_per_frame = torch.zeros(B, L, device=t.device, dtype=t.dtype)
        t_per_frame[:, P:] = t.unsqueeze(1).expand(-1, L - P)
        
        # Get embeddings
        emb = self.t_embed(t_per_frame)  # (B, L, 1, dim)
        
        return emb
    
    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        P: int = 0,
        text_emb: Optional[torch.Tensor] = None,
        cyclic_offset: int = 0,
        kv_cache_manager: Optional[KVCacheManager] = None,
        cache_write: bool = False,
        return_cache: bool = False,
    ) -> dict:
        """
        Forward pass for Ca2-VDM.
        
        Args:
            z: latent input, shape (B, L, C, H, W) or (B, C, L, H, W)
            t: diffusion timestep, shape (B,)
            P: number of clean prefix frames (0 for pure generation during training without prefix)
            text_emb: text embeddings for cross-attention (B, T, text_dim)
            cyclic_offset: offset for CyclicTPE
            kv_cache_manager: optional KVCacheManager for inference
            cache_write: if True, compute and return KV caches
            return_cache: alias for cache_write
        
        Returns:
            dict with:
                'output': denoised prediction (B, L, C, H, W)
                'temporal_caches': list of (K,V) per layer (if cache_write)
                'spatial_caches': list of (K,V) per layer (if cache_write)
        """
        # Handle input format
        if z.shape[1] == self.in_channels:
            # (B, C, L, H, W) -> (B, L, C, H, W)
            z = z.permute(0, 2, 1, 3, 4)
        
        B, L, C, H, W = z.shape
        S = H * W
        
        # Project input to embedding dim
        z_flat = z.permute(0, 1, 3, 4, 2).reshape(B, L, S, C)  # (B, L, S, C)
        x = self.input_proj(z_flat)  # (B, L, S, dim)
        
        # Add positional embeddings
        pos = self.pos_emb(L, cyclic_offset)  # (1, L, S, dim)
        x = x + pos
        
        # Get timestep embedding
        t_emb = self._get_timestep_embedding(t, P, L)  # (B, L, 1, dim)
        
        # Project text embeddings if provided
        if text_emb is not None and self.use_text_cond:
            text_feat = self.text_proj(text_emb)  # (B, T, dim)
        else:
            text_feat = None
        
        # Lists for caching
        temporal_kv_list = []
        spatial_kv_list = []
        
        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            # Get cached KVs if provided
            temporal_cache = None
            spatial_cache = None
            
            if kv_cache_manager is not None:
                temporal_cache = kv_cache_manager.get_temporal_kv(i)
                spatial_cache = kv_cache_manager.get_spatial_kv(i)
            
            x, temp_kv, spat_kv = block(
                x=x,
                c=t_emb,
                text_emb=text_feat,
                temporal_cache=temporal_cache,
                spatial_cache=spatial_cache,
                P=P,
                return_kv=cache_write or return_cache,
            )
            
            if cache_write or return_cache:
                temporal_kv_list.append(temp_kv)
                spatial_kv_list.append(spat_kv)
        
        # Final layer norm and output projection
        x = self.final_norm(x)
        x = self.output_proj(x)  # (B, L, S, out_channels)
        
        # Reshape to original format
        out_channels = self.in_channels * (2 if self.learn_sigma else 1)
        x = x.reshape(B, L, H, W, out_channels)
        x = x.permute(0, 1, 4, 2, 3)  # (B, L, out_C, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, out_C, L, H, W)
        
        result = {'output': x}
        if cache_write or return_cache:
            result['temporal_caches'] = temporal_kv_list
            result['spatial_caches'] = spatial_kv_list
        
        return result


class TransformerBlock(nn.Module):
    """
    Single Transformer block with AdaLN-Zero modulation.
    
    Order (as in Figure 3(c)):
    1. Causal Temporal Attention
    2. Prefix-Enhanced Spatial Attention  
    3. Cross Attention (text conditioning)
    4. Feed-Forward Network
    
    Uses AdaLN (adaptive layer norm) with zero-initialized gating.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        prefix_len: int = 3,
        dropout: float = 0.0,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_cross_attn = use_cross_attn
        
        # AdaLN-Zero modulation
        # 4 sets of (shift, scale, gate) for: temporal attn, spatial attn, cross attn, FFN
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 12 * dim),  # 4 modules * 3 params
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6) if use_cross_attn else None
        self.norm4 = nn.LayerNorm(dim, eps=1e-6)
        
        # Attention layers
        self.temporal_attn = CausalTemporalAttention(dim, num_heads, dropout)
        self.spatial_attn = PrefixEnhancedSpatialAttention(dim, num_heads, prefix_len, dropout)
        
        if use_cross_attn:
            self.cross_attn_q = nn.Linear(dim, dim)
            self.cross_attn_kv = nn.Linear(dim, dim * 2)
            self.cross_attn_proj = nn.Linear(dim, dim)
        
        # Feed-forward network
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        temporal_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        spatial_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        P: int = 0,
        return_kv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[Tuple]]:
        """
        Args:
            x: (B, L, S, dim)
            c: (B, L, 1, dim) timestep conditioning
            text_emb: (B, T, dim) text features for cross-attn
            temporal_cache: KV cache for temporal attn
            spatial_cache: KV cache for spatial attn
            P: number of clean prefix frames
            return_kv: return KV for caching
        
        Returns:
            out, temporal_kv, spatial_kv
        """
        B, L, S, D = x.shape
        
        # Compute modulation: 12 * dim
        modulation = self.adaLN_modulation(c)  # (B, L, 1, 12*D)
        
        # Split into 4 groups of (shift, scale, gate)
        chunks = modulation.chunk(12, dim=-1)
        (
            shift_temp, scale_temp, gate_temp,
            shift_spat, scale_spat, gate_spat,
            shift_cross, scale_cross, gate_cross,
            shift_ffn, scale_ffn, gate_ffn,
        ) = chunks
        
        # --- 1. Causal Temporal Attention ---
        norm_x = self.norm1(x)
        norm_x = norm_x * (1 + scale_temp) + shift_temp
        attn_out, temp_kv = self.temporal_attn(norm_x, temporal_cache)
        x = x + gate_temp * attn_out
        
        # --- 2. Prefix-Enhanced Spatial Attention ---
        norm_x = self.norm2(x)
        norm_x = norm_x * (1 + scale_spat) + shift_spat
        attn_out, spat_kv = self.spatial_attn(norm_x, spatial_cache, P)
        x = x + gate_spat * attn_out
        
        # --- 3. Cross Attention (text) ---
        if text_emb is not None and self.use_cross_attn:
            norm_x = self.norm3(x)
            norm_x = norm_x * (1 + scale_cross) + shift_cross
            
            # Cross-attention operates on (B*L, S, dim) with text (B, T, dim)
            # Treat (B, L) as batch
            x_flat = norm_x.reshape(B * L, S, D)
            
            # Query from video, K/V from text
            q = self.cross_attn_q(x_flat)
            
            # Expand text to batch
            if text_emb.shape[0] == 1:
                text_expanded = text_emb.expand(B * L, -1, -1)
            else:
                text_expanded = text_emb.unsqueeze(1).expand(-1, L, -1, -1).reshape(B * L, -1, D)
            
            kv = self.cross_attn_kv(text_expanded)
            k, v = kv.chunk(2, dim=-1)
            
            # Scaled dot-product attention
            head_dim = D // self.num_heads
            scale = head_dim ** -0.5
            
            # Reshape for multi-head: (B*L, S, nH, d) -> (B*L*nH, S, d) -> matmul
            q_mh = q.reshape(B * L, S, self.num_heads, head_dim).permute(0, 2, 1, 3).reshape(B * L * self.num_heads, S, head_dim)
            k_mh = k.reshape(B * L, -1, self.num_heads, head_dim).permute(0, 2, 1, 3).reshape(B * L * self.num_heads, -1, head_dim)
            v_mh = v.reshape(B * L, -1, self.num_heads, head_dim).permute(0, 2, 1, 3).reshape(B * L * self.num_heads, -1, head_dim)
            
            attn_weights = torch.matmul(q_mh, k_mh.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_out = torch.matmul(attn_weights, v_mh)
            attn_out = attn_out.reshape(B * L, self.num_heads, S, head_dim).permute(0, 2, 1, 3).reshape(B * L, S, D)
            attn_out = self.cross_attn_proj(attn_out)
            
            attn_out = attn_out.reshape(B, L, S, D)
            x = x + gate_cross * attn_out
        
        # --- 4. Feed-Forward Network ---
        norm_x = self.norm4(x)
        norm_x = norm_x * (1 + scale_ffn) + shift_ffn
        ffn_out = self.ffn(norm_x)
        x = x + gate_ffn * ffn_out
        
        temporal_kv = temp_kv if return_kv else None
        spatial_kv = spat_kv if return_kv else None
        
        return x, temporal_kv, spatial_kv
