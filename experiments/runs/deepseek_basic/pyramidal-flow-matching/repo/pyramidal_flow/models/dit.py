"""
Diffusion Transformer (DiT) for Pyramidal Flow Matching.

Implements the MM-DiT architecture based on SD3 Medium (Esser et al., 2024),
with blockwise causal attention for autoregressive video generation.

Reference: Section 3.4 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation (adaLN)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for spatial dimensions."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Create 2D sinusoidal position embeddings of shape (h*w, dim)."""
        # Each spatial dimension gets dim/2 channels
        half_dim = self.dim // 4  # Split across sin/cos for y and x
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        ).to(device)
        
        # Create grid
        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        
        # Compute embeddings for y
        y_emb = y_grid.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # (h, w, half_dim)
        y_emb = torch.cat([torch.sin(y_emb), torch.cos(y_emb)], dim=-1)  # (h, w, half_dim*2)
        
        # Compute embeddings for x
        x_emb = x_grid.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # (h, w, half_dim)
        x_emb = torch.cat([torch.sin(x_emb), torch.cos(x_emb)], dim=-1)  # (h, w, half_dim*2)
        
        # Concatenate y and x embeddings along the last dim
        emb = torch.cat([y_emb, x_emb], dim=-1)  # (h, w, dim)
        emb = emb.reshape(h * w, self.dim)
        
        return emb


class RotaryPositionEmbedding(nn.Module):
    """1D Rotary Position Embedding (RoPE) for temporal dimension."""
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding.
    x: (B, num_heads, N, head_dim)
    cos, sin: (N, head_dim)
    """
    x_rot = x.float()
    d = x_rot.shape[-1]
    x1, x2 = x_rot[..., :d//2], x_rot[..., d//2:]
    cos1 = cos.unsqueeze(0).unsqueeze(0)[..., :d//2]
    sin1 = sin.unsqueeze(0).unsqueeze(0)[..., :d//2]
    out1 = x1 * cos1 - x2 * sin1
    out2 = x1 * sin1 + x2 * cos1
    return torch.cat([out1, out2], dim=-1).to(x.dtype)


class BlockwiseCausalAttention(nn.Module):
    """Blockwise causal attention for autoregressive video generation."""
    
    def __init__(self, dim: int, num_heads: int = 24, qkv_bias: bool = True, use_rope: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        assert dim % num_heads == 0
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.use_rope:
            cos, sin = self.rope(N, x.device)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_bias = attention_mask.unsqueeze(0).unsqueeze(0) if attention_mask is not None else None
        
        if hasattr(F, 'scaled_dot_product_attention'):
            x_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, scale=scale)
        else:
            attn = (q @ k.transpose(-2, -1)) * scale
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = F.softmax(attn, dim=-1)
            x_out = attn @ v
        
        x_out = x_out.transpose(1, 2).reshape(B, N, D)
        return self.proj(x_out)


class MMDiTBlock(nn.Module):
    """Single MM-DiT Transformer block with adaptive layer norm."""
    
    def __init__(self, dim: int, num_heads: int = 24, mlp_ratio: float = 4.0, use_causal_attention: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = BlockwiseCausalAttention(dim, num_heads) if use_causal_attention else nn.MultiheadAttention(dim, num_heads, batch_first=True)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(approximate='tanh'), nn.Linear(hidden_dim, dim))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        if isinstance(self.attn, BlockwiseCausalAttention):
            attn_out = self.attn(x_norm, attention_mask=attention_mask)
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class PyramidalDiT(nn.Module):
    """
    Pyramidal Diffusion Transformer.
    MM-DiT architecture for pyramidal flow matching.
    24 transformer layers, ~2B parameters.
    """
    
    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 3072,
        num_heads: int = 24,
        num_layers: int = 24,
        text_embed_dim: int = 4096,
        use_causal_attention: bool = True,
        num_spatial_stages: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_spatial_stages = num_spatial_stages
        
        self.x_embedder = nn.Linear(input_dim, hidden_dim)
        self.text_proj = nn.Sequential(nn.Linear(text_embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.t_embedder = nn.Sequential(SinusoidalTimestepEmbedding(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.stage_embedder = nn.Embedding(num_spatial_stages, hidden_dim)
        self.pos_embed = SinusoidalPositionEmbedding(hidden_dim)
        
        self.blocks = nn.ModuleList([
            MMDiTBlock(dim=hidden_dim, num_heads=num_heads, mlp_ratio=4.0, use_causal_attention=use_causal_attention)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_proj = nn.Linear(hidden_dim, input_dim)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        t: float,
        stage_idx: int,
        conditioning: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Convert image-like to token sequence
        if x.dim() == 4:
            B, C, H, W = x.shape
            N = H * W
            x = x.permute(0, 2, 3, 1).reshape(B, N, C)
        elif x.dim() == 5:
            B, C, T, H, W = x.shape
            N = T * H * W
            x = x.permute(0, 2, 3, 4, 1).reshape(B, N, C)
        else:
            B, N, C = x.shape
        
        # Embed input tokens
        x = self.x_embedder(x)
        
        # Concatenate history
        if history is not None:
            hist_emb = self.x_embedder(history)
            x = torch.cat([hist_emb, x], dim=1)
        
        # Position encoding for current tokens
        if x.dim() == 3 and history is not None:
            curr_tokens = N
            H_curr = W_curr = int(math.sqrt(curr_tokens)) if curr_tokens > 0 else 1
        else:
            H_curr = W_curr = int(math.sqrt(N)) if N > 0 else 1
        
        if H_curr * W_curr == N:
            pos_emb = self.pos_embed(H_curr, W_curr, x.device)  # (N, hidden_dim)
        else:
            # Non-square, use simple approach
            pos_emb = torch.zeros(N, self.hidden_dim, device=x.device)
        
        # Add pos emb for current part
        hist_len = x.shape[1] - N if history is not None else 0
        if hist_len > 0:
            hist_pos = torch.zeros(hist_len, self.hidden_dim, device=x.device)
            pos_emb = torch.cat([hist_pos, pos_emb], dim=0)
        
        x = x + pos_emb.unsqueeze(0)
        
        # Conditioning vector
        t_tensor = torch.full((B,), t, device=x.device, dtype=torch.float32)
        c = self.t_embedder(t_tensor)
        stage_tensor = torch.full((B,), stage_idx, device=x.device, dtype=torch.long)
        c = c + self.stage_embedder(stage_tensor)
        if conditioning is not None:
            c = c + self.text_proj(conditioning.mean(dim=1))
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, attention_mask=attention_mask)
        
        x = self.final_norm(x)
        x = self.final_proj(x)
        
        # Remove history tokens from output
        if history is not None:
            x = x[:, -N:, :]
        
        return x


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim)
        args = t.unsqueeze(1).float() * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
