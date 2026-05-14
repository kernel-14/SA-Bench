"""
Masked Diffusion Model (MDM) - Transformer Denoising Network
=============================================================
Implements the denoising network p_theta(x_0^i | x_t) for MDMs.
Uses a bidirectional transformer (BERT-style) without time embedding,
as the masked tokens implicitly encode the noise level.

Based on: Shi et al. (2024), Sahoo et al. (2025)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 use_rope: bool = False, max_seq_len: int = 512):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, n_heads, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.use_rope:
            cos, sin = self.rope(q, L)
            q, k = apply_rotary_emb(q, k, cos, sin)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.0, use_rope: bool = False,
                 max_seq_len: int = 512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, use_rope, max_seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.ff(self.norm2(x))
        return x


class MDMTransformer(nn.Module):
    """
    Bidirectional transformer for masked diffusion modeling.
    
    This is the denoising network p_theta(x_0^i | x_t) that predicts
    the original token at each masked position given the partially masked sequence.
    
    Key design choices:
    - Bidirectional attention (not causal) since MDM can attend to all positions
    - No time embedding (time is implicitly encoded by the number of masked tokens)
    - Learnable positional embeddings (or RoPE)
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int = None, max_seq_len: int = 512, dropout: float = 0.1,
                 use_rope: bool = False, use_learnable_pos: bool = True):
        """
        Args:
            vocab_size: size of vocabulary (including mask token 0)
            d_model: model dimension
            n_heads: number of attention heads
            n_layers: number of transformer layers
            d_ff: feed-forward dimension (defaults to 4 * d_model)
            max_seq_len: maximum sequence length
            dropout: dropout rate
            use_rope: whether to use RoPE positional embeddings
            use_learnable_pos: whether to use learnable positional embeddings
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        self.use_learnable_pos = use_learnable_pos
        self.use_rope = use_rope
        if use_learnable_pos and not use_rope:
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.emb_dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_rope, max_seq_len)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following GPT-2 style."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: input token ids (B, L), with 0 for masked tokens
            attention_mask: optional attention mask (B, 1, L, L)
        
        Returns:
            logits: (B, L, vocab_size) - predicted token probabilities for all positions
        """
        B, L = x.shape
        
        h = self.token_emb(x)
        
        if self.use_learnable_pos and not self.use_rope:
            positions = torch.arange(L, device=x.device).unsqueeze(0)
            h = h + self.pos_emb(positions)
        
        h = self.emb_dropout(h)
        
        for layer in self.layers:
            h = layer(h, attention_mask)
        
        h = self.norm(h)
        logits = self.lm_head(h)
        
        return logits
    
    def get_token_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability distribution over tokens for each position.
        
        Args:
            x: input token ids (B, L)
        
        Returns:
            probs: (B, L, vocab_size) - softmax probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def count_parameters(self) -> int:
        """Count non-embedding parameters."""
        total = sum(p.numel() for p in self.parameters())
        emb = sum(p.numel() for p in self.token_emb.parameters())
        return total - emb


def create_mdm_6m(vocab_size: int, max_seq_len: int = 512, use_rope: bool = True) -> MDMTransformer:
    """Create a ~6M parameter MDM (for Sudoku experiments)."""
    return MDMTransformer(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=max_seq_len,
        dropout=0.1,
        use_rope=use_rope,
        use_learnable_pos=not use_rope,
    )


def create_mdm_19m(vocab_size: int, max_seq_len: int = 512, use_rope: bool = True) -> MDMTransformer:
    """Create a ~19M parameter MDM (for Zebra/L&O-NAE-SAT experiments)."""
    return MDMTransformer(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,
        max_seq_len=max_seq_len,
        dropout=0.1,
        use_rope=use_rope,
        use_learnable_pos=not use_rope,
    )


def create_mdm_170m(vocab_size: int, max_seq_len: int = 2048, use_rope: bool = False) -> MDMTransformer:
    """Create a ~170M parameter MDM (for text experiments)."""
    return MDMTransformer(
        vocab_size=vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        max_seq_len=max_seq_len,
        dropout=0.1,
        use_rope=use_rope,
        use_learnable_pos=not use_rope,
    )
