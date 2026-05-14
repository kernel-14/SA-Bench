"""
Context Length Extension with YaRN
====================================
Implements the context length extension experiments from Sec 4.4 of the paper.

The paper shows that SDPA gating facilitates context length extension:
  - Models trained on 3.5T tokens with 4k context
  - Extended to 32k by increasing RoPE base from 10k to 1M, training 80B more tokens
  - Further extended to 128k using YaRN

Key finding: Gated models significantly outperform baseline at 64k and 128k context lengths
(Table 5 in the paper).

YaRN (Peng et al., 2023): Efficient context window extension via NTK-aware scaling
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def compute_yarn_scaling_factor(
    seq_len: int,
    original_max_position: int,
    beta_fast: int = 32,
    beta_slow: int = 1,
    scale: float = 1.0,
) -> float:
    """
    Compute YaRN scaling factor for context length extension.
    
    YaRN uses NTK-aware interpolation to extend context length.
    """
    return scale * seq_len / original_max_position


def apply_yarn_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply YaRN-scaled rotary position embeddings."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Standard RoPE rotation
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    
    rotated = torch.cat([
        x1 * cos[..., ::2] - x2 * sin[..., ::2],
        x2 * cos[..., 1::2] + x1 * sin[..., 1::2],
    ], dim=-1)
    
    return rotated


class YaRNRoPE(nn.Module):
    """
    YaRN (Yet another RoPE extensioN) for context length extension.
    
    From: "YaRN: Efficient Context Window Extension of Large Language Models"
    (Peng et al., 2023)
    
    Used in the paper to extend context from 32k to 128k.
    """
    
    def __init__(
        self,
        head_dim: int,
        original_max_seq_len: int = 4096,
        extended_max_seq_len: int = 131072,  # 128k
        original_rope_base: float = 10000.0,
        extended_rope_base: float = 1000000.0,  # 1M
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.original_max_seq_len = original_max_seq_len
        self.extended_max_seq_len = extended_max_seq_len
        self.original_rope_base = original_rope_base
        self.extended_rope_base = extended_rope_base
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        
        # Build YaRN frequency table
        self._build_yarn_cache(extended_max_seq_len)
    
    def _get_mscale(self, scale: float = 1.0) -> float:
        """Compute magnitude scaling factor for YaRN."""
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0
    
    def _build_yarn_cache(self, seq_len: int):
        """Build YaRN-scaled frequency cache."""
        # Compute scaling factor
        scale = self.extended_max_seq_len / self.original_max_seq_len
        
        # Original frequencies
        freq_extra = 1.0 / (
            self.original_rope_base ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
            )
        )
        
        # Extended frequencies (with higher base)
        freq_inter = 1.0 / (
            self.extended_rope_base ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
            )
        )
        
        # YaRN: interpolate between original and extended frequencies
        # based on wavelength relative to original context length
        low = math.floor(
            self.head_dim * math.log(self.original_max_seq_len / (self.beta_fast * 2 * math.pi))
            / (2 * math.log(self.original_rope_base))
        )
        high = math.ceil(
            self.head_dim * math.log(self.original_max_seq_len / (self.beta_slow * 2 * math.pi))
            / (2 * math.log(self.original_rope_base))
        )
        
        # Smooth interpolation mask
        dim_indices = torch.arange(0, self.head_dim // 2, dtype=torch.float32)
        ramp = torch.clamp((dim_indices - low) / (high - low), 0, 1)
        
        # Interpolated frequencies
        freq_yarn = freq_inter * (1 - ramp) + freq_extra * ramp
        
        # Magnitude scaling
        mscale = self._get_mscale(scale)
        
        # Build position-frequency table
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, freq_yarn)
        
        # cos and sin with magnitude scaling
        self.register_buffer("rope_cos", (freqs.cos() * mscale).float(), persistent=False)
        self.register_buffer("rope_sin", (freqs.sin() * mscale).float(), persistent=False)
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply YaRN RoPE to input tensor.
        
        Args:
            x: (batch, seq, num_heads, head_dim)
            offset: Position offset for KV cache
        """
        seq_len = x.shape[1]
        cos = self.rope_cos[offset:offset + seq_len]
        sin = self.rope_sin[offset:offset + seq_len]
        
        # Reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim/2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        
        out = torch.empty_like(x)
        out[..., ::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x2 * cos + x1 * sin
        
        return out


def extend_model_context(
    model: nn.Module,
    new_max_seq_len: int,
    new_rope_base: float = 1000000.0,
    use_yarn: bool = True,
) -> nn.Module:
    """
    Extend the context length of a trained model.
    
    From the paper (Sec 4.4):
    1. Increase RoPE base from 10k to 1M
    2. Continue training on 32k sequences for 80B tokens
    3. Apply YaRN for further extension to 128k
    
    Args:
        model: Trained transformer model
        new_max_seq_len: New maximum sequence length
        new_rope_base: New RoPE base frequency (1M for 32k extension)
        use_yarn: Whether to use YaRN for extension
    
    Returns:
        Model with extended context length
    """
    for layer in model.layers:
        attn = layer.attn
        
        if use_yarn:
            # Replace RoPE with YaRN
            yarn_rope = YaRNRoPE(
                head_dim=attn.head_dim,
                original_max_seq_len=attn.max_seq_len,
                extended_max_seq_len=new_max_seq_len,
                original_rope_base=attn.rope_base,
                extended_rope_base=new_rope_base,
            )
            attn.yarn_rope = yarn_rope
            attn.use_yarn = True
        else:
            # Simple RoPE base extension
            attn.rope_base = new_rope_base
            attn.max_seq_len = new_max_seq_len
            attn._build_rope_cache(new_max_seq_len)
    
    return model


# Context length extension experiment results from Table 5
CONTEXT_EXTENSION_RESULTS = {
    "baseline": {
        "4k": 88.89,
        "8k": 85.88,
        "16k": 83.15,
        "32k": 79.50,
    },
    "sdpa_gate": {
        "4k": 90.56,
        "8k": 87.11,
        "16k": 84.61,
        "32k": 79.77,
    },
    "baseline_yarn": {
        "4k": 82.90,   # -6.0 from baseline
        "8k": 71.52,   # -14.4
        "16k": 61.23,  # -21.9
        "32k": 37.94,  # -41.56
        "64k": 37.51,
        "128k": 31.65,
    },
    "sdpa_gate_yarn": {
        "4k": 88.13,   # -2.4 from sdpa_gate
        "8k": 80.01,   # -7.1
        "16k": 76.74,  # -7.87
        "32k": 72.88,  # -6.89
        "64k": 66.60,
        "128k": 58.82,
    },
}
