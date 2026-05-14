```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import math
import logging
from collections import defaultdict

from diffusers.models.autoencoder_kl import AutoencoderKL
from transformers import CLIPTextModel, AutoTokenizer

from config import Config
from models.attention_blocks import CausalTemporalAttention, PrefixEnhancedSpatialAttention
from utils.kv_cache_manager import KVCacheManager

logger = logging.getLogger(__name__)

# --- Conceptual Open-Sora Transformer Blocks (Mockup for demonstrating injection) ---
# Since we don't have the actual Open-Sora v1.0 code, we'll create a simplified
# conceptual structure that mimics a Transformer-based diffusion model block.
# In a real implementation, this would be replaced by actual Open-Sora classes
# and the attention modules (`CausalTemporalAttention`, `PrefixEnhancedSpatialAttention`)
# would be injected/replaced within that structure.

class Ca2VDMTemporalBlock(nn.Module):
    """
    Conceptual Temporal Attention Block that uses CausalTemporalAttention.
    In a real Open-Sora model, this would be part of a larger Transformer block.
    """
    def __init__(self, in_channels: int, n_heads: int, head_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.attn = CausalTemporalAttention(in_channels, n_heads, head_dim)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels)
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                cached_kv_pairs: Optional[Dict[str, torch.Tensor]] = None, **kwargs) -> torch.Tensor:
        # x: (B*H*W, L, C)
        identity = x
        x = self.norm(x)
        # CausalTemporalAttention's forward takes query, key, value, mask, and cached_kv_pairs
        x = self.attn(x, x, x, attention_mask=attention_mask, cached_kv_pairs=cached_kv_pairs)
        x = identity + x
        x = identity + self.ff(self.norm(x))
        return x

class Ca2VDMSpatialBlock(nn.Module):
    """
    Conceptual Spatial Attention Block that uses PrefixEnhancedSpatialAttention.
    In a real Open-Sora model, this would be part of a larger Transformer block.
    """
    def __init__(self, in_channels: int, n_heads: int, head_dim: int, prefix_sub_len