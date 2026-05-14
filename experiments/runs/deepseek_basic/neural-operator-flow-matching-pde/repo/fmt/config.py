"""FMT (Flow Marching Transformer) configuration."""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class FMTConfig:
    """Configuration for the Flow Marching Transformer.
    
    Architecture: SiT-style Transformer with AdaLN-Zero conditioning,
    RMSNorm, SwiGLU activation, FlashAttention v2.
    
    Frame processing uses latent temporal pyramids with
    downsampling factors [8, 4, 2, 1] for frames [0, 1, 2, 3].
    """
    
    # Model sizes
    embed_dim: int = 512  # 256 for Small, 512 for Base, 768 for Large
    head_dim: int = 64
    num_layers: int = 12  # estimated from model size
    
    # num_heads is derived automatically from embed_dim / head_dim
    # but can be overridden for custom configurations
    _num_heads: Optional[int] = None
    
    # Latent spatial dimensions (from P2VAE)
    latent_channels: int = 16
    latent_size: int = 16
    
    # Flow marching
    num_diffusion_steps: int = 100  # N = 100 for evaluation
    dt: float = 0.01
    
    # Temporal pyramid: downsample factors for 4 frames
    # Frame 0: Down(·, 8) -> (2, 2) spatial tokens
    # Frame 1: Down(·, 4) -> (4, 4) spatial tokens
    # Frame 2: Down(·, 2) -> (8, 8) spatial tokens
    # Frame 3: full resolution (16, 16) spatial tokens
    temporal_pyramid_factors: Tuple[int, ...] = (8, 4, 2, 1)
    
    # Diffusion forcing GRU
    gru_hidden_dim: Optional[int] = None  # defaults to embed_dim
    
    # Time conditioning
    use_adaln: bool = True  # AdaLN-Zero from SiT/DiT
    
    # Training
    use_fp16: bool = True
    dropout: float = 0.0
    
    # This will be set in __post_init__
    num_heads: int = field(init=False)
    
    def __post_init__(self):
        """Compute derived attributes and validate."""
        assert self.embed_dim % self.head_dim == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by head_dim ({self.head_dim})"
        
        if self._num_heads is not None:
            self.num_heads = self._num_heads
        else:
            self.num_heads = self.embed_dim // self.head_dim
        
        if self.gru_hidden_dim is None:
            self.gru_hidden_dim = self.embed_dim
        
        # Token counts at each pyramid level
        # Total tokens: (latent_size/f)^2 per frame at down factor f
        self.tokens_per_level = tuple(
            (self.latent_size // f) ** 2 for f in self.temporal_pyramid_factors
        )
        # (2)^2 + (4)^2 + (8)^2 + (16)^2 = 4 + 16 + 64 + 256 = 340 tokens
        self.total_tokens = sum(self.tokens_per_level)
    
    @property
    def efficiency_gain(self) -> float:
        """Compute efficiency gain vs vanilla video diffusion.
        
        η = (4 × 16²)² / ((2²)² + (4²)² + (8²)² + (16²)²) = 15
        """
        full_tokens = 4 * self.latent_size ** 2  # 1024 tokens
        vanilla_cost = full_tokens ** 2
        pyramid_cost = sum(t ** 2 for t in self.tokens_per_level)
        return vanilla_cost / pyramid_cost


def fmt_small_config() -> FMTConfig:
    """FMT-S (6M parameters)."""
    return FMTConfig(
        embed_dim=256,
        num_layers=6,
    )


def fmt_base_config() -> FMTConfig:
    """FMT-B (42M parameters)."""
    return FMTConfig(
        embed_dim=512,
        num_layers=12,
    )


def fmt_large_config() -> FMTConfig:
    """FMT-L (138M parameters)."""
    return FMTConfig(
        embed_dim=768,
        num_layers=24,
    )
