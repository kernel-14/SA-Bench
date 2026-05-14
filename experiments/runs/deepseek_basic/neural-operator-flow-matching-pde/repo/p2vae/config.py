"""P2VAE configuration."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class P2VAEConfig:
    """Configuration for the Pretrained Physics Variational Autoencoder.
    
    Based on SD-VAE architecture [Rombach et al., 2022].
    Input: c3p128 (3 channels, 128x128 spatial)
    Latent: c16p16 (16 channels, 16x16 spatial, 12x compression)
    """
    
    # Input/Output
    in_channels: int = 3
    out_channels: int = 3
    spatial_size: int = 128
    
    # Latent space
    latent_channels: int = 16
    latent_size: int = 16  # 128 / 8 = 16
    
    # Architecture sizes
    base_dim: int = 64  # 64 for 16M, 128 for 87M
    ch_mult: Tuple[int, ...] = (1, 2, 4, 4)  # channel multipliers per resolution
    num_res_blocks: int = 2
    
    # VAE specifics
    z_channels: int = 16  # latent dimension before quantization
    kl_weight: float = 1e-3
    
    # Training
    use_fp16: bool = True
    
    def __post_init__(self):
        """Compute derived attributes."""
        # Token count comparison: 12x compression means
        # input tokens: 3 * 128 * 128 = 49152 scalar values
        # latent tokens: 16 * 16 * 16 = 4096 scalar values
        # compression ratio: 49152 / 4096 = 12x
        pass
    
    @property
    def compression_ratio(self) -> float:
        """Compute the compression ratio."""
        input_elements = self.in_channels * self.spatial_size * self.spatial_size
        latent_elements = self.latent_channels * self.latent_size * self.latent_size
        return input_elements / latent_elements


def p2vae_16m_config() -> P2VAEConfig:
    """P2VAE-16M configuration."""
    return P2VAEConfig(base_dim=64)


def p2vae_87m_config() -> P2VAEConfig:
    """P2VAE-87M configuration."""
    return P2VAEConfig(base_dim=128)
