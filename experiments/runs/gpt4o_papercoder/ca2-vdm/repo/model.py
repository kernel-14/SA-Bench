## model.py
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple, Optional
from utils.kvcache import KVCache
from transformers import AutoTokenizer
from config import Config


class SpatialTemporalTransformer(nn.Module):
    """Core architecture of Ca2-VDM with Causal Temporal Attention and Prefix-Enhanced Spatial Attention."""

    def __init__(self, config: Dict) -> None:
        """
        Initialize the Spatial-Temporal Transformer with configured parameters.

        Args:
            config (Dict): Configuration dictionary from config.yaml.
        """
        super(SpatialTemporalTransformer, self).__init__()

        self.latent_resolution = config.get("vae.resolution", 256) // config.get("vae.downsample_factor", 8)
        self.latent_channels = 3  # Latent space channels (assumed from VAE configuration)

        self.chunk_length = config.get("model.chunk_length_t2v", 16)
        self.prefix_length_max = config.get("model.prefix_length_max_t2v", 49)
        self.kv_cache_enabled = config.get("model.kv_cache.enabled", True)
        self.kv_cache_max_length = config.get("model.kv_cache.max_length", 49)

        # Temporal attention configurations
        self.temporal_attention_type = config.get("model.temporal_attention", "Causal")
        self.spatial_attention_type = config.get("model.spatial_attention", "PrefixEnhanced")

        # Transformers components
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.latent_channels,
                nhead=8,
                dim_feedforward=512,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=6,
        )

        # Sinusoidal Spatial Positional Embeddings (SPEs)
        self.spe_embeddings = self._generate_sinusoidal_embeddings(
            spatial_size=(self.latent_resolution, self.latent_resolution),
            dim=self.latent_channels,
        )

        # Temporal Positional Embeddings (TPEs)
        self.use_cyclic_tpe = config.get("model.positional_embeddings.cyclic_tpe", True)
        self.tpe_embeddings = nn.Embedding(self.prefix_length_max + self.chunk_length, self.latent_channels)

        # KV-cache
        if self.kv_cache_enabled:
            self.temporal_kv_cache = KVCache(self.kv_cache_max_length, (self.latent_resolution, self.latent_resolution), self.latent_channels)
            self.spatial_kv_cache: Optional[KVCache] = None  # Initialized during spatial attention

    def _generate_sinusoidal_embeddings(self, spatial_size: Tuple[int, int], dim: int) -> Tensor:
        """
        Creates sinusoidal spatial positional embeddings for spatial attention.

        Args:
            spatial_size (Tuple[int, int]): Height and Width of the spatial dimensions.
            dim (int): Embedding dimension.

        Returns:
            Tensor: Sinusoidal spatial embeddings of shape (H*W, dim).
        """
        height, width = spatial_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pe = torch.zeros(height, width, dim, device=device)

        # Sinusoidal pattern for each feature dimension
        y_pos = torch.arange(0, height, device=device).unsqueeze(1).repeat(1, width).flatten().unsqueeze(1)
        x_pos = torch.arange(0, width, device=device).repeat(height).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(torch.log(torch.tensor(10000.0, device=device)) / dim))

        pe[:, :, 0::2] = torch.sin(x_pos * div_term)
        pe[:, :, 1::2] = torch.cos(y_pos * div_term)

        return pe.view(-1, dim)  # Flatten to (H*W, dim)

    def forward(self, latent: Tensor, t: int, cache: Optional[KVCache] = None) -> Tensor:
        """
        Forward pass for training or inference.

        Args:
            latent (Tensor): Input latent tensor of shape (L, H, W, C).
            t (int): Current timestep in the diffusion process.
            cache (Optional[KVCache]): KV-cache for temporal and spatial attention layers.

        Returns:
            Tensor: Updated latent tensor after the forward pass.
        """
        batch_size, latent_seq_len, channels = latent.size()

        # Positional Embeddings for Temporal Dimension (Training or AR)
        if self.use_cyclic_tpe:
            temporal_positions = torch.arange(latent_seq_len, device=latent.device) % self.prefix_length_max
        else:
            temporal_positions = torch.arange(latent_seq_len, device=latent.device)
        temporal_embedding = self.tpe_embeddings(temporal_positions)

        # Add temporal and spatial positional embeddings
        latent = latent + temporal_embedding

        # Temporal KV-Cache: Pre-compute or reuse prior contexts
        if self.kv_cache_enabled and cache is not None:
            temporal_keys, temporal_values = cache.read_current_cache()
            latent = torch.cat([temporal_keys, latent], dim=0)

        # Apply Transformer Layers
        latent = self.transformer_layers(latent)

        # Update Temporal Cache in Inference Mode
        if self.kv_cache_enabled and cache is not None:
            cache.enqueue(latent, latent, is_temporal=True)

        return latent

    def train_mode(self) -> None:
        """Switch the model to training mode."""
        self.train()
        if self.kv_cache_enabled and self.temporal_kv_cache:
            self.temporal_kv_cache.reset()

    def inference_mode(self) -> None:
        """Switch the model to inference mode."""
        self.eval()
        if self.kv_cache_enabled and self.temporal_kv_cache:
            self.temporal_kv_cache.reset()

    def reset_cache(self) -> None:
        """Reset both temporal and spatial KV caches."""
        if self.temporal_kv_cache:
            self.temporal_kv_cache.reset()
        if self.spatial_kv_cache:
            self.spatial_kv_cache.reset()
