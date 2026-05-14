## inference.py

import torch
from torch import Tensor
from typing import Dict, Optional
from transformers import AutoTokenizer
from config import Config
from model import SpatialTemporalTransformer
from utils.kvcache import KVCache

class Inference:
    """
    Autoregressive video generation pipeline using the Ca2-VDM architecture.
    Handles efficient frame-wise generation with causal temporal attention,
    prefix-enhanced spatial attention, and KV-cache mechanisms.
    """

    def __init__(self, model: SpatialTemporalTransformer, config: Config) -> None:
        """
        Initialize the inference pipeline.

        Args:
            model (SpatialTemporalTransformer): The pre-trained Ca2-VDM model.
            config (Config): Configuration object.
        """
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval().to(self.device)

        # Extract inference-specific configurations
        self.chunk_length = self.config.get("model.chunk_length_t2v", 16)
        self.prefix_length_max = self.config.get("model.prefix_length_max_t2v", 49)
        self.kv_cache_enabled = self.config.get("model.kv_cache.enabled", True)
        self.vae_resolution = self.config.get("vae.resolution", 256)

        # Initialize temporal KV-cache
        if self.kv_cache_enabled:
            latent_resolution = self.vae_resolution // self.config.get("vae.downsample_factor", 8)
            latent_channels = self.model.latent_channels
            self.kv_cache = KVCache(
                max_length=self.prefix_length_max,
                spatial_size=(latent_resolution, latent_resolution),
                channels=latent_channels,
            )

        # Prepare text tokenizer for text-based tasks
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")

    def generate_video(
        self, initial_frame: Tensor, num_frames: int, cache: Optional[KVCache] = None
    ) -> Tensor:
        """
        Generate a video sequence autoregressively.

        Args:
            initial_frame (Tensor): The first frame or clean prefix of shape (1, 3, H, W).
            num_frames (int): Total number of frames to generate.
            cache (Optional[KVCache]): Temporal KV-cache object (defaults to self.kv_cache).

        Returns:
            Tensor: Generated video of shape (num_frames, 3, H, W).
        """
        assert initial_frame.shape[2] == self.vae_resolution and initial_frame.shape[3] == self.vae_resolution, (
            "Initial frame resolution must match VAE resolution."
        )

        # Initialize KV-cache
        cache = self.kv_cache if cache is None else cache
        if cache is not None:
            cache.reset()

        # Move initial frame to device
        initial_frame = initial_frame.to(self.device)

        # Encode the initial frame to latent space using pretrained VAE encoder
        latent = self.model.vae_encoder(initial_frame)
        prefix_frames = [latent]  # Cache initial frame for conditioning

        # Prepare TPEs cyclically if required
        num_chunks = num_frames // self.chunk_length
        generated_video = []

        # Autoregressive chunk-wise generation
        for chunk_idx in range(num_chunks):
            # Prepare prefix and denoising target for this chunk
            clean_prefix = torch.cat(prefix_frames, dim=0) if len(prefix_frames) > 1 else prefix_frames[0]
            target_chunk_length = min(self.chunk_length, num_frames - len(generated_video))

            # Create cyclic temporal positional embeddings (Cyclic-TPEs)
            t_embeddings = self._get_temporal_embeddings(len(clean_prefix) + target_chunk_length)

            # Forward pass through the model with causal temporal attention
            latent_chunk = self._denoise_latent_chunk(clean_prefix, target_chunk_length, cache, t_embeddings)

            # Decode latent chunk back to pixel space
            decoded_frames = self.model.vae_decoder(latent_chunk)
            generated_video.append(decoded_frames)

            # Update prefix frames and KV-cache for the next chunk
            prefix_frames.append(latent_chunk)
            if cache is not None:
                cache.enqueue(latent_chunk, latent_chunk, is_temporal=True)

        # Concatenate video chunks to form the full generated video
        return torch.cat(generated_video, dim=0)

    def _denoise_latent_chunk(
        self, clean_prefix: Tensor, target_chunk_length: int, cache: Optional[KVCache], t_embeddings: Tensor
    ) -> Tensor:
        """
        Denoise a latent chunk with prefix conditioning and causal temporal attention.

        Args:
            clean_prefix (Tensor): The clean prefix latent tensor.
            target_chunk_length (int): Length of the chunk to denoise.
            cache (Optional[KVCache]): Temporal KV-cache.
            t_embeddings (Tensor): Temporal positional embeddings.

        Returns:
            Tensor: Denoised latent chunk.
        """
        latent_input = torch.cat([clean_prefix, torch.zeros((target_chunk_length, *clean_prefix.shape[1:])).to(self.device)], dim=0)
        output = self.model.forward(latent_input, t=0, cache=cache)

        # Apply temporal positional embeddings
        return output + t_embeddings

    def _get_temporal_embeddings(self, seq_len: int) -> Tensor:
        """
        Compute temporal positional embeddings for a given sequence length.

        Args:
            seq_len (int): The sequence length for which TPE is required.

        Returns:
            Tensor: Temporal positional embeddings.
        """
        max_length = self.prefix_length_max + self.chunk_length
        positions = torch.arange(0, seq_len, device=self.device)
        if self.config.get("model.positional_embeddings.cyclic_tpe", True):
            positions = positions % max_length
        return self.model.tpe_embeddings(positions)

