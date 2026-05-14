"""
Frequency-guided Residual-quantized VAE (FR-VAE) for NFIG.
Implements the image tokenizer described in Section 3.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math

from .frequency_ops import FrequencyDecomposer, FrequencyComposer, FrequencyResidualQuantizer


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels: int, num_groups: int = 32):
        super().__init__(num_groups=num_groups, num_channels=num_channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = GroupNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = GroupNorm(out_channels)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return F.relu(x + residual)


class Encoder(nn.Module):
    """VQGAN-style encoder that produces a feature map f = E(x)."""

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        hidden_channels: int = 128,
        latent_channels: int = 256,
        num_res_blocks: int = 2,
        downsampling_factor: int = 16,
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_channels = latent_channels

        num_downsamples = int(math.log2(downsampling_factor))
        channels = [hidden_channels * (2**i) for i in range(num_downsamples + 1)]
        channels = [min(c, latent_channels) for c in channels]

        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        for i in range(num_downsamples):
            block = nn.ModuleList()
            ch_in = channels[i]
            ch_out = channels[i + 1]
            for _ in range(num_res_blocks):
                block.append(ResidualBlock(ch_in, ch_in))
            block.append(
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1)
            )
            self.down_blocks.append(block)

        # Final blocks
        self.final_blocks = nn.Sequential(
            *[ResidualBlock(channels[-1], channels[-1]) for _ in range(num_res_blocks)],
            GroupNorm(channels[-1]),
            nn.ReLU(),
            nn.Conv2d(channels[-1], latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block_list in self.down_blocks:
            for layer in block_list:
                x = layer(x)
        x = self.final_blocks(x)
        return x


class Decoder(nn.Module):
    """VQGAN-style decoder that reconstructs the image from latent features."""

    def __init__(
        self,
        image_size: int = 256,
        out_channels: int = 3,
        hidden_channels: int = 128,
        latent_channels: int = 256,
        num_res_blocks: int = 2,
        downsampling_factor: int = 16,
    ):
        super().__init__()
        self.image_size = image_size

        num_upsamples = int(math.log2(downsampling_factor))
        channels = [hidden_channels * (2**i) for i in range(num_upsamples + 1)]
        channels = [min(c, latent_channels) for c in channels]
        channels = channels[::-1]

        # Initial conv
        self.conv_in = nn.Conv2d(latent_channels, channels[0], kernel_size=3, padding=1)
        self.initial_blocks = nn.Sequential(
            *[ResidualBlock(channels[0], channels[0]) for _ in range(num_res_blocks)]
        )

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in range(num_upsamples):
            block = nn.ModuleList()
            ch_in = channels[i]
            ch_out = channels[i + 1]
            block.append(
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1)
            )
            for _ in range(num_res_blocks):
                block.append(ResidualBlock(ch_out, ch_out))
            self.up_blocks.append(block)

        # Final conv
        self.final = nn.Sequential(
            GroupNorm(channels[-1]),
            nn.ReLU(),
            nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.conv_in(z)
        z = self.initial_blocks(z)
        for block_list in self.up_blocks:
            for layer in block_list:
                z = layer(z)
        z = self.final(z)
        return z


class FRVAE(nn.Module):
    """
    Frequency-guided Residual-quantized VAE.
    Complete image tokenizer with frequency decomposition, residual quantization,
    and reconstruction via frequency composer + decoder.
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        hidden_channels: int = 128,
        latent_channels: int = 256,
        codebook_size: int = 4096,
        codebook_dim: int = 32,
        downsampling_factor: int = 16,
        scale_factors: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.feature_size = image_size // downsampling_factor
        self.scale_factors = scale_factors
        self.num_bands = len(scale_factors)

        # Encoder
        self.encoder = Encoder(
            image_size=image_size,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            downsampling_factor=downsampling_factor,
        )

        # Frequency decomposition
        self.freq_decomposer = FrequencyDecomposer(
            feature_size=self.feature_size,
            scale_factors=scale_factors,
            num_channels=latent_channels,
        )

        # Residual quantization
        self.residual_quantizer = FrequencyResidualQuantizer(
            feature_size=self.feature_size,
            scale_factors=scale_factors,
            codebook_size=codebook_size,
            latent_dim=codebook_dim,
            num_channels=latent_channels,
        )

        # Frequency composer
        self.freq_composer = FrequencyComposer(target_size=self.feature_size)

        # Decoder
        self.decoder = Decoder(
            image_size=image_size,
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            downsampling_factor=downsampling_factor,
        )

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Encode image to frequency tokens.
        Args:
            x: (B, 3, 256, 256)
        Returns:
            quantized_components: List of quantized feature maps per scale
            all_tokens: List of token tensors per scale
            commit_loss: Total commitment loss
        """
        f = self.encoder(x)

        # Decompose into frequency components
        freq_components = self.freq_decomposer(f)

        # Residual quantize each component
        quantized_components, all_tokens, commit_loss = self.residual_quantizer(freq_components)

        return quantized_components, all_tokens, commit_loss

    def decode(self, quantized_components: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode quantized frequency components back to an image.
        Args:
            quantized_components: List of quantized feature maps (B, latent_channels, h_i, w_i)
        Returns:
            Reconstructed image (B, 3, 256, 256)
        """
        composed = self.freq_composer(quantized_components)
        return self.decoder(composed)

    def decode_from_tokens(self, token_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode from token tensors (from transformer generation) to image.
        Reconstructs quantized components using codebook and scale decoders.

        Args:
            token_list: List of (B, h_i, w_i) tensors with codebook indices
        Returns:
            Reconstructed image (B, 3, 256, 256)
        """
        codebook = self.get_codebook()
        quantized_components = []
        for i, tokens in enumerate(token_list):
            B, h, w = tokens.shape
            # Lookup codebook: (B, h, w) -> (B, h, w, latent_dim)
            z_q = F.embedding(tokens, codebook)
            z_q = z_q.permute(0, 3, 1, 2)  # (B, latent_dim, h, w)
            # Apply scale decoder to convert back to num_channels
            v_decoded = self.residual_quantizer.scale_decoders[i](z_q)
            quantized_components.append(v_decoded)
        return self.decode(quantized_components)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        quantized_components, all_tokens, commit_loss = self.encode(x)
        reconstructed = self.decode(quantized_components)
        return reconstructed, all_tokens, commit_loss

    def get_codebook(self) -> torch.Tensor:
        return self.residual_quantizer.codebook

    def get_total_tokens(self) -> int:
        total = 0
        for s in self.scale_factors:
            total += s * s
        return total
