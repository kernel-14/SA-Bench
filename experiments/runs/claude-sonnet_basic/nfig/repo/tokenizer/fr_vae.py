"""
Frequency-guided Residual-quantized VAE (FR-VAE)
Core image tokenizer for NFIG framework.

Based on: "NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering"

Key design:
- scale_factors = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16] represent the spatial resolution
  (h_i, w_i) of each frequency band's token map.
- Band i has scale_factors[i]^2 tokens.
- Total tokens = 1+4+9+16+25+36+64+100+169+256 = 680 for 256x256 images.
- The lowest frequency band (band 0) has just 1 token (1x1 spatial resolution).
- The highest frequency band (band 9) has 256 tokens (16x16 spatial resolution,
  matching the full latent size for 256x256 images with 16x downsampling).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class FrequencyDecomposer(nn.Module):
    """
    Decomposes a feature map into n frequency bands using FFT-based masks.

    The frequency band boundaries are computed based on the token counts:
    sigma_i = sigma_{i-1} + (h_i * w_i) / sum(h_j * w_j) * sigma_max

    where h_i = w_i = scale_factors[i] (spatial resolution of band i).
    """

    def __init__(self, n_bands: int, scale_factors: List[int]):
        """
        Args:
            n_bands: number of frequency bands
            scale_factors: list of spatial resolutions [s1, s2, ..., sn] where
                           band i has s_i x s_i tokens.
        """
        super().__init__()
        self.n_bands = n_bands
        self.scale_factors = scale_factors

    def _make_radial_mask(self, H: int, W: int, low: float, high: float,
                          device) -> torch.Tensor:
        """
        Create a 2-D radial frequency mask that passes frequencies in [low, high).
        Frequencies are normalised so that the maximum radial frequency = 1.
        """
        cy, cx = H // 2, W // 2
        ys = torch.arange(H, device=device).float() - cy
        xs = torch.arange(W, device=device).float() - cx
        r = torch.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
        max_r = math.sqrt(cy ** 2 + cx ** 2) + 1e-6
        r = r / max_r
        mask = ((r >= low) & (r < high)).float()
        return mask

    def _compute_band_boundaries(self) -> List[Tuple[float, float]]:
        """
        Compute frequency band boundaries based on the token counts.
        sigma_i = sigma_{i-1} + (h_i * w_i) / sum(h_j * w_j) * sigma_max
        where sigma_max = 1 (normalised).
        """
        areas = [s * s for s in self.scale_factors]
        total = sum(areas)
        boundaries = [0.0]
        for a in areas:
            boundaries.append(boundaries[-1] + a / total)
        return [(boundaries[i], boundaries[i + 1]) for i in range(self.n_bands)]

    def forward(self, f: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            f: feature map (B, C, H', W')
        Returns:
            list of n frequency-band feature maps, each (B, C, H', W')
        """
        B, C, H, W = f.shape
        boundaries = self._compute_band_boundaries()

        # FFT (shift so DC is at centre)
        F_freq = torch.fft.fft2(f)
        F_shifted = torch.fft.fftshift(F_freq, dim=(-2, -1))

        components = []
        for low, high in boundaries:
            mask = self._make_radial_mask(H, W, low, high, f.device)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            F_masked = F_shifted * mask
            F_unshifted = torch.fft.ifftshift(F_masked, dim=(-2, -1))
            f_band = torch.fft.ifft2(F_unshifted).real
            components.append(f_band)

        return components  # list of (B, C, H', W')


class FrequencyComposer(nn.Module):
    """Reconstructs the full feature map by summing interpolated frequency components."""

    def forward(self, components: List[torch.Tensor],
                target_H: int, target_W: int) -> torch.Tensor:
        """
        Args:
            components: list of (B, C, h_i, w_i) tensors
            target_H, target_W: target spatial size
        Returns:
            reconstructed feature map (B, C, target_H, target_W)
        """
        out = None
        for comp in components:
            if comp.shape[-2] != target_H or comp.shape[-1] != target_W:
                comp_up = F.interpolate(comp, size=(target_H, target_W),
                                        mode="bilinear", align_corners=False)
            else:
                comp_up = comp
            out = comp_up if out is None else out + comp_up
        return out


class VectorQuantizer(nn.Module):
    """Standard VQ with straight-through estimator. Shared codebook across all bands."""

    def __init__(self, codebook_size: int, embed_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, C, h, w) continuous feature map
        Returns:
            z_q: (B, C, h, w) quantized feature map
            indices: (B, h*w) codebook indices
            loss: scalar VQ loss
        """
        B, C, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)
        d = (z_flat.pow(2).sum(1, keepdim=True)
             - 2 * z_flat @ self.embedding.weight.t()
             + self.embedding.weight.pow(2).sum(1))
        indices = d.argmin(dim=1)
        z_q_flat = self.embedding(indices)
        z_q = z_q_flat.reshape(B, h, w, C).permute(0, 3, 1, 2)
        indices = indices.reshape(B, h * w)
        loss = (F.mse_loss(z_q.detach(), z)
                + self.commitment_cost * F.mse_loss(z_q, z.detach()))
        z_q = z + (z_q - z).detach()
        return z_q, indices, loss

    def lookup(self, indices: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Decode indices back to feature map."""
        B = indices.shape[0]
        z_q = self.embedding(indices)
        return z_q.reshape(B, h, w, self.embed_dim).permute(0, 3, 1, 2)


class FrequencyResidualQuantizer(nn.Module):
    """
    Frequency-guided residual quantization across multiple bands.

    For each band i:
    1. Downsample the band's feature map to (h_i, w_i) = (scale_factors[i], scale_factors[i])
    2. Compute residual from accumulated previous bands
    3. Quantize the residual
    """

    def __init__(self, codebook_size: int, embed_dim: int,
                 scale_factors: List[int], commitment_cost: float = 0.25):
        """
        Args:
            codebook_size: K, number of codebook entries
            embed_dim: C, feature dimension
            scale_factors: list of spatial resolutions per band [s1, ..., sn]
                           where band i has s_i x s_i tokens
            commitment_cost: VQ commitment loss weight
        """
        super().__init__()
        self.scale_factors = scale_factors
        self.n_bands = len(scale_factors)
        self.quantizer = VectorQuantizer(codebook_size, embed_dim, commitment_cost)

    def forward(self, components: List[torch.Tensor]):
        """
        Args:
            components: list of n (B, C, H', W') frequency-band feature maps
        Returns:
            quantized_maps: list of (B, C, H', W') upsampled quantized maps
            all_indices: list of (B, h_i*w_i) index tensors
            total_vq_loss: scalar
        """
        B, C, H, W = components[0].shape
        quantized_maps = []
        all_indices = []
        total_vq_loss = 0.0
        R_prev = None  # accumulated residual at full resolution

        for i, (f_band, s) in enumerate(zip(components, self.scale_factors)):
            hi, wi = s, s  # spatial resolution of band i

            # Downsample band feature to (hi, wi)
            v_i = F.interpolate(f_band, size=(hi, wi), mode="bilinear", align_corners=False)

            # Compute target for quantization
            if i == 0:
                target = v_i
            else:
                # Downsample accumulated residual to (hi, wi) and add current band
                R_prev_down = F.interpolate(R_prev, size=(hi, wi),
                                            mode="bilinear", align_corners=False)
                target = R_prev_down + v_i

            # Quantize
            v_q, indices, vq_loss = self.quantizer(target)
            total_vq_loss = total_vq_loss + vq_loss
            all_indices.append(indices)

            # Upsample quantized map back to (H, W)
            v_q_up = F.interpolate(v_q, size=(H, W), mode="bilinear", align_corners=False)
            quantized_maps.append(v_q_up)

            # Update residual: R_i = R_{i-1} + (f_band - v_q_up)
            if i == 0:
                R_prev = f_band - v_q_up
            else:
                R_prev = R_prev + (f_band - v_q_up)

        return quantized_maps, all_indices, total_vq_loss

    def decode_indices(self, all_indices: List[torch.Tensor],
                       H: int, W: int) -> List[torch.Tensor]:
        """Decode a list of index tensors back to upsampled feature maps."""
        decoded = []
        for indices, s in zip(all_indices, self.scale_factors):
            hi, wi = s, s
            v_q = self.quantizer.lookup(indices, hi, wi)
            v_q_up = F.interpolate(v_q, size=(H, W), mode="bilinear", align_corners=False)
            decoded.append(v_q_up)
        return decoded


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    """
    CNN encoder: image -> latent feature map.
    For 256x256 images, downsamples by 16x to produce 16x16 latent.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 128,
                 channel_mult=(1, 2, 4, 4), latent_dim: int = 256,
                 n_res_blocks: int = 2):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        channels = base_channels
        layers = []
        for mult in channel_mult:
            out_ch = base_channels * mult
            for _ in range(n_res_blocks):
                layers.append(ResBlock(channels))
            if channels != out_ch:
                layers.append(nn.Conv2d(channels, out_ch, 1))
                channels = out_ch
            layers.append(Downsample(channels))
        self.down = nn.Sequential(*layers)
        self.mid = nn.Sequential(ResBlock(channels), ResBlock(channels))
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        h = self.down(h)
        h = self.mid(h)
        h = self.norm_out(h)
        h = F.silu(h)
        return self.conv_out(h)


class Decoder(nn.Module):
    """CNN decoder: latent feature map -> image."""

    def __init__(self, out_channels: int = 3, base_channels: int = 128,
                 channel_mult=(1, 2, 4, 4), latent_dim: int = 256,
                 n_res_blocks: int = 2):
        super().__init__()
        channels = base_channels * channel_mult[-1]
        self.conv_in = nn.Conv2d(latent_dim, channels, 3, padding=1)
        self.mid = nn.Sequential(ResBlock(channels), ResBlock(channels))
        layers = []
        for mult in reversed(channel_mult):
            out_ch = base_channels * mult
            layers.append(Upsample(channels))
            if channels != out_ch:
                layers.append(nn.Conv2d(channels, out_ch, 1))
                channels = out_ch
            for _ in range(n_res_blocks):
                layers.append(ResBlock(channels))
        self.up = nn.Sequential(*layers)
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid(h)
        h = self.up(h)
        h = self.norm_out(h)
        h = F.silu(h)
        return torch.tanh(self.conv_out(h))


class FRVAE(nn.Module):
    """
    Frequency-guided Residual-quantized VAE.

    scale_factors from paper: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    These represent the spatial resolution (h_i, w_i) of each band's token map.
    Total tokens = 1+4+9+16+25+36+64+100+169+256 = 680 for 256x256 images.

    The encoder downsamples 256x256 -> 16x16 (16x downsampling).
    The highest frequency band (band 9) has 16x16 tokens = full latent resolution.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        base_channels: int = 128,
        channel_mult=(1, 2, 4, 4),
        n_res_blocks: int = 2,
        codebook_size: int = 4096,
        scale_factors: Optional[List[int]] = None,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        if scale_factors is None:
            # From paper: spatial resolutions per band
            scale_factors = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
        self.scale_factors = scale_factors
        self.n_bands = len(scale_factors)
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, base_channels, channel_mult, latent_dim, n_res_blocks)
        self.decoder = Decoder(in_channels, base_channels, channel_mult, latent_dim, n_res_blocks)
        self.decomposer = FrequencyDecomposer(self.n_bands, scale_factors)
        self.composer = FrequencyComposer()
        self.quantizer = FrequencyResidualQuantizer(codebook_size, latent_dim, scale_factors, commitment_cost)

    def encode(self, x: torch.Tensor):
        """
        Encode image to quantized frequency tokens.
        Returns:
            all_indices: list of (B, s_i^2) index tensors per band
            vq_loss: scalar
        """
        f = self.encoder(x)
        components = self.decomposer(f)
        _, all_indices, vq_loss = self.quantizer(components)
        return all_indices, vq_loss

    def decode(self, all_indices: List[torch.Tensor], latent_H: int, latent_W: int) -> torch.Tensor:
        """Decode frequency token indices back to image."""
        decoded_maps = self.quantizer.decode_indices(all_indices, latent_H, latent_W)
        f_hat = self.composer(decoded_maps, latent_H, latent_W)
        return self.decoder(f_hat)

    def forward(self, x: torch.Tensor):
        """
        Full forward pass for training.
        Returns:
            x_hat: reconstructed image
            vq_loss: VQ commitment loss
            f: encoder feature map (for frequency-guided loss)
            f_hat: reconstructed feature map
        """
        f = self.encoder(x)
        B, C, H, W = f.shape
        components = self.decomposer(f)
        quantized_maps, all_indices, vq_loss = self.quantizer(components)
        f_hat = self.composer(quantized_maps, H, W)
        x_hat = self.decoder(f_hat)
        return x_hat, vq_loss, f, f_hat

    def get_token_counts(self, latent_H: int = None, latent_W: int = None) -> List[int]:
        """Return number of tokens per frequency band.
        
        Note: token counts are determined by scale_factors, not latent size.
        latent_H and latent_W are kept for API compatibility but not used.
        """
        return [s * s for s in self.scale_factors]

    def total_tokens(self, latent_H: int = None, latent_W: int = None) -> int:
        return sum(self.get_token_counts())
