"""
vae_tokenizer.py

VAETokenizer – loads the pre‑trained MAR KL‑16 Variational Autoencoder,
provides encoding/decoding of images to/from continuous latent tokens,
and performs no vector quantization.

The VAE is kept frozen and used under ``torch.no_grad()``.  Two image
resolutions are supported: 128 × 128 (low‑res, 8 × 8 latent) and
256 × 256 (high‑res, 16 × 16 latent).  The latent dimension is always 16.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Lightweight KL‑16 Autoencoder (fallback / trainable placeholder)
# ---------------------------------------------------------------------------

def _normalize(in_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm that falls back to batch size when channels are few."""
    # Ensure number of groups does not exceed channels
    gn = nn.GroupNorm(num_groups=min(num_groups, in_channels), num_channels=in_channels)
    return gn


class _ResidualBlock(nn.Module):
    """Simple residual block with two convolutions, used in encoder/decoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + residual


class _DownsampleBlock(nn.Module):
    """Downsamples by 2× via strided convolution + residual block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.res = _ResidualBlock(out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.res(x)
        return x


class _UpsampleBlock(nn.Module):
    """Upsamples by 2× via nearest-neighbour + conv + residual block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res = _ResidualBlock(out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        x = self.res(x)
        return x


class _KLEncoder(nn.Module):
    """Encoder: 3 → 16 latent channels, 16× downsampling."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=1)

        self.down1 = _DownsampleBlock(128, 128)   # 128 -> 64
        self.down2 = _DownsampleBlock(128, 256)   # 64  -> 32
        self.down3 = _DownsampleBlock(256, 512)   # 32  -> 16
        self.down4 = _DownsampleBlock(512, 512)   # 16  -> 8

        # Final projections to mean and log‑variance
        self.norm_out = _normalize(512)
        self.conv_mean = nn.Conv2d(512, latent_dim, kernel_size=3, padding=1)
        self.conv_logvar = nn.Conv2d(512, latent_dim, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        h = self.norm_out(h)
        h = F.silu(h)
        mean = self.conv_mean(h)
        logvar = self.conv_logvar(h)
        return mean, logvar


class _KLDecoder(nn.Module):
    """Decoder: 16 latent channels → 3 RGB, 16× upsampling."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_dim, 512, kernel_size=3, padding=1)

        self.up1 = _UpsampleBlock(512, 512)  # 8  -> 16
        self.up2 = _UpsampleBlock(512, 256)  # 16 -> 32
        self.up3 = _UpsampleBlock(256, 128)  # 32 -> 64
        self.up4 = _UpsampleBlock(128, 128)  # 64 -> 128

        self.norm_out = _normalize(128)
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.norm_out(h)
        h = F.silu(h)
        x_rec = self.conv_out(h)
        return x_rec


class KL16Autoencoder(nn.Module):
    """
    A self‑contained KL‑16 VAE with 16‑channel latents and 16× downsampling.

    This can serve as a fallback if the pre‑trained checkpoint is missing,
    but training from scratch would be required to match Hi‑MAR’s performance.
    """

    def __init__(self, latent_dim: int = 16, scale_factor: float = 1.0):
        super().__init__()
        self.encoder = _KLEncoder(latent_dim)
        self.decoder = _KLDecoder(latent_dim)
        # Latent scaling factor, stored to allow compatibility with checkpoints
        # that apply a post‑encoding multiplier (e.g., 0.18215 in SD).
        self.scale_factor = scale_factor

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode an image batch (range [-1, 1]) into latent mean (range unnormalised).

        Returns the mean of the approximate posterior (the “token” used in Hi‑MAR).
        """
        mean, _ = self.encoder(x)
        return mean

    def decode(self, z: Tensor) -> Tensor:
        """
        Reconstruct an image from latent vector.
        Assumes the latent is the unscaled representation (scale_factor applied
        internally if needed).
        """
        z = z * self.scale_factor   # apply scaling that was stripped after encoding
        x_rec = self.decoder(z)
        return x_rec

    @property
    def latent_dim(self) -> int:
        return self.encoder.conv_mean.out_channels

    @property
    def downsample_factor(self) -> int:
        # Hard‑coded 16, but could compute from strides
        return 16


# ---------------------------------------------------------------------------
#  VAETokenizer wrapper
# ---------------------------------------------------------------------------

class VAETokenizer:
    """
    Loads a pre‑trained MAR KL‑16 VAE (or falls back to random weights) and
    provides encoding/decoding with automatic normalization.

    Usage::

        tok = VAETokenizer(vae_path, device='cuda')
        latents = tok.encode(images)          # (B, 16, h, w)
        rec = tok.decode(latents)             # (B, 3, H, W) in [0,1]
        h, w = tok.get_hw(256)               # (16, 16)
    """

    def __init__(self, vae_path: str, device: torch.device):
        """
        Args:
            vae_path:  Path to the saved VAE checkpoint (``.pt``).
            device:    Torch device to run the VAE on.
        """
        self.device = device
        self.model: nn.Module = self._load_vae(vae_path)
        self._freeze_model()
        # Scaling factor applied to latents before decoding.  If the loaded model
        # already contains a ``scale_factor`` attribute, use it; otherwise default 1.0.
        self.latent_scale = getattr(self.model, 'scale_factor', 1.0)
        # Also store a buffer (on CPU) for safe device migration
        self.register_buffer = nn.Module()  # we don't actually need register_buffer as non‑nn.Module, but we can't
        # Instead, simply keep as a float.  For device consistency we use a zero‑dim tensor.
        self.latent_scale_tensor = torch.tensor(self.latent_scale).to(device)

    # ------------------------------------------------------------------ loading
    def _load_vae(self, vae_path: str) -> nn.Module:
        """
        Load the VAE from disk.  Supports:
          - Full ``nn.Module`` objects.
          - State dictionaries: creates a ``KL16Autoencoder`` and loads weights.
          - Missing file: falls back to a freshly initialised ``KL16Autoencoder``
            with a warning.
        """
        try:
            checkpoint = torch.load(vae_path, map_location=self.device)
        except FileNotFoundError:
            logger.warning(
                "VAE checkpoint not found at '%s'. Creating a randomly initialised "
                "KL‑16 autoencoder.  Fine‑tuning or training from scratch will be "
                "necessary to reproduce Hi‑MAR results.",
                vae_path,
            )
            autoenc = KL16Autoencoder(latent_dim=16, scale_factor=1.0)
            autoenc.to(self.device)
            return autoenc

        if isinstance(checkpoint, nn.Module):
            model = checkpoint
        elif isinstance(checkpoint, dict):
            # Attempt to carve out a state_dict
            if "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state = checkpoint["model"]
            else:
                state = checkpoint
            # Create a default model and load
            model = KL16Autoencoder(latent_dim=16, scale_factor=1.0)
            # Possibly the checkpoint contains keys with 'module.' prefix
            state = self._strip_ddp_prefix(state)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                logger.info(
                    "VAE state_dict loaded with mismatched keys. "
                    "Missing: %s, Unexpected: %s",
                    missing, unexpected,
                )
        else:
            raise TypeError(
                f"Unsupported VAE checkpoint type: {type(checkpoint)}. "
                "Expected nn.Module or dict."
            )

        model.to(self.device)
        return model

    @staticmethod
    def _strip_ddp_prefix(state_dict: dict) -> dict:
        """Remove 'module.' prefix inserted by DistributedDataParallel."""
        stripped = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith("module."):
                new_key = new_key[7:]
            stripped[new_key] = v
        return stripped

    def _freeze_model(self) -> None:
        """Set model to eval mode and freeze all parameters."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------ encode / decode

    @torch.no_grad()
    def encode(self, image: Tensor) -> Tensor:
        """
        Convert a batch of RGB images into continuous latent tokens.

        Args:
            image: Tensor of shape (B, 3, H, W), values in [0, 1].

        Returns:
            Latent tensor of shape (B, 16, H//16, W//16), unscaled.
        """
        # Normalise to [-1, 1] – standard VAE input range
        img = (image.to(self.device, non_blocking=True) - 0.5) * 2.0
        if hasattr(self.model, "encode"):
            latent = self.model.encode(img)
        else:
            # Fallback if model is not our defined class but still has encoder/decoder
            latent = self.model.encoder(img)
            if isinstance(latent, tuple):
                latent = latent[0]  # (mean, logvar)
        return latent

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        """
        Reconstruct RGB images from latent tokens.

        Args:
            latent: Tensor of shape (B, 16, h, w) in the unscaled latent space
                    (i.e., the output of ``encode``).

        Returns:
            RGB images tensor of shape (B, 3, H, W) with values in [0, 1].
        """
        z = latent.to(self.device, non_blocking=True)
        # Apply the latent scaling that was stripped after encoding
        if hasattr(self.model, "decode"):
            rec = self.model.decode(z)
        else:
            rec = self.model.decoder(z)

        # Transform from [-1, 1] → [0, 1]
        rec = (rec + 1.0) / 2.0
        rec = torch.clamp(rec, 0.0, 1.0)
        return rec

    # ------------------------------------------------------------------ utility

    def get_hw(self, res: int) -> Tuple[int, int]:
        """
        Return latent spatial dimensions for a given image resolution.

        The VAE downsampling factor is taken from the underlying model if
        available, otherwise 16.

        Args:
            res: Image side length (128 or 256).

        Returns:
            Tuple (h, w) of the latent map.
        """
        downsample = getattr(self.model, 'downsample_factor', 16)
        h = w = res // downsample
        return h, w

    def to(self, device: torch.device) -> 'VAETokenizer':
        """Move the underlying VAE to a different device (non‑module helper)."""
        self.device = device
        self.model.to(device)
        self.latent_scale_tensor = self.latent_scale_tensor.to(device)
        return self

