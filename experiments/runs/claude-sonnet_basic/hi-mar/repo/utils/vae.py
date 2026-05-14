"""
VAE utilities for Hi-MAR.
Loads the KL-16 VAE from MAR for image tokenization.
"""

import torch
import torch.nn as nn


def load_vae(vae_path):
    """
    Load the KL-16 VAE used in MAR.

    The KL-16 VAE encodes 256x256 images into 16x16 latent maps with 16 channels.
    For 128x128 images, it produces 8x8 latent maps.

    Args:
        vae_path: path to the VAE checkpoint

    Returns:
        vae: loaded VAE model
    """
    try:
        # Try to load from diffusers
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(vae_path)
        return vae
    except Exception:
        pass

    try:
        # Try to load from checkpoint
        from diffusers import AutoencoderKL
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=('DownEncoderBlock2D',) * 4,
            up_block_types=('UpDecoderBlock2D',) * 4,
            block_out_channels=(128, 256, 512, 512),
            latent_channels=16,
            layers_per_block=2,
        )
        checkpoint = torch.load(vae_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['state_dict'])
        else:
            vae.load_state_dict(checkpoint)
        return vae
    except Exception as e:
        raise RuntimeError(f'Failed to load VAE from {vae_path}: {e}')


class VAETokenizer:
    """
    Wrapper around VAE for tokenizing images into latent sequences.
    """

    def __init__(self, vae, vae_stride=16, latent_channels=16, scale_factor=0.18215):
        self.vae = vae
        self.vae_stride = vae_stride
        self.latent_channels = latent_channels
        self.scale_factor = scale_factor

    @torch.no_grad()
    def encode(self, images):
        """
        Encode images to latent tokens.

        Args:
            images: [B, 3, H, W] images in [-1, 1]

        Returns:
            tokens: [B, N, C] latent tokens
        """
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.scale_factor

        B, C, H, W = latents.shape
        tokens = latents.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return tokens

    @torch.no_grad()
    def decode(self, tokens, img_size):
        """
        Decode latent tokens to images.

        Args:
            tokens: [B, N, C] latent tokens
            img_size: output image size

        Returns:
            images: [B, 3, H, W] images in [-1, 1]
        """
        B, N, C = tokens.shape
        h = w = img_size // self.vae_stride

        latents = tokens.reshape(B, h, w, C).permute(0, 3, 1, 2)
        latents = latents / self.scale_factor

        images = self.vae.decode(latents).sample
        return images
