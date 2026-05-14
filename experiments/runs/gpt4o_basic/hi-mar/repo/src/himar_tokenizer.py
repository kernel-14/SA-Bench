import torch
import torch.nn as nn

class HiMARTokenizer:
    def __init__(self, vae_model):
        """
        Tokenizer for hierarchical image generation using pre-trained Variational Autoencoder (VAE).

        Args:
            vae_model: Pre-trained VAE for embedding images into latent representations.
        """
        self.vae = vae_model

    def encode(self, image):
        """
        Encode the input image into latent representations.

        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Latent representations of shape (B, D, h, w).
        """
        with torch.no_grad():
            latent = self.vae.encode(image)
        return latent

    def decode(self, latent):
        """
        Decode latent representations back into images.

        Args:
            latent (torch.Tensor): Latent representations of shape (B, D, h, w).

        Returns:
            torch.Tensor: Reconstructed image tensor of shape (B, C, H, W).
        """
        with torch.no_grad():
            image = self.vae.decode(latent)
        return image
