import torch
import torch.nn as nn
from himar_tokenizer import HiMARTokenizer
from himar_transformer import ScaleAwareTransformerBlock
from himar_diffusion import MLPDiffusionHead, DiffusionTransformerHead

class HiMAR(nn.Module):
    def __init__(self, vae_model, hidden_size, diffusion_hidden_size):
        """
        Hi-MAR Model implementing hierarchical masked autoregressive modeling.

        Args:
            vae_model: Pre-trained VAE tokenizer.
            hidden_size (int): Size of hidden layers in transformer blocks.
            diffusion_hidden_size (int): Size of hidden layers in diffusion heads.
        """
        super(HiMAR, self).__init__()
        self.tokenizer = HiMARTokenizer(vae_model)

        # Transformer blocks and diffusion heads
        self.low_res_transformer = ScaleAwareTransformerBlock(hidden_size)
        self.high_res_transformer = ScaleAwareTransformerBlock(hidden_size)
        self.mlp_diffusion_head = MLPDiffusionHead(hidden_size, diffusion_hidden_size)
        self.transformer_diffusion_head = DiffusionTransformerHead(hidden_size)

    def forward(self, low_res_image, high_res_image, scale_vector):
        """
        Forward pass for Hi-MAR model.

        Args:
            low_res_image (torch.Tensor): Low-resolution input images.
            high_res_image (torch.Tensor): High-resolution input images.
            scale_vector (torch.Tensor): Scale vector for normalization.

        Returns:
            torch.Tensor: Generated high-resolution image tokens.
        """
        # Tokenize inputs
        low_res_tokens = self.tokenizer.encode(low_res_image)
        high_res_tokens = self.tokenizer.encode(high_res_image)

        # Phase 1: Low-resolution modeling
        conditional_low_res_tokens = self.low_res_transformer(low_res_tokens, scale_vector)
        optimized_low_res_tokens = self.mlp_diffusion_head(conditional_low_res_tokens)

        # Phase 2: High-resolution refinement
        conditional_high_res_tokens = self.high_res_transformer(high_res_tokens + optimized_low_res_tokens, scale_vector)
        refined_high_res_tokens = self.transformer_diffusion_head(conditional_high_res_tokens)

        return refined_high_res_tokens
