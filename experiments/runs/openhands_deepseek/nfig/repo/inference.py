"""
Inference script for NFIG: Generate images using the trained model.
Uses Classifier-Free Guidance (CFG=4.5) and top-k sampling (k=990)
as specified in the paper.
"""

import os
import sys
import argparse
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NFIGTransformerConfig, FRVAEConfig
from models.fr_vae import FRVAE
from models.transformer import NFIGTransformer
from utils.setup import load_checkpoint
from data import IMAGENET_MEAN, IMAGENET_STD


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """Convert normalized images back to [0, 1] range for saving."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return torch.clamp(images * std + mean, 0, 1)


class NFIGGenerator:
    """Complete NFIG image generator."""

    def __init__(
        self,
        vae_checkpoint: str,
        transformer_checkpoint: str,
        vae_config: Optional[FRVAEConfig] = None,
        transformer_config: Optional[NFIGTransformerConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.vae_config = vae_config or FRVAEConfig()
        self.transformer_config = transformer_config or NFIGTransformerConfig()

        # Load VAE
        self.vae = FRVAE(
            image_size=self.vae_config.image_size,
            latent_channels=self.vae_config.latent_channels,
            codebook_size=self.vae_config.codebook_size,
            codebook_dim=self.vae_config.codebook_dim,
            downsampling_factor=self.vae_config.downsampling_factor,
            scale_factors=self.vae_config.scale_factors,
        ).to(self.device)
        load_checkpoint(vae_checkpoint, self.vae, device=self.device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # Load Transformer
        self.transformer = NFIGTransformer(
            vocab_size=self.transformer_config.vocab_size,
            hidden_dim=self.transformer_config.hidden_dim,
            num_heads=self.transformer_config.num_heads,
            num_layers=self.transformer_config.num_layers,
            num_classes=self.transformer_config.num_classes,
            scale_factors=self.transformer_config.scale_factors,
            feature_map_size=self.transformer_config.feature_map_size,
            dropout=0.0,  # No dropout during inference
            use_adaln=self.transformer_config.use_adaln,
        ).to(self.device)
        load_checkpoint(transformer_checkpoint, self.transformer, device=self.device)
        self.transformer.eval()
        for p in self.transformer.parameters():
            p.requires_grad = False

        print(f"Loaded models. VAE: {sum(p.numel() for p in self.vae.parameters()):,} params")
        print(f"Transformer: {sum(p.numel() for p in self.transformer.parameters()):,} params")

    @torch.no_grad()
    def generate(
        self,
        class_ids: torch.Tensor,
        cfg_scale: float = 4.5,
        top_k: int = 990,
        temperature: float = 1.0,
        use_cfg: bool = True,
    ) -> torch.Tensor:
        """
        Generate images for given class labels.

        Args:
            class_ids: (B,) tensor of class indices [0, 999]
            cfg_scale: Classifier-free guidance scale (default 4.5 from paper)
            top_k: Top-k sampling (default 990 from paper)
            temperature: Sampling temperature
            use_cfg: Whether to use CFG

        Returns:
            Generated images (B, 3, 256, 256) in normalized [-1, 1] range
        """
        class_ids = class_ids.to(self.device)

        # Generate tokens via transformer
        tokens = self.transformer.generate(
            class_ids=class_ids,
            cfg_scale=cfg_scale,
            top_k=top_k,
            temperature=temperature,
            use_cfg=use_cfg,
        )

        # Decode tokens to image using the VAE's decode_from_tokens method
        images = self.vae.decode_from_tokens(tokens)
        return images

    @torch.no_grad()
    def generate_and_save(
        self,
        class_ids: torch.Tensor,
        output_path: str,
        nrow: int = 8,
        **generate_kwargs,
    ):
        """Generate images and save as a grid."""
        images = self.generate(class_ids, **generate_kwargs)
        images = denormalize_images(images)

        vutils.save_image(images, output_path, nrow=nrow, normalize=False)
        print(f"Saved {len(class_ids)} images to {output_path}")

    @torch.no_grad()
    def generate_single(
        self,
        class_id: int,
        num_samples: int = 1,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate images for a single class."""
        class_ids = torch.full((num_samples,), class_id, dtype=torch.long)
        return self.generate(class_ids, **generate_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="NFIG: Next-Frequency Image Generation"
    )
    parser.add_argument(
        "--vae_checkpoint", type=str, required=True,
        help="Path to FR-VAE checkpoint"
    )
    parser.add_argument(
        "--transformer_checkpoint", type=str, required=True,
        help="Path to NFIG Transformer checkpoint"
    )
    parser.add_argument(
        "--output", type=str, default="generated.png",
        help="Output image path"
    )
    parser.add_argument(
        "--class_id", type=int, nargs="+", default=[0],
        help="ImageNet class indices to generate"
    )
    parser.add_argument(
        "--num_samples", type=int, default=16,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=4.5,
        help="CFG scale (default 4.5 from paper)"
    )
    parser.add_argument(
        "--top_k", type=int, default=990,
        help="Top-k sampling (default 990 from paper)"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on"
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    generator = NFIGGenerator(
        vae_checkpoint=args.vae_checkpoint,
        transformer_checkpoint=args.transformer_checkpoint,
        device=device,
    )

    if len(args.class_id) == 1:
        class_ids = torch.full(
            (args.num_samples,), args.class_id[0], dtype=torch.long
        )
    else:
        class_ids = torch.tensor(args.class_id, dtype=torch.long)
        if len(class_ids) < args.num_samples:
            class_ids = class_ids.repeat(
                (args.num_samples + len(class_ids) - 1) // len(class_ids)
            )[:args.num_samples]

    generator.generate_and_save(
        class_ids=class_ids,
        output_path=args.output,
        cfg_scale=args.cfg_scale,
        top_k=args.top_k,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
