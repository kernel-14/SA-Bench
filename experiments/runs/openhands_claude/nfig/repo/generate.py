"""
Image generation script for NFIG.

Generates class-conditional images using the trained NFIG transformer
and FR-VAE decoder. Supports CFG and top-k sampling.
"""

import argparse
import os
from typing import List, Optional

import torch
import torchvision.utils as vutils

from config import NFIGConfig, config_600m
from models.fr_vae import FRVAE
from models.transformer import NFIGTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with NFIG")
    parser.add_argument("--tokenizer-ckpt", type=str, required=True)
    parser.add_argument("--transformer-ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./generated")
    parser.add_argument("--model-size", type=str, default="310M", choices=["310M", "600M"])
    parser.add_argument("--class-labels", type=int, nargs="+", default=None,
                        help="Class indices to generate. If None, generates all 1000 classes.")
    parser.add_argument("--num-samples-per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--top-k", type=int, default=990)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-grid", action="store_true",
                        help="Save a grid of generated images")
    return parser.parse_args()


def load_tokenizer(ckpt_path: str, cfg: NFIGConfig, device: torch.device) -> FRVAE:
    model = FRVAE(
        image_size=cfg.tokenizer.image_size,
        in_channels=cfg.tokenizer.in_channels,
        z_channels=cfg.tokenizer.z_channels,
        ch=cfg.tokenizer.ch,
        ch_mult=cfg.tokenizer.ch_mult,
        num_res_blocks=cfg.tokenizer.num_res_blocks,
        attn_resolutions=cfg.tokenizer.attn_resolutions,
        codebook_size=cfg.tokenizer.codebook_size,
        scale_factors=cfg.tokenizer.scale_factors,
        feature_map_size=cfg.tokenizer.feature_map_size,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def load_transformer(
    ckpt_path: str, cfg: NFIGConfig, model_size: str, device: torch.device
) -> NFIGTransformer:
    if model_size == "600M":
        t_cfg = config_600m.transformer
    else:
        t_cfg = cfg.transformer

    model = NFIGTransformer(
        vocab_size=t_cfg.vocab_size,
        num_classes=t_cfg.num_classes,
        depth=t_cfg.depth,
        embed_dim=t_cfg.embed_dim,
        num_heads=t_cfg.num_heads,
        mlp_ratio=t_cfg.mlp_ratio,
        scale_factors=t_cfg.scale_factors,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def generate_images(
    transformer: NFIGTransformer,
    tokenizer: FRVAE,
    class_labels: torch.Tensor,
    cfg_scale: float = 4.5,
    top_k: int = 990,
    temperature: float = 1.0,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Generate images for a batch of class labels.

    Args:
        transformer: trained NFIG transformer
        tokenizer: trained FR-VAE
        class_labels: (B,) class indices
        cfg_scale: classifier-free guidance scale
        top_k: top-k sampling
        temperature: sampling temperature
        device: compute device

    Returns:
        (B, 3, H, W) generated images in [-1, 1]
    """
    class_labels = class_labels.to(device)

    # Generate token sequences
    indices_list = transformer.generate(
        class_labels=class_labels,
        cfg_scale=cfg_scale,
        top_k=top_k,
        temperature=temperature,
    )

    # Decode tokens to images
    images = tokenizer.decode_from_indices(indices_list)
    return images.clamp(-1, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = NFIGConfig()

    print("Loading models...")
    tokenizer = load_tokenizer(args.tokenizer_ckpt, cfg, device)
    transformer = load_transformer(args.transformer_ckpt, cfg, args.model_size, device)

    # Determine which classes to generate
    if args.class_labels is None:
        class_list = list(range(1000))
    else:
        class_list = args.class_labels

    print(f"Generating {args.num_samples_per_class} samples for {len(class_list)} classes...")

    all_images = []
    for class_idx in class_list:
        class_dir = os.path.join(args.output_dir, f"class_{class_idx:04d}")
        os.makedirs(class_dir, exist_ok=True)

        num_generated = 0
        while num_generated < args.num_samples_per_class:
            batch_size = min(args.batch_size, args.num_samples_per_class - num_generated)
            labels = torch.full((batch_size,), class_idx, dtype=torch.long)

            images = generate_images(
                transformer=transformer,
                tokenizer=tokenizer,
                class_labels=labels,
                cfg_scale=args.cfg_scale,
                top_k=args.top_k,
                temperature=args.temperature,
                device=device,
            )

            # Save individual images
            for i, img in enumerate(images):
                img_path = os.path.join(class_dir, f"sample_{num_generated + i:04d}.png")
                # Convert from [-1, 1] to [0, 1]
                img_save = (img * 0.5 + 0.5).clamp(0, 1)
                vutils.save_image(img_save, img_path)

            if args.save_grid:
                all_images.append(images.cpu())

            num_generated += batch_size

        print(f"Class {class_idx}: generated {num_generated} samples")

    if args.save_grid and all_images:
        # Save a grid of first sample from each class (up to 100 classes)
        grid_images = torch.cat([imgs[:1] for imgs in all_images[:100]], dim=0)
        grid_images = (grid_images * 0.5 + 0.5).clamp(0, 1)
        grid_path = os.path.join(args.output_dir, "sample_grid.png")
        vutils.save_image(grid_images, grid_path, nrow=10)
        print(f"Saved sample grid to {grid_path}")

    print(f"Generation complete. Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
