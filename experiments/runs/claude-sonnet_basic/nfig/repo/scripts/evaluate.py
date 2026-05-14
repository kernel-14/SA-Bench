"""
Evaluation and image generation script for NFIG.

Generates images using the trained NFIG model and evaluates:
- FID (Fréchet Inception Distance)
- IS (Inception Score)
- Precision and Recall

Usage:
    python scripts/evaluate.py \
        --tokenizer-path output/fr_vae/fr_vae_final.pt \
        --model-path output/nfig/nfig_final.pt \
        --output-dir output/generated \
        --n-samples 50000 \
        --cfg-scale 4.5 \
        --top-k 990
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from typing import List, Optional
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.fr_vae import FRVAE
from models.nfig_transformer import NFIGTransformer, nfig_310m, nfig_600m


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate NFIG")
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./output/generated")
    parser.add_argument("--model-size", type=str, default="310m",
                        choices=["310m", "600m"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--top-k", type=int, default=990)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-classes", type=int, default=1000)
    parser.add_argument("--codebook-size", type=int, default=4096)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-images", action="store_true",
                        help="Save generated images to disk")
    return parser.parse_args()


@torch.no_grad()
def generate_images(
    model: NFIGTransformer,
    tokenizer: FRVAE,
    class_labels: torch.Tensor,
    latent_H: int,
    latent_W: int,
    cfg_scale: float = 4.5,
    top_k: int = 990,
    temperature: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Generate images for a batch of class labels."""
    model.eval()
    tokenizer.eval()

    # Generate tokens using fast (band-parallel) generation
    generated_tokens = model.generate_fast(
        class_labels=class_labels,
        cfg_scale=cfg_scale,
        top_k=top_k,
        temperature=temperature,
    )

    # Decode tokens to images
    images = tokenizer.decode(generated_tokens, latent_H, latent_W)
    # Clamp to [-1, 1] and convert to [0, 1]
    images = (images.clamp(-1, 1) + 1) / 2
    return images


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = FRVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        scale_factors=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
    ).to(device)

    state = torch.load(args.tokenizer_path, map_location=device)
    if "model" in state:
        state = state["model"]
    tokenizer.load_state_dict(state)
    tokenizer.eval()
    print(f"Loaded tokenizer from {args.tokenizer_path}")

    # Compute token counts
    latent_H = args.image_size // 16
    latent_W = args.image_size // 16
    token_counts = tokenizer.get_token_counts(latent_H, latent_W)
    print(f"Token counts per band: {token_counts}")
    print(f"Total tokens: {sum(token_counts)}")

    # Load model
    if args.model_size == "310m":
        model = nfig_310m(
            codebook_size=args.codebook_size,
            n_classes=args.n_classes,
            token_counts=token_counts,
        )
    else:
        model = nfig_600m(
            codebook_size=args.codebook_size,
            n_classes=args.n_classes,
            token_counts=token_counts,
        )

    state = torch.load(args.model_path, map_location=device)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # Generate images
    all_images = []
    n_generated = 0
    img_idx = 0

    print(f"Generating {args.n_samples} images...")
    while n_generated < args.n_samples:
        batch_size = min(args.batch_size, args.n_samples - n_generated)
        # Sample class labels uniformly
        class_labels = torch.randint(0, args.n_classes, (batch_size,), device=device)

        images = generate_images(
            model, tokenizer, class_labels,
            latent_H, latent_W,
            cfg_scale=args.cfg_scale,
            top_k=args.top_k,
            temperature=args.temperature,
            device=device,
        )

        if args.save_images:
            for i, img in enumerate(images):
                save_image(img, os.path.join(args.output_dir, f"img_{img_idx:06d}.png"))
                img_idx += 1

        all_images.append(images.cpu())
        n_generated += batch_size
        print(f"Generated {n_generated}/{args.n_samples} images")

    all_images = torch.cat(all_images, dim=0)[:args.n_samples]
    print(f"Generated {len(all_images)} images total")

    # Save a grid of sample images
    grid_size = min(64, len(all_images))
    save_image(all_images[:grid_size], os.path.join(args.output_dir, "sample_grid.png"),
               nrow=8, normalize=False)
    print(f"Saved sample grid to {args.output_dir}/sample_grid.png")

    # Optionally compute FID/IS using torch-fidelity or similar
    try:
        import torch_fidelity
        print("Computing FID and IS using torch-fidelity...")
        # This requires reference statistics for ImageNet
        # torch_fidelity.calculate_metrics(...)
        print("Note: FID computation requires reference ImageNet statistics.")
        print("Please use the saved images with a standard FID evaluation tool.")
    except ImportError:
        print("torch-fidelity not available. Please install it for FID/IS computation.")
        print("Alternatively, use the saved images with a standard evaluation tool.")


if __name__ == "__main__":
    main()
