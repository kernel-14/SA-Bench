#!/usr/bin/env python3
"""
Generate images using a trained NFIG model.

Usage:
    python scripts/generate.py \
        --frvae_ckpt ./checkpoints/frvae_final.pt \
        --nfig_ckpt ./checkpoints/nfig_final.pt \
        --class_id 0 \
        --num_images 16 \
        --output_dir ./generated \
        --top_k 990 \
        --cfg_scale 4.5
"""

import argparse
import torch
import os
import sys
from torchvision.utils import save_image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nfig import FRVAE, NFIGTransformer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frvae_ckpt', type=str, required=True,
                        help='Path to FR-VAE checkpoint')
    parser.add_argument('--nfig_ckpt', type=str, required=True,
                        help='Path to NFIG transformer checkpoint')
    parser.add_argument('--class_id', type=int, default=None,
                        help='Class ID to generate (None for random)')
    parser.add_argument('--num_images', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='./generated')
    parser.add_argument('--top_k', type=int, default=990)
    parser.add_argument('--cfg_scale', type=float, default=4.5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load FR-VAE
    scale_factors = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    scales = [(s, s) for s in scale_factors]
    
    frvae = FRVAE(
        scales=scales,
        codebook_size=4096,
        codebook_dim=256,
        latent_dim=256,
        image_size=args.image_size,
        use_dino_disc=False,  # Not needed for generation
    ).to(device)
    
    frvae_ckpt = torch.load(args.frvae_ckpt, map_location=device)
    frvae.load_state_dict(frvae_ckpt['model_state_dict'])
    frvae.eval()
    
    # Load NFIG Transformer
    model = NFIGTransformer(
        scales=scales,
        codebook_size=4096,
        dim=1024,
        depth=16,
        num_heads=16,
        num_classes=1000,
        cond_drop_prob=0.0,
    ).to(device)
    
    nfig_ckpt = torch.load(args.nfig_ckpt, map_location=device)
    model.load_state_dict(nfig_ckpt['model_state_dict'])
    model.eval()
    
    print(f"Models loaded. Generating {args.num_images} images...")
    
    # Prepare class IDs
    if args.class_id is not None:
        class_ids = torch.full((args.num_images,), args.class_id, device=device)
    else:
        class_ids = torch.randint(0, 1000, (args.num_images,), device=device)
    
    # Generate tokens
    token_seqs = model.generate(
        class_ids=class_ids,
        top_k=args.top_k,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
    )
    
    # Decode tokens to images
    images = []
    for i in range(args.num_images):
        # Get tokens for this image
        img_tokens = [tokens[i:i+1] for tokens in token_seqs]
        img = frvae.decode_from_tokens(img_tokens)
        images.append(img)
    
    images = torch.cat(images, dim=0)
    
    # Save images
    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1.0) / 2.0
    images = torch.clamp(images, 0, 1)
    
    for i, img in enumerate(images):
        save_image(img, os.path.join(args.output_dir, f'generated_{i:04d}.png'))
    
    # Save a grid
    save_image(images, os.path.join(args.output_dir, 'generated_grid.png'), 
               nrow=int(args.num_images ** 0.5))
    
    print(f"Generated {args.num_images} images saved to {args.output_dir}")


if __name__ == '__main__':
    main()
