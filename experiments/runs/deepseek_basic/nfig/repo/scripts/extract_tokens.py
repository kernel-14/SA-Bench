#!/usr/bin/env python3
"""
Extract discrete tokens from images using a trained FR-VAE.

Usage:
    python scripts/extract_tokens.py \
        --frvae_ckpt ./checkpoints/frvae_final.pt \
        --data_path /path/to/imagenet/train \
        --output_dir ./tokens \
        --batch_size 128
"""

import argparse
import torch
import os
import sys
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nfig import FRVAE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frvae_ckpt', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load FR-VAE
    scale_factors = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    scales = [(s, s) for s in scale_factors]
    
    frvae = FRVAE(
        scales=scales,
        codebook_size=4096,
        codebook_dim=256,
        latent_dim=256,
        image_size=args.image_size,
        use_dino_disc=False,
    ).to(device)
    
    checkpoint = torch.load(args.frvae_ckpt, map_location=device)
    frvae.load_state_dict(checkpoint['model_state_dict'])
    frvae.eval()
    
    # Data
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    
    total_tokens = frvae.get_total_tokens()
    print(f"Extracting tokens from {len(dataset)} images")
    print(f"Total tokens per image: {total_tokens}")
    
    idx = 0
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        
        with torch.no_grad():
            token_list, _, _ = frvae.encode(images)
        
        # Flatten tokens
        flat_tokens = frvae.get_token_sequence(token_list)
        
        # Save each sample
        for i in range(images.shape[0]):
            np.savez(
                os.path.join(args.output_dir, f'tokens_{idx:08d}.npz'),
                tokens=flat_tokens[i].cpu().numpy(),
                class_id=labels[i].item(),
            )
            idx += 1
    
    print(f"Extracted {idx} token files to {args.output_dir}")


if __name__ == '__main__':
    main()
