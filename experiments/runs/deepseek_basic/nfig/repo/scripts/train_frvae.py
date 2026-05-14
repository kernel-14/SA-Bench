#!/usr/bin/env python3
"""
Train FR-VAE tokenizer on ImageNet.

Usage:
    python scripts/train_frvae.py --data_path /path/to/imagenet --output_dir ./checkpoints
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nfig import FRVAE
from nfig.trainer import FRVAETrainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ImageNet dataset')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--codebook_size', type=int, default=4096)
    parser.add_argument('--codebook_dim', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    dataset = datasets.ImageFolder(
        os.path.join(args.data_path, 'train'), transform=transform
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Model
    # Scales from paper: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    scale_factors = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    scales = [(s, s) for s in scale_factors]
    
    model = FRVAE(
        scales=scales,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        use_dino_disc=True,
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    trainer = FRVAETrainer(model, device, lr=args.lr)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Training loop
    print(f"Training FR-VAE on {len(dataset)} images for {args.epochs} epochs")
    print(f"Scales: {scales}")
    print(f"Total tokens per image: {model.get_total_tokens()}")
    print(f"Codebook size: {args.codebook_size}")
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_idx, (images, _) in enumerate(dataloader):
            metrics = trainer.train_step(images)
            total_loss += metrics['g_loss']
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"G_loss={metrics['g_loss']:.4f}, "
                      f"VQ_loss={metrics['vq_loss']:.4f}, "
                      f"D_loss={metrics['d_loss']:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                os.path.join(args.output_dir, f'frvae_epoch_{epoch+1}.pt')
            )
    
    # Save final model
    trainer.save_checkpoint(
        os.path.join(args.output_dir, 'frvae_final.pt')
    )
    print("Training complete!")


if __name__ == '__main__':
    main()
