#!/usr/bin/env python3
"""
Training script for Hi-MAR on ImageNet (class-conditional generation).

Usage:
    python scripts/train_imagenet.py --config configs/imagenet_himar_b.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from himar import HiMAR
from himar.model import create_himar_model
from himar.training import create_trainer
from himar.data import ImageNetLatentDataset, create_dataloader, get_vae


def parse_args():
    parser = argparse.ArgumentParser(description='Train Hi-MAR on ImageNet')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Merge CLI args
    config['output_dir'] = args.output_dir
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating Hi-MAR model...")
    model_cfg = config['model']
    model = create_himar_model(model_cfg)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load VAE
    vae = get_vae('kl-16')
    if vae is not None:
        vae = vae.to(device)
        vae.eval()
    
    # Create dataset
    data_cfg = config['data']
    train_dataset = ImageNetLatentDataset(
        root=data_cfg['data_path'],
        split='train',
        image_size=data_cfg['image_size'],
        low_res_size=data_cfg['low_res_size'],
        vae=vae,
    )
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Create trainer
    trainer = create_trainer(config['training'], model)
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = trainer.load_checkpoint(args.resume)
    
    # Train
    print(f"Starting training from epoch {start_epoch}...")
    trainer.train(train_loader, start_epoch=start_epoch)
    
    print("Training complete!")


if __name__ == '__main__':
    main()
