#!/usr/bin/env python3
"""
Train NFIG Autoregressive Transformer on ImageNet tokens.

The transformer is trained on discrete tokens extracted by the pre-trained FR-VAE.

Usage:
    python scripts/train_nfig.py --token_data /path/to/tokens --output_dir ./checkpoints --frvae_ckpt ./checkpoints/frvae_final.pt
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nfig import NFIGTransformer
from nfig.trainer import NFIGTrainer
from nfig import FRVAE


class TokenDataset(Dataset):
    """Dataset of pre-computed FR-VAE tokens."""
    
    def __init__(self, token_dir: str, max_samples: int = None):
        self.token_dir = token_dir
        self.files = sorted([f for f in os.listdir(token_dir) if f.endswith('.npz')])
        if max_samples:
            self.files = self.files[:max_samples]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(os.path.join(self.token_dir, self.files[idx]))
        tokens = torch.from_numpy(data['tokens']).long()  # (total_tokens,)
        class_id = int(data['class_id'])
        return tokens, class_id


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_data', type=str, required=True,
                        help='Path to pre-computed token directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=768,
                        help='Batch size (paper uses 768)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='Number of epochs (paper uses 350)')
    parser.add_argument('--lr', type=float, default=8e-5,
                        help='Learning rate (paper uses 8e-5)')
    parser.add_argument('--dim', type=int, default=768,
                        help='Transformer hidden dimension')
    parser.add_argument('--depth', type=int, default=16,
                        help='Number of transformer blocks (paper uses 16)')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--codebook_size', type=int, default=4096)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--cond_drop_prob', type=float, default=0.1,
                        help='CFG dropout probability')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dataset
    dataset = TokenDataset(args.token_data)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Model
    scale_factors = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    scales = [(s, s) for s in scale_factors]
    
    model = NFIGTransformer(
        scales=scales,
        codebook_size=args.codebook_size,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        cond_drop_prob=args.cond_drop_prob,
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    trainer = NFIGTrainer(model, device, lr=args.lr)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    total_tokens = sum(h * w for h, w in scales)
    num_params = model.get_num_params()
    print(f"Training NFIG Transformer:")
    print(f"  Parameters: {num_params / 1e6:.1f}M")
    print(f"  Scales: {scales}")
    print(f"  Total tokens per image: {total_tokens}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_idx, (tokens, class_ids) in enumerate(dataloader):
            metrics = trainer.train_step(tokens, class_ids)
            total_loss += metrics['loss']
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: loss={metrics['loss']:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 50 == 0:
            trainer.save_checkpoint(
                os.path.join(args.output_dir, f'nfig_epoch_{epoch+1}.pt')
            )
    
    trainer.save_checkpoint(
        os.path.join(args.output_dir, 'nfig_final.pt')
    )
    print("Training complete!")


if __name__ == '__main__':
    main()
