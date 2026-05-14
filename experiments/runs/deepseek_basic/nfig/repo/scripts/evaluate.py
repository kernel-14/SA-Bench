#!/usr/bin/env python3
"""
Evaluate NFIG generation quality using FID, IS, Precision, Recall.

Usage:
    python scripts/evaluate.py \
        --frvae_ckpt ./checkpoints/frvae_final.pt \
        --nfig_ckpt ./checkpoints/nfig_final.pt \
        --real_data /path/to/imagenet/val \
        --num_samples 50000 \
        --batch_size 64 \
        --output_dir ./eval_results
"""

import argparse
import torch
import os
import sys
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nfig import FRVAE, NFIGTransformer
from nfig.evaluation import Evaluator
from nfig.frequency_utils import compute_frequency_keep_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frvae_ckpt', type=str, required=True)
    parser.add_argument('--nfig_ckpt', type=str, required=True)
    parser.add_argument('--real_data', type=str, default=None,
                        help='Path to real ImageNet validation images')
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--top_k', type=int, default=990)
    parser.add_argument('--cfg_scale', type=float, default=4.5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load models
    scale_factors = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    scales = [(s, s) for s in scale_factors]
    
    print("Loading FR-VAE...")
    frvae = FRVAE(
        scales=scales,
        codebook_size=4096,
        codebook_dim=256,
        latent_dim=256,
        image_size=args.image_size,
        use_dino_disc=False,
    ).to(device)
    
    frvae_ckpt = torch.load(args.frvae_ckpt, map_location=device)
    frvae.load_state_dict(frvae_ckpt['model_state_dict'])
    frvae.eval()
    
    print("Loading NFIG Transformer...")
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
    
    # Compute reconstruction FID
    print("Computing reconstruction FID (rFID)...")
    
    # Load real images
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    # Collect real and reconstructed images
    real_images = []
    recon_images = []
    
    evaluator = Evaluator(device=device)
    
    if args.real_data:
        val_dataset = datasets.ImageFolder(args.real_data, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=args.num_workers)
        
        for images, _ in val_loader:
            images = images.to(device)
            with torch.no_grad():
                recon, _, _ = frvae(images)
            
            real_images.append(images.cpu())
            recon_images.append(recon.cpu())
            
            if len(torch.cat(real_images)) >= min(10000, args.num_samples // 5):
                break
        
        real_images = torch.cat(real_images)[:10000]
        recon_images = torch.cat(recon_images)[:10000]
        
        # Compute rFID
        metrics_rfid = evaluator.compute_all_metrics(real_images, recon_images)
        rfid = metrics_rfid.get('fid', float('nan'))
        print(f"  rFID: {rfid:.4f}")
    
    # Generate samples for gFID
    print(f"Generating {args.num_samples} samples for gFID...")
    
    generated_images = []
    total_gen = 0
    
    while total_gen < args.num_samples:
        class_ids = torch.randint(0, 1000, (args.batch_size,), device=device)
        
        with torch.no_grad():
            token_seqs = model.generate(
                class_ids=class_ids,
                top_k=args.top_k,
                cfg_scale=args.cfg_scale,
            )
            
            batch_imgs = []
            for i in range(args.batch_size):
                img_tokens = [tokens[i:i+1] for tokens in token_seqs]
                img = frvae.decode_from_tokens(img_tokens)
                batch_imgs.append(img)
            
            batch_imgs = torch.cat(batch_imgs, dim=0)
            generated_images.append(batch_imgs.cpu())
            total_gen += args.batch_size
            
            if total_gen % 1000 == 0:
                print(f"  Generated {total_gen}/{args.num_samples}")
    
    generated_images = torch.cat(generated_images)[:args.num_samples]
    
    # Compute gFID and other metrics
    if args.real_data:
        print("Computing generation metrics...")
        # Load more real images for FID
        real_all = []
        val_dataset_full = datasets.ImageFolder(args.real_data, transform=transform)
        val_loader_full = DataLoader(val_dataset_full, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers)
        for images, _ in val_loader_full:
            real_all.append(images)
            if len(torch.cat(real_all)) >= args.num_samples:
                break
        real_all = torch.cat(real_all)[:args.num_samples]
        
        metrics = evaluator.compute_all_metrics(real_all, generated_images)
        
        results = {
            'rfid': rfid if args.real_data else None,
            'gfid': metrics.get('fid', float('nan')),
            'is_mean': metrics.get('is_mean', float('nan')),
            'is_std': metrics.get('is_std', float('nan')),
            'precision': metrics.get('precision', float('nan')),
            'recall': metrics.get('recall', float('nan')),
        }
        
        print(f"\n Results:")
        print(f"  rFID:     {results['rfid']:.4f}")
        print(f"  gFID:     {results['gfid']:.4f}")
        print(f"  IS:       {results['is_mean']:.2f} ± {results['is_std']:.2f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    # Frequency Keep Score
    if args.real_data and len(recon_images) > 0:
        print("\n Computing Frequency Keep Score (FKS)...")
        psd_error, fks, band_scores = compute_frequency_keep_score(
            real_images[:100].to(device), 
            recon_images[:100].to(device)
        )
        
        fks_results = {
            'psd_error': float(psd_error),
            'fks': float(fks),
            'low_score': float(band_scores['Low']),
            'middle_score': float(band_scores['Middle']),
            'high_score': float(band_scores['High']),
        }
        
        print(f"  PSD Error: {psd_error:.4f}")
        print(f"  FKS:       {fks*100:.1f}%")
        print(f"  Low:       {band_scores['Low']*100:.1f}%")
        print(f"  Middle:    {band_scores['Middle']*100:.1f}%")
        print(f"  High:      {band_scores['High']*100:.1f}%")
        
        with open(os.path.join(args.output_dir, 'fks_results.json'), 'w') as f:
            json.dump(fks_results, f, indent=2)
    
    print(f"\n Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
