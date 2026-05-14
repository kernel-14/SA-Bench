#!/usr/bin/env python3
"""Many-shot training script for PEFT methods.

Evaluates PEFT methods on full-size datasets:
- CIFAR-100 (50K training images, 100 classes)
- RESISC (25.2K training images, 45 classes) 
- Clevr-Distance (70K training images, 6 classes)

Following Section 5 and Appendix A of the paper.
"""

import argparse
import os
import sys
import json
import torch
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods import METHOD_MAP
from methods.model_builder import build_model
from utils.data import get_many_shot_dataset
from utils.training import train_model


def main():
    parser = argparse.ArgumentParser(description='Many-shot PEFT training')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'resisc', 'clevr_distance'])
    parser.add_argument('--method', type=str, default='lora')
    parser.add_argument('--data_dir', type=str, default='/data')
    parser.add_argument('--output_dir', type=str, default='./results/many_shot')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading {args.dataset}...")
    train_dataset, num_classes = get_many_shot_dataset(
        args.dataset, data_dir=args.data_dir, split='train', seed=args.seed
    )
    val_dataset, _ = get_many_shot_dataset(
        args.dataset, data_dir=args.data_dir, split='val', seed=args.seed
    )
    test_dataset, _ = get_many_shot_dataset(
        args.dataset, data_dir=args.data_dir, split='test', seed=args.seed
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)
    
    print(f"Training {args.method} on {args.dataset} ({num_classes} classes)")
    
    # Build model
    model = build_model(
        method_name=args.method,
        num_classes=num_classes,
        backbone_type='in21k',
        drop_path_rate=0.0,  # Many-shot: drop path can be 0
    )
    model = model.to(args.device)
    
    # Train
    result = train_model(
        model, train_loader, val_loader, test_loader,
        method_name=args.method,
        dataset_name=args.dataset,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        device=args.device,
    )
    
    # Save result
    param_count = result['trainable_params_millions']
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    param_pct = param_count / total_params * 100
    
    print(f"\nResults for {args.method} on {args.dataset}:")
    print(f"  Test accuracy: {result['test_accuracy']:.2f}%")
    print(f"  Trainable params: {param_count:.2f}M ({param_pct:.1f}% of {total_params:.1f}M)")
    
    output = {
        'method': args.method,
        'dataset': args.dataset,
        'test_accuracy': result['test_accuracy'],
        'val_accuracy': result['best_val_accuracy'],
        'trainable_params_m': param_count,
        'trainable_params_pct': param_pct,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
    }
    
    save_path = os.path.join(args.output_dir, f"{args.dataset}_{args.method}.json")
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {save_path}")


if __name__ == '__main__':
    main()
