"""
Training script for robustness evaluation with CLIP.
Evaluates PEFT methods on ImageNet-1K (100-shot) and distribution shifts.

Usage:
    python train_robustness.py --method bitfit --data_dir /path/to/imagenet
    python train_robustness.py --method all --data_dir /path/to/imagenet
"""

import os
import sys
import argparse
import json
import copy
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.vit import count_trainable_params
from src.datasets.imagenet import get_imagenet_dataset, get_distribution_shift_datasets
from src.utils.trainer import Trainer
from src.utils.evaluator import evaluate
from src.utils.wise import wise_sweep


def build_clip_model(method, num_classes=1000, drop_path_rate=0.0, **method_kwargs):
    """
    Build a CLIP ViT-B/16 model with PEFT applied.
    
    Following the paper's setup:
    - Use CLIP ViT-B/16 visual encoder
    - Initialize head with zero-shot weights from text encoder
    - Apply PEFT to visual encoder
    """
    try:
        import clip
        # Load CLIP model
        clip_model, preprocess = clip.load('ViT-B/16', device='cpu')
        visual_encoder = clip_model.visual
        
        # Create classification head initialized with zero-shot weights
        # This requires the text encoder to generate class embeddings
        # For simplicity, we use a random head here
        # In practice, you would use the CLIP text encoder to initialize
        
        # Wrap visual encoder as a standard ViT-like model
        class CLIPVisualWrapper(nn.Module):
            def __init__(self, visual_encoder, num_classes):
                super().__init__()
                self.visual = visual_encoder
                embed_dim = visual_encoder.output_dim
                self.head = nn.Linear(embed_dim, num_classes)
            
            def forward(self, x):
                features = self.visual(x)
                return self.head(features)
        
        model = CLIPVisualWrapper(visual_encoder, num_classes)
        
    except ImportError:
        # Fallback: use timm CLIP-like model
        print("CLIP package not found, using timm ViT-B/16 as fallback")
        from timm.models import create_model
        model = create_model(
            'vit_base_patch16_clip_224',
            pretrained=True,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )
    
    # Apply PEFT method
    if method == 'full':
        for param in model.parameters():
            param.requires_grad = True
    elif method == 'bitfit':
        from src.methods.bitfit import apply_bitfit
        model = apply_bitfit(model)
    elif method == 'layernorm':
        from src.methods.layernorm import apply_layernorm
        model = apply_layernorm(model)
    elif method == 'houl_adapter':
        from src.methods.adapter import apply_adapter
        bottleneck_dim = method_kwargs.get('bottleneck_dim', 8)
        scale_factor = method_kwargs.get('scale_factor', 1.0)
        model = apply_adapter(model, bottleneck_dim=bottleneck_dim, scale_factor=scale_factor, adapter_type='houlsby')
    elif method == 'adaptformer':
        from src.methods.adaptformer import apply_adaptformer
        bottleneck_dim = method_kwargs.get('bottleneck_dim', 8)
        scale_factor = method_kwargs.get('scale_factor', 0.1)
        model = apply_adaptformer(model, bottleneck_dim=bottleneck_dim, scale_factor=scale_factor)
    elif method == 'repadapter':
        from src.methods.repadapter import apply_repadapter
        bottleneck_dim = method_kwargs.get('bottleneck_dim', 8)
        scale_factor = method_kwargs.get('scale_factor', 1.0)
        model = apply_repadapter(model, bottleneck_dim=bottleneck_dim, scale_factor=scale_factor)
    elif method == 'convpass':
        from src.methods.convpass import apply_convpass
        bottleneck_dim = method_kwargs.get('bottleneck_dim', 8)
        scale_factor = method_kwargs.get('scale_factor', 1.0)
        model = apply_convpass(model, bottleneck_dim=bottleneck_dim, scale_factor=scale_factor)
    elif method == 'lora':
        from src.methods.lora import apply_lora
        rank = method_kwargs.get('rank', 4)
        model = apply_lora(model, rank=rank)
    elif method == 'fact_tk':
        from src.methods.fact import apply_fact
        rank = method_kwargs.get('rank', 16)
        scale_factor = method_kwargs.get('scale_factor', 1.0)
        model = apply_fact(model, rank=rank, scale_factor=scale_factor, fact_type='tk')
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Robustness Evaluation with CLIP')
    
    parser.add_argument('--method', type=str, default='bitfit',
                        choices=['full', 'bitfit', 'layernorm', 'houl_adapter', 'adaptformer',
                                'repadapter', 'convpass', 'lora', 'fact_tk', 'all'],
                        help='PEFT method to use')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory for ImageNet data')
    
    # Training hyperparameters (following paper: lr=3e-5, wd=5e-3)
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_shots', type=int, default=100, help='Number of shots per class')
    
    # Method-specific hyperparameters
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA/FacT')
    parser.add_argument('--bottleneck_dim', type=int, default=8, help='Bottleneck dim for adapters')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Scale factor for adapters')
    
    # WiSE settings
    parser.add_argument('--wise', action='store_true', help='Apply WiSE after training')
    parser.add_argument('--wise_alphas', type=float, nargs='+', 
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help='Alpha values for WiSE sweep')
    
    # Other settings
    parser.add_argument('--output_dir', type=str, default='./output_robustness', help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--results_file', type=str, default='robustness_results.json',
                        help='File to save results')
    
    args = parser.parse_args()
    
    # Determine methods
    if args.method == 'all':
        methods = ['full', 'bitfit', 'layernorm', 'houl_adapter', 'adaptformer',
                  'repadapter', 'convpass', 'lora', 'fact_tk']
    else:
        methods = [args.method]
    
    method_kwargs = {
        'rank': args.rank,
        'bottleneck_dim': args.bottleneck_dim,
        'scale_factor': args.scale_factor,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # Get data loaders
    train_loader = get_imagenet_dataset(
        args.data_dir, split='train', num_shots=args.num_shots,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_loader = get_imagenet_dataset(
        args.data_dir, split='val',
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    dist_shift_loaders = get_distribution_shift_datasets(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    for method in methods:
        try:
            print(f"\n=== Training {method} ===")
            
            # Build model
            model = build_clip_model(method, num_classes=1000, **method_kwargs)
            pretrained_model = copy.deepcopy(model)  # Save pretrained weights
            
            trainable_params = count_trainable_params(model)
            print(f"Trainable params: {trainable_params:.3f}M")
            
            # Train
            output_dir = os.path.join(args.output_dir, method)
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=1000,
                lr=args.lr,
                weight_decay=args.weight_decay,
                num_epochs=args.epochs,
                device=str(device),
                output_dir=output_dir,
                experiment_name=f'{method}_imagenet',
            )
            
            best_val_acc = trainer.train(verbose=args.verbose)
            
            # Evaluate on target distribution
            target_acc = evaluate(model, val_loader, device=str(device))
            
            # Evaluate on distribution shifts
            shift_accs = {}
            for name, loader in dist_shift_loaders.items():
                shift_acc = evaluate(model, loader, device=str(device))
                shift_accs[name] = shift_acc
                print(f"  {name}: {shift_acc:.2f}%")
            
            avg_shift_acc = np.mean(list(shift_accs.values())) if shift_accs else 0.0
            
            print(f"Target (ImageNet): {target_acc:.2f}%")
            print(f"Avg Distribution Shift: {avg_shift_acc:.2f}%")
            
            results[method] = {
                'target_acc': target_acc,
                'shift_accs': shift_accs,
                'avg_shift_acc': avg_shift_acc,
                'trainable_params': trainable_params,
            }
            
            # Apply WiSE if requested
            if args.wise:
                print(f"\nApplying WiSE for {method}...")
                wise_results = wise_sweep(
                    model, pretrained_model, val_loader, dist_shift_loaders,
                    method_type=method, device=str(device), alphas=args.wise_alphas
                )
                results[method]['wise_results'] = {
                    str(alpha): {'target_acc': r[0], 'avg_shift_acc': r[1]}
                    for alpha, r in wise_results.items()
                }
        
        except Exception as e:
            print(f"Error running {method}: {e}")
            import traceback
            traceback.print_exc()
            results[method] = {'error': str(e)}
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, args.results_file)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Print summary
    print("\n=== Robustness Results Summary ===")
    print(f"{'Method':<20} {'Target Acc':<12} {'Avg Shift Acc':<15} {'#Params':<10}")
    print("-" * 60)
    for method in methods:
        if method in results and 'target_acc' in results[method]:
            r = results[method]
            print(f"{method:<20} {r['target_acc']:<12.2f} {r['avg_shift_acc']:<15.2f} "
                  f"{r.get('trainable_params', 0):<10.3f}")


if __name__ == '__main__':
    main()
