"""
Main training script for VTAB-1K experiments.
Evaluates PEFT methods on 19 classification tasks from VTAB-1K.

Usage:
    python train_vtab.py --method bitfit --dataset caltech101 --data_dir /path/to/vtab
    python train_vtab.py --method lora --dataset dtd --lr 1e-3 --rank 8
    python train_vtab.py --method all --data_dir /path/to/vtab  # Run all methods on all datasets
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

from src.models.vit import create_vit_model, count_trainable_params
from src.datasets.vtab import get_vtab_dataset, VTAB_DATASETS, VTAB_NUM_CLASSES
from src.utils.trainer import Trainer
from src.utils.evaluator import evaluate


def get_method_config():
    """Get default hyperparameter configurations for each method."""
    return {
        'linear': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
        },
        'full': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
        },
        'bitfit': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
        },
        'layernorm': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
        },
        'difffit': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
        },
        'ssf': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
        },
        'vpt_shallow': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
            'num_prompts_options': [5, 10, 50, 100, 200],
        },
        'vpt_deep': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
            'num_prompts_options': [5, 10, 50, 100],
        },
        'pfeif_adapter': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
            'bottleneck_options': [4, 8, 16, 32],
            'scale_options': [0.01, 0.1, 1.0, 10.0],
        },
        'houl_adapter': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
            'bottleneck_options': [4, 8, 16, 32],
            'scale_options': [0.01, 0.1, 1.0, 10.0],
        },
        'adaptformer': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
            'bottleneck_options': [4, 16, 32],
            'scale_options': [0.05, 0.1, 0.2],
        },
        'convpass': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
            'bottleneck_options': [8, 16],
            'scale_options': [0.01, 0.1, 1.0, 10.0, 100.0],
            'xavier_init_options': [True, False],
        },
        'repadapter': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
            'bottleneck_options': [8, 16, 32],
            'scale_options': [0.1, 0.5, 1.0, 5.0, 10.0],
        },
        'lora': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
            'rank_options': [1, 8, 16, 32],
        },
        'fact_tt': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
            'rank_options': [8, 16, 32],
            'scale_options': [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        'fact_tk': {
            'lr_options': [1e-3, 1e-2],
            'wd_options': [1e-4, 1e-3],
            'rank_options': [16, 32, 64],
            'scale_options': [0.01, 0.1, 1.0, 10.0, 100.0],
        },
    }


def build_model(method, num_classes, drop_path_rate=0.1, **method_kwargs):
    """Build a model with the specified PEFT method applied."""
    # Create base ViT model
    model = create_vit_model(
        model_name='vit_base_patch16_224_in21k',
        pretrained=True,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    
    if method == 'linear':
        # Linear probing: freeze all except head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
    
    elif method == 'full':
        # Full fine-tuning: all parameters trainable
        for param in model.parameters():
            param.requires_grad = True
    
    elif method == 'bitfit':
        from src.methods.bitfit import apply_bitfit
        model = apply_bitfit(model)
    
    elif method == 'layernorm':
        from src.methods.layernorm import apply_layernorm
        model = apply_layernorm(model)
    
    elif method == 'difffit':
        from src.methods.difffit import apply_difffit
        model = apply_difffit(model)
    
    elif method == 'ssf':
        from src.methods.ssf import apply_ssf
        model = apply_ssf(model)
    
    elif method == 'vpt_shallow':
        from src.methods.vpt import apply_vpt
        num_prompts = method_kwargs.get('num_prompts', 10)
        model = apply_vpt(model, num_prompts=num_prompts, deep=False)
    
    elif method == 'vpt_deep':
        from src.methods.vpt import apply_vpt
        num_prompts = method_kwargs.get('num_prompts', 10)
        model = apply_vpt(model, num_prompts=num_prompts, deep=True)
    
    elif method == 'pfeif_adapter':
        from src.methods.adapter import apply_adapter
        bottleneck_dim = method_kwargs.get('bottleneck_dim', 8)
        scale_factor = method_kwargs.get('scale_factor', 1.0)
        model = apply_adapter(model, bottleneck_dim=bottleneck_dim, scale_factor=scale_factor, adapter_type='pfeiffer')
    
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
    
    elif method == 'convpass':
        from src.methods.convpass import apply_convpass
        bottleneck_dim = method_kwargs.get('bottleneck_dim', 8)
        scale_factor = method_kwargs.get('scale_factor', 1.0)
        xavier_init = method_kwargs.get('xavier_init', False)
        model = apply_convpass(model, bottleneck_dim=bottleneck_dim, scale_factor=scale_factor, xavier_init=xavier_init)
    
    elif method == 'repadapter':
        from src.methods.repadapter import apply_repadapter
        bottleneck_dim = method_kwargs.get('bottleneck_dim', 8)
        scale_factor = method_kwargs.get('scale_factor', 1.0)
        model = apply_repadapter(model, bottleneck_dim=bottleneck_dim, scale_factor=scale_factor)
    
    elif method == 'lora':
        from src.methods.lora import apply_lora
        rank = method_kwargs.get('rank', 4)
        model = apply_lora(model, rank=rank)
    
    elif method == 'fact_tt':
        from src.methods.fact import apply_fact
        rank = method_kwargs.get('rank', 8)
        scale_factor = method_kwargs.get('scale_factor', 1.0)
        model = apply_fact(model, rank=rank, scale_factor=scale_factor, fact_type='tt')
    
    elif method == 'fact_tk':
        from src.methods.fact import apply_fact
        rank = method_kwargs.get('rank', 16)
        scale_factor = method_kwargs.get('scale_factor', 1.0)
        model = apply_fact(model, rank=rank, scale_factor=scale_factor, fact_type='tk')
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return model


def train_single(method, dataset_name, data_dir, args, method_kwargs=None):
    """Train a single method on a single dataset."""
    if method_kwargs is None:
        method_kwargs = {}
    
    num_classes = VTAB_NUM_CLASSES[dataset_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    model = build_model(method, num_classes, drop_path_rate=args.drop_path_rate, **method_kwargs)
    
    trainable_params = count_trainable_params(model)
    print(f"Method: {method}, Dataset: {dataset_name}, Trainable params: {trainable_params:.3f}M")
    
    # Get data loaders
    train_loader = get_vtab_dataset(data_dir, dataset_name, split='train', 
                                     batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = get_vtab_dataset(data_dir, dataset_name, split='val',
                                   batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = get_vtab_dataset(data_dir, dataset_name, split='test',
                                    batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Train
    output_dir = os.path.join(args.output_dir, method, dataset_name)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        device=str(device),
        output_dir=output_dir,
        experiment_name=f'{method}_{dataset_name}',
    )
    
    best_val_acc = trainer.train(verbose=args.verbose)
    
    # Evaluate on test set
    test_acc = evaluate(model, test_loader, device=str(device))
    
    print(f"Method: {method}, Dataset: {dataset_name}, "
          f"Best Val Acc: {best_val_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    return test_acc, trainable_params


def main():
    parser = argparse.ArgumentParser(description='VTAB-1K PEFT Evaluation')
    
    # Method and dataset
    parser.add_argument('--method', type=str, default='bitfit',
                        choices=['linear', 'full', 'bitfit', 'layernorm', 'difffit', 'ssf',
                                'vpt_shallow', 'vpt_deep', 'pfeif_adapter', 'houl_adapter',
                                'adaptformer', 'convpass', 'repadapter', 'lora', 'fact_tt', 'fact_tk', 'all'],
                        help='PEFT method to use')
    parser.add_argument('--dataset', type=str, default='caltech101',
                        help='Dataset name or "all" for all datasets')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory for VTAB-1K data')
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help='Drop path rate (0.1 recommended, 0 = disabled)')
    
    # Method-specific hyperparameters
    parser.add_argument('--rank', type=int, default=8, help='Rank for LoRA/FacT')
    parser.add_argument('--bottleneck_dim', type=int, default=8, help='Bottleneck dim for adapters')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Scale factor for adapters')
    parser.add_argument('--num_prompts', type=int, default=10, help='Number of prompts for VPT')
    
    # Other settings
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--results_file', type=str, default='vtab_results.json',
                        help='File to save results')
    
    args = parser.parse_args()
    
    # Determine methods and datasets to run
    if args.method == 'all':
        methods = ['linear', 'full', 'bitfit', 'layernorm', 'difffit', 'ssf',
                  'vpt_shallow', 'vpt_deep', 'pfeif_adapter', 'houl_adapter',
                  'adaptformer', 'convpass', 'repadapter', 'lora', 'fact_tt', 'fact_tk']
    else:
        methods = [args.method]
    
    if args.dataset == 'all':
        datasets = []
        for group_datasets in VTAB_DATASETS.values():
            datasets.extend(group_datasets)
    else:
        datasets = [args.dataset]
    
    # Method-specific kwargs
    method_kwargs = {
        'rank': args.rank,
        'bottleneck_dim': args.bottleneck_dim,
        'scale_factor': args.scale_factor,
        'num_prompts': args.num_prompts,
    }
    
    # Run experiments
    results = {}
    for method in methods:
        results[method] = {}
        for dataset in datasets:
            try:
                test_acc, trainable_params = train_single(
                    method, dataset, args.data_dir, args, method_kwargs
                )
                results[method][dataset] = {
                    'test_acc': test_acc,
                    'trainable_params': trainable_params,
                }
            except Exception as e:
                print(f"Error running {method} on {dataset}: {e}")
                results[method][dataset] = {'error': str(e)}
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, args.results_file)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Print summary table
    print("\n=== Results Summary ===")
    print(f"{'Method':<20} {'Avg Acc':<10} {'#Params':<10}")
    print("-" * 40)
    for method in methods:
        accs = [v['test_acc'] for v in results[method].values() if 'test_acc' in v]
        params = [v['trainable_params'] for v in results[method].values() if 'trainable_params' in v]
        if accs:
            avg_acc = np.mean(accs)
            avg_params = np.mean(params) if params else 0
            print(f"{method:<20} {avg_acc:<10.2f} {avg_params:<10.3f}")


if __name__ == '__main__':
    main()
