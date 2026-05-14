"""
Hyperparameter search script for PEFT methods.
Performs grid search over learning rate, weight decay, and method-specific parameters.

Usage:
    python hyperparameter_search.py --method bitfit --dataset caltech101 --data_dir /path/to/vtab
"""

import os
import sys
import argparse
import json
import itertools
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets.vtab import get_vtab_dataset, VTAB_NUM_CLASSES
from src.utils.trainer import Trainer
from src.utils.evaluator import evaluate
from train_vtab import build_model


def grid_search(method, dataset_name, data_dir, args):
    """
    Perform grid search over hyperparameters.
    
    Returns:
        best_config: Dict with best hyperparameters
        best_val_acc: Best validation accuracy
    """
    num_classes = VTAB_NUM_CLASSES[dataset_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get data loaders
    train_loader = get_vtab_dataset(data_dir, dataset_name, split='train',
                                     batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = get_vtab_dataset(data_dir, dataset_name, split='val',
                                   batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Define hyperparameter grids
    lr_options = [1e-3, 1e-2]
    wd_options = [1e-4, 1e-3]
    drop_path_options = [0.0, 0.1]  # Key finding: drop path helps!
    
    # Method-specific options
    method_specific = {}
    if method in ['pfeif_adapter', 'houl_adapter']:
        method_specific = {
            'bottleneck_dim': [4, 8, 16, 32],
            'scale_factor': [0.01, 0.1, 1.0, 10.0],
        }
    elif method == 'adaptformer':
        method_specific = {
            'bottleneck_dim': [4, 16, 32],
            'scale_factor': [0.05, 0.1, 0.2],
        }
    elif method == 'convpass':
        method_specific = {
            'bottleneck_dim': [8, 16],
            'scale_factor': [0.01, 0.1, 1.0, 10.0, 100.0],
            'xavier_init': [True, False],
        }
    elif method == 'repadapter':
        method_specific = {
            'bottleneck_dim': [8, 16, 32],
            'scale_factor': [0.1, 0.5, 1.0, 5.0, 10.0],
        }
    elif method == 'lora':
        method_specific = {
            'rank': [1, 8, 16, 32],
        }
    elif method in ['fact_tt', 'fact_tk']:
        method_specific = {
            'rank': [8, 16, 32] if method == 'fact_tt' else [16, 32, 64],
            'scale_factor': [0.01, 0.1, 1.0, 10.0, 100.0],
        }
    elif method in ['vpt_shallow', 'vpt_deep']:
        method_specific = {
            'num_prompts': [5, 10, 50, 100] if method == 'vpt_deep' else [5, 10, 50, 100, 200],
        }
    
    # Generate all combinations
    base_params = list(itertools.product(lr_options, wd_options, drop_path_options))
    
    if method_specific:
        keys = list(method_specific.keys())
        values = list(method_specific.values())
        method_combos = list(itertools.product(*values))
    else:
        keys = []
        method_combos = [()]
    
    best_val_acc = 0.0
    best_config = {}
    all_results = []
    
    total_configs = len(base_params) * len(method_combos)
    print(f"Total configurations to try: {total_configs}")
    
    config_idx = 0
    for lr, wd, drop_path in base_params:
        for method_combo in method_combos:
            config_idx += 1
            
            # Build method kwargs
            method_kwargs = {}
            for k, v in zip(keys, method_combo):
                method_kwargs[k] = v
            
            print(f"\nConfig {config_idx}/{total_configs}: "
                  f"lr={lr}, wd={wd}, drop_path={drop_path}, {method_kwargs}")
            
            try:
                # Build model
                model = build_model(method, num_classes, 
                                   drop_path_rate=drop_path, **method_kwargs)
                
                # Train
                output_dir = os.path.join(args.output_dir, 'search', method, dataset_name, 
                                          f'config_{config_idx}')
                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_classes=num_classes,
                    lr=lr,
                    weight_decay=wd,
                    num_epochs=args.epochs,
                    device=str(device),
                    output_dir=output_dir,
                    experiment_name=f'search_{config_idx}',
                )
                
                val_acc = trainer.train(verbose=False)
                
                config = {
                    'lr': lr,
                    'weight_decay': wd,
                    'drop_path_rate': drop_path,
                    **method_kwargs,
                    'val_acc': val_acc,
                }
                all_results.append(config)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_config = config.copy()
                    print(f"  New best! Val acc: {val_acc:.2f}%")
                else:
                    print(f"  Val acc: {val_acc:.2f}% (best: {best_val_acc:.2f}%)")
            
            except Exception as e:
                print(f"  Error: {e}")
    
    return best_config, best_val_acc, all_results


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Search for PEFT')
    
    parser.add_argument('--method', type=str, required=True,
                        help='PEFT method to search')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset for hyperparameter search')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory for VTAB-1K data')
    
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs per config')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    print(f"Searching hyperparameters for {args.method} on {args.dataset}")
    
    best_config, best_val_acc, all_results = grid_search(
        args.method, args.dataset, args.data_dir, args
    )
    
    print(f"\n=== Best Configuration ===")
    print(f"Val Acc: {best_val_acc:.2f}%")
    print(f"Config: {best_config}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f'search_{args.method}_{args.dataset}.json')
    with open(results_path, 'w') as f:
        json.dump({
            'best_config': best_config,
            'best_val_acc': best_val_acc,
            'all_results': all_results,
        }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
