#!/usr/bin/env python3
"""VTAB-1K Training Script.

Trains PEFT methods on VTAB-1K benchmark (19 tasks, 3 groups).
Performs systematic hyperparameter tuning following the paper.

Usage:
    python scripts/train_vtab1k.py --dataset caltech101 --method lora
    python scripts/train_vtab1k.py --all_datasets --method bitfit
    python scripts/train_vtab1k.py --all_datasets --all_methods
"""

import argparse
import os
import sys
import json
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods import METHOD_MAP, METHOD_CATEGORIES
from methods.model_builder import build_model
from utils.data import get_vtab1k_dataset, VTAB_DATASETS, ALL_VTAB_TASKS
from utils.training import train_model, hyperparameter_search, get_vtab_hparam_grid
from utils.wise import apply_wise_to_model


def main():
    parser = argparse.ArgumentParser(description='Train PEFT methods on VTAB-1K')
    parser.add_argument('--dataset', type=str, default='caltech101',
                        help='Dataset name (or "all" for all 19 tasks)')
    parser.add_argument('--method', type=str, default='lora',
                        help='PEFT method name (or "all" for all 14 methods)')
    parser.add_argument('--data_dir', type=str, default='/data/vtab-1k')
    parser.add_argument('--output_dir', type=str, default='./results/vtab1k')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (if not specified, searched from [1e-3, 1e-2])')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay (if not specified, searched from [1e-4, 1e-3])')
    parser.add_argument('--drop_path_rate', type=float, default=None,
                        help='Drop path rate (0.0 or 0.1, searched if not specified)')
    parser.add_argument('--no_hparam_search', action='store_true',
                        help='Disable hyperparameter search, use defaults')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine datasets to run
    if args.dataset == 'all':
        datasets = ALL_VTAB_TASKS
    else:
        datasets = [args.dataset]
    
    # Determine methods to run
    if args.method == 'all':
        methods = list(METHOD_MAP.keys())
    else:
        methods = [args.method]
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        dataset_results = {}
        
        # Load dataset
        train_dataset, num_classes = get_vtab1k_dataset(
            dataset_name, data_dir=args.data_dir, split='train', seed=args.seed
        )
        val_dataset, _ = get_vtab1k_dataset(
            dataset_name, data_dir=args.data_dir, split='val', seed=args.seed
        )
        test_dataset, _ = get_vtab1k_dataset(
            dataset_name, data_dir=args.data_dir, split='test', seed=args.seed
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)
        
        for method_name in methods:
            print(f"\n  Method: {method_name}")
            
            if args.no_hparam_search:
                # Use default parameters
                model = build_model(
                    method_name=method_name,
                    num_classes=num_classes,
                    backbone_type='in21k',
                    drop_path_rate=0.1,
                )
                model = model.to(args.device)
                
                result = train_model(
                    model, train_loader, val_loader, test_loader,
                    method_name=method_name, dataset_name=dataset_name,
                    lr=args.lr or 5e-3,
                    weight_decay=args.weight_decay or 1e-4,
                    num_epochs=args.num_epochs,
                    device=args.device,
                )
            else:
                # Hyperparameter search
                hparam_grid = get_vtab_hparam_grid(method_name)
                
                # Remove lr and wd from grid if specified
                if args.lr is not None:
                    hparam_grid.pop('lr', None)
                if args.weight_decay is not None:
                    hparam_grid.pop('weight_decay', None)
                if args.drop_path_rate is not None:
                    hparam_grid.pop('drop_path_rate', None)
                
                def build_fn(**kwargs):
                    return build_model(
                        method_name=method_name,
                        num_classes=num_classes,
                        backbone_type='in21k',
                        drop_path_rate=kwargs.pop('drop_path_rate', args.drop_path_rate or 0.1),
                        **kwargs,
                    )
                
                # For lr/wd, do a simpler search
                lr_values = [args.lr] if args.lr else [1e-3, 5e-3, 1e-2]
                wd_values = [args.weight_decay] if args.weight_decay else [1e-4, 5e-4, 1e-3]
                
                best_result = None
                best_val_acc = 0.0
                
                for lr in lr_values:
                    for wd in wd_values:
                        result, hparams, _ = hyperparameter_search(
                            build_fn, train_loader, val_loader, test_loader,
                            hparam_grid, method_name, dataset_name,
                            num_classes,
                            device=args.device,
                            num_epochs=args.num_epochs,
                            verbose=False,
                        )
                        result['lr'] = lr
                        result['weight_decay'] = wd
                        
                        if result['best_val_accuracy'] > best_val_acc:
                            best_val_acc = result['best_val_accuracy']
                            best_result = result
                
                result = best_result
            
            dataset_results[method_name] = result
            
            # Save individual result
            save_path = os.path.join(args.output_dir, 
                                     f"{dataset_name}_{method_name}.json")
            with open(save_path, 'w') as f:
                json.dump({
                    'method': result['method'],
                    'dataset': result['dataset'],
                    'test_accuracy': result['test_accuracy'],
                    'best_val_accuracy': result['best_val_accuracy'],
                    'trainable_params_millions': result['trainable_params_millions'],
                }, f, indent=2)
            
            print(f"    Test accuracy: {result['test_accuracy']:.2f}%")
            print(f"    Trainable params: {result['trainable_params_millions']:.2f}M")
        
        all_results[dataset_name] = dataset_results
    
    # Save summary
    summary = {}
    for dataset_name, dataset_results in all_results.items():
        summary[dataset_name] = {
            method: {
                'test_accuracy': r['test_accuracy'],
                'trainable_params_m': r['trainable_params_millions'],
            }
            for method, r in dataset_results.items()
        }
    
    summary_path = os.path.join(args.output_dir, 'vtab1k_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
