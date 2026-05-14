"""
Training script for many-shot experiments.
Evaluates PEFT methods on full-size datasets:
- CIFAR-100 (50K training images, 100 classes)
- RESISC45 (25.2K training samples, 45 classes)
- Clevr-Distance (70K samples, 6 classes)

Usage:
    python train_manyshot.py --method bitfit --dataset cifar100 --data_dir /path/to/data
    python train_manyshot.py --method all --dataset all --data_dir /path/to/data
"""

import os
import sys
import argparse
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.vit import create_vit_model, count_trainable_params
from src.datasets.manyshot import get_manyshot_dataset
from src.utils.trainer import Trainer
from src.utils.evaluator import evaluate
from train_vtab import build_model


MANYSHOT_NUM_CLASSES = {
    'cifar100': 100,
    'resisc45': 45,
    'clevr_distance': 6,
}


def main():
    parser = argparse.ArgumentParser(description='Many-shot PEFT Evaluation')
    
    parser.add_argument('--method', type=str, default='bitfit',
                        choices=['linear', 'full', 'bitfit', 'layernorm', 'difffit', 'ssf',
                                'vpt_shallow', 'vpt_deep', 'pfeif_adapter', 'houl_adapter',
                                'adaptformer', 'convpass', 'repadapter', 'lora', 'fact_tt', 'fact_tk', 'all'],
                        help='PEFT method to use')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'resisc45', 'clevr_distance', 'all'],
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory for datasets')
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='Drop path rate')
    
    # Method-specific hyperparameters
    parser.add_argument('--rank', type=int, default=8, help='Rank for LoRA/FacT')
    parser.add_argument('--bottleneck_dim', type=int, default=8, help='Bottleneck dim for adapters')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Scale factor for adapters')
    parser.add_argument('--num_prompts', type=int, default=10, help='Number of prompts for VPT')
    
    # Other settings
    parser.add_argument('--output_dir', type=str, default='./output_manyshot', help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--results_file', type=str, default='manyshot_results.json',
                        help='File to save results')
    
    args = parser.parse_args()
    
    # Determine methods and datasets
    if args.method == 'all':
        methods = ['linear', 'full', 'bitfit', 'layernorm', 'difffit', 'ssf',
                  'vpt_shallow', 'vpt_deep', 'pfeif_adapter', 'houl_adapter',
                  'adaptformer', 'convpass', 'repadapter', 'lora', 'fact_tt', 'fact_tk']
    else:
        methods = [args.method]
    
    if args.dataset == 'all':
        datasets = ['cifar100', 'resisc45', 'clevr_distance']
    else:
        datasets = [args.dataset]
    
    method_kwargs = {
        'rank': args.rank,
        'bottleneck_dim': args.bottleneck_dim,
        'scale_factor': args.scale_factor,
        'num_prompts': args.num_prompts,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    for method in methods:
        results[method] = {}
        for dataset_name in datasets:
            try:
                num_classes = MANYSHOT_NUM_CLASSES[dataset_name]
                
                # Build model
                model = build_model(method, num_classes, 
                                   drop_path_rate=args.drop_path_rate, **method_kwargs)
                
                trainable_params = count_trainable_params(model)
                print(f"\nMethod: {method}, Dataset: {dataset_name}, "
                      f"Trainable params: {trainable_params:.3f}M")
                
                # Get data loaders
                train_loader = get_manyshot_dataset(
                    args.data_dir, dataset_name, split='train',
                    batch_size=args.batch_size, num_workers=args.num_workers
                )
                val_loader = get_manyshot_dataset(
                    args.data_dir, dataset_name, split='val',
                    batch_size=args.batch_size, num_workers=args.num_workers
                )
                test_loader = get_manyshot_dataset(
                    args.data_dir, dataset_name, split='test',
                    batch_size=args.batch_size, num_workers=args.num_workers
                )
                
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
                test_acc = evaluate(model, test_loader, device=str(device))
                
                print(f"Method: {method}, Dataset: {dataset_name}, "
                      f"Best Val Acc: {best_val_acc:.2f}%, Test Acc: {test_acc:.2f}%")
                
                results[method][dataset_name] = {
                    'test_acc': test_acc,
                    'val_acc': best_val_acc,
                    'trainable_params': trainable_params,
                }
                
            except Exception as e:
                print(f"Error running {method} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                results[method][dataset_name] = {'error': str(e)}
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, args.results_file)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Print summary
    print("\n=== Many-Shot Results Summary ===")
    for dataset_name in datasets:
        print(f"\n{dataset_name}:")
        print(f"{'Method':<20} {'Test Acc':<10} {'#Params':<10}")
        print("-" * 40)
        for method in methods:
            if dataset_name in results.get(method, {}):
                r = results[method][dataset_name]
                if 'test_acc' in r:
                    print(f"{method:<20} {r['test_acc']:<10.2f} {r.get('trainable_params', 0):<10.3f}")


if __name__ == '__main__':
    main()
