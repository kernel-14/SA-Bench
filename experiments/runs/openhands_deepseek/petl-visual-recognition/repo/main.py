"""Main entry point for PEFT study experiments.

Usage:
    # VTAB-1K experiment
    python main.py --mode vtab1k --method lora --dataset cifar100

    # Many-shot experiment
    python main.py --mode many_shot --method lora --dataset cifar100

    # Robustness experiment
    python main.py --mode robustness --method lora

    # HP sweep
    python main.py --mode hp_sweep --method lora --dataset cifar100

    # WiSE evaluation
    python main.py --mode wise --method lora --checkpoint path/to/model.pt

    # Ensemble evaluation
    python main.py --mode ensemble --checkpoints_dir path/to/checkpoints/
"""

import argparse
import torch
import yaml
import os
import json
import itertools
import numpy as np

from models.vit import vit_base_patch16_224
from models.peft_vit import build_peft_vit
from models.peft import count_trainable_params
from train import build_model, train_vtab1k, train_many_shot, train_robustness
from evaluate import (
    evaluate_model, compute_prediction_similarity,
    majority_vote_ensemble, logit_average_ensemble,
    apply_wise_to_model, evaluate_wise, WiSEPEFT,
    compute_ranking_frequency,
)
from data.vtab import (
    VTAB_TASKS, VTAB_NUM_CLASSES,
    create_vtab_loaders, create_vtab_full_train_loader,
)
from data.datasets import (
    create_many_shot_loaders,
    create_imagenet_100shot_loaders, get_clip_zero_shot_weights,
)

PEFT_METHODS = [
    'vpt_shallow', 'vpt_deep',
    'bitfit', 'difffit', 'layernorm', 'ssf',
    'pfeif_adapter', 'houl_adapter', 'adaptformer',
    'repadapter', 'convpass',
    'lora', 'fact_tt', 'fact_tk',
]


def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='PEFT Study for Visual Recognition')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['vtab1k', 'many_shot', 'robustness',
                                 'hp_sweep', 'wise', 'ensemble'])
    parser.add_argument('--method', type=str, default=None,
                        help='PEFT method name (or "linear", "full")')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--checkpoints_dir', type=str, default=None)
    parser.add_argument('--backbone', type=str, default='in21k',
                        choices=['in21k', 'clip'])
    parser.add_argument('--pretrained_path', type=str, default=None)
    # PEFT hyperparams
    parser.add_argument('--prompt_num', type=int, default=None)
    parser.add_argument('--bottle_neck', type=int, default=None)
    parser.add_argument('--scale_factor', type=float, default=None)
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


def run_vtab1k(args, config):
    """Run VTAB-1K experiment for one method on one dataset."""
    num_classes = VTAB_NUM_CLASSES[args.dataset]
    peft_config = build_peft_config(args, config)

    # Build model
    model = build_model(
        num_classes=num_classes,
        peft_method=args.method if args.method not in ('linear', 'full') else
                     (None if args.method == 'linear' else 'full'),
        peft_config=peft_config,
        pretrained_path=args.pretrained_path,
        backbone=args.backbone,
    )
    model = model.to(args.device)

    # Create data loaders
    train_loader, val_loader, test_loader = create_vtab_loaders(
        args.data_root, args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Train
    lr = args.lr or 1e-3
    wd = args.weight_decay or 1e-4
    epochs = args.epochs or 100

    _, val_acc, test_acc = train_vtab1k(
        model, train_loader, val_loader, test_loader,
        epochs=epochs, lr=lr, weight_decay=wd, device=args.device,
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'method': args.method,
        'dataset': args.dataset,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'trainable_params': count_trainable_params(model),
        'peft_config': peft_config,
    }
    result_path = os.path.join(
        args.output_dir, f'vtab1k_{args.method}_{args.dataset}.json'
    )
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Results saved to {result_path}')

    return result


def run_hp_sweep(args, config):
    """Run hyperparameter sweep for one method on one dataset."""
    num_classes = VTAB_NUM_CLASSES[args.dataset]
    cfg = config['peft'].get(args.method, {})

    # Define HP grid
    hp_grids = {
        'vpt_shallow': {'prompt_num': cfg.get('prompt_num', [5, 10, 50, 100, 200])},
        'vpt_deep': {'prompt_num': cfg.get('prompt_num', [5, 10, 50, 100])},
        'pfeif_adapter': {
            'bottle_neck': cfg.get('bottle_neck', [4, 8, 16, 32]),
            'scale_factor': cfg.get('scale_factor', [0.01, 0.1, 1, 10]),
        },
        'houl_adapter': {
            'bottle_neck': cfg.get('bottle_neck', [4, 8, 16, 32]),
            'scale_factor': cfg.get('scale_factor', [0.01, 0.1, 1, 10]),
        },
        'adaptformer': {
            'bottle_neck': cfg.get('bottle_neck', [4, 16, 32]),
            'scale_factor': cfg.get('scale_factor', [0.05, 0.1, 0.2]),
        },
        'repadapter': {
            'bottle_neck': cfg.get('bottle_neck', [8, 16, 32]),
            'scale_factor': cfg.get('scale_factor', [0.1, 0.5, 1, 5, 10]),
        },
        'convpass': {
            'bottle_neck': cfg.get('bottle_neck', [8, 16]),
            'scale_factor': cfg.get('scale_factor', [0.01, 0.1, 1, 10, 100]),
        },
        'lora': {'rank': cfg.get('rank', [1, 8, 16, 32])},
        'fact_tt': {
            'bottle_neck': cfg.get('bottle_neck', [8, 16, 32]),
            'scale_factor': cfg.get('scale_factor', [0.01, 0.1, 1, 10, 100]),
        },
        'fact_tk': {
            'bottle_neck': cfg.get('bottle_neck', [16, 32, 64]),
            'scale_factor': cfg.get('scale_factor', [0.01, 0.1, 1, 10, 100]),
        },
    }

    train_lr = config['training']['vtab1k']['lr']
    train_wd = config['training']['vtab1k']['weight_decay']

    if args.method in hp_grids:
        hp_dict = hp_grids[args.method]
        keys = list(hp_dict.keys())
        values = list(hp_dict.values())
    else:
        hp_dict = {}
        keys = []
        values = []

    dp_rates = config['training']['vtab1k']['drop_path_rate']

    all_combos = list(itertools.product(*values)) if values else [()]
    best_val_acc = 0
    best_config = None
    results = []

    train_loader, val_loader, test_loader = create_vtab_loaders(
        args.data_root, args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    for combo in all_combos:
        peft_config = {k: v for k, v in zip(keys, combo)}
        peft_config['drop_path_rate'] = 0.1  # default nonzero

        for dp_rate in dp_rates:
            peft_config['drop_path_rate'] = dp_rate

            for lr in train_lr:
                for wd in train_wd:
                    model = build_model(
                        num_classes=num_classes,
                        peft_method=args.method,
                        peft_config=peft_config,
                        pretrained_path=args.pretrained_path,
                        backbone=args.backbone,
                    )
                    model = model.to(args.device)

                    _, val_acc, test_acc = train_vtab1k(
                        model, train_loader, val_loader, test_loader,
                        epochs=args.epochs or 100, lr=lr, weight_decay=wd,
                        device=args.device,
                    )

                    result_entry = {
                        'peft_config': peft_config,
                        'lr': lr,
                        'weight_decay': wd,
                        'val_acc': val_acc,
                        'test_acc': test_acc,
                        'params': count_trainable_params(model),
                    }
                    results.append(result_entry)
                    print(f'Config: {peft_config}, lr={lr}, wd={wd}, '
                          f'Val={val_acc:.2f}%, Test={test_acc:.2f}%')

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_config = result_entry

    print(f'Best val acc: {best_val_acc:.2f}%')
    print(f'Best config: {best_config}')

    os.makedirs(args.output_dir, exist_ok=True)
    sweep_path = os.path.join(
        args.output_dir, f'hp_sweep_{args.method}_{args.dataset}.json'
    )
    with open(sweep_path, 'w') as f:
        json.dump({'best': best_config, 'all': results}, f, indent=2)

    return best_config


def run_many_shot(args, config):
    """Run many-shot experiment."""
    train_loader, val_loader, test_loader, num_classes = create_many_shot_loaders(
        args.data_root, args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    peft_config = build_peft_config(args, config)

    model = build_model(
        num_classes=num_classes,
        peft_method=args.method if args.method not in ('linear', 'full') else
                     (None if args.method == 'linear' else 'full'),
        peft_config=peft_config,
        pretrained_path=args.pretrained_path,
        backbone=args.backbone,
    )
    model = model.to(args.device)

    lr = args.lr or 5e-4
    wd = args.weight_decay or 1e-4
    epochs = args.epochs or 40

    _, val_acc, test_acc = train_many_shot(
        model, train_loader, val_loader, test_loader,
        epochs=epochs, lr=lr, weight_decay=wd, device=args.device,
    )

    result = {
        'method': args.method,
        'dataset': args.dataset,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'trainable_params': count_trainable_params(model),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(
        args.output_dir, f'many_shot_{args.method}_{args.dataset}.json'
    )
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def run_robustness(args, config):
    """Run robustness experiment with CLIP ViT."""
    train_loader, imagenet_val_loader, shift_loaders = create_imagenet_100shot_loaders(
        args.data_root, n_shot_per_class=100,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = 1000
    peft_config = build_peft_config(args, config)
    peft_config['drop_path_rate'] = 0.0  # robustness uses no drop path

    model = build_model(
        num_classes=num_classes,
        peft_method=args.method if args.method not in ('linear', 'full') else
                     (None if args.method == 'linear' else 'full'),
        peft_config=peft_config,
        pretrained_path=args.pretrained_path,
        backbone='clip',
    )
    model = model.to(args.device)

    _, results = train_robustness(
        model, train_loader, imagenet_val_loader, shift_loaders,
        epochs=args.epochs or 10,
        lr=args.lr or 3e-5,
        weight_decay=args.weight_decay or 5e-3,
        device=args.device,
    )

    results['method'] = args.method

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(
        args.output_dir, f'robustness_{args.method}.json'
    )
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_wise(args, config):
    """Run WiSE evaluation between fine-tuned and pre-trained models."""
    # Load fine-tuned model
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise ValueError(f'Checkpoint not found: {args.checkpoint}')

    num_classes = 1000  # ImageNet
    peft_config = build_peft_config(args, config)

    finetuned_model = build_model(
        num_classes=num_classes,
        peft_method=args.method,
        peft_config=peft_config,
        backbone=args.backbone,
    )
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    finetuned_model.load_state_dict(checkpoint['model_state_dict']
                                     if 'model_state_dict' in checkpoint
                                     else checkpoint)
    finetuned_model = finetuned_model.to(args.device)

    pretrained_model = build_model(
        num_classes=num_classes,
        peft_method=None,
        backbone=args.backbone,
        pretrained_path=args.pretrained_path,
    )
    pretrained_model = pretrained_model.to(args.device)

    _, imagenet_val_loader, shift_loaders = create_imagenet_100shot_loaders(
        args.data_root, batch_size=args.batch_size,
    )

    alphas = config.get('wise', {}).get('alphas',
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    # Evaluate on target
    target_results = evaluate_wise(
        None, pretrained_model, finetuned_model,
        imagenet_val_loader, alphas, args.device
    )

    shift_results = {}
    for shift_name, shift_loader in shift_loaders.items():
        shift_results[shift_name] = evaluate_wise(
            None, pretrained_model, finetuned_model,
            shift_loader, alphas, args.device
        )

    results = {
        'method': args.method,
        'target': target_results,
        'shifts': shift_results,
        'avg_shift': {alpha: np.mean([shift_results[s][alpha]
                       for s in shift_results])
                      for alpha in alphas},
    }

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, f'wise_{args.method}.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_ensemble(args, config):
    """Run ensemble evaluation across multiple PEFT checkpoints."""
    checkpoints_dir = args.checkpoints_dir
    if not checkpoints_dir or not os.path.exists(checkpoints_dir):
        raise ValueError(f'Checkpoints directory not found: {checkpoints_dir}')

    # Load all model predictions
    results_dict = {}

    for ckpt_file in sorted(os.listdir(checkpoints_dir)):
        if not ckpt_file.endswith('.pt'):
            continue
        method_name = os.path.splitext(ckpt_file)[0]
        checkpoint = torch.load(os.path.join(checkpoints_dir, ckpt_file),
                               map_location='cpu')

        # Infer dataset from checkpoint
        dataset = checkpoint.get('dataset', args.dataset)
        num_classes = VTAB_NUM_CLASSES.get(dataset, 1000)

        # Build and load model
        model = build_model(num_classes=num_classes, peft_method=method_name)
        model.load_state_dict(checkpoint['model_state_dict']
                              if 'model_state_dict' in checkpoint
                              else checkpoint)
        model = model.to(args.device)

        # Get test loader
        _, _, test_loader = create_vtab_loaders(args.data_root, dataset)

        eval_results = evaluate_model(model, test_loader, args.device)
        results_dict[method_name] = eval_results

    # Compute prediction similarity
    overlap_matrix, method_names = compute_prediction_similarity(results_dict)
    print('Prediction overlap matrix:')
    print(overlap_matrix)

    # Ensemble
    maj_acc, _ = majority_vote_ensemble(results_dict)
    logit_acc, _ = logit_average_ensemble(results_dict)
    print(f'Majority vote ensemble: {maj_acc:.2f}%')
    print(f'Logit average ensemble: {logit_acc:.2f}%')

    # Ranking frequency
    acc_dict = {n: results_dict[n]['acc'] for n in method_names}
    freq_matrix, method_order, mean_ranks = compute_ranking_frequency(
        {args.dataset: acc_dict}
    )
    print(f'Mean ranks: {mean_ranks}')

    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'overlap_matrix': overlap_matrix.tolist(),
        'method_names': method_names,
        'majority_vote_acc': maj_acc,
        'logit_average_acc': logit_acc,
        'mean_ranks': mean_ranks,
    }
    result_path = os.path.join(args.output_dir, f'ensemble_{args.dataset}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def build_peft_config(args, config):
    """Build PEFT config dict from args and config file."""
    peft_config = {'drop_path_rate': args.drop_path_rate}

    if args.prompt_num is not None:
        peft_config['prompt_num'] = args.prompt_num
    elif args.method in ('vpt_shallow', 'vpt_deep'):
        cfg = config.get('peft', {}).get(args.method, {})
        peft_config['prompt_num'] = cfg.get('prompt_num', [10])[0] if isinstance(
            cfg.get('prompt_num'), list) else cfg.get('prompt_num', 10)

    if args.bottle_neck is not None:
        peft_config['bottle_neck'] = args.bottle_neck
    if args.scale_factor is not None:
        peft_config['scale_factor'] = args.scale_factor
    if args.rank is not None:
        peft_config['rank'] = args.rank

    return peft_config


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    mode_fns = {
        'vtab1k': run_vtab1k,
        'hp_sweep': run_hp_sweep,
        'many_shot': run_many_shot,
        'robustness': run_robustness,
        'wise': run_wise,
        'ensemble': run_ensemble,
    }

    fn = mode_fns[args.mode]
    result = fn(args, config)
    print(f'{args.mode} completed successfully.')
