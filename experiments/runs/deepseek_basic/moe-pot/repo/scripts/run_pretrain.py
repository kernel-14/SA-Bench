"""
Pre-training script for MoE-POT.

Trains the MoE-POT model on 6 public PDE datasets:
- FNO-NS (1e-5)
- FNO-NS (1e-3)
- PDEBench-CNS (0.1, 0.01)
- PDEBench-SWE
- PDEBench-DR
- CFDBench

Usage:
    python scripts/run_pretrain.py --config configs/moe_pot_tiny.yaml
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from moe_pot import MoEPOT, create_moe_pot_tiny, create_moe_pot_small, create_moe_pot_medium
from moe_pot.training import MoEPOTTrainer, MultiPDEDataset, BalancedBatchSampler
from moe_pot.data_utils import prepare_pre_training_datasets, standardize_resolution, pad_channels


def parse_args():
    parser = argparse.ArgumentParser(description='Pre-train MoE-POT')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--synthetic', action='store_true', default=True,
                        help='Use synthetic data for testing')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to real PDE data')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading configuration for {config['model']['name']}")
    print(f"Model: dim={config['model']['dim']}, layers={config['model']['num_layers']}, "
          f"heads={config['model']['num_heads']}")
    print(f"MoE: {config['model']['num_routed_experts']} routed experts, "
          f"{config['model']['num_shared_experts']} shared, top-k={config['model']['top_k']}")
    
    # Prepare datasets
    print("\nPreparing pre-training datasets...")
    datasets = prepare_pre_training_datasets(
        data_dir=args.data_dir,
        synthetic=args.synthetic,
        spatial_size=config['data']['spatial_size'],
        num_timesteps=20,
    )
    
    print(f"Loaded {len(datasets)} datasets:")
    for name, data in datasets.items():
        print(f"  {name}: {data.shape}")
    
    # Standardize and unify channels
    max_channels = max(d.shape[2] for d in datasets.values())
    print(f"\nMax channels across datasets: {max_channels}")
    print(f"Padding all datasets to {max_channels} channels")
    
    unified_datasets = []
    for name, data in datasets.items():
        if data.shape[2] < max_channels:
            data = pad_channels(data, max_channels)
        print(f"  {name}: {data.shape}")
        unified_datasets.append(data)
    
    # Create model
    print(f"\nCreating {config['model']['name']}...")
    model_cfg = config['model']
    model = MoEPOT(
        in_channels=max_channels,
        out_channels=max_channels,
        spatial_size=config['data']['spatial_size'],
        patch_size=model_cfg['patch_size'],
        T=config['data']['T'],
        dim=model_cfg['dim'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        mode=model_cfg['mode'],
        num_routed_experts=model_cfg['num_routed_experts'],
        num_shared_experts=model_cfg['num_shared_experts'],
        top_k=model_cfg['top_k'],
        dropout=model_cfg.get('dropout', 0.0),
        use_positional_encoding=model_cfg.get('use_positional_encoding', True),
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create multi-dataset with balanced sampling
    dataset = MultiPDEDataset(
        datasets=unified_datasets,
        T=config['data']['T'],
        weights=[config['training']['weight_dataset']] * len(unified_datasets),
        dataset_names=list(datasets.keys()),
    )
    
    sampler = BalancedBatchSampler(
        dataset=dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
    )
    
    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
    )
    
    # Create trainer
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    trainer = MoEPOTTrainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training']['betas']),
        noise_epsilon=config['training']['noise_epsilon'],
        load_balance_weight=config['training']['load_balance_weight'],
    )
    
    # Pre-train
    print(f"\nStarting pre-training for {config['training']['num_epochs']} epochs...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    trainer.pre_train(
        train_loader=train_loader,
        num_epochs=config['training']['num_epochs'],
        warmup_epochs=config['training']['warmup_epochs'],
        save_path=os.path.join(args.output_dir, f"{config['model']['name']}_pretrained.pt"),
        log_interval=10,
    )
    
    print("\nPre-training complete!")


if __name__ == '__main__':
    main()
