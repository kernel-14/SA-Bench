"""
Main script to run all experiments from the paper.

Usage:
    python run_experiments.py --experiment all
    python run_experiments.py --experiment out_of_sample
    python run_experiments.py --experiment input_extension
    python run_experiments.py --experiment multiphysics
    python run_experiments.py --experiment out_of_sample --config configs/default.yaml
"""

import argparse
import os
import sys
import json
import yaml
import torch


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_out_of_sample(config: dict):
    """Run out-of-sample parameter values experiments."""
    from experiments.experiment_out_of_sample import run_burgers_experiment
    
    exp_config = config.get('experiments', {}).get('out_of_sample', {})
    data_config = config.get('data', {})
    train_config = config.get('training', {})
    
    results = run_burgers_experiment(
        n_pretrain=data_config.get('n_pretrain', 800),
        n_finetune=data_config.get('n_finetune', 200),
        n_test=data_config.get('n_test', 200),
        n_epochs_pretrain=train_config.get('n_epochs_pretrain', 100),
        n_epochs_finetune=train_config.get('n_epochs_finetune', 50),
        batch_size=data_config.get('batch_size', 32),
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        save_dir=os.path.join(config.get('paths', {}).get('checkpoints', 'checkpoints'), 'out_of_sample'),
    )
    
    return results


def run_input_extension(config: dict):
    """Run input function set extension experiments."""
    from experiments.experiment_input_extension import run_heat_extension_experiment
    
    data_config = config.get('data', {})
    train_config = config.get('training', {})
    
    results = run_heat_extension_experiment(
        n_pretrain=data_config.get('n_pretrain', 800),
        n_finetune=data_config.get('n_finetune', 200),
        n_test=data_config.get('n_test', 200),
        n_epochs_pretrain=train_config.get('n_epochs_pretrain', 100),
        n_epochs_finetune=train_config.get('n_epochs_finetune', 50),
        batch_size=data_config.get('batch_size', 32),
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        save_dir=os.path.join(config.get('paths', {}).get('checkpoints', 'checkpoints'), 'input_extension'),
    )
    
    return results


def run_multiphysics(config: dict):
    """Run multi-physics pretraining experiments."""
    from experiments.experiment_multiphysics import run_multiphysics_experiment
    
    data_config = config.get('data', {})
    train_config = config.get('training', {})
    mp_config = config.get('experiments', {}).get('multiphysics', {})
    
    results = run_multiphysics_experiment(
        n_pretrain_per_physics=mp_config.get('n_pretrain_per_physics', 500),
        n_finetune=data_config.get('n_finetune', 200),
        n_test=data_config.get('n_test', 200),
        n_epochs_pretrain=train_config.get('n_epochs_pretrain', 100),
        n_epochs_finetune=train_config.get('n_epochs_finetune', 50),
        batch_size=data_config.get('batch_size', 32),
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        save_dir=os.path.join(config.get('paths', {}).get('checkpoints', 'checkpoints'), 'multiphysics'),
        pdebench_path=mp_config.get('pdebench_path', None),
    )
    
    return results


def save_results(results: dict, output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to serializable format
    serializable = {}
    for model_name, metrics in results.items():
        serializable[model_name] = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in metrics.items()
        }
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run neural operator experiments from the paper"
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='all',
        choices=['all', 'out_of_sample', 'input_extension', 'multiphysics'],
        help='Which experiment to run'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        print(f"Config file {args.config} not found, using defaults")
        config = {}
    
    # Set device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    all_results = {}
    
    if args.experiment in ['all', 'out_of_sample']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: Out-of-Sample Parameter Values")
        print("=" * 70)
        results = run_out_of_sample(config)
        all_results['out_of_sample'] = results
        save_results(results, os.path.join(args.output_dir, 'out_of_sample.json'))
    
    if args.experiment in ['all', 'input_extension']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: Input Function Set Extension")
        print("=" * 70)
        results = run_input_extension(config)
        all_results['input_extension'] = results
        save_results(results, os.path.join(args.output_dir, 'input_extension.json'))
    
    if args.experiment in ['all', 'multiphysics']:
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: Multi-Physics Pretraining")
        print("=" * 70)
        results = run_multiphysics(config)
        all_results['multiphysics'] = results
        save_results(results, os.path.join(args.output_dir, 'multiphysics.json'))
    
    # Save combined results
    if all_results:
        save_results(
            {k: v for exp_results in all_results.values() for k, v in exp_results.items()},
            os.path.join(args.output_dir, 'all_results.json')
        )
    
    print("\nAll experiments completed!")
    return all_results


if __name__ == "__main__":
    main()
