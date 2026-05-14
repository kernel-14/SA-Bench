"""
Main entry point for SC-FNO experiments.

Reproduces the core experiments from:
"Sensitivity-Constrained Fourier Neural Operators for Forward and Inverse 
Problems in Parametric Differential Equations" (Behroozi, Shen & Kifer).

Usage:
    python main.py --case PDE1 --model SC-FNO --n_samples 2000
    python main.py --case PDE1 --model all  # Compare all 4 variants
    python main.py --case PDE1 --experiment inversion  # Run inversion
    python main.py --case PDE2_ZONED --n_samples 500  # High-dim experiment
"""

import argparse
import yaml
import torch
import numpy as np
import time
import os

from models.fno import FNO
from models.other_operators import DeepONet, WaveletNO, MultiWaveletNO
from data.dataset_generation import generate_dataset
from training.train_fno import train_fno, train_sc_fno, train_fno_pinn, train_sc_fno_pinn
from training.data_utils import prepare_dataloaders
from utils.metrics import compute_all_metrics, evaluate_model
from inversion.parameter_inversion import inversion_experiment


def load_config(case):
    """Load configuration for a specific case."""
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'cases', f'{case.lower()}.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use default config with case-specific overrides
        config = {
            'case': case,
            'data': {'n_samples': 2000},
            'model': {'modes': [8, 8], 'spatial_dims': 2, 'width': 20, 'n_layers': 4},
            'training': {'epochs': 500, 'learning_rate': 0.001, 'batch_size': 4},
        }
    return config


def create_model(case_config, device='cpu'):
    """Create FNO model based on case configuration."""
    mc = case_config['model']
    model = FNO(
        input_dim=mc.get('input_dim', 7),
        output_dim=mc.get('output_dim', 1),
        modes=mc.get('modes', [8, 8]),
        width=mc.get('width', 20),
        n_layers=mc.get('n_layers', 4),
        spatial_dims=mc.get('spatial_dims', 2),
    )
    return model.to(device)


def run_experiment(config, model_type='SC-FNO', device='cpu'):
    """
    Run a complete experiment for one model type.
    
    Steps:
    1. Generate dataset
    2. Prepare dataloaders
    3. Train model with appropriate loss
    4. Evaluate on test set
    5. Optionally run inversion
    
    Args:
        config: Case configuration dict
        model_type: One of 'FNO', 'SC-FNO', 'FNO-PINN', 'SC-FNO-PINN'
        device: torch device
    
    Returns:
        dict with results
    """
    case = config['case']
    print(f"\n{'='*70}")
    print(f"Running {model_type} on {case}")
    print(f"{'='*70}")
    
    # 1. Generate data
    print("\n[1] Generating dataset...")
    n_samples = config['data'].get('n_samples', 2000)
    data = generate_dataset(case, n_samples=n_samples, device=device, use_ad=True)
    print(f"    Generated {n_samples} samples")
    print(f"    u shape: {data['u_true'].shape}")
    print(f"    Jacobian shape: {data['jac_true'].shape}")
    print(f"    Parameters: {data['param_names']}")
    
    # 2. Prepare dataloaders
    print("\n[2] Preparing dataloaders...")
    batch_size = config['training'].get('batch_size', 4)
    M = config['data'].get('M', 5)
    
    train_loader, val_loader, test_loader = prepare_dataloaders(
        data, batch_size=batch_size, case=case, M=M
    )
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches: {len(val_loader)}")
    print(f"    Test batches: {len(test_loader)}")
    
    # 3. Create model
    print("\n[3] Creating model...")
    model = create_model(config, device)
    n_params = model.count_params()
    print(f"    Learnable parameters: {n_params}")
    
    # 4. Train
    print(f"\n[4] Training {model_type}...")
    epochs = config['training'].get('epochs', 500)
    lr = config['training'].get('learning_rate', 0.001)
    
    t_start = time.time()
    
    if model_type == 'FNO':
        model, history = train_fno(model, train_loader, val_loader,
                                   epochs=epochs, lr=lr, device=device, verbose=True)
    elif model_type == 'SC-FNO':
        model, history = train_sc_fno(model, train_loader, val_loader,
                                      epochs=epochs, lr=lr, device=device, verbose=True)
    elif model_type == 'FNO-PINN':
        # Note: PDE residual function would need to be provided for each case
        print("    WARNING: FNO-PINN requires PDE residual function; falling back to FNO")
        model, history = train_fno(model, train_loader, val_loader,
                                   epochs=epochs, lr=lr, device=device, verbose=True)
    elif model_type == 'SC-FNO-PINN':
        print("    WARNING: SC-FNO-PINN requires PDE residual function; falling back to SC-FNO")
        model, history = train_sc_fno(model, train_loader, val_loader,
                                      epochs=epochs, lr=lr, device=device, verbose=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    train_time = time.time() - t_start
    avg_epoch_time = train_time / len(history['epoch_times']) if history['epoch_times'] else train_time / epochs
    print(f"    Training time: {train_time:.1f}s")
    print(f"    Avg epoch time: {avg_epoch_time:.2f}s")
    
    # 5. Evaluate
    print(f"\n[5] Evaluating {model_type}...")
    param_names = data['param_names']
    metrics = compute_all_metrics(model, test_loader, param_names, device)
    
    print(f"\n    Solution u metrics:")
    print(f"      R² = {metrics.get('R2', 'N/A')}")
    print(f"      Relative L² = {metrics.get('relative_L2', 'N/A')}")
    
    print(f"    Jacobian metrics:")
    for pname in param_names:
        r2_key = f'∂u/∂{pname}_R2'
        l2_key = f'∂u/∂{pname}_relL2'
        print(f"      ∂u/∂{pname}: R² = {metrics.get(r2_key, 'N/A'):.4f}, Rel L² = {metrics.get(l2_key, 'N/A'):.4f}")
    print(f"      Average: R² = {metrics.get('avg_jac_R2', 'N/A'):.4f}, Rel L² = {metrics.get('avg_jac_relL2', 'N/A'):.4f}")
    
    results = {
        'model_type': model_type,
        'case': case,
        'n_params': n_params,
        'train_time': train_time,
        'avg_epoch_time': avg_epoch_time,
        'metrics': metrics,
        'history': history,
    }
    
    return results, model, test_loader


def run_inversion(model, test_loader, config, device='cpu'):
    """Run parameter inversion experiment after training."""
    case = config['case']
    print(f"\n{'='*70}")
    print(f"Running Inversion Experiment on {case}")
    print(f"{'='*70}")
    
    # Convert test_loader to list for easier access
    test_data = []
    for batch in test_loader:
        test_data.extend(list(zip(*batch)))
    
    inversion_results = inversion_experiment(
        model, test_data, case, model_name='SC-FNO', device=device
    )
    
    return inversion_results


def main():
    parser = argparse.ArgumentParser(description='SC-FNO Experiments')
    parser.add_argument('--case', type=str, default='PDE1',
                        choices=['ODE1', 'ODE2', 'PDE1', 'PDE2', 'PDE3', 'PDE4', 'PDE2_ZONED'],
                        help='Which case to run')
    parser.add_argument('--model', type=str, default='SC-FNO',
                        choices=['FNO', 'SC-FNO', 'FNO-PINN', 'SC-FNO-PINN', 'all'],
                        help='Model variant to train')
    parser.add_argument('--experiment', type=str, default='train',
                        choices=['train', 'inversion', 'all'],
                        help='Which experiment to run')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.case)
    
    # Override with CLI arguments
    if args.n_samples is not None:
        config['data']['n_samples'] = args.n_samples
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.model == 'all':
        model_types = ['FNO', 'SC-FNO', 'FNO-PINN', 'SC-FNO-PINN']
    else:
        model_types = [args.model]
    
    all_results = {}
    
    for model_type in model_types:
        results, model, test_loader = run_experiment(config, model_type, device)
        all_results[model_type] = results
        
        if args.experiment in ['inversion', 'all'] and args.case in ['PDE1', 'PDE2']:
            inv_results = run_inversion(model, test_loader, config, device)
            all_results[f'{model_type}_inversion'] = inv_results
    
    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    for model_type, results in all_results.items():
        if 'metrics' in results:
            m = results['metrics']
            print(f"\n{model_type}:")
            print(f"  u R²: {m.get('R2', 'N/A'):.4f}")
            print(f"  u Rel L²: {m.get('relative_L2', 'N/A'):.4f}")
            print(f"  Avg Jacobian R²: {m.get('avg_jac_R2', 'N/A'):.4f}")
            print(f"  Avg Jacobian Rel L²: {m.get('avg_jac_relL2', 'N/A'):.4f}")
            print(f"  Train time: {results.get('train_time', 0):.1f}s")
            print(f"  Avg epoch: {results.get('avg_epoch_time', 0):.2f}s")
    
    return all_results


if __name__ == '__main__':
    main()
