#!/usr/bin/env python3
"""
Ablation study on history horizon M and forecast horizon N (Section A.4.1).

Reproduces the heatmap analysis from Fig. S8:
- Left: relative autoregressive prediction error for different (M, N) combos
- Right: training time for different (M, N) combos

Usage:
    python scripts/ablation_horizons.py --data_path /path/to/data
"""

import os
import sys
import argparse
import numpy as np
import torch
import time
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from rwm.world_model import RoboticWorldModel
from rwm.training import TrajectoryDataset, WorldModelTrainer, create_dataloader
from rwm.evaluation import compare_models


def parse_args():
    parser = argparse.ArgumentParser(description='Ablation study on M and N')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to trajectory data')
    parser.add_argument('--output_dir', type=str, default='./outputs/ablation',
                       help='Output directory')
    parser.add_argument('--M_values', type=int, nargs='+', 
                       default=[8, 16, 32, 64, 128],
                       help='History horizon values to test')
    parser.add_argument('--N_values', type=int, nargs='+',
                       default=[1, 2, 4, 8, 16],
                       help='Forecast horizon values to test')
    parser.add_argument('--max_iterations', type=int, default=500,
                       help='Training iterations per config')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def load_data(data_path):
    """Load trajectories."""
    import pickle
    data_path = Path(data_path)
    if data_path.suffix == '.pkl':
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    elif data_path.suffix == '.npy':
        data = np.load(data_path, allow_pickle=True).item()
    else:
        raise ValueError(f"Unsupported format: {data_path.suffix}")
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'trajectories' in data:
        return data['trajectories']
    return [data]


def run_ablation(args):
    """Run ablation study over M and N values."""
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load data
    trajectories = load_data(args.data_path)
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Split
    n = len(trajectories)
    n_train = int(n * 0.8)
    train_traj = trajectories[:n_train]
    val_traj = trajectories[n_train:]
    
    sample = train_traj[0]
    obs_dim = sample['obs'].shape[-1]
    act_dim = sample['act'].shape[-1]
    priv_dim = sample.get('priv', np.zeros((1, 0))).shape[-1]
    
    M_values = args.M_values
    N_values = args.N_values
    
    # Store results
    error_grid = np.zeros((len(M_values), len(N_values)))
    time_grid = np.zeros((len(M_values), len(N_values)))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, M in enumerate(M_values):
        for j, N in enumerate(N_values):
            print(f"\n{'='*60}")
            print(f"Testing M={M}, N={N}")
            print(f"{'='*60}")
            
            # Create model
            model = RoboticWorldModel(
                obs_dim=obs_dim,
                act_dim=act_dim,
                priv_dim=priv_dim,
                history_horizon=M,
                forecast_horizon=N,
            )
            model.to(device)
            
            # Create trainer
            trainer = WorldModelTrainer(
                model=model,
                device=device,
                learning_rate=args.learning_rate,
                weight_decay=1e-5,
            )
            
            # Create dataloader
            train_loader = create_dataloader(
                train_traj,
                history_horizon=M,
                forecast_horizon=N,
                batch_size=args.batch_size,
                shuffle=True,
            )
            
            # Measure training time
            start_time = time.time()
            
            for iteration in range(args.max_iterations):
                batch = next(iter(train_loader))
                trainer.train_step(batch, mode='autoregressive')
            
            training_time = time.time() - start_time
            time_grid[i, j] = training_time
            
            # Evaluate prediction error
            val_dataset = TrajectoryDataset(val_traj, M, N)
            eval_results = trainer.evaluate_prediction_error(
                val_dataset, num_steps=N, batch_size=args.batch_size
            )
            error_grid[i, j] = eval_results['mean_error']
            
            print(f"M={M}, N={N}: error={error_grid[i,j]:.6f}, time={training_time:.1f}s")
    
    # Save results
    results = {
        'M_values': M_values,
        'N_values': N_values,
        'error_grid': error_grid,
        'time_grid': time_grid,
    }
    
    np.savez(output_dir / 'ablation_results.npz', **results)
    print(f"\nResults saved to {output_dir}/ablation_results.npz")
    
    # Print summary table
    print("\nRelative Prediction Error Grid:")
    print("M\\N", end="\t")
    for n_val in N_values:
        print(f"N={n_val}", end="\t")
    print()
    for i, M in enumerate(M_values):
        print(f"M={M}", end="\t")
        for j, N in enumerate(N_values):
            print(f"{error_grid[i,j]:.4f}", end="\t")
        print()
    
    print("\nTraining Time Grid (seconds):")
    print("M\\N", end="\t")
    for n_val in N_values:
        print(f"N={n_val}", end="\t")
    print()
    for i, M in enumerate(M_values):
        print(f"M={M}", end="\t")
        for j, N in enumerate(N_values):
            print(f"{time_grid[i,j]:.1f}", end="\t")
        print()
    
    return results


if __name__ == '__main__':
    args = parse_args()
    run_ablation(args)
