#!/usr/bin/env python3
"""
Noise robustness evaluation (Section 4.2).

Tests RWM and MLP baseline under Gaussian noise perturbations 
applied to both observations and actions across different noise levels.

Usage:
    python scripts/evaluate_noise.py --data_path /path/to/data --checkpoint_rwm /path/to/rwm.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rwm.world_model import RoboticWorldModel
from rwm.baselines import MLPWorldModel, create_mlp_baseline
from rwm.training import TrajectoryDataset, WorldModelTrainer
from rwm.evaluation import evaluate_noise_robustness, autoregressive_rollout, compute_relative_error


def parse_args():
    parser = argparse.ArgumentParser(description='Noise robustness evaluation')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to trajectory data')
    parser.add_argument('--checkpoint_rwm', type=str, default=None,
                       help='Path to RWM checkpoint (if None, trains from scratch)')
    parser.add_argument('--output_dir', type=str, default='./outputs/noise_eval',
                       help='Output directory')
    parser.add_argument('--noise_levels', type=float, nargs='+',
                       default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
                       help='Noise standard deviations to test')
    parser.add_argument('--history_horizon', type=int, default=32)
    parser.add_argument('--forecast_horizon', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
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


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    print(f"Dims - obs: {obs_dim}, act: {act_dim}, priv: {priv_dim}")
    
    M = args.history_horizon
    N = args.forecast_horizon
    
    # Create or load RWM
    rwm = RoboticWorldModel(
        obs_dim=obs_dim, act_dim=act_dim, priv_dim=priv_dim,
        history_horizon=M, forecast_horizon=N,
    )
    
    if args.checkpoint_rwm and Path(args.checkpoint_rwm).exists():
        print(f"Loading RWM from {args.checkpoint_rwm}")
        checkpoint = torch.load(args.checkpoint_rwm, map_location=device)
        rwm.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Training RWM from scratch...")
        from rwm.training import WorldModelTrainer, create_dataloader
        
        rwm.to(device)
        trainer = WorldModelTrainer(rwm, device=device)
        train_loader = create_dataloader(train_traj, M, N, batch_size=256)
        
        for i in range(500):
            batch = next(iter(train_loader))
            trainer.train_step(batch, mode='autoregressive')
            if i % 100 == 0:
                print(f"  RWM pre-training iteration {i}")
    
    rwm.to(device)
    rwm.eval()
    
    # Create MLP baseline
    mlp = create_mlp_baseline(obs_dim, act_dim)
    
    print("Training MLP baseline...")
    mlp.to(device)
    mlp_trainer = WorldModelTrainer(mlp, device=device)
    train_loader = create_dataloader(train_traj, M, N, batch_size=256)
    
    for i in range(500):
        batch = next(iter(train_loader))
        mlp_trainer.train_step(batch, mode='autoregressive')
        if i % 100 == 0:
            print(f"  MLP pre-training iteration {i}")
    
    mlp.eval()
    
    # Prepare test batch
    val_dataset = TrajectoryDataset(val_traj, M, N)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    test_batch = next(iter(val_loader))
    obs = test_batch['obs'].to(device)
    act = test_batch['act'].to(device)
    
    obs_history = obs[:, :M, :]
    act_history = act[:, :M, :]
    act_future = act[:, M:M+N, :]
    
    # Evaluate noise robustness for both models
    print(f"\nEvaluating RWM noise robustness...")
    rwm_noise_results = evaluate_noise_robustness(
        rwm, obs_history, act_history, act_future,
        noise_levels=args.noise_levels,
        noise_seed=args.seed,
    )
    
    print(f"Evaluating MLP noise robustness...")
    mlp_noise_results = evaluate_noise_robustness(
        mlp, obs_history, act_history, act_future,
        noise_levels=args.noise_levels,
        noise_seed=args.seed,
    )
    
    # Print results
    print("\n" + "="*60)
    print("NOISE ROBUSTNESS RESULTS")
    print("="*60)
    print(f"{'Noise':>10s} | {'RWM Error':>12s} | {'MLP Error':>12s} | {'Ratio':>8s}")
    print("-"*60)
    for noise_level in args.noise_levels:
        rwm_err = rwm_noise_results[noise_level].mean()
        mlp_err = mlp_noise_results[noise_level].mean()
        ratio = mlp_err / (rwm_err + 1e-8)
        print(f"{noise_level:10.3f} | {rwm_err:12.6f} | {mlp_err:12.6f} | {ratio:8.2f}x")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_dir / 'noise_results.npz',
        noise_levels=args.noise_levels,
        rwm_results={k: v for k, v in rwm_noise_results.items()},
        mlp_results={k: v for k, v in mlp_noise_results.items()},
    )
    
    print(f"\nResults saved to {output_dir}/noise_results.npz")


if __name__ == '__main__':
    main()
