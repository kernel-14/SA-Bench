#!/usr/bin/env python3
"""
Training script for Robotic World Model (RWM).

Implements the autoregressive training procedure described in Section 3.2:
- Sliding window data construction (M + N steps)
- Dual-autoregressive training
- Multi-step prediction loss with forecast decay

Usage:
    python scripts/train_rwm.py --robot anymal_d --data_path /path/to/data
"""

import os
import sys
import argparse
import numpy as np
import torch
import pickle
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rwm.world_model import RoboticWorldModel, create_rwm_anymal_d, create_rwm_unitree_g1
from rwm.training import TrajectoryDataset, WorldModelTrainer, create_dataloader
from rwm.baselines import (
    MLPWorldModel, RSSM, TransformerWorldModel,
    create_mlp_baseline, create_rssm_baseline, create_transformer_baseline
)
from rwm.evaluation import compare_models, evaluate_noise_robustness


def parse_args():
    parser = argparse.ArgumentParser(description='Train RWM world model')
    parser.add_argument('--robot', type=str, default='anymal_d',
                       choices=['anymal_d', 'unitree_g1'],
                       help='Robot type')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to trajectory data (.pkl or .npy)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--history_horizon', type=int, default=32,
                       help='History horizon M')
    parser.add_argument('--forecast_horizon', type=int, default=8,
                       help='Forecast horizon N')
    parser.add_argument('--forecast_decay', type=float, default=1.0,
                       help='Forecast decay factor alpha')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--max_iterations', type=int, default=2500,
                       help='Maximum training iterations')
    parser.add_argument('--mode', type=str, default='autoregressive',
                       choices=['autoregressive', 'teacher_forcing'],
                       help='Training mode')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    return parser.parse_args()


def load_data(data_path: str):
    """Load trajectory data."""
    data_path = Path(data_path)
    
    if data_path.suffix == '.pkl':
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    elif data_path.suffix == '.npy':
        data = np.load(data_path, allow_pickle=True).item()
    elif data_path.suffix == '.npz':
        data = dict(np.load(data_path, allow_pickle=True))
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")
    
    # Expect data to be either a list of trajectories or a dict with 'trajectories' key
    if isinstance(data, list):
        trajectories = data
    elif isinstance(data, dict):
        if 'trajectories' in data:
            trajectories = data['trajectories']
        else:
            trajectories = [data]
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")
    
    return trajectories


def split_trajectories(trajectories, train_ratio=0.8):
    """Split trajectories into train and validation sets."""
    n = len(trajectories)
    n_train = int(n * train_ratio)
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    return [trajectories[i] for i in train_idx], [trajectories[i] for i in val_idx]


def create_model(robot_type, history_horizon, forecast_horizon, forecast_decay):
    """Create the appropriate RWM model for the robot."""
    if robot_type == 'anymal_d':
        return create_rwm_anymal_d()
    elif robot_type == 'unitree_g1':
        return create_rwm_unitree_g1()
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")


def train(args):
    """Main training function."""
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}")
    trajectories = load_data(args.data_path)
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Split data
    train_traj, val_traj = split_trajectories(trajectories)
    print(f"Train trajectories: {len(train_traj)}, Val trajectories: {len(val_traj)}")
    
    # Determine observation and action dimensions from data
    sample = train_traj[0]
    obs_dim = sample['obs'].shape[-1]
    act_dim = sample['act'].shape[-1]
    priv_dim = sample.get('priv', np.zeros((1, 0))).shape[-1]
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}, Privileged dim: {priv_dim}")
    
    # Create model
    model = RoboticWorldModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        priv_dim=priv_dim,
        gru_hidden_dims=(256, 256),
        head_hidden_dim=128,
        history_horizon=args.history_horizon,
        forecast_horizon=args.forecast_horizon,
        forecast_decay=args.forecast_decay,
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = WorldModelTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        forecast_decay=args.forecast_decay,
    )
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_traj,
        history_horizon=args.history_horizon,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_traj,
        history_horizon=args.history_horizon,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    print(f"Training with {len(train_loader)} batches per epoch")
    print(f"Mode: {args.mode}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for iteration in range(args.max_iterations):
        # Train on one batch
        batch = next(iter(train_loader))
        loss_dict = trainer.train_step(batch, mode=args.mode)
        train_losses.append(loss_dict['total_loss'])
        
        # Validation
        if iteration % 100 == 0 or iteration == args.max_iterations - 1:
            val_batch = next(iter(val_loader))
            with torch.no_grad():
                val_loss_dict = trainer.compute_loss(val_batch, mode=args.mode)
            val_loss = val_loss_dict['total_loss'].item()
            val_losses.append(val_loss)
            
            print(f"Iteration {iteration}: train_loss={loss_dict['total_loss']:.6f}, "
                  f"val_loss={val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'train_loss': loss_dict['total_loss'],
                    'val_loss': val_loss,
                    'args': vars(args),
                }, output_dir / 'best_model.pt')
    
    # Final save
    torch.save({
        'iteration': args.max_iterations,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'args': vars(args),
    }, output_dir / 'final_model.pt')
    
    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to {output_dir}")
    
    return model, trainer


def evaluate(args):
    """Evaluate trained model(s)."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print(f"Loading data from {args.data_path}")
    trajectories = load_data(args.data_path)
    _, val_traj = split_trajectories(trajectories)
    
    sample = val_traj[0]
    obs_dim = sample['obs'].shape[-1]
    act_dim = sample['act'].shape[-1]
    priv_dim = sample.get('priv', np.zeros((1, 0))).shape[-1]
    
    # Load RWM model
    rwm = RoboticWorldModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        priv_dim=priv_dim,
        history_horizon=args.history_horizon,
        forecast_horizon=args.forecast_horizon,
        forecast_decay=args.forecast_decay,
    )
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        rwm.load_state_dict(checkpoint['model_state_dict'])
    
    # Create baseline models for comparison
    models = {
        'RWM-AR': rwm,
        'RWM-TF': RoboticWorldModel(obs_dim, act_dim, priv_dim, 
                                     history_horizon=args.history_horizon,
                                     forecast_horizon=args.forecast_horizon),
        'MLP': create_mlp_baseline(obs_dim, act_dim),
        'RSSM': create_rssm_baseline(obs_dim, act_dim),
        'Transformer': create_transformer_baseline(obs_dim, act_dim),
    }
    
    print("Comparing models...")
    results = compare_models(
        models, val_traj,
        history_horizon=args.history_horizon,
        forecast_horizon=args.forecast_horizon,
        device=device,
    )
    
    print("\nResults (relative prediction error):")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:20s}: mean_error={metrics['mean_error']:.6f}, "
              f"final_step={metrics['final_step_error']:.6f}")
    
    return results


if __name__ == '__main__':
    args = parse_args()
    
    if args.eval_only:
        evaluate(args)
    else:
        train(args)
