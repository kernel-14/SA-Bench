#!/usr/bin/env python3
"""
Run the low-data regime experiment from Section 5 of the LUNO paper.

Trains an FNO on 25 trajectories of Burgers' equation and evaluates
uncertainty quantification methods.

From Table 1 in the paper:
  Method          RMSE        chi2    NLL
  Input Perturb.  3.63e-2    0.894  -1.8720
  Ensemble        3.49e-2    5.597  -0.8145
  Sample-Iso      3.72e-2    0.977  -1.9341
  LUNO-Iso        3.62e-2    0.864  -1.9488
  Sample-LA       5.59e-2    2.774  -1.1572
  LUNO-LA         3.62e-2    1.022  -2.0787  (best NLL)
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.pde_data import generate_burgers_data
from data.dataset import create_dataloaders
from models.fno import FNO1d
from experiments.train_fno import create_fno_1d, train_fno, train_ensemble
from experiments.evaluate_uq import evaluate_all_methods, print_results_table
import flax.nnx as nnx


def main():
    print("=" * 60)
    print("LUNO: Low-Data Regime Experiment (Burgers' Equation)")
    print("=" * 60)

    # Configuration from Appendix D.1.1 and D.2
    config = {
        'n_train': 25,
        'n_val': 250,
        'n_test': 250,
        'spatial_res': 256,
        'temporal_res': 59,
        'n_time_in': 10,
        'n_epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'hidden_channels': 18,
        'n_modes': 12,
        'n_layers': 4,
        'n_ensemble': 10,
        'ggn_rank': 500,
        'n_samples': 200,
        'seed': 42,
    }

    print("\nGenerating Burgers' equation data...")
    data = generate_burgers_data(
        n_train=config['n_train'],
        n_val=config['n_val'],
        n_test=config['n_test'],
        spatial_res=config['spatial_res'],
        temporal_res=config['temporal_res'],
        seed=config['seed'],
    )

    train_dataset, val_dataset, test_dataset = create_dataloaders(
        train_data=data['train'],
        val_data=data['val'],
        test_data=data['test'],
        normalize=True,
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    # Input channels: 10 time steps
    in_channels = config['n_time_in']
    out_channels = 1

    print("\nTraining FNO...")
    model = create_fno_1d(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=config['hidden_channels'],
        n_modes=config['n_modes'],
        n_layers=config['n_layers'],
        seed=config['seed'],
    )

    history = train_fno(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_epochs=config['n_epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        seed=config['seed'],
        verbose=True,
    )

    print("\nTraining ensemble...")
    ensemble = train_ensemble(
        in_channels=in_channels,
        out_channels=out_channels,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_members=config['n_ensemble'],
        n_epochs=config['n_epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        is_2d=False,
        verbose=False,
    )

    print("\nEvaluating UQ methods...")
    results = evaluate_all_methods(
        model=model,
        ensemble=ensemble,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        ggn_rank=config['ggn_rank'],
        n_samples=config['n_samples'],
        seed=config['seed'],
    )

    print("\nResults (Table 1 from paper):")
    print_results_table(results)

    return results


if __name__ == '__main__':
    main()
