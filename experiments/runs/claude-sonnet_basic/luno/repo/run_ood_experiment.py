#!/usr/bin/env python3
"""
Run the out-of-distribution (OOD) experiment from Section 5 of the LUNO paper.

Trains an FNO on 1000 trajectories of the Base Advection-Diffusion-Reaction equation
and evaluates uncertainty quantification methods on OOD datasets.

From Table 2 in the paper:
  Method          Base    Flip    Pos-Neg-Flip
  Input Perturb.  -2.586  2.573   494.935
  Ensemble        -5.313  3.825   -1.014
  Sample-Iso      -2.921  4.071   43.362
  LUNO-Iso        -2.892  3.450   37.733
  Sample-LA       -2.576  4.395   27.046
  LUNO-LA         -2.934  -1.126  1.164  (best on OOD)
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.pde_data import generate_advection_diffusion_data
from data.dataset import create_dataloaders, PDEDataset
from experiments.train_fno import create_fno_2d, train_fno, train_ensemble
from experiments.evaluate_uq import evaluate_all_methods, print_results_table
import flax.nnx as nnx


def main():
    print("=" * 60)
    print("LUNO: OOD Experiment (Advection-Diffusion-Reaction)")
    print("=" * 60)

    # Configuration from Appendix D.1.2 and D.2
    config = {
        'n_train': 1000,
        'n_val': 250,
        'n_test': 250,
        'spatial_res': 100,
        'temporal_res': 59,
        'n_time_in': 10,
        'n_epochs': 1000,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'hidden_channels': 18,
        'n_modes': 12,
        'n_layers': 4,
        'n_ensemble': 10,
        'ggn_rank': 500,
        'n_samples': 200,
        'seed': 42,
        'alpha': 0.026,  # Diffusion coefficient
    }

    # Input channels: 10 time steps + vx + vy + R = 13
    in_channels = config['n_time_in'] + 3
    out_channels = 1

    print("\nGenerating Base dataset...")
    base_data = generate_advection_diffusion_data(
        variant='base',
        n_train=config['n_train'],
        n_val=config['n_val'],
        n_test=config['n_test'],
        spatial_res=config['spatial_res'],
        temporal_res=config['temporal_res'],
        alpha=config['alpha'],
        seed=config['seed'],
    )

    train_dataset, val_dataset, test_base_dataset = create_dataloaders(
        train_data=base_data['train'],
        val_data=base_data['val'],
        test_data=base_data['test'],
        normalize=True,
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test (Base): {len(test_base_dataset)} samples")

    print("\nTraining FNO on Base dataset...")
    model = create_fno_2d(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=config['hidden_channels'],
        n_modes_x=config['n_modes'],
        n_modes_y=config['n_modes'],
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
        is_2d=True,
        verbose=False,
    )

    # Evaluate on all OOD variants
    ood_variants = ['base', 'flip', 'pos_neg_flip']
    all_results = {}

    for variant in ood_variants:
        print(f"\nGenerating {variant} OOD dataset...")

        if variant == 'base':
            test_dataset = test_base_dataset
        else:
            ood_data = generate_advection_diffusion_data(
                variant=variant,
                n_val=config['n_val'],
                n_test=config['n_test'],
                spatial_res=config['spatial_res'],
                temporal_res=config['temporal_res'],
                alpha=config['alpha'],
                seed=config['seed'] + 1,
            )
            # Apply training normalization
            test_inputs = (ood_data['test'][0] - train_dataset.input_mean) / train_dataset.input_std
            test_targets = (ood_data['test'][1] - train_dataset.target_mean) / train_dataset.target_std
            test_dataset = PDEDataset(ood_data['test'][0], ood_data['test'][1], normalize=False)
            test_dataset.inputs_normalized = test_inputs
            test_dataset.targets_normalized = test_targets

        print(f"\nEvaluating UQ methods on {variant}...")
        results = evaluate_all_methods(
            model=model,
            ensemble=ensemble,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            ggn_rank=config['ggn_rank'],
            n_samples=config['n_samples'],
            seed=config['seed'],
            is_2d=True,
        )

        all_results[variant] = results
        print(f"\nResults for {variant}:")
        print_results_table(results)

    # Print summary table (NLL only, matching Table 2)
    print("\n" + "=" * 70)
    print("Summary: Expected Marginal NLL (Table 2 from paper)")
    print("=" * 70)
    print(f"{'Method':<25} {'Base':>10} {'Flip':>10} {'Pos-Neg-Flip':>15}")
    print("-" * 70)

    methods = ['Input Perturbations', 'Ensemble', 'Sample-Iso', 'LUNO-Iso', 'Sample-LA', 'LUNO-LA']
    for method in methods:
        row = f"{method:<25}"
        for variant in ood_variants:
            if method in all_results.get(variant, {}):
                nll = all_results[variant][method].get('nll', float('nan'))
                row += f" {nll:>10.3f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    print("=" * 70)

    return all_results


if __name__ == '__main__':
    main()
