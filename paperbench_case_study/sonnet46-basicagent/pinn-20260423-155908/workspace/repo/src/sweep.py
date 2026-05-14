"""
Hyperparameter sweep script for PINN experiments.

Reproduces the systematic hyperparameter search described in the paper:
  - For each PDE, sweep over Adam learning rates, seeds, and network widths
  - Select the configuration with the smallest L2RE for spectral density plots

Usage:
  python sweep.py --pde convection --mode optimizer_comparison
  python sweep.py --pde convection --mode nncg_comparison
"""

import argparse
import os
import sys
import itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import run_experiment


# Hyperparameter grids from the paper
ADAM_LRS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
WIDTHS = [50, 100, 200, 400]
SEEDS = [123, 234, 345, 456, 567]  # 5 random seeds
SWITCH_ITERS = [1000, 11000, 31000]  # Adam+L-BFGS switch points

# Best configurations found by the authors (from addendum)
BEST_CONFIGS = {
    'convection': {'width': 200, 'lr': 1e-4, 'seed': 345},
    'reaction': {'width': 200, 'lr': 1e-3, 'seed': 456},
    'wave': {'width': 200, 'lr': 1e-3, 'seed': 567},
}


def sweep_optimizer_comparison(pde_name, device='cpu', save_dir='results'):
    """
    Sweep over all optimizer configurations for a given PDE.
    Reproduces Figure 8 (Adam vs L-BFGS vs Adam+L-BFGS comparison).
    """
    print(f"\n=== Optimizer comparison sweep for {pde_name} ===")

    # Adam: sweep over LRs, widths, seeds
    for width in WIDTHS:
        for lr in ADAM_LRS:
            for seed in SEEDS:
                run_experiment(
                    pde_name=pde_name,
                    optimizer_name='adam',
                    width=width,
                    lr=lr,
                    seed=seed,
                    device=device,
                    save_dir=save_dir,
                )

    # L-BFGS: sweep over widths, seeds (no LR to tune)
    for width in WIDTHS:
        for seed in SEEDS:
            run_experiment(
                pde_name=pde_name,
                optimizer_name='lbfgs',
                width=width,
                lr=1e-3,  # placeholder, not used
                seed=seed,
                device=device,
                save_dir=save_dir,
            )

    # Adam+L-BFGS: sweep over LRs, widths, seeds, switch points
    for width in WIDTHS:
        for lr in ADAM_LRS:
            for seed in SEEDS:
                for switch_iter in SWITCH_ITERS:
                    run_experiment(
                        pde_name=pde_name,
                        optimizer_name='adam_lbfgs',
                        width=width,
                        lr=lr,
                        seed=seed,
                        switch_iter=switch_iter,
                        device=device,
                        save_dir=save_dir,
                    )


def sweep_nncg_comparison(pde_name, device='cpu', save_dir='results'):
    """
    Run NNCG and GD comparison after Adam+L-BFGS.
    Uses the best configuration (switch at 11k) for each PDE.
    Reproduces Figure 4 and Table 2.
    """
    print(f"\n=== NNCG comparison sweep for {pde_name} ===")

    best = BEST_CONFIGS[pde_name]
    width = best['width']
    lr = best['lr']
    seed = best['seed']

    # Adam+L-BFGS (baseline)
    run_experiment(
        pde_name=pde_name,
        optimizer_name='adam_lbfgs',
        width=width,
        lr=lr,
        seed=seed,
        switch_iter=11000,
        device=device,
        save_dir=save_dir,
    )

    # Adam+L-BFGS+NNCG with different mu values
    for mu in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        run_experiment(
            pde_name=pde_name,
            optimizer_name='adam_lbfgs_nncg',
            width=width,
            lr=lr,
            seed=seed,
            switch_iter=11000,
            nncg_mu=mu,
            device=device,
            save_dir=save_dir,
        )

    # Adam+L-BFGS+GD
    run_experiment(
        pde_name=pde_name,
        optimizer_name='adam_lbfgs_gd',
        width=width,
        lr=lr,
        seed=seed,
        switch_iter=11000,
        device=device,
        save_dir=save_dir,
    )


def find_best_config(pde_name, save_dir='results'):
    """
    Find the best configuration (lowest L2RE) for a given PDE.
    Used to select configurations for spectral density plots (Figures 3, 7).
    """
    best_l2re = float('inf')
    best_config = None

    for fname in os.listdir(save_dir):
        if not fname.startswith(pde_name) or not fname.endswith('.npy'):
            continue
        if 'adam_lbfgs' not in fname or 'nncg' in fname or 'gd' in fname:
            continue
        if 'sw11000' not in fname:
            continue

        results = np.load(os.path.join(save_dir, fname), allow_pickle=True).item()
        if results['final_l2re'] < best_l2re:
            best_l2re = results['final_l2re']
            best_config = results

    return best_config


def main():
    parser = argparse.ArgumentParser(description='PINN hyperparameter sweep')
    parser.add_argument('--pde', type=str, default='convection',
                        choices=['convection', 'reaction', 'wave'])
    parser.add_argument('--mode', type=str, default='optimizer_comparison',
                        choices=['optimizer_comparison', 'nncg_comparison', 'find_best'])
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()

    if args.mode == 'optimizer_comparison':
        sweep_optimizer_comparison(args.pde, args.device, args.save_dir)
    elif args.mode == 'nncg_comparison':
        sweep_nncg_comparison(args.pde, args.device, args.save_dir)
    elif args.mode == 'find_best':
        best = find_best_config(args.pde, args.save_dir)
        if best:
            print(f"Best config for {args.pde}:")
            print(f"  width={best['width']}, lr={best['lr']}, seed={best['seed']}")
            print(f"  L2RE={best['final_l2re']:.3e}, loss={best['final_loss']:.3e}")
        else:
            print(f"No results found for {args.pde}")


if __name__ == '__main__':
    main()
