"""Experiment runner for multiphysics neural operator pretraining.

Implements the three experiment scenarios described in Section 4:

1. Out-of-sample parameter values: pretrain and fine-tune on the same equation
   type but with different coefficient values (Burgers', Gray-Scott, Navier-Stokes).

2. Input function set extension: extend equations with additional terms
   (heat + convection, reaction-diffusion + advection).

3. General multi-physics learning: transfer knowledge from advection and
   Burgers' equation to reaction-diffusion.

Usage:
    python experiments.py --scenario 1 --model mamba_fno
    python experiments.py --scenario 2 --model perceiver
    python experiments.py --scenario 3 --model codano
"""
import argparse
import torch
import numpy as np
import time
import json
import os
from typing import Dict

from models import (
    FNO, MambaFNO, PerceiverFNO, CoDANO, SwinTransformerNO
)
from models.adapters import LiftAdapter, ProjAdapter, LocalAttnFNO
from data.pde_generators import (
    BurgersDataset, GrayScottDataset, NavierStokesDataset,
    HeatEquationDataset, AdvectionDataset, ReactionDiffusionDataset,
    ReactionDiffusionAdvectionDataset,
)
from data.pde_dataset import create_dataloader
from training.metrics import compute_metrics
from training.pretrain import MultiPhysicsPretrainer
from training.finetune import FineTuner, train_from_scratch
from utils.helpers import count_parameters, set_seed


MODEL_REGISTRY = {
    'fno': FNO,
    'mamba_fno': MambaFNO,
    'perceiver': PerceiverFNO,
    'codano': CoDANO,
    'swin': SwinTransformerNO,
    'local_attn_fno': LocalAttnFNO,
}


def get_model_factory(model_name: str, hidden_channels: int = 32):
    """Create a factory function for a given model type."""
    ModelClass = MODEL_REGISTRY[model_name]

    def factory(input_channels, output_channels, **kwargs):
        return ModelClass(
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=hidden_channels,
            **kwargs,
        )

    return factory


def run_scenario_1(args):
    """Out-of-sample parameter values scenario.

    Pretrain on one set of parameter ranges, fine-tune on different ranges.
    Uses Burgers', Gray-Scott, and Navier-Stokes equations.
    """
    print("=" * 60)
    print("Scenario 1: Out-of-sample parameter values")
    print(f"Model: {args.model}")
    print("=" * 60)

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Model factory
    model_factory = get_model_factory(args.model, hidden_channels=args.hidden_channels)

    # Create a shared core model for pretraining
    core_model = model_factory(input_channels=2, output_channels=1)

    pretrainer = MultiPhysicsPretrainer(
        model_factory=model_factory,
        core_model=core_model,
        device=device,
        learning_rate=args.lr,
    )

    # Add three physics problems for pretraining
    print("\n--- Setting up pretraining datasets ---")

    # Burgers: pretrain with nu in [0.005, 0.01]
    burgers_train = BurgersDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                    split='train', seed=args.seed)
    burgers_val = BurgersDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                  split='val', seed=args.seed)
    pretrainer.add_physics(
        'burgers', input_channels=2, output_channels=1,
        train_loader=create_dataloader(burgers_train, args.batch_size),
        val_loader=create_dataloader(burgers_val, args.batch_size, shuffle=False),
    )

    # Gray-Scott: pretrain with standard parameters
    gs_train = GrayScottDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                 split='train', seed=args.seed)
    gs_val = GrayScottDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                               split='val', seed=args.seed)
    pretrainer.add_physics(
        'grayscott', input_channels=4, output_channels=2,
        train_loader=create_dataloader(gs_train, args.batch_size),
        val_loader=create_dataloader(gs_val, args.batch_size, shuffle=False),
    )

    # Navier-Stokes: pretrain with nu in [0.005, 0.01]
    ns_train = NavierStokesDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                    split='train', seed=args.seed)
    ns_val = NavierStokesDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                  split='val', seed=args.seed)
    pretrainer.add_physics(
        'navierstokes', input_channels=2, output_channels=1,
        train_loader=create_dataloader(ns_train, args.batch_size),
        val_loader=create_dataloader(ns_val, args.batch_size, shuffle=False),
    )

    # Pretraining
    print(f"\n--- Pretraining for {args.pretrain_epochs} epochs ---")
    for epoch in range(args.pretrain_epochs):
        metrics = pretrainer.pretrain_epoch()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.pretrain_epochs}, Loss: {metrics['pretrain_loss']:.6f}")

    # Evaluate pretrained models
    print("\n--- Pretraining evaluation ---")
    pretrain_results = pretrainer.evaluate()
    for name, metrics in pretrain_results.items():
        print(f"  {name}: MSE={metrics['mse']:.6e}, NMAE={metrics['nmae_pct']:.4f}%")

    # Fine-tuning: test on out-of-sample parameters
    print("\n--- Fine-tuning on out-of-sample Burgers (nu in [0.001, 0.005]) ---")

    # Create fine-tuning model (same architecture, new adapters)
    ft_model = model_factory(input_channels=2, output_channels=1)

    # Copy pretrained core weights
    pretrained_burgers = pretrainer.models['burgers']
    if hasattr(ft_model, 'fno_blocks') and hasattr(pretrained_burgers, 'fno_blocks'):
        ft_model.fno_blocks.load_state_dict(pretrained_burgers.fno_blocks.state_dict())
    if hasattr(ft_model, 'mamba') and hasattr(pretrained_burgers, 'mamba'):
        ft_model.mamba.load_state_dict(pretrained_burgers.mamba.state_dict())

    ft_train = BurgersDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                               split='train', seed=args.seed + 1)
    ft_val = BurgersDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                             split='val', seed=args.seed + 1)

    finetuner = FineTuner(ft_model, device=device, learning_rate=args.lr)
    ft_loader = create_dataloader(ft_train, args.batch_size)
    ft_val_loader = create_dataloader(ft_val, args.batch_size, shuffle=False)

    ft_times = []
    for epoch in range(args.finetune_epochs):
        t0 = time.time()
        metrics = finetuner.train_epoch(ft_loader)
        epoch_time = time.time() - t0
        ft_times.append(epoch_time)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.finetune_epochs}, Loss: {metrics['finetune_loss']:.6f}, Time: {epoch_time:.2f}s")

    ft_results = finetuner.evaluate(ft_val_loader)
    ft_results['avg_epoch_time'] = np.mean(ft_times)
    ft_results['n_params'] = count_parameters(ft_model)
    print(f"\n  Fine-tuned results: MSE={ft_results['mse']:.6e}, NMAE={ft_results['nmae_pct']:.4f}%")
    print(f"  Avg epoch time: {ft_results['avg_epoch_time']:.2f}s")

    # Training from scratch for comparison
    print("\n--- Training from scratch for comparison ---")
    scratch_model = model_factory(input_channels=2, output_channels=1)
    scratch_train_loader = create_dataloader(ft_train, args.batch_size)
    scratch_val_loader = create_dataloader(ft_val, args.batch_size, shuffle=False)

    scratch_results = train_from_scratch(
        scratch_model, scratch_train_loader, scratch_val_loader,
        epochs=args.finetune_epochs, lr=args.lr, device=device,
    )
    scratch_results['n_params'] = count_parameters(scratch_model)
    print(f"  From scratch: MSE={scratch_results['mse']:.6e}, NMAE={scratch_results['nmae_pct']:.4f}%")
    print(f"  Avg epoch time: {scratch_results['avg_epoch_time']:.2f}s")

    # Save results
    results = {
        'scenario': 1,
        'model': args.model,
        'pretrained': ft_results,
        'from_scratch': scratch_results,
        'pretrain_eval': pretrain_results,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f'scenario1_{args.model}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_scenario_2(args):
    """Input function set extension scenario.

    Pretrain on simpler equations, fine-tune on extended equations with
    additional terms (heat + convection, reaction-diffusion + advection).
    """
    print("=" * 60)
    print("Scenario 2: Input function set extension")
    print(f"Model: {args.model}")
    print("=" * 60)

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_factory = get_model_factory(args.model, hidden_channels=args.hidden_channels)
    core_model = model_factory(input_channels=2, output_channels=1)

    pretrainer = MultiPhysicsPretrainer(
        model_factory=model_factory,
        core_model=core_model,
        device=device,
        learning_rate=args.lr,
    )

    # Pretrain on heat equation (2 input channels: u0, alpha)
    heat_train = HeatEquationDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                      split='train', seed=args.seed, with_convection=False)
    heat_val = HeatEquationDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                    split='val', seed=args.seed, with_convection=False)
    pretrainer.add_physics(
        'heat', input_channels=2, output_channels=1,
        train_loader=create_dataloader(heat_train, args.batch_size),
        val_loader=create_dataloader(heat_val, args.batch_size, shuffle=False),
    )

    # Pretrain on reaction-diffusion (4 input channels: u0, v0, f, k)
    rd_train = ReactionDiffusionDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                         split='train', seed=args.seed)
    rd_val = ReactionDiffusionDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                       split='val', seed=args.seed)
    pretrainer.add_physics(
        'reaction_diffusion', input_channels=4, output_channels=2,
        train_loader=create_dataloader(rd_train, args.batch_size),
        val_loader=create_dataloader(rd_val, args.batch_size, shuffle=False),
    )

    # Pretraining
    print(f"\n--- Pretraining for {args.pretrain_epochs} epochs ---")
    for epoch in range(args.pretrain_epochs):
        metrics = pretrainer.pretrain_epoch()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.pretrain_epochs}, Loss: {metrics['pretrain_loss']:.6f}")

    # Fine-tune on heat + convection (extended: 4 input channels)
    print("\n--- Fine-tuning on Heat + Convection (extended equation) ---")
    ft_model = model_factory(input_channels=4, output_channels=1)

    pretrained_heat = pretrainer.models['heat']
    if hasattr(ft_model, 'fno_blocks') and hasattr(pretrained_heat, 'fno_blocks'):
        ft_model.fno_blocks.load_state_dict(pretrained_heat.fno_blocks.state_dict())

    ft_train = HeatEquationDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                    split='train', seed=args.seed + 1, with_convection=True)
    ft_val = HeatEquationDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                  split='val', seed=args.seed + 1, with_convection=True)

    finetuner = FineTuner(ft_model, device=device, learning_rate=args.lr)
    ft_loader = create_dataloader(ft_train, args.batch_size)
    ft_val_loader = create_dataloader(ft_val, args.batch_size, shuffle=False)

    ft_times = []
    for epoch in range(args.finetune_epochs):
        t0 = time.time()
        metrics = finetuner.train_epoch(ft_loader)
        ft_times.append(time.time() - t0)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.finetune_epochs}, Loss: {metrics['finetune_loss']:.6f}")

    ft_results_heat = finetuner.evaluate(ft_val_loader)
    ft_results_heat['avg_epoch_time'] = np.mean(ft_times)
    print(f"  Heat+Conv (pretrained): MSE={ft_results_heat['mse']:.6e}, NMAE={ft_results_heat['nmae_pct']:.4f}%")

    # Fine-tune on reaction-diffusion + advection (extended: 6 input channels)
    print("\n--- Fine-tuning on Reaction-Diffusion + Advection (extended equation) ---")
    ft_model2 = model_factory(input_channels=6, output_channels=2)

    pretrained_rd = pretrainer.models['reaction_diffusion']
    if hasattr(ft_model2, 'fno_blocks') and hasattr(pretrained_rd, 'fno_blocks'):
        ft_model2.fno_blocks.load_state_dict(pretrained_rd.fno_blocks.state_dict())

    rda_train = ReactionDiffusionAdvectionDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                                    split='train', seed=args.seed)
    rda_val = ReactionDiffusionAdvectionDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                                  split='val', seed=args.seed)

    finetuner2 = FineTuner(ft_model2, device=device, learning_rate=args.lr)
    rda_loader = create_dataloader(rda_train, args.batch_size)
    rda_val_loader = create_dataloader(rda_val, args.batch_size, shuffle=False)

    ft_times2 = []
    for epoch in range(args.finetune_epochs):
        t0 = time.time()
        metrics = finetuner2.train_epoch(rda_loader)
        ft_times2.append(time.time() - t0)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.finetune_epochs}, Loss: {metrics['finetune_loss']:.6f}")

    ft_results_rd = finetuner2.evaluate(rda_val_loader)
    ft_results_rd['avg_epoch_time'] = np.mean(ft_times2)
    print(f"  RD+Adv (pretrained): MSE={ft_results_rd['mse']:.6e}, NMAE={ft_results_rd['nmae_pct']:.4f}%")

    # From scratch baselines
    print("\n--- Training from scratch baselines ---")
    scratch_heat = model_factory(input_channels=4, output_channels=1)
    scratch_results_heat = train_from_scratch(
        scratch_heat, ft_loader, ft_val_loader,
        epochs=args.finetune_epochs, lr=args.lr, device=device,
    )
    print(f"  Heat+Conv (scratch): MSE={scratch_results_heat['mse']:.6e}, NMAE={scratch_results_heat['nmae_pct']:.4f}%")

    scratch_rd = model_factory(input_channels=6, output_channels=2)
    scratch_results_rd = train_from_scratch(
        scratch_rd, rda_loader, rda_val_loader,
        epochs=args.finetune_epochs, lr=args.lr, device=device,
    )
    print(f"  RD+Adv (scratch): MSE={scratch_results_rd['mse']:.6e}, NMAE={scratch_results_rd['nmae_pct']:.4f}%")

    results = {
        'scenario': 2,
        'model': args.model,
        'heat_convection': {'pretrained': ft_results_heat, 'from_scratch': scratch_results_heat},
        'rd_advection': {'pretrained': ft_results_rd, 'from_scratch': scratch_results_rd},
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f'scenario2_{args.model}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_scenario_3(args):
    """General multi-physics learning scenario.

    Pretrain on advection and Burgers', fine-tune on reaction-diffusion.
    This tests cross-domain transfer between fundamentally different PDE types.
    """
    print("=" * 60)
    print("Scenario 3: Cross-domain multi-physics transfer")
    print(f"Model: {args.model}")
    print("=" * 60)

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_factory = get_model_factory(args.model, hidden_channels=args.hidden_channels)
    core_model = model_factory(input_channels=2, output_channels=1)

    pretrainer = MultiPhysicsPretrainer(
        model_factory=model_factory,
        core_model=core_model,
        device=device,
        learning_rate=args.lr,
    )

    # Pretrain on advection (3 input channels: u0, vx, vy)
    adv_train = AdvectionDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                  split='train', seed=args.seed)
    adv_val = AdvectionDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                split='val', seed=args.seed)
    pretrainer.add_physics(
        'advection', input_channels=3, output_channels=1,
        train_loader=create_dataloader(adv_train, args.batch_size),
        val_loader=create_dataloader(adv_val, args.batch_size, shuffle=False),
    )

    # Pretrain on Burgers (2 input channels: u0, nu)
    burgers_train = BurgersDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                    split='train', seed=args.seed)
    burgers_val = BurgersDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                  split='val', seed=args.seed)
    pretrainer.add_physics(
        'burgers', input_channels=2, output_channels=1,
        train_loader=create_dataloader(burgers_train, args.batch_size),
        val_loader=create_dataloader(burgers_val, args.batch_size, shuffle=False),
    )

    # Pretraining
    print(f"\n--- Pretraining for {args.pretrain_epochs} epochs ---")
    for epoch in range(args.pretrain_epochs):
        metrics = pretrainer.pretrain_epoch()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.pretrain_epochs}, Loss: {metrics['pretrain_loss']:.6f}")

    # Fine-tune on reaction-diffusion (4 input channels: u0, v0, f, k)
    print("\n--- Fine-tuning on Reaction-Diffusion (cross-domain transfer) ---")
    ft_model = model_factory(input_channels=4, output_channels=2)

    # Copy pretrained core from any model
    pretrained_adv = pretrainer.models['advection']
    if hasattr(ft_model, 'fno_blocks') and hasattr(pretrained_adv, 'fno_blocks'):
        ft_model.fno_blocks.load_state_dict(pretrained_adv.fno_blocks.state_dict())
    if hasattr(ft_model, 'mamba') and hasattr(pretrained_adv, 'mamba'):
        ft_model.mamba.load_state_dict(pretrained_adv.mamba.state_dict())
    if hasattr(ft_model, 'coda_blocks') and hasattr(pretrained_adv, 'coda_blocks'):
        ft_model.coda_blocks.load_state_dict(pretrained_adv.coda_blocks.state_dict())
    if hasattr(ft_model, 'perceiver_blocks') and hasattr(pretrained_adv, 'perceiver_blocks'):
        ft_model.perceiver_blocks.load_state_dict(pretrained_adv.perceiver_blocks.state_dict())

    rd_train = ReactionDiffusionDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                         split='train', seed=args.seed + 1)
    rd_val = ReactionDiffusionDataset(n_samples=args.n_samples, nx=args.nx, ny=args.ny,
                                       split='val', seed=args.seed + 1)

    finetuner = FineTuner(ft_model, device=device, learning_rate=args.lr)
    ft_loader = create_dataloader(rd_train, args.batch_size)
    ft_val_loader = create_dataloader(rd_val, args.batch_size, shuffle=False)

    ft_times = []
    for epoch in range(args.finetune_epochs):
        t0 = time.time()
        metrics = finetuner.train_epoch(ft_loader)
        ft_times.append(time.time() - t0)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.finetune_epochs}, Loss: {metrics['finetune_loss']:.6f}")

    ft_results = finetuner.evaluate(ft_val_loader)
    ft_results['avg_epoch_time'] = np.mean(ft_times)
    ft_results['n_params'] = count_parameters(ft_model)
    print(f"  Fine-tuned: MSE={ft_results['mse']:.6e}, NMAE={ft_results['nmae_pct']:.4f}%")
    print(f"  Avg epoch time: {ft_results['avg_epoch_time']:.2f}s")

    # From scratch
    print("\n--- Training from scratch baseline ---")
    scratch_model = model_factory(input_channels=4, output_channels=2)
    scratch_results = train_from_scratch(
        scratch_model, ft_loader, ft_val_loader,
        epochs=args.finetune_epochs, lr=args.lr, device=device,
    )
    scratch_results['n_params'] = count_parameters(scratch_model)
    print(f"  From scratch: MSE={scratch_results['mse']:.6e}, NMAE={scratch_results['nmae_pct']:.4f}%")
    print(f"  Avg epoch time: {scratch_results['avg_epoch_time']:.2f}s")

    results = {
        'scenario': 3,
        'model': args.model,
        'pretrained': ft_results,
        'from_scratch': scratch_results,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f'scenario3_{args.model}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description='Multiphysics Neural Operator Experiments')

    parser.add_argument('--scenario', type=int, default=1, choices=[1, 2, 3],
                        help='Experiment scenario (1, 2, or 3)')
    parser.add_argument('--model', type=str, default='mamba_fno',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture')
    parser.add_argument('--hidden_channels', type=int, default=32,
                        help='Hidden channels')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples per dataset')
    parser.add_argument('--nx', type=int, default=64,
                        help='Spatial grid size (x)')
    parser.add_argument('--ny', type=int, default=64,
                        help='Spatial grid size (y)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--pretrain_epochs', type=int, default=50,
                        help='Number of pretraining epochs')
    parser.add_argument('--finetune_epochs', type=int, default=100,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')

    args = parser.parse_args()

    if args.scenario == 1:
        run_scenario_1(args)
    elif args.scenario == 2:
        run_scenario_2(args)
    elif args.scenario == 3:
        run_scenario_3(args)


if __name__ == '__main__':
    main()
