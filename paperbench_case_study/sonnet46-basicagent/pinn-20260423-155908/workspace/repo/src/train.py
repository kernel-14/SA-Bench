"""
Main training script for PINN experiments.

Supports:
  - Adam
  - L-BFGS
  - Adam+L-BFGS (with switch at 1k, 11k, or 31k iterations)
  - Adam+L-BFGS+NNCG
  - Adam+L-BFGS+GD

Usage:
  python train.py --pde convection --optimizer adam_lbfgs --switch_iter 11000
                  --width 200 --lr 1e-4 --seed 345
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MLP
from src.pdes import convection, reaction, wave
from src.optimizers.nncg import NNCG


PDE_MODULES = {
    'convection': convection,
    'reaction': reaction,
    'wave': wave,
}

TOTAL_ITERS = 41000


def get_pde_module(pde_name):
    return PDE_MODULES[pde_name]


def move_data_to_device(data, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()}


def train_adam(model, data, pde_module, lr, n_iters, device):
    """Train with Adam for n_iters iterations."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    grad_norms = []

    for i in range(n_iters):
        optimizer.zero_grad()
        loss, _, _ = pde_module.pinn_loss(model, data)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        optimizer.step()
        losses.append(loss.item())
        grad_norms.append(grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm))

    return losses, grad_norms


def train_lbfgs(model, data, pde_module, n_iters, device):
    """Train with L-BFGS for n_iters iterations."""
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=n_iters,
        history_size=100,
        line_search_fn='strong_wolfe'
    )
    losses = []
    grad_norms = []

    def closure():
        optimizer.zero_grad()
        loss, _, _ = pde_module.pinn_loss(model, data)
        loss.backward()
        return loss

    # L-BFGS runs all iterations in one step call
    optimizer.step(closure)

    # Record final loss
    with torch.no_grad():
        loss, _, _ = pde_module.pinn_loss(model, data)
        losses.append(loss.item())

    return losses, grad_norms, optimizer


def train_lbfgs_iterative(model, data, pde_module, n_iters, device):
    """
    Train with L-BFGS, recording loss at each iteration.
    Uses max_iter=1 per step call to track progress.
    """
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,  # inner iterations per step
        history_size=100,
        line_search_fn='strong_wolfe'
    )
    losses = []
    grad_norms = []

    n_outer = max(1, n_iters // 20)

    for i in range(n_outer):
        def closure():
            optimizer.zero_grad()
            loss, _, _ = pde_module.pinn_loss(model, data)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            loss, _, _ = pde_module.pinn_loss(model, data)
            losses.append(loss.item())

        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms.append(total_norm ** 0.5)

    return losses, grad_norms, optimizer


def train_gd(model, data, pde_module, lr, n_iters, device):
    """Train with gradient descent for n_iters iterations."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    losses = []
    grad_norms = []

    for i in range(n_iters):
        optimizer.zero_grad()
        loss, _, _ = pde_module.pinn_loss(model, data)
        loss.backward()
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        optimizer.step()
        losses.append(loss.item())
        grad_norms.append(grad_norm)

    return losses, grad_norms


def run_experiment(pde_name, optimizer_name, width, lr, seed, switch_iter=11000,
                   nncg_mu=1e-2, device='cpu', save_dir='results'):
    """
    Run a single PINN training experiment.

    Args:
        pde_name: 'convection', 'reaction', or 'wave'
        optimizer_name: 'adam', 'lbfgs', 'adam_lbfgs', 'adam_lbfgs_nncg', 'adam_lbfgs_gd'
        width: MLP hidden layer width
        lr: Adam learning rate
        seed: random seed
        switch_iter: iteration to switch from Adam to L-BFGS (for adam_lbfgs)
        nncg_mu: damping parameter for NNCG
        device: torch device
        save_dir: directory to save results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    pde_module = get_pde_module(pde_name)
    data = pde_module.sample_points(seed=seed)
    data = move_data_to_device(data, device)

    model = MLP(input_dim=2, width=width, output_dim=1).to(device)

    all_losses = []
    all_grad_norms = []
    switch_points = {}  # iteration -> description

    if optimizer_name == 'adam':
        losses, grad_norms = train_adam(model, data, pde_module, lr, TOTAL_ITERS, device)
        all_losses.extend(losses)
        all_grad_norms.extend(grad_norms)

    elif optimizer_name == 'lbfgs':
        losses, grad_norms, lbfgs_opt = train_lbfgs_iterative(
            model, data, pde_module, TOTAL_ITERS, device)
        all_losses.extend(losses)
        all_grad_norms.extend(grad_norms)

    elif optimizer_name in ('adam_lbfgs', 'adam_lbfgs_nncg', 'adam_lbfgs_gd'):
        # Phase 1: Adam
        adam_iters = switch_iter
        losses_adam, grad_norms_adam = train_adam(
            model, data, pde_module, lr, adam_iters, device)
        all_losses.extend(losses_adam)
        all_grad_norms.extend(grad_norms_adam)
        switch_points[adam_iters] = 'adam->lbfgs'

        # Phase 2: L-BFGS
        lbfgs_iters = TOTAL_ITERS - switch_iter
        losses_lbfgs, grad_norms_lbfgs, lbfgs_opt = train_lbfgs_iterative(
            model, data, pde_module, lbfgs_iters, device)
        all_losses.extend(losses_lbfgs)
        all_grad_norms.extend(grad_norms_lbfgs)

        if optimizer_name == 'adam_lbfgs_nncg':
            # Phase 3: NNCG
            switch_points[TOTAL_ITERS] = 'lbfgs->nncg'

            def loss_fn():
                return pde_module.pinn_loss(model, data)[0]

            nncg = NNCG(
                params=list(model.parameters()),
                loss_fn=loss_fn,
                n_iterations=2000,
                sketch_size=60,
                precond_update_freq=20,
                mu=nncg_mu,
                cg_tol=1e-16,
                cg_max_iter=1000,
                eta=1.0,
                armijo_alpha=0.1,
                armijo_beta=0.5
            )
            losses_nncg, grad_norms_nncg = nncg.step()
            all_losses.extend(losses_nncg)
            all_grad_norms.extend(grad_norms_nncg)

        elif optimizer_name == 'adam_lbfgs_gd':
            # Phase 3: GD (gradient descent)
            switch_points[TOTAL_ITERS] = 'lbfgs->gd'
            # Use same lr as Adam for GD
            losses_gd, grad_norms_gd = train_gd(
                model, data, pde_module, lr, 2000, device)
            all_losses.extend(losses_gd)
            all_grad_norms.extend(grad_norms_gd)

    # Compute final metrics
    model.eval()
    with torch.no_grad():
        final_loss, final_loss_res, final_loss_bc = pde_module.pinn_loss(model, data)
    final_l2re = pde_module.compute_l2re(model, device=device)

    results = {
        'pde': pde_name,
        'optimizer': optimizer_name,
        'width': width,
        'lr': lr,
        'seed': seed,
        'switch_iter': switch_iter,
        'nncg_mu': nncg_mu,
        'losses': all_losses,
        'grad_norms': all_grad_norms,
        'final_loss': final_loss.item(),
        'final_l2re': final_l2re,
        'switch_points': switch_points,
    }

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{pde_name}_{optimizer_name}_w{width}_lr{lr}_s{seed}"
    if 'lbfgs' in optimizer_name:
        fname += f"_sw{switch_iter}"
    if 'nncg' in optimizer_name:
        fname += f"_mu{nncg_mu}"
    np.save(os.path.join(save_dir, fname + '.npy'), results)

    print(f"[{pde_name}] {optimizer_name} w={width} lr={lr} seed={seed} | "
          f"loss={final_loss.item():.3e} L2RE={final_l2re:.3e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train PINN')
    parser.add_argument('--pde', type=str, default='convection',
                        choices=['convection', 'reaction', 'wave'])
    parser.add_argument('--optimizer', type=str, default='adam_lbfgs',
                        choices=['adam', 'lbfgs', 'adam_lbfgs',
                                 'adam_lbfgs_nncg', 'adam_lbfgs_gd'])
    parser.add_argument('--width', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=345)
    parser.add_argument('--switch_iter', type=int, default=11000,
                        help='Iteration to switch from Adam to L-BFGS')
    parser.add_argument('--nncg_mu', type=float, default=1e-2,
                        help='NNCG damping parameter')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()

    run_experiment(
        pde_name=args.pde,
        optimizer_name=args.optimizer,
        width=args.width,
        lr=args.lr,
        seed=args.seed,
        switch_iter=args.switch_iter,
        nncg_mu=args.nncg_mu,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == '__main__':
    main()
