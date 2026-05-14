"""
Spectral density computation for PINN Hessians.

Implements the Stochastic Lanczos Quadrature (SLQ) algorithm for computing
the spectral density (eigenvalue distribution) of the Hessian and
L-BFGS preconditioned Hessian.

Used to reproduce Figures 3 and 7 from the paper.

Reference: Rathore et al., "Challenges in Training PINNs: A Loss Landscape Perspective", ICML 2024.
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MLP
from src.pdes import convection, reaction, wave
from src.utils.lbfgs_precond import (
    unroll_lbfgs_update, lbfgs_precond_matvec, extract_lbfgs_state
)


PDE_MODULES = {
    'convection': convection,
    'reaction': reaction,
    'wave': wave,
}


def compute_hvp(model, params, loss_fn, v):
    """
    Compute Hessian-vector product H @ v using autograd.

    Args:
        model: neural network
        params: list of model parameters
        loss_fn: callable returning scalar loss
        v: flat vector of size p

    Returns:
        hvp: flat vector of size p
    """
    loss = loss_fn()
    grad = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.view(-1) for g in grad])
    hvp = torch.autograd.grad(flat_grad, params,
                               grad_outputs=v.view_as(flat_grad),
                               retain_graph=False)
    return torch.cat([h.view(-1) for h in hvp]).detach()


def lanczos_iteration(matvec_fn, p, n_iter, device):
    """
    Lanczos iteration for tridiagonalization.

    Args:
        matvec_fn: function computing matrix-vector product
        p: dimension of the matrix
        n_iter: number of Lanczos iterations
        device: torch device

    Returns:
        alphas: diagonal elements of tridiagonal matrix (n_iter,)
        betas: off-diagonal elements (n_iter-1,)
    """
    # Random starting vector
    v = torch.randn(p, device=device)
    v = v / torch.linalg.norm(v)

    alphas = []
    betas = []
    v_prev = torch.zeros(p, device=device)
    v_curr = v

    for j in range(n_iter):
        w = matvec_fn(v_curr)
        alpha = torch.dot(v_curr, w).item()
        alphas.append(alpha)

        if j < n_iter - 1:
            w = w - alpha * v_curr - (betas[-1] if betas else 0.0) * v_prev
            beta = torch.linalg.norm(w).item()
            betas.append(beta)
            if beta < 1e-10:
                # Early termination
                break
            v_prev = v_curr
            v_curr = w / beta

    return np.array(alphas), np.array(betas)


def slq_spectral_density(matvec_fn, p, n_iter=100, n_probes=10, device='cpu',
                          sigma=0.01, grid_points=1000, grid_min=None, grid_max=None):
    """
    Stochastic Lanczos Quadrature for spectral density estimation.

    Estimates the spectral density phi(lambda) = (1/p) * sum_i delta(lambda - lambda_i)
    using Gaussian kernel smoothing.

    Args:
        matvec_fn: matrix-vector product function
        p: matrix dimension
        n_iter: number of Lanczos iterations per probe
        n_probes: number of random probe vectors
        device: torch device
        sigma: Gaussian kernel bandwidth
        grid_points: number of grid points for density estimation
        grid_min: minimum eigenvalue for grid (auto-detected if None)
        grid_max: maximum eigenvalue for grid (auto-detected if None)

    Returns:
        grid: eigenvalue grid
        density: spectral density at each grid point
        all_eigenvalues: list of eigenvalue arrays from each probe
        all_weights: list of weight arrays from each probe
    """
    all_eigenvalues = []
    all_weights = []

    for probe_idx in range(n_probes):
        alphas, betas = lanczos_iteration(matvec_fn, p, n_iter, device)

        # Build tridiagonal matrix
        n = len(alphas)
        T = np.diag(alphas) + np.diag(betas[:n-1], 1) + np.diag(betas[:n-1], -1)

        # Eigendecomposition of T
        eigvals, eigvecs = np.linalg.eigh(T)

        # Weights are squares of first components of eigenvectors
        weights = eigvecs[0, :] ** 2

        all_eigenvalues.append(eigvals)
        all_weights.append(weights)

    # Determine grid range
    all_eigs = np.concatenate(all_eigenvalues)
    if grid_min is None:
        grid_min = np.percentile(all_eigs, 1)
    if grid_max is None:
        grid_max = np.percentile(all_eigs, 99)

    grid = np.linspace(grid_min, grid_max, grid_points)

    # Compute density using Gaussian kernel smoothing
    density = np.zeros(grid_points)
    for eigvals, weights in zip(all_eigenvalues, all_weights):
        for lam, w in zip(eigvals, weights):
            density += w * np.exp(-0.5 * ((grid - lam) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    density /= n_probes

    return grid, density, all_eigenvalues, all_weights


def compute_spectral_density(model, data, pde_module, device='cpu',
                              n_iter=100, n_probes=10, sigma=0.01,
                              compute_precond=False, lbfgs_optimizer=None):
    """
    Compute spectral density of the PINN Hessian (and optionally the
    L-BFGS preconditioned Hessian).

    Args:
        model: trained PINN model
        data: training data dict
        pde_module: PDE module
        device: torch device
        n_iter: Lanczos iterations
        n_probes: number of probe vectors
        sigma: Gaussian kernel bandwidth
        compute_precond: whether to also compute preconditioned Hessian density
        lbfgs_optimizer: L-BFGS optimizer (needed for preconditioned density)

    Returns:
        results dict with 'hessian' and optionally 'precond_hessian' keys
    """
    params = list(model.parameters())
    p = sum(param.numel() for param in params)

    def loss_fn():
        return pde_module.pinn_loss(model, data)[0]

    def hvp_fn(v):
        return compute_hvp(model, params, loss_fn, v)

    # Compute Hessian spectral density
    print(f"Computing Hessian spectral density (p={p}, n_iter={n_iter}, n_probes={n_probes})...")
    grid_h, density_h, eigs_h, weights_h = slq_spectral_density(
        hvp_fn, p, n_iter=n_iter, n_probes=n_probes, device=device, sigma=sigma
    )

    results = {
        'hessian': {
            'grid': grid_h,
            'density': density_h,
            'eigenvalues': eigs_h,
            'weights': weights_h,
        }
    }

    if compute_precond and lbfgs_optimizer is not None:
        # Extract L-BFGS state
        y_list, s_list, rho_list = extract_lbfgs_state(lbfgs_optimizer)

        if len(y_list) > 0:
            # Unroll L-BFGS update
            tilde_Y, tilde_V, tilde_S, gamma = unroll_lbfgs_update(y_list, s_list, rho_list)
            m = tilde_Y.shape[1]
            p_ext = p + m

            def precond_matvec(v):
                return lbfgs_precond_matvec(v, tilde_Y, tilde_V, tilde_S, gamma, hvp_fn)

            print(f"Computing preconditioned Hessian spectral density (p_ext={p_ext})...")
            grid_ph, density_ph, eigs_ph, weights_ph = slq_spectral_density(
                precond_matvec, p_ext, n_iter=n_iter, n_probes=n_probes,
                device=device, sigma=sigma
            )

            results['precond_hessian'] = {
                'grid': grid_ph,
                'density': density_ph,
                'eigenvalues': eigs_ph,
                'weights': weights_ph,
                'gamma': gamma,
                'm': m,
            }

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute PINN spectral density')
    parser.add_argument('--pde', type=str, default='convection',
                        choices=['convection', 'reaction', 'wave'])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--n_iter', type=int, default=100,
                        help='Number of Lanczos iterations')
    parser.add_argument('--n_probes', type=int, default=10,
                        help='Number of probe vectors for SLQ')
    parser.add_argument('--sigma', type=float, default=0.01,
                        help='Gaussian kernel bandwidth')
    parser.add_argument('--width', type=int, default=200)
    parser.add_argument('--seed', type=int, default=345)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='results/spectral')
    args = parser.parse_args()

    pde_module = PDE_MODULES[args.pde]
    data = pde_module.sample_points(seed=args.seed)
    data = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()}

    model = MLP(input_dim=2, width=args.width, output_dim=1).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    results = compute_spectral_density(
        model, data, pde_module,
        device=args.device,
        n_iter=args.n_iter,
        n_probes=args.n_probes,
        sigma=args.sigma,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    fname = f"{args.pde}_spectral_density.npy"
    np.save(os.path.join(args.save_dir, fname), results)
    print(f"Saved spectral density to {os.path.join(args.save_dir, fname)}")


if __name__ == '__main__':
    main()
