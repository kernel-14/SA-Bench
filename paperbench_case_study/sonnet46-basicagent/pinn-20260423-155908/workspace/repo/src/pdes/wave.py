"""
Wave PDE:
  d^2u/dt^2 - 4 * d^2u/dx^2 = 0,  x in (0, 1), t in (0, 1)
  u(x, 0) = sin(pi*x) + 0.5*sin(beta*pi*x),  x in [0, 1]
  du/dt(x, 0) = 0,  x in [0, 1]
  u(0, t) = u(1, t) = 0,  t in [0, 1]

Analytical solution: u(x,t) = sin(pi*x)*cos(2*pi*t) + 0.5*sin(beta*pi*x)*cos(2*beta*pi*t)
We use beta = 5.
"""

import numpy as np
import torch


BETA = 5.0
C = 2.0  # wave speed (c^2 = 4)


def exact_solution(x, t):
    return (np.sin(np.pi * x) * np.cos(2 * np.pi * t) +
            0.5 * np.sin(BETA * np.pi * x) * np.cos(2 * BETA * np.pi * t))


def sample_points(n_res=10000, n_ic=257, n_bc=101, seed=None):
    rng = np.random.default_rng(seed)

    # Interior: x in (0,1), t in (0,1) — 255x100 grid
    x_grid = np.linspace(0, 1, 257)[1:-1]
    t_grid = np.linspace(0, 1, 102)[1:-1]
    XX, TT = np.meshgrid(x_grid, t_grid, indexing='ij')
    all_pts = np.stack([XX.ravel(), TT.ravel()], axis=1)

    idx = rng.choice(len(all_pts), size=n_res, replace=False)
    res_pts = all_pts[idx]

    # IC: t=0, 257 equally spaced x in [0,1]
    x_ic = np.linspace(0, 1, n_ic)
    ic_pts = np.stack([x_ic, np.zeros(n_ic)], axis=1)
    ic_vals = np.sin(np.pi * x_ic) + 0.5 * np.sin(BETA * np.pi * x_ic)

    # IC for du/dt: t=0, same x points
    ic_dt_vals = np.zeros(n_ic)  # du/dt(x,0) = 0

    # BC: x=0 and x=1, 101 equally spaced t in [0,1]
    t_bc = np.linspace(0, 1, n_bc)
    bc_x0 = np.stack([np.zeros(n_bc), t_bc], axis=1)
    bc_x1 = np.stack([np.ones(n_bc), t_bc], axis=1)

    return {
        'res': torch.tensor(res_pts, dtype=torch.float32),
        'ic_pts': torch.tensor(ic_pts, dtype=torch.float32),
        'ic_vals': torch.tensor(ic_vals, dtype=torch.float32),
        'ic_dt_vals': torch.tensor(ic_dt_vals, dtype=torch.float32),
        'bc_x0': torch.tensor(bc_x0, dtype=torch.float32),
        'bc_x1': torch.tensor(bc_x1, dtype=torch.float32),
    }


def pinn_loss(model, data):
    # Residual: u_tt - 4*u_xx = 0
    x_r = data['res'][:, 0:1].requires_grad_(True)
    t_r = data['res'][:, 1:2].requires_grad_(True)
    xt_r = torch.cat([x_r, t_r], dim=1)
    u = model(xt_r)

    u_t = torch.autograd.grad(u, t_r, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_r, grad_outputs=torch.ones_like(u_t),
                                create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_r, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_r, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]
    residual = u_tt - 4.0 * u_xx
    loss_res = 0.5 * torch.mean(residual ** 2)

    # IC: u(x,0) = sin(pi*x) + 0.5*sin(beta*pi*x)
    ic_pts = data['ic_pts']
    ic_vals = data['ic_vals'].unsqueeze(1)
    u_ic = model(ic_pts)
    loss_ic = 0.5 * torch.mean((u_ic - ic_vals) ** 2)

    # IC: du/dt(x,0) = 0
    x_ic_t = ic_pts[:, 0:1].requires_grad_(True)
    t_ic_t = ic_pts[:, 1:2].requires_grad_(True)
    xt_ic = torch.cat([x_ic_t, t_ic_t], dim=1)
    u_ic2 = model(xt_ic)
    u_ic_t = torch.autograd.grad(u_ic2, t_ic_t, grad_outputs=torch.ones_like(u_ic2),
                                  create_graph=True)[0]
    ic_dt_vals = data['ic_dt_vals'].unsqueeze(1)
    loss_ic_dt = 0.5 * torch.mean((u_ic_t - ic_dt_vals) ** 2)

    # BC: u(0,t) = 0, u(1,t) = 0
    u_bc0 = model(data['bc_x0'])
    u_bc1 = model(data['bc_x1'])
    loss_bc = 0.5 * torch.mean(u_bc0 ** 2) + 0.5 * torch.mean(u_bc1 ** 2)
    loss_bc = loss_bc / 2.0  # average over two BCs

    n_ic = data['ic_pts'].shape[0]
    n_bc = data['bc_x0'].shape[0] + data['bc_x1'].shape[0]
    n_total_bc = 2 * n_ic + n_bc  # IC + IC_dt + BC

    loss_bc_total = (loss_ic * n_ic + loss_ic_dt * n_ic +
                     (0.5 * torch.mean(u_bc0 ** 2) * data['bc_x0'].shape[0] +
                      0.5 * torch.mean(u_bc1 ** 2) * data['bc_x1'].shape[0])) / n_total_bc

    return loss_res + loss_bc_total, loss_res, loss_bc_total


def compute_l2re(model, device='cpu'):
    x_int = np.linspace(0, 1, 257)[1:-1]
    t_int = np.linspace(0, 1, 102)[1:-1]
    XX, TT = np.meshgrid(x_int, t_int, indexing='ij')
    pts_int = np.stack([XX.ravel(), TT.ravel()], axis=1)

    x_ic_pts = np.linspace(0, 1, 257)
    pts_ic = np.stack([x_ic_pts, np.zeros(257)], axis=1)

    t_bc_pts = np.linspace(0, 1, 101)
    pts_bc0 = np.stack([np.zeros(101), t_bc_pts], axis=1)
    pts_bc1 = np.stack([np.ones(101), t_bc_pts], axis=1)

    all_pts = np.vstack([pts_int, pts_ic, pts_bc0, pts_bc1])
    x_all = all_pts[:, 0]
    t_all = all_pts[:, 1]
    u_exact = exact_solution(x_all, t_all)

    pts_tensor = torch.tensor(all_pts, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        u_pred = model(pts_tensor).cpu().numpy().ravel()

    l2re = np.sqrt(np.sum((u_pred - u_exact) ** 2) / np.sum(u_exact ** 2))
    return l2re
