"""
Reaction ODE:
  du/dt - rho*u*(1-u) = 0,  x in (0, 2*pi), t in (0, 1)
  u(x, 0) = exp(-(x-pi)^2 / (2*(pi/4)^2)),  x in [0, 2*pi]
  u(0, t) = u(2*pi, t),  t in [0, 1]

Analytical solution: u(x,t) = h(x)*exp(rho*t) / (h(x)*exp(rho*t) + 1 - h(x))
  where h(x) = exp(-(x-pi)^2 / (2*(pi/4)^2))
We use rho = 5.
"""

import numpy as np
import torch


RHO = 5.0


def h(x):
    return np.exp(-((x - np.pi) ** 2) / (2 * (np.pi / 4) ** 2))


def exact_solution(x, t):
    hx = h(x)
    return hx * np.exp(RHO * t) / (hx * np.exp(RHO * t) + 1 - hx)


def sample_points(n_res=10000, n_ic=257, n_bc=101, seed=None):
    rng = np.random.default_rng(seed)

    x_grid = np.linspace(0, 2 * np.pi, 257)[1:-1]
    t_grid = np.linspace(0, 1, 102)[1:-1]
    XX, TT = np.meshgrid(x_grid, t_grid, indexing='ij')
    all_pts = np.stack([XX.ravel(), TT.ravel()], axis=1)

    idx = rng.choice(len(all_pts), size=n_res, replace=False)
    res_pts = all_pts[idx]

    x_ic = np.linspace(0, 2 * np.pi, n_ic)
    ic_pts = np.stack([x_ic, np.zeros(n_ic)], axis=1)
    ic_vals = h(x_ic)

    t_bc = np.linspace(0, 1, n_bc)
    bc_x0 = np.stack([np.zeros(n_bc), t_bc], axis=1)
    bc_x2pi = np.stack([np.full(n_bc, 2 * np.pi), t_bc], axis=1)

    return {
        'res': torch.tensor(res_pts, dtype=torch.float32),
        'ic_pts': torch.tensor(ic_pts, dtype=torch.float32),
        'ic_vals': torch.tensor(ic_vals, dtype=torch.float32),
        'bc_x0': torch.tensor(bc_x0, dtype=torch.float32),
        'bc_x2pi': torch.tensor(bc_x2pi, dtype=torch.float32),
    }


def pinn_loss(model, data):
    x_r = data['res'][:, 0:1].requires_grad_(True)
    t_r = data['res'][:, 1:2].requires_grad_(True)
    xt_r = torch.cat([x_r, t_r], dim=1)
    u = model(xt_r)

    u_t = torch.autograd.grad(u, t_r, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    residual = u_t - RHO * u * (1 - u)
    loss_res = 0.5 * torch.mean(residual ** 2)

    ic_pts = data['ic_pts']
    ic_vals = data['ic_vals'].unsqueeze(1)
    u_ic = model(ic_pts)
    loss_ic = 0.5 * torch.mean((u_ic - ic_vals) ** 2)

    u_bc0 = model(data['bc_x0'])
    u_bc2pi = model(data['bc_x2pi'])
    loss_bc = 0.5 * torch.mean((u_bc0 - u_bc2pi) ** 2)

    n_bc = data['ic_pts'].shape[0] + data['bc_x0'].shape[0] + data['bc_x2pi'].shape[0]
    loss_bc_total = (loss_ic * data['ic_pts'].shape[0] +
                     loss_bc * (data['bc_x0'].shape[0] + data['bc_x2pi'].shape[0])) / n_bc

    return loss_res + loss_bc_total, loss_res, loss_bc_total


def compute_l2re(model, device='cpu'):
    x_int = np.linspace(0, 2 * np.pi, 257)[1:-1]
    t_int = np.linspace(0, 1, 102)[1:-1]
    XX, TT = np.meshgrid(x_int, t_int, indexing='ij')
    pts_int = np.stack([XX.ravel(), TT.ravel()], axis=1)

    x_ic_pts = np.linspace(0, 2 * np.pi, 257)
    pts_ic = np.stack([x_ic_pts, np.zeros(257)], axis=1)

    t_bc_pts = np.linspace(0, 1, 101)
    pts_bc0 = np.stack([np.zeros(101), t_bc_pts], axis=1)
    pts_bc2pi = np.stack([np.full(101, 2 * np.pi), t_bc_pts], axis=1)

    all_pts = np.vstack([pts_int, pts_ic, pts_bc0, pts_bc2pi])
    x_all = all_pts[:, 0]
    t_all = all_pts[:, 1]
    u_exact = exact_solution(x_all, t_all)

    pts_tensor = torch.tensor(all_pts, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        u_pred = model(pts_tensor).cpu().numpy().ravel()

    l2re = np.sqrt(np.sum((u_pred - u_exact) ** 2) / np.sum(u_exact ** 2))
    return l2re
