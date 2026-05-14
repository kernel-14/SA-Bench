"""
Utilities for computing the L-BFGS preconditioned Hessian spectral density.

Implements Algorithms 2 and 3 from the paper appendix:
  - Algorithm 2: Unrolling the L-BFGS Update
  - Algorithm 3: Matrix-vector product for preconditioned Hessian

Reference: Rathore et al., "Challenges in Training PINNs: A Loss Landscape Perspective", ICML 2024.
"""

import torch
import numpy as np


def unroll_lbfgs_update(y_list, s_list, rho_list):
    """
    Algorithm 2: Unrolling the L-BFGS Update.

    Computes the columns of tilde_Y, tilde_V, tilde_S from the stored
    L-BFGS directions.

    Args:
        y_list: list of y_k vectors (gradient differences), most recent first
        s_list: list of s_k vectors (iterate differences), most recent first
        rho_list: list of rho_k = 1/(y_k^T s_k), most recent first

    Returns:
        tilde_Y: matrix of columns rho_i * y_i
        tilde_V: matrix of columns tilde_v_i
        tilde_S: matrix of columns tilde_s_i
        gamma: scaling factor for H_k^0 = gamma * I
    """
    m = len(y_list)
    if m == 0:
        return None, None, None, 1.0

    p = y_list[0].shape[0]
    device = y_list[0].device

    tilde_Y = torch.zeros(p, m, device=device)
    tilde_V = torch.zeros(p, m, device=device)
    tilde_S = torch.zeros(p, m, device=device)

    # gamma = s_{k-1}^T y_{k-1} / (y_{k-1}^T y_{k-1})
    gamma = (torch.dot(s_list[0], y_list[0]) / torch.dot(y_list[0], y_list[0])).item()

    # i = k-1 (most recent, index 0)
    tilde_Y[:, 0] = rho_list[0] * y_list[0]
    tilde_V[:, 0] = s_list[0]
    tilde_S[:, 0] = torch.sqrt(torch.tensor(rho_list[0], device=device)) * s_list[0]

    for i in range(1, m):
        tilde_Y[:, i] = rho_list[i] * y_list[i]
        # alpha = sum_{j=0}^{i-1} (tilde_Y[:,j]^T s_list[i]) * tilde_V[:,j]
        alpha = torch.zeros(p, device=device)
        for j in range(i):
            alpha += torch.dot(tilde_Y[:, j], s_list[i]) * tilde_V[:, j]
        tilde_V[:, i] = s_list[i] - alpha
        tilde_S[:, i] = torch.sqrt(torch.tensor(rho_list[i], device=device)) * tilde_V[:, i]

    return tilde_Y, tilde_V, tilde_S, gamma


def lbfgs_precond_matvec(v, tilde_Y, tilde_V, tilde_S, gamma, hvp_fn):
    """
    Algorithm 3: Matrix-vector product for tilde_H_k^T H_L tilde_H_k.

    This computes the matrix-vector product needed for spectral density
    computation of the preconditioned Hessian.

    The preconditioned Hessian H_k H_L(w) has the same non-zero eigenvalues
    as tilde_H_k^T H_L(w) tilde_H_k, where H_k = tilde_H_k tilde_H_k^T.

    tilde_H_k = [sqrt(gamma) * (I - tilde_Y tilde_V^T), tilde_S]

    For a vector v = [v1 (size p), v2 (size m)]:
      v' = sqrt(gamma) * (v1 - tilde_V @ tilde_Y^T @ v1) + tilde_S @ v2
      v'' = H_L @ v'  (Hessian-vector product)
      v''' = [sqrt(gamma) * (v'' - tilde_Y @ tilde_V^T @ v''), tilde_S^T @ v'']

    Args:
        v: input vector of size p + m
        tilde_Y: (p, m) matrix
        tilde_V: (p, m) matrix
        tilde_S: (p, m) matrix
        gamma: scalar
        hvp_fn: Hessian-vector product function

    Returns:
        result vector of size p + m
    """
    p = tilde_Y.shape[0]
    m = tilde_Y.shape[1]

    v1 = v[:p]
    v2 = v[p:]

    sqrt_gamma = torch.sqrt(torch.tensor(gamma, device=v.device))

    # v' = sqrt(gamma) * (v1 - tilde_V @ tilde_Y^T @ v1) + tilde_S @ v2
    v_prime = sqrt_gamma * (v1 - tilde_V @ (tilde_Y.t() @ v1)) + tilde_S @ v2

    # v'' = H_L @ v'
    v_double_prime = hvp_fn(v_prime)

    # v''' = [sqrt(gamma) * (v'' - tilde_Y @ tilde_V^T @ v''), tilde_S^T @ v'']
    part1 = sqrt_gamma * (v_double_prime - tilde_Y @ (tilde_V.t() @ v_double_prime))
    part2 = tilde_S.t() @ v_double_prime

    return torch.cat([part1, part2])


def extract_lbfgs_state(optimizer):
    """
    Extract the stored L-BFGS state (y, s, rho vectors) from a PyTorch L-BFGS optimizer.

    Returns:
        y_list: list of y vectors (most recent first)
        s_list: list of s vectors (most recent first)
        rho_list: list of rho values (most recent first)
    """
    state = optimizer.state[optimizer._params[0]]

    if 'old_dirs' not in state:
        return [], [], []

    old_dirs = state['old_dirs']   # list of s vectors
    old_stps = state['old_stps']   # list of y vectors
    ro = state['ro']               # list of rho values

    # PyTorch stores them oldest first; we want most recent first
    y_list = list(reversed(old_stps))
    s_list = list(reversed(old_dirs))
    rho_list = list(reversed([r.item() for r in ro]))

    return y_list, s_list, rho_list
