"""
Dataset classes and input construction for SC-FNO.

Each equation requires a different input format:
  - ODE (1D): input [u(0:M), p_repeated, t_coords] → output u(M:N)
  - PDE 1D (2D): input [u(x,0:M), p_repeated, x_coords, t_coords] → output u(x,M:N)
  - PDE 2D (3D): input [omega(x,y,0), p_repeated, x_coords, y_coords] → output omega(x,y,T)

Data split: 70% train, 15% val, 15% test (paper Section 3.1)
"""

import math
import os
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ODEDataset(Dataset):
    """
    Dataset for ODE problems (ODE1, ODE2).

    Input to FNO: (batch, T_in, in_channels)
      - u values at first M time steps
      - parameters p repeated along time axis
      - time coordinates

    Output: (batch, T_out, 1)
      - u values at remaining N-M time steps
    """

    def __init__(
        self,
        params: torch.Tensor,
        u: torch.Tensor,
        jacobian: torch.Tensor,
        M: int,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ):
        """
        Args:
            params: (n_samples, n_params)
            u: (n_samples, N) solution trajectories
            jacobian: (n_samples, N, n_params) Jacobians ∂u/∂p
            M: number of initial time steps given as input
            t_start, t_end: temporal domain bounds
        """
        self.params = params
        self.u = u
        self.jacobian = jacobian
        self.M = M
        self.N = u.shape[1]
        self.n_params = params.shape[1]
        self.t = torch.linspace(t_start, t_end, self.N)

    def __len__(self) -> int:
        return self.params.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.params[idx]          # (n_params,)
        u_full = self.u[idx]          # (N,)
        jac_full = self.jacobian[idx] # (N, n_params)

        u_in = u_full[: self.M]       # (M,)
        u_out = u_full[self.M :]      # (N-M,)
        jac_out = jac_full[self.M :]  # (N-M, n_params)

        T_in = self.M
        T_out = self.N - self.M

        # Build input tensor: (T_in, 1 + n_params + 1)
        # [u_value, p_1, ..., p_n, t_coord]
        t_in = self.t[: T_in]
        p_rep = p.unsqueeze(0).expand(T_in, -1)  # (T_in, n_params)
        x_in = torch.stack([u_in, t_in], dim=1)  # (T_in, 2)
        fno_input = torch.cat([x_in, p_rep], dim=1)  # (T_in, 2 + n_params)

        return {
            "fno_input": fno_input,       # (T_in, 2+n_params)
            "u_out": u_out.unsqueeze(-1), # (T_out, 1)
            "jac_out": jac_out,           # (T_out, n_params)
            "params": p,                  # (n_params,)
            "u_full": u_full,             # (N,)
            "jac_full": jac_full,         # (N, n_params)
        }


class PDE1DDataset(Dataset):
    """
    Dataset for 1D PDEs (PDE1, PDE2, PDE4).

    Input to FNO: (batch, Sx, T_in, in_channels)
      - u values at first M time steps
      - parameters p repeated along spatial-temporal axes
      - spatial and temporal coordinates

    Output: (batch, Sx, T_out, 1)
      - u values at remaining N-M time steps
    """

    def __init__(
        self,
        params: torch.Tensor,
        u: torch.Tensor,
        jacobian: torch.Tensor,
        M: int,
        x_start: float = 0.0,
        x_end: float = 1.0,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ):
        """
        Args:
            params: (n_samples, n_params)
            u: (n_samples, Sx, N) solution trajectories
            jacobian: (n_samples, Sx, N, n_params) Jacobians ∂u/∂p
            M: number of initial time steps given as input
        """
        self.params = params
        self.u = u
        self.jacobian = jacobian
        self.M = M
        self.Sx = u.shape[1]
        self.N = u.shape[2]
        self.n_params = params.shape[1]

        self.x = torch.linspace(x_start, x_end, self.Sx)
        self.t = torch.linspace(t_start, t_end, self.N)

    def __len__(self) -> int:
        return self.params.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.params[idx]           # (n_params,)
        u_full = self.u[idx]           # (Sx, N)
        jac_full = self.jacobian[idx]  # (Sx, N, n_params)

        u_in = u_full[:, : self.M]     # (Sx, M)
        u_out = u_full[:, self.M :]    # (Sx, N-M)
        jac_out = jac_full[:, self.M :, :]  # (Sx, N-M, n_params)

        T_in = self.M
        T_out = self.N - self.M

        # Build input tensor: (Sx, T_in, 1 + n_params + 2)
        # [u_value, x_coord, t_coord, p_1, ..., p_n]
        x_grid = self.x.unsqueeze(1).expand(-1, T_in)   # (Sx, T_in)
        t_grid = self.t[:T_in].unsqueeze(0).expand(self.Sx, -1)  # (Sx, T_in)
        p_rep = p.unsqueeze(0).unsqueeze(0).expand(self.Sx, T_in, -1)  # (Sx, T_in, n_params)

        fno_input = torch.cat([
            u_in.unsqueeze(-1),          # (Sx, T_in, 1)
            x_grid.unsqueeze(-1),        # (Sx, T_in, 1)
            t_grid.unsqueeze(-1),        # (Sx, T_in, 1)
            p_rep,                       # (Sx, T_in, n_params)
        ], dim=-1)  # (Sx, T_in, 3 + n_params)

        return {
            "fno_input": fno_input,          # (Sx, T_in, 3+n_params)
            "u_out": u_out.unsqueeze(-1),    # (Sx, T_out, 1)
            "jac_out": jac_out,              # (Sx, T_out, n_params)
            "params": p,                     # (n_params,)
            "u_full": u_full,                # (Sx, N)
            "jac_full": jac_full,            # (Sx, N, n_params)
        }


class PDE2DDataset(Dataset):
    """
    Dataset for 2D PDEs (PDE3: Navier-Stokes).

    Input to FNO2d: (batch, Sx, Sy, in_channels)
      - vorticity at initial time step
      - parameters p repeated along spatial axes
      - spatial coordinates x, y

    Output: (batch, Sx, Sy, 1)
      - vorticity at final time step

    FNO2d treats (Sx, Sy) as the two spatial dimensions (no explicit time axis
    since we map IC → final state directly, per Table C.7 which shows no t-mode
    for PDE3).
    """

    def __init__(
        self,
        params: torch.Tensor,
        omega0: torch.Tensor,
        omega_final: torch.Tensor,
        jacobian: torch.Tensor,
        x_start: float = 0.0,
        x_end: float = 1.0,
        y_start: float = 0.0,
        y_end: float = 1.0,
    ):
        """
        Args:
            params: (n_samples, 2)
            omega0: (n_samples, Sx, Sy) initial vorticity
            omega_final: (n_samples, Sx, Sy) final vorticity
            jacobian: (n_samples, Sx, Sy, 2) Jacobians ∂ω_final/∂p
        """
        self.params = params
        self.omega0 = omega0
        self.omega_final = omega_final
        self.jacobian = jacobian
        self.Sx = omega0.shape[1]
        self.Sy = omega0.shape[2]
        self.n_params = params.shape[1]

        self.x = torch.linspace(x_start, x_end, self.Sx)
        self.y = torch.linspace(y_start, y_end, self.Sy)

    def __len__(self) -> int:
        return self.params.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.params[idx]                # (2,)
        w0 = self.omega0[idx]               # (Sx, Sy)
        w_final = self.omega_final[idx]     # (Sx, Sy)
        jac = self.jacobian[idx]            # (Sx, Sy, 2)

        # Build input tensor: (Sx, Sy, 1 + 2 + n_params)
        # [omega_value, x_coord, y_coord, p_1, p_2]
        X, Y = torch.meshgrid(self.x, self.y, indexing="ij")  # (Sx, Sy)
        p_rep = p.unsqueeze(0).unsqueeze(0).expand(self.Sx, self.Sy, -1)  # (Sx, Sy, 2)

        fno_input = torch.cat([
            w0.unsqueeze(-1),    # (Sx, Sy, 1)
            X.unsqueeze(-1),     # (Sx, Sy, 1)
            Y.unsqueeze(-1),     # (Sx, Sy, 1)
            p_rep,               # (Sx, Sy, 2)
        ], dim=-1)  # (Sx, Sy, 3+n_params)

        return {
            "fno_input": fno_input,              # (Sx, Sy, 3+n_params)
            "u_out": w_final.unsqueeze(-1),      # (Sx, Sy, 1)
            "jac_out": jac,                      # (Sx, Sy, 2)
            "params": p,                         # (2,)
            "omega0": w0,                        # (Sx, Sy)
        }


def split_dataset(
    dataset: Dataset,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train/val/test (70/15/15 as per paper)."""
    n = len(dataset)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


def make_dataloaders(
    dataset: Dataset,
    batch_size: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders."""
    train_ds, val_ds, test_ds = split_dataset(dataset, train_frac, val_frac, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def perturb_params(
    params: torch.Tensor,
    param_ranges: Dict[str, Tuple[float, float]],
    perturbation_ratio: float,
) -> torch.Tensor:
    """
    Perturb parameters beyond their training range by perturbation_ratio λ.

    For each parameter with range [a, b], the perturbed range is [b, (1+λ)b].
    This tests model generalization beyond training distribution.

    Args:
        params: (n_samples, n_params) parameters in original range
        param_ranges: dict of {name: (lo, hi)} for each parameter
        perturbation_ratio: λ, fraction to extend beyond upper bound

    Returns:
        perturbed_params: (n_samples, n_params)
    """
    perturbed = params.clone()
    for i, (name, (lo, hi)) in enumerate(param_ranges.items()):
        # Sample from [hi, (1+λ)*hi]
        n = params.shape[0]
        perturbed[:, i] = (
            torch.rand(n, device=params.device) * perturbation_ratio * hi + hi
        )
    return perturbed


def save_dataset(data: Dict[str, torch.Tensor], path: str) -> None:
    """Save dataset to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)


def load_dataset(path: str) -> Dict[str, torch.Tensor]:
    """Load dataset from disk."""
    return torch.load(path, map_location="cpu")
