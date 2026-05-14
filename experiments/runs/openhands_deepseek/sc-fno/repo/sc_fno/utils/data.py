"""Data preparation utilities for neural operator training.

Prepares input tensors from solution + parameter datasets.
Available operators:
- FNO/SC-FNO: input includes coordinates, initial conditions, and parameters.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def build_input_tensor_ode(
    u_init: torch.Tensor,
    t: torch.Tensor,
    params: Dict[str, float],
    param_names: List[str],
) -> torch.Tensor:
    """Build input tensor for ODE problems (1D temporal only).

    The input includes:
    - Time coordinate t
    - Initial solution values u_init (first M time steps)
    - Parameters p

    Args:
        u_init: Initial solution (N_t_init,) tensor.
        t: Full time grid (N_t,) tensor.
        params: Parameter dict.
        param_names: Ordered parameter names.

    Returns:
        Input tensor of shape (input_channels, N_t).
    """
    param_vals = torch.tensor([params[n] for n in param_names])

    t_in = t.unsqueeze(0)  # (1, N_t)
    u_in = torch.zeros_like(t)
    u_in[: len(u_init)] = u_init

    u_in = u_in.unsqueeze(0)  # (1, N_t)
    p_in = param_vals.unsqueeze(-1).expand(-1, len(t))  # (n_params, N_t)

    return torch.cat([t_in, u_in, p_in], dim=0)  # (1 + 1 + n_params, N_t)


def build_input_tensor_pde(
    u_init: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    params: Dict[str, float],
    param_names: List[str],
    M: int,
) -> torch.Tensor:
    """Build input tensor for 1D+time PDE problems.

    Maps from first M time steps + params to next N-M time steps.

    Args:
        u_init: Initial solution (N_x, M) tensor.
        x: Spatial grid (N_x,) tensor.
        t: Full time grid (N_t,) tensor.
        params: Parameter dict.
        param_names: Ordered parameter names.
        M: Number of initial time steps provided as input.

    Returns:
        Input tensor of shape (input_channels, N_x, N_t_all)
        where N_t_all = N_t - M (the output time steps).
    """
    param_vals = torch.tensor([params[n] for n in param_names])

    N_x = len(x)
    N_t_total = len(t)
    N_t_out = N_t_total - M

    x_mesh = x.unsqueeze(-1).expand(N_x, N_t_out)  # (N_x, N_t_out)
    t_mesh = t[M:].unsqueeze(0).expand(N_x, N_t_out)  # (N_x, N_t_out)

    u_past = torch.zeros(N_x, N_t_out)
    for i in range(min(M, N_t_out)):
        u_past[:, i] = u_init[:, i]
    for i in range(M, N_t_out):
        u_past[:, i] = u_init[:, M - 1]

    p_tensor = param_vals.unsqueeze(-1).unsqueeze(-1).expand(-1, N_x, N_t_out)

    return torch.cat([
        x_mesh.unsqueeze(0),
        t_mesh.unsqueeze(0),
        u_past.unsqueeze(0),
        p_tensor,
    ], dim=0)


def build_input_tensor_pde3(
    omega_init: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    params: Dict[str, float],
    param_names: List[str],
) -> torch.Tensor:
    """Build input tensor for PDE3 (Navier-Stokes, 2D).

    Maps from initial condition to final time t=3.

    Args:
        omega_init: Initial vorticity (N_x, N_y) tensor.
        x: x-grid (N_x,).
        y: y-grid (N_y,).
        params: Parameter dict.
        param_names: Ordered parameter names.

    Returns:
        Input tensor of shape (input_channels, N_x, N_y).
    """
    param_vals = torch.tensor([params[n] for n in param_names])

    N_x, N_y = len(x), len(y)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    p_tensor = param_vals.unsqueeze(-1).unsqueeze(-1).expand(-1, N_x, N_y)

    return torch.cat([
        xx.unsqueeze(0),
        yy.unsqueeze(0),
        omega_init.unsqueeze(0),
        p_tensor,
    ], dim=0)


def prepare_dataloaders(
    dataset: List[Dict],
    param_names: List[str],
    equation: str,
    M: int = 5,
    input_fn = None,
    batch_size: int = 4,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.device]:
    """Prepare train/val/test dataloaders from solver outputs.

    Args:
        dataset: List of dictionaries from generate_dataset().
        param_names: Ordered parameter names.
        equation: Equation type ("ode1", "ode2", "pde1", "pde2", "pde3", "pde4").
        M: Number of initial time steps for input.
        input_fn: Optional custom input builder.
        batch_size: Batch size.
        train_ratio, val_ratio, test_ratio: Data split ratios.
        seed: Random seed.

    Returns:
        (train_loader, val_loader, test_loader, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "ode" in equation:
        input_tensors, output_tensors, jacobian_tensors = _prepare_ode_data(
            dataset, param_names, M, input_fn, device
        )
    elif equation == "pde3":
        input_tensors, output_tensors, jacobian_tensors = _prepare_pde3_data(
            dataset, param_names, device
        )
    else:
        input_tensors, output_tensors, jacobian_tensors = _prepare_pde_data(
            dataset, param_names, M, input_fn, device
        )

    n_total = len(input_tensors)
    generator = torch.Generator().manual_seed(seed)

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    all_data = [input_tensors, output_tensors] + jacobian_tensors
    full_dataset = TensorDataset(*all_data)

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, device


def _prepare_ode_data(
    dataset: List[Dict],
    param_names: List[str],
    M: int,
    input_fn,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    inputs, outputs, jacobians = [], [], {name: [] for name in param_names}

    for sample in dataset:
        u = sample["u"]
        t = sample["t"]
        params = sample["params"]

        x_in = build_input_tensor_ode(u[:M], t, params, param_names)
        y_out = u[M:]

        inputs.append(x_in)
        outputs.append(y_out)

        for name in param_names:
            jac = sample.get(f"du_d{name}", torch.zeros_like(u[M:]))
            jacobians[name].append(jac)

    inputs = torch.stack(inputs).to(device)
    outputs = torch.stack(outputs).unsqueeze(1).to(device)

    jac_tensors = []
    for name in param_names:
        jac_tensors.append(torch.stack(jacobians[name]).unsqueeze(1).to(device))

    return inputs, outputs, jac_tensors


def _prepare_pde_data(
    dataset: List[Dict],
    param_names: List[str],
    M: int,
    input_fn,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    inputs, outputs, jacobians = [], [], {name: [] for name in param_names}

    for sample in dataset:
        u = sample["u"]
        x = sample["x"]
        t = sample["t"]
        params = sample["params"]

        x_in = build_input_tensor_pde(u[:, :M], x, t, params, param_names, M)
        y_out = u[:, M:]

        inputs.append(x_in)
        outputs.append(y_out)

        for name in param_names:
            jac = sample.get(f"du_d{name}", torch.zeros_like(u[:, M:]))
            jacobians[name].append(jac)

    inputs = torch.stack(inputs).to(device)
    outputs = torch.stack(outputs).unsqueeze(1).to(device)

    jac_tensors = []
    for name in param_names:
        jac_tensors.append(torch.stack(jacobians[name]).unsqueeze(1).to(device))

    return inputs, outputs, jac_tensors


def _prepare_pde3_data(
    dataset: List[Dict],
    param_names: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    inputs, outputs, jacobians = [], [], {name: [] for name in param_names}

    for sample in dataset:
        omega = sample["omega"]
        x = sample["x"]
        y = sample["y"]
        params = sample["params"]

        x_in = build_input_tensor_pde3(omega, x, y, params, param_names)
        y_out = omega

        inputs.append(x_in)
        outputs.append(y_out)

        for name in param_names:
            jac = sample.get(f"domega_d{name}", torch.zeros_like(omega))
            jacobians[name].append(jac)

    inputs = torch.stack(inputs).to(device)
    outputs = torch.stack(outputs).unsqueeze(1).to(device)

    jac_tensors = []
    for name in param_names:
        jac_tensors.append(torch.stack(jacobians[name]).unsqueeze(1).to(device))

    return inputs, outputs, jac_tensors
