"""Parameter inversion via optimization of surrogate models.

Implements single-parameter and multi-parameter inversion as described
in Section 3.1.

Given observed solution paths u_obs and a trained surrogate model F(u_0, x, t, p),
we find p* = argmin_p || F(u_0, x, t, p) - u_obs ||^2 using gradient-based
optimization.
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn


def invert_parameters(
    model: nn.Module,
    u_target: torch.Tensor,
    x_input: torch.Tensor,
    param_names: List[str],
    param_init: Dict[str, float],
    param_bounds: Dict[str, List[float]],
    n_iterations: int = 200,
    lr: float = 0.01,
    verbose: bool = False,
) -> Dict[str, float]:
    """Invert model parameters to match target solution.

    Args:
        model: Trained neural operator.
        u_target: Target solution tensor.
        x_input: Base input tensor (without parameters embedded).
        param_names: Names of parameters to optimize.
        param_init: Initial guess for parameters.
        param_bounds: Dict mapping param name -> [low, high].
        n_iterations: Number of optimization iterations.
        lr: Learning rate for Adam optimizer.
        verbose: Print progress.

    Returns:
        Dict of optimized parameter values.
    """
    device = next(model.parameters()).device
    u_target = u_target.to(device)
    x_input = x_input.to(device).clone()

    params_tensor = torch.tensor(
        [param_init[n] for n in param_names],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    optimizer = torch.optim.Adam([params_tensor], lr=lr)
    bounds_low = torch.tensor(
        [param_bounds[n][0] for n in param_names], device=device
    )
    bounds_high = torch.tensor(
        [param_bounds[n][1] for n in param_names], device=device
    )

    def clamp_params():
        with torch.no_grad():
            params_tensor.clamp_(bounds_low, bounds_high)

    clamp_params()

    for it in range(n_iterations):
        optimizer.zero_grad()

        x_with_params = _embed_params(x_input, params_tensor)
        u_pred = model(x_with_params)

        loss = torch.nn.functional.mse_loss(u_pred, u_target)
        loss.backward()
        optimizer.step()
        clamp_params()

        if verbose and it % 50 == 0:
            print(f"  Inversion iter {it:4d} | Loss = {loss.item():.6e}")

    result = {}
    for i, name in enumerate(param_names):
        result[name] = params_tensor[i].item()

    if verbose:
        print(f"  Final loss = {loss.item():.6e}")
        for name, val in result.items():
            print(f"    {name} = {val:.4f}")

    return result


def invert_parameters_multi(
    model: nn.Module,
    u_target: torch.Tensor,
    x_input: torch.Tensor,
    param_names: List[str],
    param_init: Dict[str, float],
    param_bounds: Dict[str, List[float]],
    n_iterations: int = 200,
    lr: float = 0.01,
    n_restarts: int = 3,
    verbose: bool = False,
) -> Dict[str, float]:
    """Multi-start parameter inversion.

    Args:
        model: Trained neural operator.
        u_target: Target solution tensor.
        x_input: Base input tensor.
        param_names: Parameter names to optimize.
        param_init: Initial guess.
        param_bounds: Parameter bounds.
        n_iterations: Iterations per restart.
        lr: Learning rate.
        n_restarts: Number of random restarts.
        verbose: Print progress.

    Returns:
        Best parameter dictionary.
    """
    best_loss = float("inf")
    best_params = None

    for restart in range(n_restarts):
        init = dict(param_init)
        for name in param_names:
            lo, hi = param_bounds[name]
            init[name] = float(torch.rand(1).item() * (hi - lo) + lo)

        if verbose:
            print(f"Restart {restart + 1}/{n_restarts}")

        result = invert_parameters(
            model, u_target, x_input, param_names, init, param_bounds,
            n_iterations, lr, verbose=False
        )

        x_inp = _embed_params(x_input, torch.tensor(
            [result[n] for n in param_names], device=u_target.device
        ))
        loss = nn.functional.mse_loss(model(x_inp), u_target).item()

        if loss < best_loss:
            best_loss = loss
            best_params = result

    if verbose:
        params_str = ", ".join(f"{n}={v:.4f}" for n, v in best_params.items())
        print(f"Best: {params_str}, loss={best_loss:.6e}")

    return best_params


def _embed_params(
    x_input: torch.Tensor, params: torch.Tensor
) -> torch.Tensor:
    """Embed parameter values into the input tensor.

    The parameters are concatenated with spatial-temporal coordinates
    and initial conditions, matching the SC-FNO input preparation.

    Args:
        x_input: Base tensor (B, base_channels, *grid_dims).
        params: Parameter tensor (n_params,) or (B, n_params).
    Returns:
        Input tensor with parameters embedded.
    """
    if params.dim() == 1:
        params = params.unsqueeze(0).expand(x_input.shape[0], -1)

    grid_dims = x_input.shape[2:]
    params_expanded = params.unsqueeze(-1).unsqueeze(-1)
    for _ in range(len(grid_dims) - 1):
        params_expanded = params_expanded.unsqueeze(-1)

    params_expanded = params_expanded.expand(-1, -1, *grid_dims)

    return torch.cat([x_input, params_expanded.to(x_input.device)], dim=1)
