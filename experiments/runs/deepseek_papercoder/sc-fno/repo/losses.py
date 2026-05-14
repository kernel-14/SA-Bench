# losses.py
# ============================================================================
# Purpose: Implement sensitivity‑constrained and physics‑informed loss functions
#          for the SC‑FNO framework.  Both classes are stateless and can be
#          called repeatedly during training.
# ============================================================================

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Helper: random point sampling (reimplemented here to avoid circular imports)
# ----------------------------------------------------------------------------
def _sample_random_flat_indices(
    batch_size: int,
    total_points: int,
    n_points: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate random flat indices for each sample in a batch.

    Args:
        batch_size:   Number of samples in the batch.
        total_points: Total number of spatial‑temporal output points per sample.
        n_points:     Number of points to sample (clamped to total_points).
        device:       Target device.

    Returns:
        LongTensor of shape (batch_size, n_points).
    """
    n = min(n_points, total_points)
    indices = torch.stack(
        [torch.randperm(total_points, device=device)[:n] for _ in range(batch_size)],
        dim=0,
    )
    return indices


# ============================================================================
# SensitivityLoss
# ============================================================================
class SensitivityLoss:
    r"""Compute the MSE between predicted and true Jacobians on a random subset.

    Implements the loss term :math:`L_s = \frac{1}{M}\sum_j \|\partial\hat{u}/\partial p - \partial u/\partial p\|^2`,
    where the points are re‑sampled every call to ensure eventual coverage of the
    full domain over multiple epochs.

    Args:
        output_shape:       Tuple describing the spatio‑temporal dimensions of the
                            model output (excluding batch and channel).  For example:
                            (N_time,) for ODEs, (S_x, N_time) for 1D+time PDEs.
        num_sample_points:  Number of random points to select for Jacobian evaluation.
                            Defaults to 200 (as used in the paper).
    """

    def __init__(
        self,
        output_shape: Tuple[int, ...],
        num_sample_points: int = 200,
    ) -> None:
        if num_sample_points <= 0:
            raise ValueError("num_sample_points must be positive.")
        self.num_sample_points = num_sample_points
        self.output_shape = output_shape
        # pre‑compute total number of output points
        self._total_points = 1
        for d in output_shape:
            self._total_points *= d

    def __call__(
        self,
        model: torch.nn.Module,
        u_input: torch.Tensor,
        p: torch.Tensor,
        J_true: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the sensitivity loss for a batch.

        Args:
            model:   The neural operator (e.g., FNO) that takes
                     `(u_input, p, grid)` and returns `u_pred`.
            u_input: Input initial condition segment, shape ``(B, 1, *input_shape)``.
            p:       Physical parameters, shape ``(B, num_params)``.
            J_true:  True Jacobian tensor, shape ``(B, num_params, *output_shape)``.
            grid:    Coordinate grid tensor, shape ``(B, grid_channels, *output_shape)``.
                     (Passed directly to `model.forward`.)

        Returns:
            Scalar loss (averaged over batch, points and parameters).
        """
        batch_size = u_input.shape[0]
        device = u_input.device
        n_params = p.shape[-1]

        # ----- 1. Sample random output points --------------------------------
        indices = _sample_random_flat_indices(
            batch_size, self._total_points, self.num_sample_points, device
        )  # (B, n)

        # ----- 2. Differentiable forward wrapper -----------------------------
        p_grad = p.detach().clone().requires_grad_(True)

        def f(p_in: torch.Tensor) -> torch.Tensor:
            u_out = model(u_input, p_in, grid)        # (B, 1, *out_shape)
            return u_out.reshape(batch_size, -1).gather(1, indices)  # (B, n)

        # ----- 3. Compute predicted Jacobian at the sampled points -----------
        # Prefer torch.func (>=2.0) for vectorised performance, otherwise fall
        # back to a per‑sample loop with torch.autograd.functional.jacobian.
        if hasattr(torch.func, "jacrev"):
            J_pred = torch.func.jacrev(f)(p_grad)      # (B, n, num_params)
        else:
            # Fallback: loop over batch (slower but functionally correct)
            J_pred_list: List[torch.Tensor] = []
            for i in range(batch_size):
                def _single(p_single: torch.Tensor) -> torch.Tensor:
                    u_single = model(
                        u_input[i:i + 1], p_single.unsqueeze(0), grid[i:i + 1]
                    )
                    return u_single.reshape(1, -1)[0, indices[i]]  # (n,)
                # jacobian returns (n, P)
                J_i = torch.autograd.functional.jacobian(_single, p_grad[i])
                J_pred_list.append(J_i.unsqueeze(0))
            J_pred = torch.cat(J_pred_list, dim=0)     # (B, n, P)

        # ----- 4. Gather true Jacobian at the same points --------------------
        # J_true shape: (B, P, *output_shape) -> flatten to (B, P, total_points)
        J_true_flat = J_true.reshape(batch_size, n_params, self._total_points)
        # Use advanced indexing: for each b, select (:, indices[b])
        J_true_sel = torch.stack(
            [
                J_true_flat[b, :, indices[b]].transpose(0, 1)   # (n, P)
                for b in range(batch_size)
            ],
            dim=0,
        )   # (B, n, P)

        # ----- 5. MSE loss ---------------------------------------------------
        return F.mse_loss(J_pred, J_true_sel)


# ============================================================================
# PINNLoss
# ============================================================================
class PINNLoss:
    r"""Physics‑informed loss that penalises violation of the governing PDE,
    initial condition, and (periodic) boundary conditions.

    The implementation relies on the discrete grid output of the FNO and uses
    `torch.gradient` to compute spatial and temporal derivatives.  Random
    interior collocation points are sampled each call.

    Args:
        equation_name:   ``'pde1'``, ``'pde2'``, ``'pde4'``, ``'ode1'``, ``'ode2'``.
        alpha:           Weighting factor for IC/BC losses relative to PDE loss.
        n_interior:      Number of interior collocation points.
        periodic_bc:     Whether periodic boundary conditions apply.
        t_grid:          1D tensor of output time coordinates.
        x_grid:          1D tensor of spatial coordinates (for 1D spatial PDEs).
        dx, dt:          Grid spacings (if `None`, they are inferred from the grids).
        output_shape:    Tuple describing the spatio‑temporal dimensions of the
                         model output, e.g., ``(S_x, N_time)`` for PDE1.
        param_names:     List of parameter names (used to extract the correct
                         components from the parameter vector).
    """

    def __init__(
        self,
        equation_name: str,
        alpha: float = 0.1,
        n_interior: int = 1000,
        periodic_bc: bool = True,
        t_grid: Optional[torch.Tensor] = None,
        x_grid: Optional[torch.Tensor] = None,
        dx: Optional[float] = None,
        dt: Optional[float] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
        param_names: Optional[List[str]] = None,
    ) -> None:
        self.eq = equation_name
        self.alpha = alpha
        self.n_interior = n_interior
        self.periodic = periodic_bc
        self.t_grid = t_grid
        self.x_grid = x_grid
        self.output_shape = output_shape
        self.param_names = param_names

        # Grid spacings
        if dx is None and x_grid is not None and x_grid.numel() > 1:
            dx = (x_grid[-1] - x_grid[0]) / (x_grid.numel() - 1)
        if dt is None and t_grid is not None and t_grid.numel() > 1:
            dt = (t_grid[-1] - t_grid[0]) / (t_grid.numel() - 1)
        self.dx = dx
        self.dt = dt

        # --- Determine axis order and presence of time -------------------------
        eq_lower = equation_name.lower()
        if eq_lower in ("pde1", "pde2", "pde4"):
            # Output layout: (B, 1, S_x, N_time)
            self.has_time = True
            self.space_dim = 2      # spatial axis
            self.time_dim = 3       # time axis
            self.spatial_size = output_shape[0]
            self.time_size = output_shape[1]
        elif eq_lower in ("ode1", "ode2"):
            # Output layout: (B, 1, N_time)
            self.has_time = True
            self.space_dim = None
            self.time_dim = 2       # time is the only spatial dim
            self.time_size = output_shape[0]
        else:
            raise NotImplementedError(f"PINNLoss not implemented for '{equation_name}'.")

        # Pre‑compute a coordinate mesh for collocation point evaluation (t, x).
        if self.has_time and self.space_dim is not None:
            # 2D mesh: shape (Sx, Nt) for both x and t.
            X, T = torch.meshgrid(self.x_grid, self.t_grid, indexing="ij")
            self._X_mesh = X    # (Sx, Nt)
            self._T_mesh = T
        elif self.has_time:   # ODE: 1D mesh for t only.
            self._T_mesh = self.t_grid   # (Nt,)

    # ----------------------------------------------------------------------
    # Equation‑specific PDE residuals (called element‑wise on collocation points)
    # ----------------------------------------------------------------------
    @staticmethod
    def _residual_pde1(
        u: torch.Tensor,
        u_t: torch.Tensor,
        u_x: torch.Tensor,
        u_xx: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Residual of the generalized nonlinear damped wave equation."""
        # p order: c, alpha, beta, gamma, omega
        c, alpha, beta, gamma, omega = p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4]
        return u_t - (c**2) * u_xx - alpha * u - beta * u - gamma * torch.sin(omega * u)

    @staticmethod
    def _residual_pde2(
        u: torch.Tensor,
        u_t: torch.Tensor,
        u_x: torch.Tensor,
        u_xx: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Residual of the forced Burgers' equation."""
        # p order: alpha, gamma, delta, omega
        alpha, gamma, delta, omega = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        return (1.0 / math.pi) * u_t + alpha * u * u_x - gamma * u_xx - delta * torch.sin(omega * t)

    @staticmethod
    def _residual_pde4(
        u: torch.Tensor,
        u_t: torch.Tensor,
        u_xx: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Residual of the Allen‑Cahn equation."""
        # p order: c, alpha, beta, omega, epsilon
        c, alpha, beta, omega, eps = p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4]
        return u_t - eps * u_xx - alpha * u + beta * u**3

    @staticmethod
    def _residual_ode1(
        u: torch.Tensor,
        u_t: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Residual of the composite harmonic oscillator (du/dt - RHS = 0)."""
        alpha, beta, gamma = p[..., 0], p[..., 1], p[..., 2]
        rhs = alpha * torch.sin(alpha * math.pi * t) + beta * torch.cos(beta * math.pi * t)
        return u_t - rhs

    @staticmethod
    def _residual_ode2(
        u: torch.Tensor,        # shape (n_points,)
        u_t: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Residual of the Duffing oscillator (u = position, du/dt must be computed)."""
        # Note: For the Duffing equation we need both position and velocity.
        # The paper's FNO outputs only position, so the PINN loss cannot be
        # computed from the output alone.  We raise an error here.
        raise NotImplementedError("PINN loss for Duffing oscillator requires a state vector (position + velocity).")

    # ----------------------------------------------------------------------
    # Interior collocation point generator
    # ----------------------------------------------------------------------
    def _interior_indices(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return flat indices for interior points (excluding spatial boundaries).

        For PDEs, we exclude index 0 and -1 along the spatial axis.
        For ODEs, all time indices are considered interior (no boundary loss).
        """
        if self.space_dim is not None:   # 2D output (Sx, Nt)
            Sx, Nt = self.output_shape
            # Create a mask: True for interior x, all t
            interior = torch.zeros(Sx, Nt, dtype=torch.bool, device=device)
            interior[1:-1, :] = True
            valid_flat = interior.flatten().nonzero(as_tuple=False).squeeze(-1)  # indices
        else:   # ODE: all indices are interior
            Nt = self.time_size
            valid_flat = torch.arange(Nt, device=device)
        # Sample self.n_interior points without replacement
        n = min(self.n_interior, valid_flat.numel())
        perm = torch.randperm(valid_flat.numel(), device=device)[:n]
        indices = valid_flat[perm]
        return indices.unsqueeze(0).expand(batch_size, -1)   # (B, n)

    # ----------------------------------------------------------------------
    # Main forward method
    # ----------------------------------------------------------------------
    def __call__(
        self,
        model: torch.nn.Module,
        u_input: torch.Tensor,
        p: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the total PINN loss (PDE + α·(IC + BC)).

        Args:
            model:   The FNO model.
            u_input: Initial condition segment, shape ``(B, 1, *input_shape)``.
            p:       Physical parameters, ``(B, num_params)``.
            grid:    Coordinate grid for model.

        Returns:
            Scalar loss tensor.
        """
        u_pred = model(u_input, p, grid)   # (B, 1, *output_shape)
        batch_size = u_pred.shape[0]
        device = u_pred.device

        # ---------- 1. Initial condition loss ---------------------------------
        if self.has_time:
            # For PDEs (time_dim = 3): first output time step vs last of u_input.
            # u_input shape: (B, 1, Sx, M) for PDEs, (B, 1, M) for ODEs.
            if self.space_dim is not None:
                # PDE: u_input last time index is -1 on time dim (3 for PDEs? Actually
                #      u_input has its own time dimension; we need to know its layout.
                #      In the dataset, u_input is the initial M steps. For PDE1 it is
                #      shape (Sx, M) but after batch and channel dims: (B, 1, Sx, M).
                #      Time is last dim.
                last_input = u_input[..., -1]          # (B, 1, Sx)
                first_output = u_pred[..., 0]           # (B, 1, Sx)
            else:
                # ODE: u_input shape (B, 1, M), last_input = u_input[:, :, -1]
                last_input = u_input[:, :, -1]          # (B, 1)
                first_output = u_pred[..., 0]            # (B, 1)
            L_IC = F.mse_loss(first_output, last_input)
        else:
            L_IC = torch.tensor(0.0, device=device)

        # ---------- 2. Boundary condition loss (periodic) ----------------------
        if self.space_dim is not None and self.periodic:
            # For layout (B, 1, Sx, Nt): left = [:, :, 0, :], right = [:, :, -1, :]
            left = u_pred[:, :, 0, :]
            right = u_pred[:, :, -1, :]
            L_BC = F.mse_loss(left, right)
        else:
            L_BC = torch.tensor(0.0, device=device)

        # ---------- 3. PDE residual loss ---------------------------------------
        if not self.has_time:
            raise RuntimeError("PINN loss requires a time dimension for the PDE residual.")

        # Sample interior points
        idx_interior = self._interior_indices(batch_size, device)  # (B, n_points)

        # Flatten output for gather
        u_flat = u_pred.reshape(batch_size, 1, -1).squeeze(1)     # (B, total_points)

        # Compute spatial and temporal derivatives using torch.gradient
        if self.space_dim is not None:
            # u_pred shape: (B, 1, Sx, Nt) -> need gradients along dims 2 (x) and 3 (t)
            u_x, u_t = torch.gradient(
                u_pred.squeeze(1),   # (B, Sx, Nt)
                spacing=(self.dx, self.dt),
                dim=(1, 2),
            )
            # Also compute u_xx (second spatial derivative)
            u_xx, _ = torch.gradient(u_x, spacing=self.dx, dim=1)
        else:
            # ODE: shape (B, 1, Nt) -> gradient along dim=2 with spacing dt
            u_t = torch.gradient(u_pred.squeeze(1), spacing=self.dt, dim=1)[0]   # (B, Nt)
            u_xx = None   # not needed

        # Gather sampled values
        gather_fn = lambda t: torch.gather(t.reshape(batch_size, -1), 1, idx_interior)
        u_sel = gather_fn(u_flat)        # (B, n)
        u_t_sel = gather_fn(u_t)         # (B, n)
        if u_xx is not None:
            u_xx_sel = gather_fn(u_xx)   # (B, n)

        # Gather parameter vectors (repeat for each point)
        # p shape (B, P) -> expand to (B, n, P)
        p_expanded = p.unsqueeze(1).expand(-1, idx_interior.shape[1], -1)  # (B, n, P)

        # Gather coordinates (t and optionally x) at collocation points
        if self.space_dim is not None:
            # For PDEs, we need t values and also u_x for Burgers etc.
            # X_mesh, T_mesh are (Sx, Nt); flatten and gather
            X_flat = self._X_mesh.flatten().to(device)   # (total_points,)
            T_flat = self._T_mesh.flatten().to(device)
            x_sel = torch.gather(X_flat.expand(batch_size, -1), 1, idx_interior)  # (B, n)
            t_sel = torch.gather(T_flat.expand(batch_size, -1), 1, idx_interior)
        else:
            # ODE: only time mesh
            T_flat = self._T_mesh.to(device)   # (Nt,)
            t_sel = torch.gather(T_flat.expand(batch_size, -1), 1, idx_interior)

        # Call the appropriate residual function
        eq_lower = self.eq.lower()
        if eq_lower == "pde1":
            residual = self._residual_pde1(u_sel, u_t_sel, gather_fn(u_x), u_xx_sel, p_expanded)
        elif eq_lower == "pde2":
            residual = self._residual_pde2(u_sel, u_t_sel, gather_fn(u_x), u_xx_sel, p_expanded, t_sel)
        elif eq_lower == "pde4":
            residual = self._residual_pde4(u_sel, u_t_sel, u_xx_sel, p_expanded)
        elif eq_lower == "ode1":
            residual = self._residual_ode1(u_sel, u_t_sel, p_expanded, t_sel)
        elif eq_lower == "ode2":
            raise NotImplementedError("PINN loss not supported for ODE2 (requires state vector).")
        else:
            raise ValueError(f"Unknown equation '{self.eq}' for PINN loss.")

        L_PDE = torch.mean(residual**2)

        # ---------- 4. Combine -------------------------------------------------
        total_loss = L_PDE + self.alpha * (L_IC + L_BC)
        return total_loss

