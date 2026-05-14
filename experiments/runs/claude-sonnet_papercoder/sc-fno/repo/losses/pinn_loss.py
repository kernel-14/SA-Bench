## Code: losses/pinn_loss.py

```python
## losses/pinn_loss.py
"""PINN equation loss (L_Eq) for SC-FNO experiments.

Implements the optional Physics-Informed Neural Network (PINN) equation loss
used by FNO-PINN and SC-FNO-PINN variants (Section 2.2 of the SC-FNO paper).

The loss penalizes violations of the governing differential equation at
randomly sampled collocation points, plus initial and boundary condition terms:

    L_Eq = L_PDE + α·(L_IC + L_BC)
         = (1/N) Σᵢ |N[û(xᵢ,tᵢ); p]|² + α·(L_IC + L_BC)

where N[·] is the PDE operator, α is the weighting factor for IC/BC terms,
and N is the number of collocation points.

Key design decisions:
  - Equation dispatch: a dict maps equation_type → residual method, avoiding
    long if-else chains and making it easy to add new equations.
  - Derivative computation: torch.autograd.grad with create_graph=True so that
    the PINN loss can be backpropagated through to update model weights.
  - PDE3 special case: since the model outputs only the final time step for
    Navier-Stokes, the time-derivative term cannot be computed from the model
    output. The PDE residual is skipped for PDE3 (only IC/BC applied), which
    is consistent with the paper (Table 2 shows no PINN variants for PDE3).
  - Collocation sampling: n_colloc points are sampled randomly per call,
    varying between calls to cover the full domain over training.

Configuration (config.yaml):
  - pinn.n_colloc: 256       (number of collocation points per call)
  - pinn.alpha_weight: 1.0   (weight for IC and BC terms)
  - training.loss_weights.c3: 1.0  (weight applied by Trainer)

References:
    - SC-FNO paper Section 2.2: "The PINN-Loss Equation as an Optional Regularizer"
    - Raissi et al. (2019): "Physics-informed neural networks"
    - config.yaml: pinn.n_colloc, pinn.alpha_weight
"""

import math
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PINNLoss(nn.Module):
    """PINN equation loss L_Eq for FNO-PINN and SC-FNO-PINN training.

    Computes the physics-informed loss by evaluating the PDE residual at
    randomly sampled collocation points and adding initial/boundary condition
    penalties. This is the optional regularizer described in Section 2.2 of
    the SC-FNO paper.

    The loss has no learnable parameters — it is a pure functional loss module.
    All equation-specific logic is encapsulated in private residual methods
    dispatched via a dict keyed by equation_type.

    Attributes:
        equation_type: String identifier for the governing equation.
                       One of: 'ode1', 'ode2', 'pde1', 'pde2', 'pde3', 'pde4'.
                       Sourced from cfg['equation'] in the equation sub-config.
        n_colloc: Number of collocation points sampled per forward call.
                  Sourced from config.yaml key 'pinn.n_colloc' (default 256).
        alpha_weight: Weight for the IC and BC loss terms relative to the PDE
                      residual term. Sourced from config.yaml key
                      'pinn.alpha_weight' (default 1.0).
        _residual_dispatch: Dict mapping equation_type → residual method.

    Example:
        >>> loss_fn = PINNLoss(equation_type='pde1', n_colloc=256, alpha_weight=1.0)
        >>> L_eq = loss_fn.forward(model, params, u0, coords)
        >>> L_eq.backward()   # gradients flow to model.parameters()
    """

    def __init__(
        self,
        equation_type: str = "pde1",
        n_colloc: int = 256,
        alpha_weight: float = 1.0,
    ) -> None:
        """Initializes PINNLoss.

        Args:
            equation_type: String identifier for the governing equation.
                           One of: 'ode1', 'ode2', 'pde1', 'pde2', 'pde3',
                           'pde4'. Determines which _pde_residual_* method
                           is called in forward(). Sourced from cfg['equation']
                           in the equation sub-config.
            n_colloc: Number of collocation points to sample per forward call.
                      Sourced from config.yaml key 'pinn.n_colloc' (default 256).
                      Clamped to the total grid size at runtime if the grid is
                      smaller than this value.
            alpha_weight: Weight for the IC and BC loss terms:
                          L_Eq = L_PDE + alpha_weight * (L_IC + L_BC).
                          Sourced from config.yaml key 'pinn.alpha_weight'
                          (default 1.0). Set to 0.0 to disable IC/BC terms.

        Raises:
            ValueError: If equation_type is not one of the supported values.
        """
        super().__init__()

        # Validate equation type.
        valid_equations = {"ode1", "ode2", "pde1", "pde2", "pde3", "pde4"}
        eq_lower: str = str(equation_type).lower().strip()
        if eq_lower not in valid_equations:
            raise ValueError(
                f"PINNLoss: unsupported equation_type '{equation_type}'. "
                f"Must be one of {sorted(valid_equations)}. "
                f"Check cfg['equation'] in the equation sub-config."
            )

        self.equation_type: str = eq_lower
        self.n_colloc: int = max(1, int(n_colloc))
        self.alpha_weight: float = float(alpha_weight)

        # Build the dispatch table mapping equation_type → residual method.
        # This avoids long if-else chains and makes it easy to add equations.
        self._residual_dispatch: Dict[str, Callable] = {
            "ode1": self._pde_residual_ode1,
            "ode2": self._pde_residual_ode2,
            "pde1": self._pde_residual_pde1,
            "pde2": self._pde_residual_pde2,
            "pde3": self._pde_residual_pde3,
            "pde4": self._pde_residual_pde4,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        model: nn.Module,
        params: torch.Tensor,
        u0: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the PINN equation loss L_Eq for a mini-batch.

        Full pipeline:
          1. Enable gradient tracking on coords (required for AD derivatives).
          2. Forward pass through the FNO to get û(x, t; p).
          3. Compute PDE residual at n_colloc randomly sampled collocation points.
          4. Compute IC loss: û at t=0 vs stored u0.
          5. Compute BC loss: boundary condition violations.
          6. Assemble: L_Eq = mean(residual²) + alpha_weight * (L_IC + L_BC).

        Args:
            model: The FNO model instance. Must implement forward(params, u0,
                   coords) -> Tensor. The model must NOT detach coords internally
                   so that the AD chain from u_pred back to coords is intact.
            params: Physical parameter tensor, shape [B, n_params]. Float32.
                    Does not need requires_grad for the PINN loss (we only
                    differentiate w.r.t. coords here, not params).
            u0: Initial condition tensor. Shape depends on equation:
                - ODEs:    [B, M] or [B, 1]
                - 1D PDEs: [B, M, Sx]
                - PDE3:    [B, Sx, Sy]
                Float32, same device as params.
            coords: Coordinate grid tensor (shared across batch). Shape:
                - ODEs:    [T_out, 1]
                - 1D PDEs: [T_out, Sx, 2]
                - PDE3:    [Sx, Sy, 2]
                Float32. Will have requires_grad_(True) set internally.
                The original tensor is NOT modified — a clone is used.

        Returns:
            Scalar tensor (0-dimensional) containing L_Eq. Differentiable
            with respect to the FNO model weights (via create_graph=True in
            the AD derivative computations).

        Raises:
            RuntimeError: If params, u0, and coords are on different devices.

        Example:
            >>> loss_fn = PINNLoss('pde1', n_colloc=256, alpha_weight=1.0)
            >>> L_eq = loss_fn.forward(model, params, u0, coords)
            >>> L_eq.shape   # torch.Size([]) — scalar
            >>> L_eq.backward()
        """
        device: torch.device = params.device
        dtype: torch.dtype = params.dtype

        # ------------------------------------------------------------------
        # Step 1: Clone coords and enable gradient tracking.
        # We clone to avoid modifying the original tensor (which is shared
        # across all samples in the batch and across multiple loss calls).
        # requires_grad_(True) is needed so that torch.autograd.grad can
        # compute ∂u/∂t and ∂u/∂x through the model forward pass.
        # ------------------------------------------------------------------
        coords_with_grad: torch.Tensor = (
            coords.to(device=device, dtype=dtype)
            .clone()
            .requires_grad_(True)
        )

        # Ensure u0 is on the correct device.
        u0_dev: torch.Tensor = u0.to(device=device, dtype=dtype)
        params_dev: torch.Tensor = params.to(device=device, dtype=dtype)

        # ------------------------------------------------------------------
        # Step 2: Forward pass through the FNO.
        # Must be done with gradient tracking enabled so that u_pred is a
        # differentiable function of coords_with_grad.
        # ------------------------------------------------------------------
        with torch.enable_grad():
            u_pred: torch.Tensor = model.forward(
                params_dev, u0_dev, coords_with_grad
            )
        # u_pred shape:
        #   ODEs:    [B, T_out]
        #   1D PDEs: [B, T_out, Sx]
        #   PDE3:    [B, Sx, Sy]

        B: int = u_pred.shape[0]

        # ------------------------------------------------------------------
        # Step 3: Compute PDE residual at collocation points.
        # Dispatch to the equation-specific residual method.
        # ------------------------------------------------------------------
        residual_fn: Callable = self._residual_dispatch[self.equation_type]

        with torch.enable_grad():
            residual: torch.Tensor = residual_fn(
                u_pred=u_pred,
                coords=coords_with_grad,
                params=params_dev,
            )
        # residual shape: [B, n_colloc_actual] or scalar 0.0 for PDE3.

        # PDE residual loss: mean squared residual at collocation points.
        if residual.numel() > 0:
            l_pde: torch.Tensor = (residual ** 2).mean()
        else:
            l_pde = torch.tensor(0.0, dtype=dtype, device=device)

        # ------------------------------------------------------------------
        # Step 4: Compute IC loss.
        # ------------------------------------------------------------------
        with torch.enable_grad():
            l_ic: torch.Tensor = self._ic_loss(u_pred=u_pred, u0=u0_dev)

        # ------------------------------------------------------------------
        # Step 5: Compute BC loss.
        # ------------------------------------------------------------------
        with torch.enable_grad():
            l_bc: torch.Tensor = self._bc_loss(u_pred=u_pred)

        # ------------------------------------------------------------------
        # Step 6: Assemble total equation loss.
        # L_Eq = L_PDE + alpha_weight * (L_IC + L_BC)
        # ------------------------------------------------------------------
        l_eq: torch.Tensor = (
            l_pde + self.alpha_weight * (l_ic + l_bc)
        )

        return l_eq

    # ------------------------------------------------------------------
    # Equation-specific PDE residual methods
    # ------------------------------------------------------------------

    def _pde_residual(
        self,
        u_pred: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatcher: calls the equation-specific residual method.

        This method exists for interface completeness per the design spec.
        In practice, forward() calls the dispatch dict directly for clarity.

        Args:
            u_pred: FNO predicted output, shape [B, ...].
            coords: Coordinate grid with requires_grad=True.
            params: Physical parameters, shape [B, n_params].

        Returns:
            Residual tensor at collocation points, shape [B, n_colloc_actual].
        """
        residual_fn: Callable = self._residual_dispatch[self.equation_type]
        return residual_fn(u_pred=u_pred, coords=coords, params=params)

    def _pde_residual_ode1(
        self,
        u_pred: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the PDE residual for ODE1: Composite Harmonic Oscillator.

        Equation (Appendix B):
            du/dt = α·sin(α·π·t) + β·cos(β·π·t)

        Residual at collocation point t_j:
            r_j = ∂û/∂t - α·sin(α·π·t_j) - β·cos(β·π·t_j)

        Args:
            u_pred: Predicted solution, shape [B, T_out].
            coords: Time coordinate grid with requires_grad=True.
                    Shape [T_out, 1] — last dim is the time coordinate.
            params: Physical parameters, shape [B, 3] = [α, β, γ].

        Returns:
            Residual tensor at sampled collocation points, shape [B, n_actual].
        """
        B: int = u_pred.shape[0]
        T_out: int = u_pred.shape[1]
        device: torch.device = u_pred.device
        dtype: torch.dtype = u_pred.dtype

        # Sample collocation indices from the time grid.
        n_actual: int = min(self.n_colloc, T_out)
        indices: torch.Tensor = torch.randperm(T_out, device=device)[:n_actual]

        # Extract time values at collocation points.
        # coords shape: [T_out, 1] — time is the only coordinate.
        t_colloc: torch.Tensor = coords[indices, 0]  # [n_actual]

        # Extract u_pred at collocation points: [B, n_actual].
        u_colloc: torch.Tensor = u_pred[:, indices]  # [B, n_actual]

        # Compute ∂û/∂t at collocation points via AD.
        # We sum over batch and collocation dims to get a scalar, then grad.
        du_dt_full: torch.Tensor = self._compute_time_derivative_1d(
            u=u_pred,
            coords=coords,
            time_dim_idx=0,  # coords shape [T_out, 1], time is dim 0
        )
        # du_dt_full shape: [B, T_out]
        du_dt_colloc: torch.Tensor = du_dt_full[:, indices]  # [B, n_actual]

        # Extract parameters: α = params[:, 0], β = params[:, 1].
        alpha: torch.Tensor = params[:, 0].unsqueeze(1)  # [B, 1]
        beta: torch.Tensor = params[:, 1].unsqueeze(1)   # [B, 1]

        # Broadcast t_colloc over batch: [1, n_actual] → [B, n_actual].
        t_bc: torch.Tensor = t_colloc.unsqueeze(0).expand(B, n_actual)

        # RHS of ODE1: α·sin(α·π·t) + β·cos(β·π·t)
        pi: float = math.pi
        rhs: torch.Tensor = (
            alpha * torch.sin(alpha * pi * t_bc)
            + beta * torch.cos(beta * pi * t_bc)
        )  # [B, n_actual]

        # Residual: ∂û/∂t - RHS
        residual: torch.Tensor = du_dt_colloc - rhs  # [B, n_actual]

        return residual

    def _pde_residual_ode2(
        self,
        u_pred: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the PDE residual for ODE2: Duffing Oscillator.

        Equation (Appendix B):
            ẍ + δ·ẋ + α·x + β·x³ = γ·cos(ω·t)

        Residual at collocation point t_j:
            r_j = ẍ_j + δ·ẋ_j + α·x_j + β·x_j³ - γ·cos(ω·t_j)

        where ẋ = ∂x/∂t and ẍ = ∂²x/∂t².

        Args:
            u_pred: Predicted position x(t), shape [B, T_out].
            coords: Time coordinate grid with requires_grad=True.
                    Shape [T_out, 1].
            params: Physical parameters, shape [B, 7] = [α, β, γ, δ, ω, ε, ζ].

        Returns:
            Residual tensor at sampled collocation points, shape [B, n_actual].
        """
        B: int = u_pred.shape[0]
        T_out: int = u_pred.shape[1]
        device: torch.device = u_pred.device

        # Sample collocation indices.
        n_actual: int = min(self.n_colloc, T_out)
        indices: torch.Tensor = torch.randperm(T_out, device=device)[:n_actual]

        # Time values at collocation points.
        t_colloc: torch.Tensor = coords[indices, 0]  # [n_actual]
        t_bc: torch.Tensor = t_colloc.unsqueeze(0).expand(B, n_actual)  # [B, n_actual]

        # x at collocation points.
        x_colloc: torch.Tensor = u_pred[:, indices]  # [B, n_actual]

        # First time derivative ẋ = ∂x/∂t.
        dx_dt_full: torch.Tensor = self._compute_time_derivative_1d(
            u=u_pred,
            coords=coords,
            time_dim_idx=0,
        )  # [B, T_out]
        dx_dt_colloc: torch.Tensor = dx_dt_full[:, indices]  # [B, n_actual]

        # Second time derivative ẍ = ∂²x/∂t².
        d2x_dt2_full: torch.Tensor = self._compute_time_derivative_1d(
            u=dx_dt_full,
            coords=coords,
            time_dim_idx=0,
        )  # [B, T_out]
        d2x_dt2_colloc: torch.Tensor = d2x_dt2_full[:, indices]  # [B, n_actual]

        # Extract parameters.
        alpha: torch.Tensor = params[:, 0].unsqueeze(1)  # [B, 1]
        beta: torch.Tensor = params[:, 1].unsqueeze(1)   # [B, 1]
        gamma: torch.Tensor = params[:, 2].unsqueeze(1)  # [B, 1]
        delta: torch.Tensor = params[:, 3].unsqueeze(1)  # [B, 1]
        omega: torch.Tensor = params[:, 4].unsqueeze(1)  # [B, 1]

        # Residual: ẍ + δ·ẋ + α·x + β·x³ - γ·cos(ω·t)
        residual: torch.Tensor = (
            d2x_dt2_colloc
            + delta * dx_dt_colloc
            + alpha * x_colloc
            + beta * x_colloc ** 3
            - gamma * torch.cos(omega * t_bc)
        )  # [B, n_actual]

        return residual

    def _pde_residual_pde1(
        self,
        u_pred: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the PDE residual for PDE1: Generalized Nonlinear Damped Wave.

        Equation (Appendix B):
            ∂²u/∂t² = c²·∂²u/∂x² + α·∂u/∂t + β·u + γ·sin(ω·u)

        Residual at collocation point (x_j, t_j):
            r_j = ∂²û/∂t² - c²·∂²û/∂x² - α·∂û/∂t - β·û - γ·sin(ω·û)

        Args:
            u_pred: Predicted solution, shape [B, T_out, Sx].
            coords: Spatiotemporal coordinate grid with requires_grad=True.
                    Shape [T_out, Sx, 2] — last dim is [x, t].
            params: Physical parameters, shape [B, 5] = [c, α, β, γ, ω].

        Returns:
            Residual tensor at sampled collocation points, shape [B, n_actual].
        """
        B: int = u_pred.shape[0]
        T_out: int = u_pred.shape[1]
        Sx: int = u_pred.shape[2]
        device: torch.device = u_pred.device
        dtype: torch.dtype = u_pred.dtype

        total_points: int = T_out * Sx
        n_actual: int = min(self.n_colloc, total_points)
        indices: torch.Tensor = torch.randperm(total_points, device=device)[:n_actual]

        # Compute spatial and temporal derivatives over the full grid.
        # coords shape: [T_out, Sx, 2] — dim 2 is [x_coord, t_coord].
        # x is index 0, t is index 1 in the last dimension.
        du_dt_full: torch.Tensor = self._compute_grad_2d(
            u=u_pred,
            coords=coords,
            coord_idx=1,  # t is the second coordinate (index 1)
        )  # [B, T_out, Sx]

        du_dx_full: torch.Tensor = self._compute_grad_2d(
            u=u_pred,
            coords=coords,
            coord_idx=0,  # x is the first coordinate (index 0)
        )  # [B, T_out, Sx]

        d2u_dt2_full: torch.Tensor = self._compute_grad_2d(
            u=du_dt_full,
            coords=coords,
            coord_idx=1,
        )  # [B, T_out, Sx]

        d2u_dx2_full: torch.Tensor = self._compute_grad_2d(
            u=du_dx_full,
            coords=coords,
            coord_idx=0,
        )  # [B, T_out, Sx]

        # Flatten spatial-temporal dims for indexing: [B, T_out*Sx].
        u_flat: torch.Tensor = u_pred.reshape(B, total_points)
        du_dt_flat: torch.Tensor = du_dt_full.reshape(B, total_points)
        d2u_dt2_flat: torch.Tensor = d2u_dt2_full.reshape(B, total_points)
        d2u_dx2_flat: torch.Tensor = d2u_dx2_full.reshape(B, total_points)

        # Extract values at collocation points.
        u_c: torch.Tensor = u_flat[:, indices]          # [B, n_actual]
        du_dt_c: torch.Tensor = du_dt_flat[:, indices]  # [B, n_actual]
        d2u_dt2_c: torch.Tensor = d2u_dt2_flat[:, indices]  # [B, n_actual]
        d2u_dx2_c: torch.Tensor = d2u_dx2_flat[:, indices]  # [B, n_actual]

        # Extract parameters: [c, α, β, γ, ω].
        c: torch.Tensor = params[:, 0].unsqueeze(1)      # [B, 1]
        alpha: torch.Tensor = params[:, 1].unsqueeze(1)  # [B, 1]
        beta: torch.Tensor = params[:, 2].unsqueeze(1)   # [B, 1]
        gamma: torch.Tensor = params[:, 3].unsqueeze(1)  # [B, 1]
        omega: torch.Tensor = params[:, 4].unsqueeze(1)  # [B, 1]

        # Residual: ∂²û/∂t² - c²·∂²û/∂x² - α·∂û/∂t - β·û - γ·sin(ω·û)
        residual: torch.Tensor = (
            d2u_dt2_c
            - c ** 2 * d2u_dx2_c
            - alpha * du_dt_c
            - beta * u_c
            - gamma * torch.sin(omega * u_c)
        )  # [B, n_actual]

        return residual

    def _pde_residual_pde2(
        self,
        u_pred: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the PDE residual for PDE2: Forced Burgers' Equation.

        Equation (Appendix B):
            (1/π)·∂u/∂t + α·u·∂u/∂x = γ·∂²u/∂x² + δ·sin(ω·t)

        Residual at collocation point (x_j, t_j):
            r_j = (1/π)·∂û/∂t + α·û·∂û/∂x - γ·∂²û/∂x² - δ·sin(ω·t_j)

        Handles both standard (4 params) and zoned (82 params) cases.
        For the zoned case, α and δ are spatially varying — each spatial
        point uses the parameter from its corresponding zone.

        Args:
            u_pred: Predicted solution, shape [B, T_out, Sx].
            coords: Spatiotemporal coordinate grid with requires_grad=True.
                    Shape [T_out, Sx, 2] — last dim is [x_coord, t_coord].
            params: Physical parameters. Shape [B, 4] for standard case
                    ([α, γ, δ, ω]) or [B, 82] for zoned case.

        Returns:
            Residual tensor at sampled collocation points, shape [B, n_actual].
        """
        B: int = u_pred.shape[0]
        T_out: int = u_pred.shape[1]
        Sx: int = u_pred.shape[2]
        device: torch.device = u_pred.device
        dtype: torch.dtype = u_pred.dtype

        total_points: int = T_out * Sx
        n_actual: int = min(self.n_colloc, total_points)
        indices: torch.Tensor = torch.randperm(total_points, device=device)[:n_actual]

        # Compute spatial and temporal derivatives over the full grid.
        # coords shape: [T_out, Sx, 2] — [x_coord, t_coord].
        du_dt_full: torch.Tensor = self._compute_grad_2d(
            u=u_pred, coords=coords, coord_idx=1
        )  # [B, T_out, Sx]

        du_dx_full: torch.Tensor = self._compute_grad_2d(
            u=u_pred, coords=coords, coord_idx=0
        )  # [B, T_out, Sx]

        d2u_dx2_full: torch.Tensor = self._compute_grad_2d(
            u=du_dx_full, coords=coords, coord_idx=0
        )  # [B, T_out, Sx]

        # Flatten for indexing: [B, T_out*Sx].
        u_flat: torch.Tensor = u_pred.reshape(B, total_points)
        du_dt_flat: torch.Tensor = du_dt_full.reshape(B, total_points)
        du_dx_flat: torch.Tensor = du_dx_full.reshape(B, total_points)
        d2u_dx2_flat: torch.Tensor = d2u_dx2_full.reshape(B, total_points)

        # Extract values at collocation points.
        u_c: torch.Tensor = u_flat[:, indices]           # [B, n_actual]
        du_dt_c: torch.Tensor = du_dt_flat[:, indices]   # [B, n_actual]
        du_dx_c: torch.Tensor = du_dx_flat[:, indices]   # [B, n_actual]
        d2u_dx2_c: torch.Tensor = d2u_dx2_flat[:, indices]  # [B, n_actual]

        # Determine if this is the zoned variant (82 params) or standard (4 params).
        n_params_actual: int = params.shape[1]
        is_zoned: bool = (n_params_actual > 4)

        # Compute spatial indices for each collocation point (for zoned case).
        # indices are flat indices into [T_out, Sx] grid.
        # spatial_idx = indices % Sx gives the spatial zone index.
        spatial_idx: torch.Tensor = indices % Sx  # [n_actual], values in [0, Sx)

        if is_zoned:
            # Zoned case: params layout [α_0,...,α_39, δ_0,...,δ_39, γ, ω].
            n_zones: int = Sx  # One zone per spatial point.

            # Per-zone α: params[:, :n_zones], shape [B, n_zones].
            # Index by spatial position: alpha_c[b, j] = params[b, spatial_idx[