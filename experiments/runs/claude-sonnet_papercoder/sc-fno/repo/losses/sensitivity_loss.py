## losses/sensitivity_loss.py
"""Sensitivity loss (L_s) for SC-FNO experiments.

Implements the core contribution of the SC-FNO paper: a sensitivity-based
regularization loss that supervises the Jacobian ∂û/∂p of the FNO's predicted
solution with respect to the physical parameters p.

Mathematical definition (Paper Section 2.1):

    L_s = (1/M) * Σ_j || ∂û(x_j, t_j; p)/∂p - ∂u(x_j, t_j; p)/∂p ||²

where:
  - ∂û/∂p is the Jacobian of the FNO's predicted output w.r.t. input params,
    computed via automatic differentiation (AD) through the FNO forward pass.
  - ∂u/∂p is the ground-truth Jacobian pre-computed by the differentiable
    numerical solver (or finite differences) during dataset generation.
  - M is the number of sampled spatial-temporal evaluation points.
  - j indexes the randomly sampled subset of grid points.

Key implementation details (Paper Section 2.4):
  - Instead of computing the Jacobian at ALL grid points (expensive), a random
    subset of n_sample_points is drawn per epoch. This subset varies between
    epochs to eventually cover the full solution space.
  - The AD Jacobian requires create_graph=True to allow backpropagation through
    the Jacobian computation (second-order gradients). This is what enables the
    sensitivity loss to update the FNO weights.
  - params must be detached and re-enabled for grad to isolate the Jacobian
    computation from the upstream data-loading graph.

Memory overhead (Paper Section 4):
  - FNO on PDE1: 722 MB; SC-FNO: 764 MB (42 MB overhead from create_graph=True).
  - The computation graph built by create_graph=True is freed after the outer
    loss.backward() call in the Trainer.

Configuration (config.yaml):
  - training.sensitivity_sample_fraction: 0.10  (10% of grid points per epoch)
  - training.sensitivity_sample_max: 256         (hard cap on sample count)
  - training.loss_weights.c2: 1.0               (weight applied by Trainer)

References:
    - SC-FNO paper Section 2.1: Sensitivity loss definition
    - SC-FNO paper Section 2.4: Implementation details (random sampling)
    - SC-FNO paper Algorithm 2: SC-FNO training loop pseudocode
    - SC-FNO paper Table C.8: Training time overhead (SC-FNO vs FNO)
    - config.yaml: training.sensitivity_sample_fraction, training.sensitivity_sample_max
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SensitivityLoss(nn.Module):
    """Sensitivity loss L_s for SC-FNO training.

    Computes the mean squared error between the FNO's predicted Jacobian
    ∂û/∂p (computed via AD) and the pre-computed ground-truth Jacobian
    ∂u/∂p at a randomly sampled subset of spatial-temporal grid points.

    This loss has no learnable parameters — it is a pure functional loss
    module. The only state is ``self.n_sample_points``, which controls the
    number of grid points sampled per forward call.

    The random sampling varies naturally between calls (no fixed seed) so
    that over the course of training, the full solution domain is covered.
    This matches the paper's description: "This sampling varies between
    epochs to eventually cover the full solution space."

    Attributes:
        n_sample_points: Number of spatial-temporal grid points to sample
                         per forward call. Computed by the Trainer as:
                         ``min(int(fraction * total_points), max_points)``
                         where fraction=0.10 and max_points=256 from
                         config.yaml. Clamped to total_points at runtime
                         if the grid is smaller than this value.

    Example:
        >>> loss_fn = SensitivityLoss(n_sample_points=50)
        >>> # During training (called by Trainer._compute_total_loss):
        >>> L_s = loss_fn.forward(
        ...     model=fno_model,
        ...     params=batch['params'],      # [B, n_params]
        ...     u0=batch['u0'],              # [B, M, Sx] for 1D PDEs
        ...     coords=batch['coords'],      # [T_out, Sx, 2] for 1D PDEs
        ...     j_true=batch['jacobians'],   # [B, T_out, Sx, n_params]
        ... )
        >>> L_s.backward()   # second-order gradients flow to FNO weights
    """

    def __init__(self, n_sample_points: int = 50) -> None:
        """Initializes SensitivityLoss.

        Args:
            n_sample_points: Number of spatial-temporal grid points to sample
                             per forward call for the Jacobian comparison.
                             Sourced from config.yaml as:
                               min(int(sensitivity_sample_fraction * total_grid_points),
                                   sensitivity_sample_max)
                             where sensitivity_sample_fraction=0.10 and
                             sensitivity_sample_max=256.
                             Default 50 is a safe fallback for small grids.
                             Clamped to total_points at runtime if the grid
                             is smaller than this value (e.g., ODE1 with
                             T_out=90 and n_sample_points=256 → clamped to 90).
        """
        super().__init__()

        # Validate and store the sample point count.
        if n_sample_points <= 0:
            raise ValueError(
                f"n_sample_points must be a positive integer, got {n_sample_points}. "
                f"Check config.yaml keys 'training.sensitivity_sample_fraction' "
                f"and 'training.sensitivity_sample_max'."
            )

        self.n_sample_points: int = int(n_sample_points)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        model: nn.Module,
        params: torch.Tensor,
        u0: torch.Tensor,
        coords: torch.Tensor,
        j_true: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the sensitivity loss L_s for a mini-batch.

        Full pipeline:
          1. Detach params from any upstream graph and enable requires_grad.
          2. Forward pass through the FNO to get û(x, t; p).
          3. Sample n_sample_points random flat indices into the grid.
          4. Compute the predicted Jacobian ∂û/∂p at sampled points via AD.
          5. Gather the ground-truth Jacobian ∂u/∂p at the same indices.
          6. Return MSE between predicted and true Jacobians.

        The ``create_graph=True`` flag in the AD Jacobian computation is
        essential — it builds a computation graph through the grad call,
        enabling the outer ``loss.backward()`` in the Trainer to propagate
        second-order gradients back to the FNO weights.

        Args:
            model: The FNO model instance. Must implement a ``forward`` method
                   with signature ``forward(params, u0, coords) -> Tensor``.
                   The model's weights are NOT modified here — only the
                   Jacobian w.r.t. ``params`` is computed.
            params: Physical parameter tensor for the batch, shape [B, n_params].
                    Must be float32. Will be detached and re-enabled for grad
                    internally — the original tensor is not modified.
            u0: Initial condition tensor for the batch. Shape depends on equation:
                - ODEs:    [B, M] or [B, 1]
                - 1D PDEs: [B, M, Sx]
                - PDE3:    [B, Sx, Sy]
                Must be float32 and on the same device as params.
            coords: Coordinate grid tensor (shared across batch). Shape depends
                    on equation:
                - ODEs:    [T_out, 1]
                - 1D PDEs: [T_out, Sx, 2]
                - PDE3:    [Sx, Sy, 2]
                Must be float32 and on the same device as params.
            j_true: Pre-computed ground-truth Jacobian tensor, shape:
                - ODEs:    [B, T_out, n_params]
                - 1D PDEs: [B, T_out, Sx, n_params]
                - PDE3:    [B, Sx, Sy, n_params]
                Must be float32 and on the same device as params.
                Must NOT be None — if use_jacobian=False was set in the
                dataset, this method will raise a clear error.

        Returns:
            Scalar tensor (0-dimensional) containing the mean squared error
            between predicted and true Jacobians at the sampled grid points.
            Differentiable with respect to the FNO model weights (via
            create_graph=True in the AD Jacobian computation).

        Raises:
            ValueError: If j_true is None (dataset loaded without Jacobians).
            RuntimeError: If params, u0, coords, and j_true are on different
                          devices, or if tensor shapes are inconsistent.

        Example:
            >>> loss_fn = SensitivityLoss(n_sample_points=50)
            >>> L_s = loss_fn.forward(model, params, u0, coords, j_true)
            >>> L_s.shape   # torch.Size([]) — scalar
            >>> L_s.backward()   # gradients flow to model.parameters()
        """
        # ------------------------------------------------------------------
        # Guard: j_true must be provided.
        # ------------------------------------------------------------------
        if j_true is None:
            raise ValueError(
                "SensitivityLoss.forward: j_true is None. "
                "The dataset was loaded with use_jacobian=False, but the "
                "sensitivity loss requires pre-computed Jacobians. "
                "Re-load the dataset with use_jacobian=True, or disable "
                "the sensitivity loss by using variant='fno' or 'fno_pinn'."
            )

        # ------------------------------------------------------------------
        # Step 1: Detach params from any upstream computation graph and
        # enable requires_grad so that AD can compute ∂û/∂p.
        #
        # We create a new leaf tensor (via .detach()) to avoid polluting the
        # original batch tensor with gradient information. The original
        # params tensor (from the DataLoader) is not modified.
        # ------------------------------------------------------------------
        device: torch.device = params.device
        dtype: torch.dtype = params.dtype

        params_for_jac: torch.Tensor = params.detach().requires_grad_(True)
        # params_for_jac: [B, n_params], leaf tensor with requires_grad=True

        # Ensure u0 and coords are on the correct device and dtype.
        u0_dev: torch.Tensor = u0.to(device=device, dtype=dtype)
        coords_dev: torch.Tensor = coords.to(device=device, dtype=dtype)
        j_true_dev: torch.Tensor = j_true.to(device=device, dtype=dtype)

        # ------------------------------------------------------------------
        # Step 2: Forward pass through the FNO.
        # Must be done with gradient tracking enabled (torch.enable_grad)
        # to build the computation graph for the AD Jacobian computation.
        # ------------------------------------------------------------------
        with torch.enable_grad():
            u_pred: torch.Tensor = model.forward(
                params_for_jac, u0_dev, coords_dev
            )
        # u_pred shape: [B, T_out, Sx] for 1D PDEs, [B, T_out] for ODEs,
        #               [B, Sx, Sy] for PDE3.

        # ------------------------------------------------------------------
        # Step 3: Determine total grid points per sample (excluding batch dim).
        # ------------------------------------------------------------------
        B: int = u_pred.shape[0]
        n_params: int = params_for_jac.shape[1]

        # total_points = product of all non-batch dimensions.
        # For PDE1: T_out * Sx = 25 * 20 = 500
        # For ODE1: T_out = 90
        # For PDE3: Sx * Sy = 64 * 64 = 4096
        total_points: int = u_pred[0].numel()

        # ------------------------------------------------------------------
        # Step 4: Sample random flat indices into the spatial-temporal grid.
        # Clamp to total_points if the grid is smaller than n_sample_points.
        # ------------------------------------------------------------------
        indices: torch.Tensor = self._sample_indices(total_points)
        # indices: [n_actual_sample_points], LongTensor on CPU.
        # Move to the same device as u_pred for indexing.
        indices_dev: torch.Tensor = indices.to(device=device)

        n_actual: int = indices.shape[0]

        # ------------------------------------------------------------------
        # Step 5: Compute the predicted Jacobian ∂û/∂p at sampled points.
        # ------------------------------------------------------------------
        j_pred: torch.Tensor = self._compute_jacobian(
            u_pred=u_pred,
            params=params_for_jac,
            indices=indices_dev,
            B=B,
            n_params=n_params,
            n_actual=n_actual,
            total_points=total_points,
        )
        # j_pred: [B, n_actual, n_params]

        # ------------------------------------------------------------------
        # Step 6: Gather ground-truth Jacobian at the same sampled indices.
        # j_true shape: [B, T_out, Sx, n_params] or [B, T_out, n_params]
        #               or [B, Sx, Sy, n_params].
        # Flatten all non-batch, non-param dims: [B, total_points, n_params].
        # ------------------------------------------------------------------
        j_true_flat: torch.Tensor = j_true_dev.reshape(B, total_points, n_params)
        # j_true_flat: [B, total_points, n_params]

        # Index into the flattened Jacobian at the sampled positions.
        # indices_dev: [n_actual], values in [0, total_points).
        j_true_sampled: torch.Tensor = j_true_flat[:, indices_dev, :]
        # j_true_sampled: [B, n_actual, n_params]

        # ------------------------------------------------------------------
        # Step 7: Compute MSE between predicted and true Jacobians.
        # F.mse_loss averages over all elements (batch, points, params).
        # This matches the paper's L_s = (1/M) * Σ_j ||...||² formulation.
        # ------------------------------------------------------------------
        loss: torch.Tensor = F.mse_loss(j_pred, j_true_sampled.detach())
        # loss: scalar tensor, differentiable w.r.t. model weights via
        # the create_graph=True computation graph built in _compute_jacobian.

        return loss

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_indices(self, total_points: int) -> torch.Tensor:
        """Samples a random subset of flat indices into the spatial-temporal grid.

        Uses ``torch.randperm`` without a fixed seed so that the sampled
        subset varies naturally between calls (epochs). Over the course of
        training, this ensures the full solution domain is covered.

        The number of sampled points is clamped to ``min(n_sample_points,
        total_points)`` to handle grids smaller than the requested sample
        size (e.g., ODE1 with T_out=90 and n_sample_points=256).

        Args:
            total_points: Total number of grid points per sample (product of
                          all non-batch dimensions of u_pred). For PDE1:
                          T_out * Sx = 25 * 20 = 500. For ODE1: T_out = 90.
                          For PDE3: Sx * Sy = 64 * 64 = 4096.

        Returns:
            1D LongTensor of shape [n_actual] on CPU, where
            n_actual = min(self.n_sample_points, total_points).
            Values are unique integers in [0, total_points).

        Example:
            >>> loss_fn = SensitivityLoss(n_sample_points=50)
            >>> indices = loss_fn._sample_indices(total_points=500)
            >>> indices.shape   # torch.Size([50])
            >>> indices.dtype   # torch.int64
            >>> (indices >= 0).all() and (indices < 500).all()
            True
        """
        # Clamp to available points — handles small grids gracefully.
        n_actual: int = min(self.n_sample_points, total_points)

        # torch.randperm generates a random permutation of [0, total_points).
        # Taking the first n_actual elements gives a random subset without
        # replacement. No fixed seed — varies naturally between calls.
        indices: torch.Tensor = torch.randperm(total_points)[:n_actual]
        # indices: [n_actual], LongTensor on CPU.

        return indices

    def _compute_jacobian(
        self,
        u_pred: torch.Tensor,
        params: torch.Tensor,
        indices: torch.Tensor,
        B: int,
        n_params: int,
        n_actual: int,
        total_points: int,
    ) -> torch.Tensor:
        """Computes the predicted Jacobian ∂û/∂p at sampled grid points via AD.

        Uses a loop over sampled points with one-hot ``grad_outputs`` to
        compute per-point gradients. Each iteration computes the VJP
        (vector-Jacobian product) with a one-hot vector selecting one grid
        point, which gives the gradient of that single output w.r.t. all
        input parameters.

        This approach requires ``n_actual`` backward passes through the FNO,
        each cheap (scalar output per batch element). With n_actual ≤ 256
        (from config.yaml ``training.sensitivity_sample_max``), this is
        tractable.

        The ``create_graph=True`` flag is essential — it builds a computation
        graph through the grad call, enabling the outer ``loss.backward()``
        in the Trainer to propagate second-order gradients back to the FNO
        weights. Without this, the sensitivity loss would not update the model.

        The ``retain_graph=True`` flag is used for all but the last iteration
        to prevent the computation graph from being freed mid-loop. The last
        iteration uses ``retain_graph=False`` (default) to allow cleanup.

        Args:
            u_pred: FNO predicted output, shape [B, ...] (any non-batch dims).
                    Must have been computed with params.requires_grad=True so
                    that the computation graph connects u_pred to params.
            params: Leaf parameter tensor with requires_grad=True, shape
                    [B, n_params]. This is the tensor w.r.t. which we
                    differentiate.
            indices: Flat indices of sampled grid points, shape [n_actual].
                     LongTensor on the same device as u_pred.
            B: Batch size (first dimension of u_pred).
            n_params: Number of physical parameters (last dimension of params).
            n_actual: Actual number of sampled points (≤ n_sample_points).
            total_points: Total grid points per sample (product of non-batch dims).

        Returns:
            Predicted Jacobian at sampled points, shape [B, n_actual, n_params].
            Differentiable with respect to the FNO model weights (via
            create_graph=True).

        Note:
            The loop over n_actual points is the bottleneck of SC-FNO training.
            This explains the ~1.5–2.3× training time overhead of SC-FNO over
            FNO (Table C.8). With n_actual ≤ 256, the overhead is acceptable.
        """
        device: torch.device = u_pred.device
        dtype: torch.dtype = u_pred.dtype

        # ------------------------------------------------------------------
        # Flatten u_pred to [B, total_points] for uniform indexing across
        # all equation types (ODE, 1D PDE, 2D PDE).
        # ------------------------------------------------------------------
        u_flat: torch.Tensor = u_pred.reshape(B, total_points)
        # u_flat: [B, total_points], connected to params via computation graph.

        # Extract the sampled outputs: [B, n_actual].
        u_sampled: torch.Tensor = u_flat[:, indices]
        # u_sampled: [B, n_actual], still connected to params.

        # ------------------------------------------------------------------
        # Pre-allocate the output Jacobian tensor.
        # Shape: [B, n_actual, n_params].
        # We fill this in-place during the loop.
        # ------------------------------------------------------------------
        j_pred: torch.Tensor = torch.zeros(
            B, n_actual, n_params,
            dtype=dtype,
            device=device,
        )

        # ------------------------------------------------------------------
        # Loop over sampled points, computing ∂û_j/∂p for each point j.
        #
        # For each j, we use a one-hot grad_outputs vector that selects
        # only the j-th sampled output. The VJP then gives:
        #   grad = Σ_b Σ_j' v[b, j'] * ∂u_sampled[b, j']/∂params[b, :]
        # With v[b, j] = 1 and v[b, j'] = 0 for j' ≠ j, this simplifies to:
        #   grad[b, :] = ∂u_sampled[b, j]/∂params[b, :]
        # which is exactly the j-th row of the Jacobian for each batch element.
        #
        # Note: params is a [B, n_params] tensor, so grad has shape [B, n_params].
        # ------------------------------------------------------------------
        for j in range(n_actual):
            # One-hot grad_outputs: select only the j-th sampled point.
            # Shape: [B, n_actual], all zeros except column j = 1.
            v: torch.Tensor = torch.zeros_like(u_sampled)
            v[:, j] = 1.0

            # Determine whether to retain the graph for subsequent iterations.
            # retain_graph=True for all but the last iteration.
            # On the last iteration, retain_graph is not set (defaults to False
            # when create_graph=False, but since create_graph=True, the graph
            # is retained by default for the outer backward — we still set
            # retain_graph=True to be safe, and let the outer backward clean up).
            is_last: bool = (j == n_actual - 1)

            # Compute the VJP: grad = v^T * J where J = ∂u_sampled/∂params.
            # create_graph=True: builds a graph through this grad call so that
            # the outer loss.backward() can compute ∂L_s/∂(model weights).
            # retain_graph=True: prevents the forward graph from being freed
            # before all n_actual iterations complete.
            grad_j: torch.Tensor = torch.autograd.grad(
                outputs=u_sampled,
                inputs=params,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True,   # Always retain; outer backward cleans up.
                allow_unused=True,   # Some params may not affect u (e.g., c=0 in PDE1).
            )[0]
            # grad_j: [B, n_params] — the j-th row of the Jacobian for each batch element.

            # Handle the case where grad is None (allow_unused=True).
            # This can happen if a parameter has no effect on the output
            # (e.g., c=0 in PDE1 makes the wave speed term vanish).
            if grad_j is None:
                # Zero gradient — parameter has no effect on this output.
                grad_j = torch.zeros(
                    B, n_params,
                    dtype=dtype,
                    device=device,
                )

            # Store the j-th row of the Jacobian.
            # j_pred[:, j, :] = grad_j: [B, n_params]
            j_pred[:, j, :] = grad_j

        # j_pred: [B, n_actual, n_params], differentiable w.r.t. model weights.
        return j_pred
