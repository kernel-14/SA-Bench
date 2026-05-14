## losses/data_loss.py
"""Data loss (L_u) for SC-FNO experiments.

Implements the primary solution path loss used by all four model variants:
  - FNO:          L_u only
  - SC-FNO:       L_u + L_s
  - FNO-PINN:     L_u + L_Eq
  - SC-FNO-PINN:  L_u + L_s + L_Eq

The loss is the per-sample relative L² norm, averaged over the batch:

    L_u = (1/B) * Σ_b [ ||û_b - u_b||_F / (||u_b||_F + ε) ]

where ||·||_F is the Frobenius norm over all non-batch dimensions and ε=1e-8
guards against division by zero.

This is the standard FNO training loss from Li et al. (2021). Per-sample
normalization (rather than batch-level) treats every trajectory equally
regardless of its amplitude — important when solution magnitudes vary across
the parameter space.

The same formula is exposed as both:
  - ``forward(u_pred, u_true) -> Tensor``: differentiable scalar for backprop
  - ``relative_l2(u_pred, u_true) -> float``: Python float for logging/eval

Tensor shape conventions (shared across all modules):
  - ODEs (ODE1, ODE2):          [B, T]
  - 1D PDEs (PDE1, PDE2, PDE4): [B, T, Sx]
  - 2D PDE (PDE3):              [B, Sx, Sy]

All shapes are handled uniformly by flattening non-batch dimensions before
computing norms — no equation-specific branching is needed.

References:
    - Li et al. (2021): "Fourier Neural Operator for Parametric Partial
      Differential Equations" (https://arxiv.org/abs/2010.08895)
    - SC-FNO paper Section 2.1: SC-FNO loss formulation
    - config.yaml: training.loss_weights.c1 = 1.0 (weight applied by Trainer)
"""

from typing import Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module-level epsilon constant
# ---------------------------------------------------------------------------

#: Small constant added to the denominator to prevent division by zero.
#: Applied when ||u_true||_F is near zero (e.g., PDE4 near phase boundaries).
_EPS: float = 1e-8


# ---------------------------------------------------------------------------
# Shared implementation function
# ---------------------------------------------------------------------------

def _relative_l2_impl(
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
) -> torch.Tensor:
    """Core relative L² computation shared by forward() and relative_l2().

    Computes the per-sample relative L² norm averaged over the batch:

        result = mean_b [ ||û_b - u_b||_2 / (||u_b||_2 + ε) ]

    where the norms are taken over all non-batch dimensions (flattened).

    This function is the single source of truth for the relative L² formula.
    Both ``DataLoss.forward`` and ``DataLoss.relative_l2`` delegate here to
    guarantee that training loss values and reported evaluation metrics are
    always identical.

    Args:
        u_pred: Predicted solution tensor. Shape [B, ...] where B is the
                batch size and ... represents any number of spatial/temporal
                dimensions. Must be float32 and on the same device as u_true.
                Must retain requires_grad if called from forward() during
                training — this function does not call .detach().
        u_true: True solution tensor. Same shape as u_pred. Must be float32.

    Returns:
        Scalar tensor (0-dimensional) containing the mean relative L² loss
        over the batch. Differentiable with respect to u_pred.

    Raises:
        RuntimeError: If u_pred and u_true have different shapes.

    Example:
        >>> u_pred = torch.randn(4, 30, 20)   # [B=4, T=30, Sx=20]
        >>> u_true = torch.randn(4, 30, 20)
        >>> loss = _relative_l2_impl(u_pred, u_true)
        >>> loss.shape   # torch.Size([]) — scalar
        >>> loss.item()  # some positive float
    """
    if u_pred.shape != u_true.shape:
        raise RuntimeError(
            f"Shape mismatch in relative L² computation: "
            f"u_pred.shape={tuple(u_pred.shape)} != "
            f"u_true.shape={tuple(u_true.shape)}. "
            f"Ensure the model output and target tensors have the same shape."
        )

    # Batch size — first dimension is always the batch dimension.
    B: int = u_pred.shape[0]

    # ------------------------------------------------------------------
    # Flatten all non-batch dimensions to get shape [B, D].
    # D = product of all spatial-temporal dimensions.
    # This handles ODEs [B, T], 1D PDEs [B, T, Sx], and PDE3 [B, Sx, Sy]
    # uniformly without any equation-specific branching.
    # ------------------------------------------------------------------
    u_pred_flat: torch.Tensor = u_pred.reshape(B, -1)   # [B, D]
    u_true_flat: torch.Tensor = u_true.reshape(B, -1)   # [B, D]

    # ------------------------------------------------------------------
    # Compute per-sample L² norms along the flattened dimension (dim=1).
    # torch.norm with dim=1 computes the Euclidean (Frobenius) norm over D.
    # ------------------------------------------------------------------
    # Numerator: ||û_b - u_b||_2, shape [B].
    diff_norm: torch.Tensor = torch.norm(
        u_pred_flat - u_true_flat, p=2, dim=1
    )  # [B]

    # Denominator: ||u_b||_2 + ε, shape [B].
    # The epsilon guard prevents division by zero when u_true is near zero.
    true_norm: torch.Tensor = torch.norm(
        u_true_flat, p=2, dim=1
    ) + _EPS  # [B]

    # ------------------------------------------------------------------
    # Per-sample relative L²: shape [B].
    # ------------------------------------------------------------------
    per_sample_rel_l2: torch.Tensor = diff_norm / true_norm  # [B]

    # ------------------------------------------------------------------
    # Average over the batch to get a scalar loss.
    # ------------------------------------------------------------------
    return per_sample_rel_l2.mean()  # scalar tensor


# ---------------------------------------------------------------------------
# DataLoss class
# ---------------------------------------------------------------------------

class DataLoss(nn.Module):
    """Relative L² data loss L_u for SC-FNO training and evaluation.

    Implements the primary solution path loss used by all four model variants
    (FNO, SC-FNO, FNO-PINN, SC-FNO-PINN). The loss is the per-sample relative
    L² norm averaged over the batch:

        L_u = (1/B) * Σ_b [ ||û_b - u_b||_F / (||u_b||_F + ε) ]

    This class has no learnable parameters — it is a pure functional loss.

    Usage in the training pipeline:
        - ``Trainer._compute_total_loss()`` calls ``forward(u_pred, u_true)``
          and scales the result by ``c1`` from ``cfg['training']['loss_weights']``
          (default 1.0 per config.yaml).
        - ``Evaluator.evaluate_forward()`` and ``Inverter._optimize()`` call
          ``relative_l2(u_pred, u_true)`` for metric logging and inversion.

    Attributes:
        (none — this module has no learnable parameters or mutable state)

    Example:
        >>> loss_fn = DataLoss()
        >>> u_pred = torch.randn(4, 30, 20, requires_grad=True)  # [B, T, Sx]
        >>> u_true = torch.randn(4, 30, 20)
        >>> loss = loss_fn(u_pred, u_true)
        >>> loss.backward()   # gradients flow through u_pred
        >>> # For evaluation (no grad):
        >>> rel_l2 = loss_fn.relative_l2(u_pred.detach(), u_true)
        >>> isinstance(rel_l2, float)
        True
    """

    def __init__(self) -> None:
        """Initializes DataLoss.

        No parameters are needed. This module has no learnable weights.
        The only initialization is the parent ``nn.Module.__init__()`` call.
        """
        super().__init__()
        # No learnable parameters — DataLoss is a pure functional loss module.

    def forward(
        self,
        u_pred: torch.Tensor,
        u_true: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the relative L² loss for backpropagation.

        Returns a differentiable scalar tensor so that ``.backward()`` can be
        called on it during training. Gradients flow through ``u_pred`` to the
        model parameters.

        This method is called by ``Trainer._compute_total_loss()`` on every
        mini-batch. The result is scaled by ``c1`` (default 1.0) before being
        added to the total loss.

        Args:
            u_pred: Predicted solution tensor, shape [B, ...]. Must be
                    float32 and on the same device as u_true. Should retain
                    ``requires_grad=True`` (set by the model's forward pass)
                    so that gradients can flow back to model parameters.
            u_true: True solution tensor, same shape as u_pred. Typically
                    loaded from the dataset and does not require gradients.

        Returns:
            Scalar tensor (0-dimensional) containing the mean relative L²
            loss over the batch. Differentiable with respect to u_pred.

        Raises:
            RuntimeError: If u_pred and u_true have different shapes.

        Example:
            >>> loss_fn = DataLoss()
            >>> u_pred = torch.randn(4, 30, 20, requires_grad=True)
            >>> u_true = torch.randn(4, 30, 20)
            >>> loss = loss_fn(u_pred, u_true)
            >>> loss.shape   # torch.Size([])
            >>> loss.backward()   # no error — gradients flow through u_pred
        """
        return _relative_l2_impl(u_pred, u_true)

    def relative_l2(
        self,
        u_pred: Union[torch.Tensor, "numpy.ndarray"],  # type: ignore[name-defined]
        u_true: Union[torch.Tensor, "numpy.ndarray"],  # type: ignore[name-defined]
    ) -> float:
        """Computes the relative L² metric as a Python float for logging/eval.

        Identical formula to ``forward``, but:
          - Wrapped in ``torch.no_grad()`` to avoid building a computation graph.
          - Returns a Python ``float`` (via ``.item()``) rather than a ``Tensor``.
          - Accepts numpy arrays as inputs (converted to torch.Tensor internally).

        This method is called by:
          - ``Evaluator.evaluate_forward()`` for solution quality metrics.
          - ``Evaluator.evaluate_sensitivity()`` for Jacobian quality metrics.
          - ``Inverter._optimize()`` as the inversion objective (via forward).
          - ``evaluation/metrics.py`` ``Metrics.relative_l2()`` delegates here.

        Args:
            u_pred: Predicted solution. Either a ``torch.Tensor`` of shape
                    [B, ...] or a numpy array of the same shape. If a numpy
                    array is provided, it is converted to float32 torch.Tensor
                    on CPU before computation.
            u_true: True solution. Same shape and type constraints as u_pred.

        Returns:
            Python float containing the mean relative L² over the batch.
            Always non-negative. Returns 0.0 if both inputs are identical.

        Raises:
            RuntimeError: If u_pred and u_true have different shapes after
                          type conversion.
            TypeError: If inputs cannot be converted to torch.Tensor.

        Example:
            >>> loss_fn = DataLoss()
            >>> u_pred = torch.randn(4, 30, 20)
            >>> u_true = torch.randn(4, 30, 20)
            >>> rel_l2 = loss_fn.relative_l2(u_pred, u_true)
            >>> isinstance(rel_l2, float)
            True
            >>> # Also works with numpy arrays:
            >>> import numpy as np
            >>> rel_l2_np = loss_fn.relative_l2(np.random.randn(4, 30, 20),
            ...                                  np.random.randn(4, 30, 20))
            >>> isinstance(rel_l2_np, float)
            True
        """
        # ------------------------------------------------------------------
        # Convert numpy arrays to torch.Tensor if necessary.
        # This allows the method to be called from evaluation code that
        # works with numpy arrays (e.g., after .detach().cpu().numpy()).
        # ------------------------------------------------------------------
        u_pred_tensor: torch.Tensor = self._to_tensor(u_pred)
        u_true_tensor: torch.Tensor = self._to_tensor(u_true)

        # ------------------------------------------------------------------
        # Compute the relative L² metric without building a computation graph.
        # torch.no_grad() prevents unnecessary memory allocation for gradients
        # during evaluation.
        # ------------------------------------------------------------------
        with torch.no_grad():
            loss_tensor: torch.Tensor = _relative_l2_impl(
                u_pred_tensor, u_true_tensor
            )

        return float(loss_tensor.item())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_tensor(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],  # type: ignore[name-defined]
    ) -> torch.Tensor:
        """Converts input to a float32 torch.Tensor if it is not already one.

        Handles:
          - ``torch.Tensor``: returned as-is (cast to float32 if needed).
          - ``numpy.ndarray``: converted via ``torch.from_numpy``, then cast.
          - Python list / tuple: converted via ``torch.tensor``.
          - Python scalar (int, float): wrapped in ``torch.tensor``.

        Args:
            x: Input array-like object. May be a torch.Tensor, numpy array,
               Python list, or scalar.

        Returns:
            A float32 ``torch.Tensor`` on CPU. Device transfer (if needed)
            is the responsibility of the caller — this method does not move
            tensors to GPU.

        Raises:
            TypeError: If ``x`` cannot be converted to a torch.Tensor.

        Example:
            >>> loss_fn = DataLoss()
            >>> import numpy as np
            >>> arr = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> t = loss_fn._to_tensor(arr)
            >>> isinstance(t, torch.Tensor)
            True
            >>> t.dtype
            torch.float32
        """
        if isinstance(x, torch.Tensor):
            # Already a tensor — cast to float32 if needed.
            return x.float()

        # Check for numpy array by inspecting the type's module name.
        # This avoids a hard numpy import at module level.
        type_module: str = getattr(type(x), "__module__", "")
        if type_module.startswith("numpy"):
            # numpy.ndarray: use torch.from_numpy for zero-copy when possible.
            # .copy() ensures the array is C-contiguous (required by from_numpy).
            import numpy as np  # pylint: disable=import-outside-toplevel
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x.copy()).float()

        # Python lists, tuples, and scalars.
        try:
            return torch.tensor(x, dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"DataLoss._to_tensor: cannot convert input of type "
                f"'{type(x).__name__}' to torch.Tensor. "
                f"Expected torch.Tensor, numpy.ndarray, list, tuple, or scalar."
            ) from exc
