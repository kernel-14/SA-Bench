## evaluation/metrics.py
"""Evaluation metrics for SC-FNO experiments.

Provides the Metrics class implementing R² (coefficient of determination) and
relative L² error — the two primary evaluation metrics reported in every table
of the SC-FNO paper (Tables 1, 2, 3, 4, 5, D.9–D.14).

Both metrics are computed as global scalars over the full test set (all batch,
time, and spatial dimensions flattened together), matching the single-value
reporting convention used throughout the paper.

Key numerical decisions:
  - All non-batch dimensions are flattened to 1D before norm/variance computation.
  - R² uses float64 internally to avoid catastrophic cancellation near R²=0.
  - Negative R² values are allowed (no clamping) — the paper reports values as
    low as -8.332 (Table 4, FNO Jacobians for Allen-Cahn).
  - Division-by-zero guards use epsilon=1e-10 and return 0.0 with a warning.

Usage patterns (from Evaluator):
  - Solution quality:   metrics.compute_all(u_pred, u_true)
  - Jacobian quality:   metrics.compute_all(j_pred[..., i], j_true[..., i])
  - Inversion quality:  metrics.compute_all(p_pred[:, i], p_true[:, i])

All three cases are handled uniformly by flattening to 1D before computation.

References:
    - SC-FNO paper Table 1: R² and Relative L² for PDE1 and PDE2
    - SC-FNO paper Table 3: R²=-3.11 for FNO Jacobians (Allen-Cahn)
    - SC-FNO paper Table D.13: AD vs FD solver comparison metrics
    - SC-FNO paper Figures 1, 2: Inversion scatter plots with R² annotations
"""

import math
import warnings
from typing import Dict, Union

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Module-level epsilon constant
# ---------------------------------------------------------------------------

#: Small constant added to denominators to prevent division by zero.
#: Applied when ||y_true||_2 ≈ 0 (relative L²) or SS_tot ≈ 0 (R²).
_EPS: float = 1e-10


class Metrics:
    """Stateless utility for computing R² and relative L² evaluation metrics.

    Both metrics are computed as global scalars over all elements (batch,
    time, spatial dimensions flattened together), matching the single-value
    reporting convention used in all SC-FNO paper tables.

    This class has no learnable parameters and no mutable state beyond the
    epsilon guard stored at construction. All methods accept torch.Tensor
    or numpy.ndarray inputs and return Python floats.

    Attributes:
        eps: Small constant for numerical stability in division operations.
             Default 1e-10. Sourced from the module-level _EPS constant.

    Example:
        >>> metrics = Metrics()
        >>> u_pred = torch.randn(300, 25, 20)   # [N_test, T_out, Sx] for PDE1
        >>> u_true = torch.randn(300, 25, 20)
        >>> result = metrics.compute_all(u_pred, u_true)
        >>> result.keys()
        dict_keys(['r2', 'relative_l2'])
        >>> isinstance(result['r2'], float)
        True
        >>> # Jacobian slice for one parameter:
        >>> j_pred_alpha = torch.randn(300, 25, 20)   # ∂û/∂α
        >>> j_true_alpha = torch.randn(300, 25, 20)
        >>> jac_metrics = metrics.compute_all(j_pred_alpha, j_true_alpha)
        >>> # Inversion scalar parameter:
        >>> p_pred_alpha = torch.randn(300)   # recovered α values
        >>> p_true_alpha = torch.randn(300)   # true α values
        >>> inv_metrics = metrics.compute_all(p_pred_alpha, p_true_alpha)
    """

    def __init__(self, eps: float = _EPS) -> None:
        """Initializes Metrics with a numerical stability epsilon.

        Args:
            eps: Small constant added to denominators to prevent division by
                 zero. Applied when ||y_true||_2 < eps (relative L²) or when
                 SS_tot < eps (R²). Default 1e-10 is appropriate for float64
                 intermediate computations. Sourced from module-level _EPS.
        """
        self.eps: float = float(eps)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def r2_score(
        self,
        y_pred: Union[torch.Tensor, np.ndarray],
        y_true: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """Computes the R² (coefficient of determination) score.

        Formula:
            R² = 1 - SS_res / SS_tot
            SS_res = Σ (y_true_i - y_pred_i)²
            SS_tot = Σ (y_true_i - mean(y_true))²

        Both sums are taken over ALL elements (batch × time × space flattened
        to 1D), giving a single global scalar. This matches the single-value
        reporting convention in all SC-FNO paper tables.

        Negative R² values are allowed and expected — the paper reports values
        as low as -8.332 (Table 4, FNO Jacobians for Allen-Cahn). No clamping
        is applied.

        Computation is performed in float64 to avoid catastrophic cancellation
        when SS_res ≈ SS_tot (R² near 0), which occurs frequently for FNO
        Jacobian predictions.

        Args:
            y_pred: Predicted values. Any shape [B, ...] or 1D [N]. Accepts
                    torch.Tensor (any dtype) or numpy.ndarray. Converted to
                    float64 internally for numerical stability.
            y_true: True values. Same shape as y_pred. Accepts torch.Tensor
                    or numpy.ndarray.

        Returns:
            R² score as a Python float. Range is (-∞, 1.0]:
              - R²=1.0: perfect prediction
              - R²=0.0: predicting the mean of y_true
              - R²<0.0: worse than predicting the mean (common for FNO Jacobians)
            Returns 0.0 if SS_tot < eps (degenerate case where y_true is
            nearly constant), with a warning logged to stderr.

        Raises:
            RuntimeError: If y_pred and y_true have different shapes after
                          type conversion.

        Example:
            >>> metrics = Metrics()
            >>> y_pred = torch.tensor([1.0, 2.0, 3.0])
            >>> y_true = torch.tensor([1.0, 2.0, 3.0])
            >>> metrics.r2_score(y_pred, y_true)
            1.0
            >>> y_pred_bad = torch.zeros(3)
            >>> metrics.r2_score(y_pred_bad, y_true)  # predicting zero
            # Some negative value
        """
        # Convert inputs to float64 torch tensors for numerical stability.
        pred_f64: torch.Tensor = self._to_float64_tensor(y_pred)
        true_f64: torch.Tensor = self._to_float64_tensor(y_true)

        # Validate shapes match.
        if pred_f64.shape != true_f64.shape:
            raise RuntimeError(
                f"Metrics.r2_score: shape mismatch — "
                f"y_pred.shape={tuple(pred_f64.shape)} != "
                f"y_true.shape={tuple(true_f64.shape)}. "
                f"Ensure predicted and true tensors have the same shape."
            )

        # Flatten all dimensions to 1D for global scalar computation.
        # This handles ODEs [B, T], 1D PDEs [B, T, Sx], PDE3 [B, Sx, Sy],
        # Jacobian slices [B, T, Sx], and inversion scalars [N] uniformly.
        pred_flat: torch.Tensor = pred_f64.reshape(-1)   # [N_total]
        true_flat: torch.Tensor = true_f64.reshape(-1)   # [N_total]

        # Compute global mean of y_true (scalar).
        true_mean: torch.Tensor = true_flat.mean()

        # SS_res = Σ (y_true_i - y_pred_i)²
        ss_res: torch.Tensor = torch.sum((true_flat - pred_flat) ** 2)

        # SS_tot = Σ (y_true_i - mean(y_true))²
        ss_tot: torch.Tensor = torch.sum((true_flat - true_mean) ** 2)

        # Guard against degenerate case where y_true is nearly constant.
        # This can happen for Jacobians near bifurcation points (PDE4) or
        # when a parameter has no effect on the solution.
        if float(ss_tot.item()) < self.eps:
            warnings.warn(
                f"Metrics.r2_score: SS_tot={float(ss_tot.item()):.2e} < eps={self.eps:.2e}. "
                f"y_true appears to be nearly constant. Returning R²=0.0. "
                f"This may indicate a degenerate test case (e.g., zero Jacobian).",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0

        # R² = 1 - SS_res / SS_tot
        r2: float = float((1.0 - ss_res / ss_tot).item())

        return r2

    def relative_l2(
        self,
        y_pred: Union[torch.Tensor, np.ndarray],
        y_true: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """Computes the relative L² error.

        Formula:
            Rel-L² = ||y_pred - y_true||_2 / ||y_true||_2

        Both norms are computed over ALL elements (batch × time × space
        flattened to 1D), giving a single global scalar. This matches the
        single-value reporting convention in all SC-FNO paper tables.

        This is the same formula used in DataLoss.forward() for training,
        ensuring that reported evaluation metrics are directly comparable to
        training loss values. The only difference is that this method returns
        a Python float (not a differentiable tensor) and uses the global
        (not per-sample) norm.

        Args:
            y_pred: Predicted values. Any shape [B, ...] or 1D [N]. Accepts
                    torch.Tensor (any dtype) or numpy.ndarray. Converted to
                    float32 internally (sufficient precision for evaluation).
            y_true: True values. Same shape as y_pred. Accepts torch.Tensor
                    or numpy.ndarray.

        Returns:
            Relative L² error as a Python float. Range is [0, ∞):
              - 0.0: perfect prediction
              - 1.0: prediction has the same L² norm as the error
              - >1.0: error is larger than the signal (common for FNO Jacobians)
            Returns 0.0 if ||y_true||_2 < eps (degenerate case where y_true
            is near zero), with a warning logged to stderr.

        Raises:
            RuntimeError: If y_pred and y_true have different shapes after
                          type conversion.

        Example:
            >>> metrics = Metrics()
            >>> y_pred = torch.tensor([1.0, 2.0, 3.0])
            >>> y_true = torch.tensor([1.0, 2.0, 3.0])
            >>> metrics.relative_l2(y_pred, y_true)
            0.0
            >>> y_pred_bad = torch.zeros(3)
            >>> metrics.relative_l2(y_pred_bad, y_true)
            1.0
        """
        # Convert inputs to float32 torch tensors.
        # float32 is sufficient for relative L² (no catastrophic cancellation risk).
        pred_f32: torch.Tensor = self._to_float32_tensor(y_pred)
        true_f32: torch.Tensor = self._to_float32_tensor(y_true)

        # Validate shapes match.
        if pred_f32.shape != true_f32.shape:
            raise RuntimeError(
                f"Metrics.relative_l2: shape mismatch — "
                f"y_pred.shape={tuple(pred_f32.shape)} != "
                f"y_true.shape={tuple(true_f32.shape)}. "
                f"Ensure predicted and true tensors have the same shape."
            )

        # Flatten all dimensions to 1D for global scalar computation.
        pred_flat: torch.Tensor = pred_f32.reshape(-1)   # [N_total]
        true_flat: torch.Tensor = true_f32.reshape(-1)   # [N_total]

        # Numerator: ||y_pred - y_true||_2
        diff_norm: float = float(
            torch.norm(pred_flat - true_flat, p=2).item()
        )

        # Denominator: ||y_true||_2
        true_norm: float = float(
            torch.norm(true_flat, p=2).item()
        )

        # Guard against degenerate case where y_true is near zero.
        if true_norm < self.eps:
            warnings.warn(
                f"Metrics.relative_l2: ||y_true||_2={true_norm:.2e} < eps={self.eps:.2e}. "
                f"y_true appears to be near zero. Returning relative L²=0.0. "
                f"This may indicate a degenerate test case.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0

        return diff_norm / true_norm

    def compute_all(
        self,
        y_pred: Union[torch.Tensor, np.ndarray],
        y_true: Union[torch.Tensor, np.ndarray],
    ) -> Dict[str, float]:
        """Computes both R² and relative L² metrics for a prediction-truth pair.

        This is the primary interface used by Evaluator when iterating over
        variables (u, ∂u/∂α, ∂u/∂β, etc.) and by Trainer for validation
        metric logging. Returns a dict with both metrics for easy logging
        via Logger.log_dict().

        Args:
            y_pred: Predicted values. Any shape [B, ...] or 1D [N]. Accepts
                    torch.Tensor (any dtype) or numpy.ndarray.
            y_true: True values. Same shape as y_pred. Accepts torch.Tensor
                    or numpy.ndarray.

        Returns:
            Dict with exactly two keys:
              - 'r2':          float, R² score (range (-∞, 1.0])
              - 'relative_l2': float, relative L² error (range [0, ∞))

        Raises:
            RuntimeError: If y_pred and y_true have different shapes.

        Example:
            >>> metrics = Metrics()
            >>> # Solution quality for PDE1 test set
            >>> result = metrics.compute_all(u_pred, u_true)
            >>> result
            {'r2': 0.983, 'relative_l2': 0.017}
            >>> # Jacobian quality for ∂u/∂α
            >>> jac_result = metrics.compute_all(j_pred_alpha, j_true_alpha)
            >>> jac_result
            {'r2': 0.924, 'relative_l2': 0.076}
            >>> # Inversion quality for recovered α values
            >>> inv_result = metrics.compute_all(p_pred_alpha, p_true_alpha)
            >>> inv_result
            {'r2': 0.998, 'relative_l2': 0.035}
        """
        r2_val: float = self.r2_score(y_pred, y_true)
        rel_l2_val: float = self.relative_l2(y_pred, y_true)

        return {
            "r2": r2_val,
            "relative_l2": rel_l2_val,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_float64_tensor(
        self,
        x: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """Converts input to a float64 CPU torch.Tensor.

        Used internally by r2_score() for numerically stable SS computation.
        Float64 prevents catastrophic cancellation when SS_res ≈ SS_tot
        (R² near 0), which is common for FNO Jacobian predictions.

        Handles:
          - torch.Tensor: detached, moved to CPU, cast to float64
          - numpy.ndarray: converted via torch.from_numpy, cast to float64
          - Python list / tuple: converted via torch.tensor, cast to float64
          - Python scalar (int, float): wrapped in torch.tensor, cast to float64

        Args:
            x: Input array-like object. May be a torch.Tensor, numpy array,
               Python list, tuple, or scalar.

        Returns:
            A float64 torch.Tensor on CPU. Detached from any computation graph.

        Raises:
            TypeError: If x cannot be converted to a torch.Tensor.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().to(dtype=torch.float64)

        # Check for numpy array by inspecting the type's module name.
        type_module: str = getattr(type(x), "__module__", "")
        if type_module.startswith("numpy"):
            if isinstance(x, np.ndarray):
                # Use torch.from_numpy for zero-copy when possible.
                # .copy() ensures C-contiguous layout required by from_numpy.
                return torch.from_numpy(x.copy()).to(dtype=torch.float64)

        # Python lists, tuples, and scalars.
        try:
            return torch.tensor(x, dtype=torch.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Metrics._to_float64_tensor: cannot convert input of type "
                f"'{type(x).__name__}' to torch.Tensor. "
                f"Expected torch.Tensor, numpy.ndarray, list, tuple, or scalar."
            ) from exc

    def _to_float32_tensor(
        self,
        x: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """Converts input to a float32 CPU torch.Tensor.

        Used internally by relative_l2() for evaluation metric computation.
        Float32 is sufficient for relative L² (no catastrophic cancellation
        risk since we compute a ratio of norms, not a difference of large
        numbers).

        Handles:
          - torch.Tensor: detached, moved to CPU, cast to float32
          - numpy.ndarray: converted via torch.from_numpy, cast to float32
          - Python list / tuple: converted via torch.tensor, cast to float32
          - Python scalar (int, float): wrapped in torch.tensor, cast to float32

        Args:
            x: Input array-like object. May be a torch.Tensor, numpy array,
               Python list, tuple, or scalar.

        Returns:
            A float32 torch.Tensor on CPU. Detached from any computation graph.

        Raises:
            TypeError: If x cannot be converted to a torch.Tensor.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().to(dtype=torch.float32)

        # Check for numpy array by inspecting the type's module name.
        type_module: str = getattr(type(x), "__module__", "")
        if type_module.startswith("numpy"):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x.copy()).to(dtype=torch.float32)

        # Python lists, tuples, and scalars.
        try:
            return torch.tensor(x, dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Metrics._to_float32_tensor: cannot convert input of type "
                f"'{type(x).__name__}' to torch.Tensor. "
                f"Expected torch.Tensor, numpy.ndarray, list, tuple, or scalar."
            ) from exc
