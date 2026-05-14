## evaluation/metrics.py
"""Evaluation metrics for PDE foundation model assessment.

Implements the two primary metrics described in the paper (Section 4.1):
  - L2 Relative Error (L2RE): standard metric for PDE foundation models
  - Variance-Normalized RMSE (VRMSE): suggested by The Well benchmark [34]

Also provides batch_variance for the ensemble generation experiment (Figure 3)
and a convenience wrapper compute_all returning both metrics as a dict.

All methods are static — the class is a pure namespace for metric utilities.
No state is maintained between calls. All inputs are expected to be float32
tensors on the same device; no device transfers occur inside this module.

Integration points:
  - evaluation/evaluate.py: calls compute_all (Table 1), l2_relative_error
    at specific rollout steps (Table 3), batch_variance (Figure 3)
  - training/train_p2vae.py: calls compute_all during validation
  - training/train_fmt.py: calls l2_relative_error during validation
  - training/finetune.py: calls compute_all during fine-tuning validation
"""

import logging
import warnings
from typing import Dict

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Numerical stability constant: clamps denominators to prevent division by zero.
# Applied in L2RE (zero-field targets) and VRMSE (constant-field targets).
# Value 1e-8 is standard for float32 computations.
_EPS: float = 1e-8


class Metrics:
    """Static utility class for PDE prediction evaluation metrics.

    All methods are static and stateless. No instantiation is required,
    but the class can be instantiated for use as an object attribute in
    trainer and evaluator classes (e.g., self.metrics = Metrics()).

    Supported metrics:
        L2RE:  L2 relative error, standard for PDE foundation models.
        VRMSE: Variance-normalized RMSE, from The Well benchmark.
        batch_variance: Per-pixel variance across ensemble members (Figure 3).
        compute_all: Convenience wrapper returning both L2RE and VRMSE.

    All methods operate on (B, C, H, W) tensors representing a batch of
    spatial field snapshots. Trajectory batches (B, T, C, H, W) must be
    sliced by the caller before passing to these methods.

    Numerical stability:
        - Zero denominators are clamped to _EPS=1e-8 before division.
        - Non-finite results trigger a warning log and return a zero tensor.
        - NaN/Inf inputs are detected and warned about but not silently masked.
    """

    # Class-level epsilon constant for numerical stability.
    # Exposed as a class attribute so callers can inspect or override if needed.
    EPS: float = _EPS

    @staticmethod
    def l2_relative_error(
        pred: Tensor,
        target: Tensor,
        reduce: str = "mean",
    ) -> Tensor:
        """Compute the L2 relative error between predictions and targets.

        Implements the standard PDE foundation model metric (paper Section 4.1):
            L2RE_i = ||pred_i - target_i||_2 / ||target_i||_2

        The norm is computed over all spatial and channel dimensions jointly
        (C, H, W flattened), treating the full multi-physics field as one entity.
        This matches standard PDE benchmark practice (e.g., DPOT, MPP, VICON).

        Args:
            pred: Predicted field tensor of shape (B, C, H, W). Must have the
                same shape as target. dtype: float32. Any device.
            target: Ground truth field tensor of shape (B, C, H, W). Must have
                the same shape as pred. dtype: float32. Same device as pred.
            reduce: Reduction mode for the per-sample L2RE values:
                'mean': return the mean over the batch (scalar tensor).
                    Used by all trainers and evaluators for logging.
                'none': return per-sample L2RE of shape (B,).
                    Used when per-sample breakdown is needed.
                'sum': return the sum over the batch (scalar tensor).
                    Used for accumulating metrics over multiple batches.
                Default: 'mean'.

        Returns:
            Tensor containing the L2 relative error:
                reduce='mean': scalar tensor (shape [])
                reduce='none': tensor of shape (B,)
                reduce='sum': scalar tensor (shape [])
            dtype: float32. Device: same as pred and target.
            Returns torch.zeros(1, device=pred.device) if the result is
            non-finite (with a warning log).

        Raises:
            ValueError: If pred and target have different shapes.
            ValueError: If reduce is not one of 'mean', 'none', 'sum'.
        """
        # --- Input validation ---
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape. "
                f"Got pred.shape={pred.shape}, target.shape={target.shape}."
            )
        if reduce not in ("mean", "none", "sum"):
            raise ValueError(
                f"reduce must be one of 'mean', 'none', 'sum'. Got '{reduce}'."
            )

        # Warn about non-finite inputs (do not silently mask).
        if not torch.isfinite(pred).all():
            warnings.warn(
                "l2_relative_error: pred contains non-finite values (NaN/Inf). "
                "Results may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
        if not torch.isfinite(target).all():
            warnings.warn(
                "l2_relative_error: target contains non-finite values (NaN/Inf). "
                "Results may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )

        b: int = pred.shape[0]

        # Flatten (C, H, W) → single dimension for norm computation.
        # Shape: (B, C*H*W)
        pred_flat: Tensor = pred.reshape(b, -1)
        target_flat: Tensor = target.reshape(b, -1)

        # Compute residual: (B, C*H*W)
        diff_flat: Tensor = pred_flat - target_flat

        # Numerator: ||pred - target||_2 per sample, shape (B,)
        # Using squared sum + sqrt to avoid potential issues with torch.linalg.norm
        # on older PyTorch versions.
        numerator: Tensor = (diff_flat ** 2).sum(dim=-1).sqrt()

        # Denominator: ||target||_2 per sample, shape (B,)
        denominator: Tensor = (target_flat ** 2).sum(dim=-1).sqrt()

        # Clamp denominator to avoid division by zero for zero-field targets.
        # A zero target field (e.g., initial condition) would give L2RE = inf
        # without this guard. The clamped value _EPS=1e-8 is negligible for
        # typical PDE field magnitudes (O(1) to O(100)).
        denominator_clamped: Tensor = torch.clamp(denominator, min=_EPS)

        # Per-sample L2 relative error: shape (B,)
        per_sample_l2re: Tensor = numerator / denominator_clamped

        # Apply reduction.
        result: Tensor
        if reduce == "mean":
            result = per_sample_l2re.mean()
        elif reduce == "sum":
            result = per_sample_l2re.sum()
        else:  # reduce == "none"
            result = per_sample_l2re

        # NaN/Inf guard: return zero with a warning if result is non-finite.
        # This can happen if both pred and target are zero (0/eps is fine,
        # but other edge cases may arise with extreme field values).
        if reduce in ("mean", "sum") and not torch.isfinite(result):
            logger.warning(
                "l2_relative_error: non-finite result detected (%.6f). "
                "Returning 0.0 as fallback. Check input data for anomalies.",
                result.item(),
            )
            return torch.zeros(1, dtype=pred.dtype, device=pred.device).squeeze()

        return result

    @staticmethod
    def vrmse(
        pred: Tensor,
        target: Tensor,
        reduce: str = "mean",
    ) -> Tensor:
        """Compute the variance-normalized RMSE between predictions and targets.

        Implements the VRMSE metric from The Well benchmark (paper Section 4.1,
        reference [34]):
            VRMSE_i = RMSE_i / std(target_i)

        where RMSE and std are both computed over all spatial and channel
        dimensions (C, H, W) jointly per sample.

        The normalization by the target's standard deviation makes the metric
        scale-invariant across different physical quantities (e.g., velocity
        fields with magnitude O(1) vs. pressure fields with magnitude O(100)).
        A VRMSE < 1 indicates the model outperforms a trivial constant predictor.

        Args:
            pred: Predicted field tensor of shape (B, C, H, W). Must have the
                same shape as target. dtype: float32. Any device.
            target: Ground truth field tensor of shape (B, C, H, W). Must have
                the same shape as pred. dtype: float32. Same device as pred.
            reduce: Reduction mode for the per-sample VRMSE values:
                'mean': return the mean over the batch (scalar tensor).
                'none': return per-sample VRMSE of shape (B,).
                'sum': return the sum over the batch (scalar tensor).
                Default: 'mean'.

        Returns:
            Tensor containing the variance-normalized RMSE:
                reduce='mean': scalar tensor (shape [])
                reduce='none': tensor of shape (B,)
                reduce='sum': scalar tensor (shape [])
            dtype: float32. Device: same as pred and target.
            Returns torch.zeros(1, device=pred.device) if the result is
            non-finite (with a warning log).

        Raises:
            ValueError: If pred and target have different shapes.
            ValueError: If reduce is not one of 'mean', 'none', 'sum'.
        """
        # --- Input validation ---
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape. "
                f"Got pred.shape={pred.shape}, target.shape={target.shape}."
            )
        if reduce not in ("mean", "none", "sum"):
            raise ValueError(
                f"reduce must be one of 'mean', 'none', 'sum'. Got '{reduce}'."
            )

        # Warn about non-finite inputs.
        if not torch.isfinite(pred).all():
            warnings.warn(
                "vrmse: pred contains non-finite values (NaN/Inf). "
                "Results may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
        if not torch.isfinite(target).all():
            warnings.warn(
                "vrmse: target contains non-finite values (NaN/Inf). "
                "Results may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )

        b: int = pred.shape[0]

        # Compute residual: (B, C, H, W)
        diff: Tensor = pred - target

        # --- RMSE per sample ---
        # Mean squared error over (C, H, W) jointly: shape (B,)
        # Using reshape to (B, -1) for clarity, then mean over flattened dim.
        diff_flat: Tensor = diff.reshape(b, -1)  # (B, C*H*W)
        mse_per_sample: Tensor = (diff_flat ** 2).mean(dim=-1)  # (B,)

        # RMSE per sample: sqrt of MSE, shape (B,)
        rmse_per_sample: Tensor = mse_per_sample.sqrt()

        # --- Standard deviation of target per sample ---
        # Computed over (C, H, W) jointly using Bessel-corrected std.
        # Reshape to (B, -1) for std computation.
        target_flat: Tensor = target.reshape(b, -1)  # (B, C*H*W)

        # Use unbiased=True (Bessel correction, PyTorch default) for statistical
        # correctness. With typical field sizes (C*H*W = 3*128*128 = 49152),
        # the correction is negligible but correct.
        # Guard against B=1 edge case: std with unbiased=True requires at least
        # 2 elements along the reduction dimension. Since we reduce over C*H*W
        # (not batch), this is always safe as long as C*H*W >= 2.
        std_target: Tensor = target_flat.std(dim=-1, unbiased=True)  # (B,)

        # Clamp std to avoid division by zero for constant-field targets.
        # A constant field (e.g., uniform pressure) has std=0, which would
        # give VRMSE = inf. The clamped value _EPS=1e-8 is negligible for
        # typical PDE field standard deviations (O(0.01) to O(10)).
        std_target_clamped: Tensor = torch.clamp(std_target, min=_EPS)

        # Per-sample VRMSE: shape (B,)
        per_sample_vrmse: Tensor = rmse_per_sample / std_target_clamped

        # Apply reduction.
        result: Tensor
        if reduce == "mean":
            result = per_sample_vrmse.mean()
        elif reduce == "sum":
            result = per_sample_vrmse.sum()
        else:  # reduce == "none"
            result = per_sample_vrmse

        # NaN/Inf guard: return zero with a warning if result is non-finite.
        if reduce in ("mean", "sum") and not torch.isfinite(result):
            logger.warning(
                "vrmse: non-finite result detected (%.6f). "
                "Returning 0.0 as fallback. Check input data for anomalies.",
                result.item(),
            )
            return torch.zeros(1, dtype=pred.dtype, device=pred.device).squeeze()

        return result

    @staticmethod
    def batch_variance(ensemble: Tensor) -> Tensor:
        """Compute the mean per-pixel variance across ensemble members.

        Implements the ensemble diversity metric used in Figure 3 of the paper:
        "Average of batch-wise variation of x_4 ensemble generated at different
        x_3 noise levels k_3."

        The variance is computed across the batch dimension (ensemble members)
        at each spatial location and channel, then averaged over all spatial
        locations and channels to produce a single scalar.

        Math:
            pixel_var[c, h, w] = Var_b(ensemble[b, c, h, w])  over b=0..B-1
            batch_variance = mean(pixel_var)  over all (c, h, w)

        A higher batch_variance indicates more diverse ensemble predictions
        (higher uncertainty). The paper shows this is a decreasing function
        of k_3: k_3=0 (pure noise init) → maximum variance, k_3=1 (clean
        init) → zero variance (all members identical).

        Args:
            ensemble: Ensemble of predictions, shape (B, C, H, W) where B is
                the number of ensemble members (config: ensemble.batch_size=32),
                C=3 (channels), H=W=128 (spatial resolution). Each of the B
                samples is an independent prediction for the same initial
                condition, differing only in the noise realization z ~ N(0, I)
                used for the k_3 initialization.
                dtype: float32. Any device.

        Returns:
            Scalar tensor containing the mean per-pixel variance across all
            ensemble members, averaged over all spatial locations and channels.
            Shape: [] (scalar). dtype: float32. Device: same as ensemble.
            Returns torch.zeros(1, device=ensemble.device) if B=1 (no variance
            from a single sample) or if the result is non-finite.

        Note:
            For B=1, variance is undefined (or zero by convention). This method
            uses unbiased=False (population variance) to avoid NaN for B=1.
            With B=32 (config: ensemble.batch_size=32), the difference between
            biased and unbiased variance is 1/32 ≈ 3%, which is negligible.
        """
        # Warn about non-finite inputs.
        if not torch.isfinite(ensemble).all():
            warnings.warn(
                "batch_variance: ensemble contains non-finite values (NaN/Inf). "
                "Results may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )

        b: int = ensemble.shape[0]

        # Handle degenerate case: single ensemble member has zero variance.
        if b == 1:
            logger.debug(
                "batch_variance: ensemble has B=1. Returning 0.0 (no variance "
                "from a single sample)."
            )
            return torch.zeros(1, dtype=ensemble.dtype, device=ensemble.device).squeeze()

        # Compute per-pixel variance across batch dimension.
        # ensemble shape: (B, C, H, W)
        # pixel_var shape: (C, H, W) — variance over B ensemble members
        # Use unbiased=False (population variance) to avoid NaN for small B
        # and to be consistent with the paper's "batch-wise variation" framing.
        # With B=32, the difference from unbiased=True is 1/32 ≈ 3%.
        pixel_var: Tensor = ensemble.var(dim=0, unbiased=False)  # (C, H, W)

        # Average over all spatial locations and channels.
        # This gives the "average batch-wise variation" shown in Figure 3's y-axis.
        mean_var: Tensor = pixel_var.mean()  # scalar

        # Safety clamp: floating-point arithmetic can produce tiny negatives.
        # Variance is always non-negative by definition.
        mean_var = torch.clamp(mean_var, min=0.0)

        # NaN/Inf guard.
        if not torch.isfinite(mean_var):
            logger.warning(
                "batch_variance: non-finite result detected. "
                "Returning 0.0 as fallback."
            )
            return torch.zeros(1, dtype=ensemble.dtype, device=ensemble.device).squeeze()

        return mean_var

    @staticmethod
    def compute_all(
        pred: Tensor,
        target: Tensor,
    ) -> Dict[str, float]:
        """Compute all scalar metrics and return as a Python dict.

        Convenience wrapper that calls l2_relative_error and vrmse with
        reduce='mean' and converts the results to Python floats. Used by
        trainers and evaluators for logging and checkpoint selection.

        Args:
            pred: Predicted field tensor of shape (B, C, H, W). dtype: float32.
            target: Ground truth field tensor of shape (B, C, H, W). dtype: float32.
                Must have the same shape as pred.

        Returns:
            Dictionary with keys:
                'l2re': Mean L2 relative error over the batch (Python float).
                    From config: evaluation.metrics[0] = 'l2_relative_error'.
                'vrmse': Mean variance-normalized RMSE over the batch (Python float).
                    From config: evaluation.metrics[1] = 'vrmse'.
            Both values are finite Python floats. If either metric computation
            fails (e.g., due to shape mismatch or non-finite inputs), the
            corresponding value is float('nan') and a warning is logged.
        """
        l2re_val: float
        vrmse_val: float

        try:
            l2re_tensor: Tensor = Metrics.l2_relative_error(
                pred=pred,
                target=target,
                reduce="mean",
            )
            l2re_val = float(l2re_tensor.item())
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "compute_all: l2_relative_error failed with error: %s. "
                "Returning nan.",
                exc,
            )
            l2re_val = float("nan")

        try:
            vrmse_tensor: Tensor = Metrics.vrmse(
                pred=pred,
                target=target,
                reduce="mean",
            )
            vrmse_val = float(vrmse_tensor.item())
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "compute_all: vrmse failed with error: %s. "
                "Returning nan.",
                exc,
            )
            vrmse_val = float("nan")

        return {
            "l2re": l2re_val,
            "vrmse": vrmse_val,
        }
