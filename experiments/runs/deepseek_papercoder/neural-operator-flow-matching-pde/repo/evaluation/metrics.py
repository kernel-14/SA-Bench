## evaluation/metrics.py
"""
Evaluation metrics for physical field predictions.

Provides static methods to compute:
- L2 Relative Error (L2RE)
- Variance-Normalised Root Mean Square Error (VRMSE)
- Rollout errors (step‑wise L2RE for autoregressive predictions)

All methods operate on PyTorch tensors and are stateless.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


class Metrics:
    """Static evaluation metrics for PDE field predictions."""

    @staticmethod
    def l2re(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute the L2 relative error.

        The error is defined as
            L2RE = ||pred - target||₂ / (||target||₂ + ε)
        and averaged over batch, channels, and spatial dimensions.

        Args:
            pred: Predicted field tensor of shape (..., C, H, W).
            target: Ground truth tensor of same shape.
            eps: Small value to avoid division by zero.

        Returns:
            Scalar tensor (mean L2RE over batch, channels, and space).
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Shapes of pred {pred.shape} and target {target.shape} must match."
            )

        # Compute squared L2 norm difference over spatial dimensions (H, W)
        diff = pred - target
        diff_norm2 = torch.sum(diff.pow(2), dim=(-2, -1))          # (..., C)
        target_norm2 = torch.sum(target.pow(2), dim=(-2, -1))     # (..., C)

        # Relative error per sample and channel
        rel = torch.sqrt(diff_norm2 / (target_norm2 + eps))        # (..., C)

        # Mean over all batch/extra dimensions and channels
        return rel.mean()

    @staticmethod
    def vrmse(
        pred: torch.Tensor,
        target: torch.Tensor,
        global_var: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute the Variance-Normalised Root Mean Square Error (VRMSE).

        VRMSE = sqrt( MSE / (global_var + ε) ), where MSE is the mean squared
        error over the spatial and batch dimensions, per channel, and
        global_var is a pre-computed per‑channel global variance of the target
        over the entire test set.

        Args:
            pred: Predicted field (B, C, H, W).
            target: Ground truth field (same shape).
            global_var: Per‑channel variance tensor of shape (C,) or (1, C, 1, 1).
            eps: Small value to avoid division by zero.

        Returns:
            Scalar tensor (VRMSE averaged over channels).
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Shapes of pred {pred.shape} and target {target.shape} must match."
            )
        B, C, H, W = pred.shape
        if global_var.numel() != C:
            raise ValueError(
                f"global_var must have {C} elements, got {global_var.numel()}"
            )

        # Reshape global_var to (1, C, 1, 1) for broadcasting
        if global_var.dim() == 1:
            global_var = global_var.view(1, C, 1, 1)
        elif global_var.dim() == 3:
            global_var = global_var.unsqueeze(0).unsqueeze(-1)
        # else assume it is already (1, C, 1, 1)

        # Compute per‑element MSE, then average over spatial and batch dimensions per channel
        mse = F.mse_loss(pred, target, reduction="none")  # (B, C, H, W)
        mse_per_channel = mse.mean(dim=(0, 2, 3))         # (C,)

        # Normalise by global variance (with epsilon)
        vrmse_per_channel = torch.sqrt(mse_per_channel / (global_var.view(C) + eps))

        # Average over channels
        return vrmse_per_channel.mean()

    @staticmethod
    def rollout_errors(
        pred_seq: torch.Tensor,
        true_seq: torch.Tensor,
        report_steps: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Compute step‑wise L2RE for an autoregressive rollout.

        Args:
            pred_seq: Predicted trajectory (T, B, C, H, W).
            true_seq: Ground truth trajectory (same shape).
            report_steps: List of step indices (1‑based) to include in the result.
                          Defaults to [1, 5, 10] and the last step (marked as "last_step").
                          If a step is larger than T, its value is set to NaN.

        Returns:
            Dictionary mapping step names to scalar errors (floats).
            Includes a "average" key for the L2RE over all T steps.
        """
        if pred_seq.shape != true_seq.shape:
            raise ValueError(
                f"Shapes of pred_seq {pred_seq.shape} and true_seq {true_seq.shape} must match."
            )
        T, B, C, H, W = pred_seq.shape

        # Default steps: 1, 5, 10, last
        if report_steps is None:
            report_steps = [1, 5, 10, -1]  # -1 indicates last step

        errors: Dict[str, float] = {}

        for step in report_steps:
            if step == -1:
                idx = T - 1   # zero‑based index of last step
                key = "last_step"
            else:
                idx = step - 1  # convert to 0‑based
                key = f"step_{step}"

            if 0 <= idx < T:
                err = Metrics.l2re(pred_seq[idx], true_seq[idx])
                errors[key] = err.item()
            else:
                errors[key] = float("nan")

        # Average L2RE over all T steps
        # Flatten batch and time
        pred_all = pred_seq.reshape(T * B, C, H, W)
        true_all = true_seq.reshape(T * B, C, H, W)
        avg_error = Metrics.l2re(pred_all, true_all)
        errors["average"] = avg_error.item()

        return errors
