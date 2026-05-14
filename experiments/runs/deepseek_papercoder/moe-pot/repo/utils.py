## utils.py
"""Static utility functions shared across the MoE‑POT project.

This module provides:
    * Reproducibility setup (``set_seed``).
    * Masked MSE loss for heterogeneous PDE datasets.
    * L2 relative error computation for evaluation.
    * Load‑balancing auxiliary loss for Mixture‑of‑Experts routers.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from torch import Tensor


class Utils:
    """Collection of stateless helper functions.

    All methods are static; this class acts purely as a namespace.
    """

    @staticmethod
    def set_seed(seed: int) -> None:
        """Make the experiment reproducible by seeding random number generators.

        Affects Python, NumPy, PyTorch (CPU and all visible GPUs), and
        cuDNN (deterministic mode, no benchmarking).

        Args:
            seed: integer seed for all RNGs. Typically ``42``.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Force deterministic CUDA operations (may slightly reduce performance).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def masked_mse(pred: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
        """Compute the mean squared error only on valid (mask > 0) elements.

        Each element of the mask indicates whether the corresponding
        position/channel is valid (1) or should be ignored (0). This
        allows a single loss function across datasets with different
        numbers of physical channels and irregular spatial domains.

        Args:
            pred: Predicted tensor of shape ``(B, C, H, W)``.
            target: Ground‑truth tensor, same shape as ``pred``.
            mask: Binary tensor broadcastable to ``pred``, typically of
                shape ``(B, C, H, W)``, ``(B, 1, H, W)``, or
                ``(B, H, W)``.

        Returns:
            A scalar tensor containing the masked MSE. If the total
            mask sum is zero (no valid elements), returns 0.0.

        Raises:
            ValueError: if ``pred``, ``target``, and ``mask`` cannot be
                broadcast to a common shape.
        """
        # Ensure all tensors are on the same device and have compatible shapes
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            )

        # Expand mask to full shape if needed (e.g., (B,1,H,W) -> (B,C,H,W))
        mask = mask.expand_as(pred).to(pred.dtype)

        diff = pred - target
        squared_error = diff ** 2
        masked_squared = squared_error * mask

        # Sum over all dimensions except possibly batch (depends on caller,
        # but we compute total sum and divide by total valid count).
        total_valid = mask.sum()
        if total_valid == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        loss = masked_squared.sum() / (total_valid + eps)
        return loss

    @staticmethod
    def l2_relative_error(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
        """Return the L2 relative error between prediction and target.

        The error is computed as
        :math:`\\| pred - target \\|_2 / (\\| target \\|_2 + \\epsilon)`.

        Args:
            pred: Predicted tensor (arbitrary shape).
            target: Ground‑truth tensor (same shape as ``pred``).

        Returns:
            Scalar tensor with the relative error.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            )

        diff = pred - target
        norm_diff: Tensor = torch.linalg.vector_norm(diff, ord=2)
        norm_target: Tensor = torch.linalg.vector_norm(target, ord=2)

        return norm_diff / (norm_target + eps)

    @staticmethod
    def compute_load_balancing_loss(
        routing_weights: Tensor,
        balance_weight: float = 0.1,
        eps: float = 1e-8,
    ) -> Tensor:
        """Calculate the load‑balancing auxiliary loss for one MoE layer.

        The importance of each routed expert is defined as the sum of
        its routing weights over the batch. The loss is proportional to
        the squared coefficient of variation of these importances,
        encouraging uniform expert usage.

        Args:
            routing_weights: Tensor of shape ``(batch, N_routed)``
                containing the full softmax gating probabilities
                **before** top‑K masking.
            balance_weight: Multiplier for the auxiliary loss (default
                matches ``config.load_balance_weight``, i.e., 0.1).
            eps: Small constant to avoid division by zero.

        Returns:
            A scalar tensor representing the per‑layer balancing loss.

        Note:
            This function is typically called for every MoE layer and
            the results are summed before being added to the main task
            loss.
        """
        # Compute importance of each expert over the batch
        importance: Tensor = routing_weights.sum(dim=0)  # (N_routed,)

        expert_count = importance.numel()
        if expert_count == 0:
            return torch.tensor(0.0, device=routing_weights.device, dtype=routing_weights.dtype)

        importance_mean = importance.mean()
        # Use unbiased=False to obtain the empirical standard deviation
        # (this matches the paper’s reported CV computation).
        importance_var = torch.var(importance, unbiased=False)
        importance_std = torch.sqrt(importance_var + eps)

        # Coefficient of variation
        cv = importance_std / (importance_mean + eps)

        # Loss = w_bal * CV^2
        loss = balance_weight * (cv ** 2)
        return loss
