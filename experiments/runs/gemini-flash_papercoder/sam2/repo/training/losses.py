"""
training/losses.py

This module implements the various loss functions used for training the SAM 2 model,
as described in the paper. It includes Focal Loss and Dice Loss for mask prediction,
L1 Loss for IoU prediction, and Cross-Entropy Loss for occlusion prediction.
"""

import torch
import torch.nn.functional as F
from typing import Optional


class Losses:
    """
    A collection of static methods for calculating various loss functions
    used in SAM 2 training.
    """

    @staticmethod
    def focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """
        Computes the Focal Loss.

        Args:
            inputs (torch.Tensor): Raw logits from the model's mask prediction head.
                                   Expected shape (N, H, W) or (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks.
                                    Expected shape (N, H, W) or (N, 1, H, W).
            alpha (float): Weighting factor for positive/negative examples.
                           Defaults to 0.25.
            gamma (float): Focusing parameter. Defaults to 2.0.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. 'mean' by default.

        Returns:
            torch.Tensor: The computed focal loss.
        """
        # Ensure inputs and targets are float and match dimensions
        inputs = inputs.flatten(1)  # (N, H*W)
        targets = targets.flatten(1).float()  # (N, H*W)

        # Compute Binary Cross-Entropy with Logits (per-element)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate pt, the probability of the true class
        # pt = p_t, where p_t = p if y=1 else 1-p
        # For BCE_loss: - (y * log(p) + (1-y) * log(1-p))
        # log(p) = -BCE_loss if y=1
        # log(1-p) = -BCE_loss if y=0
        # So, pt = exp(log(p)) = exp(-BCE_loss / target) IF target is 1
        # Simplified: Use sigmoid on inputs to get probabilities, then mask with targets.
        probs = torch.sigmoid(inputs)
        pt = targets * probs + (1 - targets) * (1 - probs)

        # Compute alpha_factor
        alpha_factor = torch.ones_like(targets) * alpha
        alpha_factor = torch.where(targets == 1, alpha_factor, 1 - alpha_factor)

        # Compute modulating factor
        modulating_factor = (1.0 - pt) ** gamma

        # Combine
        focal_loss_val = alpha_factor * modulating_factor * bce_loss

        if reduction == 'mean':
            return focal_loss_val.mean()
        elif reduction == 'sum':
            return focal_loss_val.sum()
        elif reduction == 'none':
            return focal_loss_val
        else:
            raise ValueError(f"Reduction method '{reduction}' not supported.")

    @staticmethod
    def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        epsilon: float = 1e-6,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """
        Computes the Dice Loss.

        Args:
            inputs (torch.Tensor): Raw logits from the model's mask prediction head.
                                   Expected shape (N, H, W) or (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks.
                                    Expected shape (N, H, W) or (N, 1, H, W).
            epsilon (float): A small constant added to the denominator for numerical stability.
                             Defaults to 1e-6.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. 'mean' by default.

        Returns:
            torch.Tensor: The computed dice loss.
        """
        # Ensure inputs and targets are float
        inputs = inputs.flatten(1)  # (N, H*W)
        targets = targets.flatten(1).float()  # (N, H*W)

        # Apply sigmoid to convert logits to probabilities
        inputs = torch.sigmoid(inputs)

        # Calculate intersection and union
        intersection = (inputs * targets).sum(dim=-1)
        union = inputs.sum(dim=-1) + targets.sum(dim=-1)

        # Compute Dice coefficient
        dice_coefficient = (2.0 * intersection + epsilon) / (union + epsilon)
        dice_loss_val = 1.0 - dice_coefficient

        if reduction == 'mean':
            return dice_loss_val.mean()
        elif reduction == 'sum':
            return dice_loss_val.sum()
        elif reduction == 'none':
            return dice_loss_val
        else:
            raise ValueError(f"Reduction method '{reduction}' not supported.")

    @staticmethod
    def l1_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """
        Computes the L1 Loss (Mean Absolute Error).

        Args:
            inputs (torch.Tensor): Predicted IoU scores from the model.
                                   Expected to be in range [0, 1] (after sigmoid).
            targets (torch.Tensor): Ground truth IoU scores.
                                    Expected to be in range [0, 1].
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. 'mean' by default.

        Returns:
            torch.Tensor: The computed L1 loss.
        """
        # Ensure inputs and targets are float
        inputs = inputs.float()
        targets = targets.float()
        
        return F.l1_loss(inputs, targets, reduction=reduction)

    @staticmethod
    def cross_entropy_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """
        Computes the Binary Cross-Entropy Loss with Logits.

        Args:
            inputs (torch.Tensor): Raw logits from the model's occlusion prediction head.
                                   Expected shape (N, 1) or (N,).
            targets (torch.Tensor): Ground truth binary labels (0 or 1).
                                    Expected shape (N, 1) or (N,).
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. 'mean' by default.

        Returns:
            torch.Tensor: The computed BCE loss.
        """
        # Ensure targets are float for F.binary_cross_entropy_with_logits
        targets = targets.float()
        return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)

