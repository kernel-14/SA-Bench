import torch
from typing import Tuple

def l2_relative_error(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Calculates the L2 Relative Error between a predicted tensor and a target tensor.
    L2RE = ||pred - target||_2 / (||target||_2 + epsilon)

    This metric is commonly used in PDE foundation models to quantify the relative
    difference, making it robust to the scale of the physical quantities.
    The norm is computed over all elements of the input tensors, yielding a single
    scalar L2RE for the entire batch/tensor.

    Args:
        pred (torch.Tensor): The predicted output tensor. Expected shape:
                             (batch_size, channels, H, W) or similar.
        target (torch.Tensor): The ground truth target tensor. Expected to
                               have the same shape as `pred`.
        epsilon (float, optional): A small value added to the denominator to prevent
                                   division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: A scalar tensor representing the L2 Relative Error.
    """
    # Ensure computations are in float32 for numerical stability, especially for norms
    pred_float32 = pred.to(torch.float32)
    target_float32 = target.to(torch.float32)

    # Calculate the L2 norm of the difference
    abs_diff = torch.abs(pred_float32 - target_float32)
    numerator = torch.norm(abs_diff, p=2)

    # Calculate the L2 norm of the target tensor
    denominator = torch.norm(target_float32, p=2)

    # Add epsilon to the denominator for numerical stability
    denominator = denominator + epsilon

    # Calculate the L2 Relative Error
    l2_re = numerator / denominator
    return l2_re


def vrmse(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Calculates the Variance-Normalized Root Mean Square Error (VRMSE).
    VRMSE = RMSE / (std(target) + epsilon)

    VRMSE is the Root Mean Square Error (RMSE) divided by the standard deviation
    of the target values. This normalization makes the error metric independent
    of the absolute scale of the physical quantities, as suggested by [34].
    The mean and standard deviation are computed over all elements of the input tensors,
    yielding a single scalar VRMSE for the entire batch/tensor.

    Args:
        pred (torch.Tensor): The predicted output tensor. Expected shape:
                             (batch_size, channels, H, W) or similar.
        target (torch.Tensor): The ground truth target tensor. Expected to
                               have the same shape as `pred`.
        epsilon (float, optional): A small value added to the denominator to prevent
                                   division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: A scalar tensor representing the Variance-Normalized RMSE.
    """
    # Ensure computations are in float32 for numerical stability
    pred_float32 = pred.to(torch.float32)
    target_float32 = target.to(torch.float32)

    # Calculate Mean Squared Error (MSE)
    squared_diff = torch.pow(pred_float32 - target_float32, 2)
    mse = torch.mean(squared_diff)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = torch.sqrt(mse)

    # Calculate the standard deviation of the target tensor
    # torch.std() computes the standard deviation of all elements by default.
    std_target = torch.std(target_float32)

    # Add epsilon to std_target for numerical stability
    denominator = std_target + epsilon

    # Calculate VRMSE
    vrmse_val = rmse / denominator
    return vrmse_val

