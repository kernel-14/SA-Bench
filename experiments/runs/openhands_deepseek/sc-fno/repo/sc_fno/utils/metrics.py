"""Evaluation metrics used in the paper.

- R² (coefficient of determination)
- Relative L² error
"""

from typing import Dict, Optional, Union

import torch


def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute R² score.

    R² = 1 - SS_res / SS_tot

    Args:
        y_pred: Predicted values.
        y_true: True values.
    Returns:
        R² as a float.
    """
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 1.0
    r2 = 1 - ss_res / ss_tot
    return max(min(r2.item(), 1.0), -20.0)


def relative_l2_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute relative L² error.

    rel_L2 = ||y_pred - y_true||_2 / ||y_true||_2

    Args:
        y_pred: Predicted values.
        y_true: True values.
    Returns:
        Relative L² error as a float.
    """
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    diff_norm = torch.norm(y_true - y_pred, p=2)
    ref_norm = torch.norm(y_true, p=2)
    if ref_norm < 1e-12:
        return float(diff_norm.item())
    return float((diff_norm / ref_norm).item())


def compute_all_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute both R² and relative L² error.

    Args:
        y_pred: Predicted values.
        y_true: True values.
        prefix: String prefix for metric names.
    Returns:
        Dict with metric names and values.
    """
    return {
        f"{prefix}_R2": r2_score(y_pred, y_true),
        f"{prefix}_relative_L2": relative_l2_error(y_pred, y_true),
    }
