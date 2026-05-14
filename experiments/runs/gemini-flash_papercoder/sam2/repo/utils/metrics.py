"""
utils/metrics.py

This module provides functions for calculating various segmentation metrics,
including Jaccard Index (IoU), F-measure, and the J&F average metric,
as well as the global G metric used in some video object segmentation benchmarks.
"""

import torch
from typing import List, Tuple

# Constants
EPSILON = torch.tensor(1e-6)  # Small float value to prevent division by zero


def _get_mask_metrics_components(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes True Positives (TP), False Positives (FP), and False Negatives (FN)
    for a single pair of binary masks.

    Args:
        pred_mask (torch.Tensor): A binary tensor representing the predicted
                                   segmentation mask (values 0 or 1, or boolean).
        gt_mask (torch.Tensor): A binary tensor representing the ground truth
                                segmentation mask (values 0 or 1, or boolean).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple (tp, fp, fn),
                                                         where each is a float tensor.
    """
    # Ensure masks are boolean for logical operations
    bool_pred = pred_mask.bool()
    bool_gt = gt_mask.bool()

    # True Positives: pixels where both predicted and ground truth are 1
    tp = (bool_pred & bool_gt).sum().float()
    # False Positives: pixels where predicted is 1 but ground truth is 0
    fp = (bool_pred & ~bool_gt).sum().float()
    # False Negatives: pixels where predicted is 0 but ground truth is 1
    fn = (~bool_pred & bool_gt).sum().float()

    return tp, fp, fn


def calculate_jaccard_from_components(
    tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the Jaccard Index (IoU) from True Positives, False Positives,
    and False Negatives.

    Args:
        tp (torch.Tensor): True Positives count.
        fp (torch.Tensor): False Positives count.
        fn (torch.Tensor): False Negatives count.

    Returns:
        torch.Tensor: The Jaccard Index score.
    """
    union = tp + fp + fn
    # Handle edge case where union is zero (both masks are empty)
    if union.item() == 0:
        return torch.tensor(1.0, device=tp.device) if tp.item() == 0 else torch.tensor(0.0, device=tp.device)
    
    jaccard = tp / (union + EPSILON.to(union.device))
    return jaccard


def calculate_fmeasure_from_components(
    tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """
    Calculates the F-measure from True Positives, False Positives, and False Negatives.

    Args:
        tp (torch.Tensor): True Positives count.
        fp (torch.Tensor): False Positives count.
        fn (torch.Tensor): False Negatives count.
        beta (float): The beta parameter for F-measure. Defaults to 1.0 (F1-score).

    Returns:
        torch.Tensor: The F-measure score.
    """
    precision_denominator = tp + fp
    recall_denominator = tp + fn

    # Calculate Precision
    if precision_denominator.item() == 0:
        precision = torch.tensor(1.0, device=tp.device) if tp.item() == 0 else torch.tensor(0.0, device=tp.device)
    else:
        precision = tp / (precision_denominator + EPSILON.to(precision_denominator.device))

    # Calculate Recall
    if recall_denominator.item() == 0:
        recall = torch.tensor(1.0, device=tp.device) if tp.item() == 0 else torch.tensor(0.0, device=tp.device)
    else:
        recall = tp / (recall_denominator + EPSILON.to(recall_denominator.device))

    f_denominator = (beta**2 * precision) + recall

    # Handle edge case where f_denominator is zero
    if f_denominator.item() == 0:
        return torch.tensor(1.0, device=tp.device) if tp.item() == 0 else torch.tensor(0.0, device=tp.device)
    
    f_measure = (1 + beta**2) * precision * recall / (f_denominator + EPSILON.to(f_denominator.device))
    return f_measure


def calculate_j_and_f(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """
    Calculates the J&F metric, which is the average of the Jaccard Index (IoU)
    and F-measure (F1-score).

    Args:
        pred_mask (torch.Tensor): A binary tensor representing the predicted
                                   segmentation mask (values 0 or 1).
        gt_mask (torch.Tensor): A binary tensor representing the ground truth
                                segmentation mask (values 0 or 1).

    Returns:
        float: The J&F score.
    """
    tp, fp, fn = _get_mask_metrics_components(pred_mask, gt_mask)
    
    jaccard_score = calculate_jaccard_from_components(tp, fp, fn)
    f_measure_score = calculate_fmeasure_from_components(tp, fp, fn, beta=1.0) # F1-score

    return ((jaccard_score + f_measure_score) / 2.0).item()


def calculate_miou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """
    Calculates the Mean IoU (mIoU) for a single pair of masks, which is
    equivalent to the Jaccard Index for binary masks.

    Args:
        pred_mask (torch.Tensor): A binary tensor representing the predicted
                                   segmentation mask (values 0 or 1).
        gt_mask (torch.Tensor): A binary tensor representing the ground truth
                                segmentation mask (values 0 or 1).

    Returns:
        float: The mIoU score.
    """
    tp, fp, fn = _get_mask_metrics_components(pred_mask, gt_mask)
    jaccard_score = calculate_jaccard_from_components(tp, fp, fn)
    return jaccard_score.item()


def calculate_g(pred_masks: List[torch.Tensor], gt_masks: List[torch.Tensor]) -> float:
    """
    Calculates the G metric, which involves accumulating True Positives,
    False Positives, and False Negatives across a sequence of masks (e.g., video frames)
    and then computing the Jaccard Index and F-measure from these global components.
    The final G score is the average of these global Jaccard and F-measure scores.

    Args:
        pred_masks (List[torch.Tensor]): A list of binary predicted masks for a sequence.
        gt_masks (List[torch.Tensor]): A list of binary ground truth masks for the same sequence.

    Returns:
        float: The G score.

    Raises:
        ValueError: If the number of predicted masks does not match the number of ground truth masks.
    """
    if len(pred_masks) != len(gt_masks):
        raise ValueError("Number of predicted masks must match number of ground truth masks.")

    if not pred_masks:
        return 0.0 # Return 0 if no masks are provided

    # Initialize global components with a tensor on the same device as input masks
    device = pred_masks[0].device
    global_tp = torch.tensor(0.0, device=device)
    global_fp = torch.tensor(0.0, device=device)
    global_fn = torch.tensor(0.0, device=device)

    for p_mask, g_mask in zip(pred_masks, gt_masks):
        frame_tp, frame_fp, frame_fn = _get_mask_metrics_components(p_mask, g_mask)
        global_tp += frame_tp
        global_fp += frame_fp
        global_fn += frame_fn

    global_jaccard_score = calculate_jaccard_from_components(global_tp, global_fp, global_fn)
    global_f_measure_score = calculate_fmeasure_from_components(global_tp, global_fp, global_fn, beta=1.0) # F1-score

    return ((global_jaccard_score + global_f_measure_score) / 2.0).item()

