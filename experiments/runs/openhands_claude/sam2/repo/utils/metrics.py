"""
Evaluation metrics for SAM 2.

J&F metric (Jaccard & F-measure) for video segmentation.
mIoU for image segmentation.

J (Jaccard / IoU): region similarity = intersection / union
F (F-measure):     boundary accuracy = 2 * precision * recall / (precision + recall)
J&F: mean of J and F scores
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Jaccard (IoU) metric
# ---------------------------------------------------------------------------

def jaccard(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Jaccard index (IoU) between binary prediction and ground truth.
    Both arrays should be boolean or 0/1.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def batch_jaccard(pred: Tensor, gt: Tensor) -> Tensor:
    """
    Compute IoU for a batch of binary masks.
    pred, gt: (B, H, W) bool tensors
    Returns: (B,) float tensor
    """
    pred = pred.bool().flatten(1)
    gt = gt.bool().flatten(1)
    intersection = (pred & gt).float().sum(1)
    union = (pred | gt).float().sum(1)
    iou = torch.where(union > 0, intersection / union, torch.ones_like(intersection))
    return iou


# ---------------------------------------------------------------------------
# F-measure (boundary accuracy)
# ---------------------------------------------------------------------------

def _seg2bmap(seg: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    """Convert segmentation mask to boundary map."""
    seg = seg.astype(bool)
    if width is not None and height is not None:
        seg = cv2.resize(seg.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    sw = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    sw[:-1, 1:] = seg[1:, :-1]

    b = seg ^ e | seg ^ s | seg ^ se | seg ^ sw
    b[-1, :] = seg[-1, :] ^ seg[-2, :]
    b[:, -1] = seg[:, -1] ^ seg[:, -2]
    b[-1, -1] = 0

    if seg.sum() == 0:
        b = np.zeros_like(seg)
    if (~seg).sum() == 0:
        b = np.zeros_like(seg)

    return b


def f_measure(pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> float:
    """
    Compute F-measure (boundary accuracy) between prediction and ground truth.

    bound_th: threshold for boundary matching (fraction of image diagonal).
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    bound_pix = max(1, round(bound_th * np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)))

    fg_boundary = _seg2bmap(pred)
    gt_boundary = _seg2bmap(gt)

    # Dilate boundaries
    from scipy.ndimage import binary_dilation
    struct = np.ones((2 * bound_pix + 1, 2 * bound_pix + 1), dtype=bool)
    fg_dil = binary_dilation(fg_boundary, structure=struct)
    gt_dil = binary_dilation(gt_boundary, structure=struct)

    # Precision and recall
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    n_fg = fg_boundary.sum()
    n_gt = gt_boundary.sum()

    if n_fg == 0 and n_gt == 0:
        return 1.0
    if n_fg == 0 or n_gt == 0:
        return 0.0

    precision = fg_match.sum() / float(n_fg)
    recall = gt_match.sum() / float(n_gt)

    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# J&F metric for video segmentation
# ---------------------------------------------------------------------------

def compute_jf_sequence(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute J&F for a single object across a video sequence.

    Args:
        pred_masks: list of (H, W) binary arrays, one per frame
        gt_masks:   list of (H, W) binary arrays, one per frame

    Returns dict with 'J', 'F', 'JF' scores.
    """
    assert len(pred_masks) == len(gt_masks)
    j_scores = []
    f_scores = []
    for pred, gt in zip(pred_masks, gt_masks):
        j_scores.append(jaccard(pred, gt))
        f_scores.append(f_measure(pred, gt))

    J = float(np.mean(j_scores))
    F = float(np.mean(f_scores))
    return {"J": J, "F": F, "JF": (J + F) / 2.0}


def compute_jf_dataset(
    all_pred: Dict[str, List[np.ndarray]],
    all_gt: Dict[str, List[np.ndarray]],
) -> Dict[str, float]:
    """
    Compute mean J&F over all sequences in a dataset.

    Args:
        all_pred: {sequence_id: [pred_mask_t0, pred_mask_t1, ...]}
        all_gt:   {sequence_id: [gt_mask_t0, gt_mask_t1, ...]}

    Returns dict with mean 'J', 'F', 'JF'.
    """
    j_list, f_list = [], []
    for seq_id in all_gt:
        if seq_id not in all_pred:
            continue
        scores = compute_jf_sequence(all_pred[seq_id], all_gt[seq_id])
        j_list.append(scores["J"])
        f_list.append(scores["F"])

    mean_j = float(np.mean(j_list)) if j_list else 0.0
    mean_f = float(np.mean(f_list)) if f_list else 0.0
    return {"J": mean_j, "F": mean_f, "JF": (mean_j + mean_f) / 2.0}


# ---------------------------------------------------------------------------
# mIoU for image segmentation
# ---------------------------------------------------------------------------

def compute_miou(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
) -> float:
    """
    Compute mean IoU over a list of (pred, gt) mask pairs.
    Used for the SA task evaluation.
    """
    ious = [jaccard(p, g) for p, g in zip(pred_masks, gt_masks)]
    return float(np.mean(ious)) if ious else 0.0


# ---------------------------------------------------------------------------
# Click-based evaluation helpers
# ---------------------------------------------------------------------------

def simulate_clicks_on_frame(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_clicks: int = 3,
) -> List[Tuple[int, int, int]]:
    """
    Simulate interactive click evaluation (Appendix F.1.2).

    First click: center of GT mask.
    Subsequent clicks: center of error region.

    Returns list of (y, x, label) tuples.
    """
    from data.utils import get_mask_center, get_error_center

    clicks = []
    current_pred = pred_mask.copy()

    # First click: center of GT
    cy, cx = get_mask_center(gt_mask)
    clicks.append((cy, cx, 1))

    # Subsequent clicks: error region
    for _ in range(num_clicks - 1):
        cy, cx, label = get_error_center(current_pred, gt_mask)
        clicks.append((cy, cx, label))

    return clicks
