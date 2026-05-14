# utils/metrics.py
"""
Evaluation metrics for SAM 2 reproduction.

Implements:
- compute_JF: region similarity J and contour accuracy F for video object segmentation.
- compute_mIoU: mean Intersection over Union for image segmentation.

Relies on opencv-python (cv2) for boundary extraction and distance transforms,
as required by the standard VOS evaluation protocol.
"""

from typing import Dict, List

import numpy as np
import cv2


def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Compute the boundary (contour) of a binary mask using a morphological gradient.

    Args:
        mask: 2D numpy array, dtype bool or uint8 with 0/1 values.

    Returns:
        boundary: binary mask (same dtype as mask) where 1 indicates boundary pixels.
    """
    # Ensure mask is uint8 for OpenCV functions.
    mask_u8 = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    boundary = (dilated > 0) & (mask_u8 == 0)
    return boundary.astype(np.uint8)


def _compute_contour_f(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: int = 2) -> float:
    """
    Compute the contour F-measure for a single frame.

    The F-measure is based on boundary precision and recall, where a predicted boundary pixel is
    considered correct if it lies within `tolerance` pixels of any ground-truth boundary pixel,
    and vice versa for recall.

    Args:
        pred_mask: Binary predicted mask (H, W).
        gt_mask: Binary ground-truth mask (H, W).
        tolerance: Distance in pixels (default 2, as used in DAVIS benchmarks).

    Returns:
        f: The F-measure (harmonic mean of precision and recall). If both boundaries are empty,
           returns 1.0. If one is empty and the other is not, returns 0.0.
    """
    pred_boundary = _extract_boundary(pred_mask)
    gt_boundary = _extract_boundary(gt_mask)

    pred_bnd_pixels = np.count_nonzero(pred_boundary)
    gt_bnd_pixels = np.count_nonzero(gt_boundary)

    # Handle degenerate cases.
    if pred_bnd_pixels == 0 and gt_bnd_pixels == 0:
        return 1.0
    if pred_bnd_pixels == 0 or gt_bnd_pixels == 0:
        return 0.0

    # Distance transform: for each pixel, distance to the nearest boundary pixel.
    # Use L2 distance (Euclidean).
    dist_gt = cv2.distanceTransform(1 - gt_boundary, cv2.DIST_L2, 3)
    dist_pred = cv2.distanceTransform(1 - pred_boundary, cv2.DIST_L2, 3)

    # Precision: fraction of predicted boundary pixels within tolerance of a GT boundary pixel.
    precision = np.mean(dist_gt[pred_boundary > 0] <= tolerance)
    # Recall: fraction of GT boundary pixels within tolerance of a predicted boundary pixel.
    recall = np.mean(dist_pred[gt_boundary > 0] <= tolerance)

    if precision + recall == 0:
        return 0.0
    f = 2 * precision * recall / (precision + recall)
    return f


def compute_JF(pred_masks: np.ndarray, gt_masks: np.ndarray) -> Dict[str, float]:
    """
    Compute J (region similarity) and F (contour accuracy) for a tracked object across a video.

    This function follows the evaluation protocol of the DAVIS and other video object segmentation
    benchmarks, averaging metrics over frames.

    Args:
        pred_masks: numpy array of shape (T, H, W), dtype bool or uint8 0/1, predicted binary masks.
        gt_masks: numpy array of shape (T, H, W), same dtype, ground-truth binary masks.
                  For frames where the object is absent, the mask should be all zeros.

    Returns:
        A dictionary with keys 'J', 'F', 'J_and_F' holding the average values over all frames.
        - J: mean Intersection-over-Union (IoU).
        - F: mean contour F-measure.
        - J_and_F: arithmetic mean of J and F.
    """
    T = pred_masks.shape[0]
    assert gt_masks.shape[0] == T, "Prediction and ground truth must have same number of frames."

    j_list = []
    f_list = []

    for t in range(T):
        pred = pred_masks[t].astype(bool)
        gt = gt_masks[t].astype(bool)

        # --- Region similarity J (IoU) ---
        intersection = np.sum(pred & gt)
        union = np.sum(pred | gt)
        if union > 0:
            j_list.append(intersection / union)
        else:
            # Both masks empty: perfect agreement.
            j_list.append(1.0)

        # --- Contour accuracy F ---
        f_list.append(_compute_contour_f(pred.astype(np.uint8), gt.astype(np.uint8)))

    J = np.mean(j_list) if j_list else 0.0
    F = np.mean(f_list) if f_list else 0.0
    J_and_F = (J + F) / 2.0

    return {"J": float(J), "F": float(F), "J_and_F": float(J_and_F)}


def compute_mIoU(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> float:
    """
    Compute mean Intersection over Union (mIoU) for a collection of object masks in an image.

    Each mask corresponds to one object instance. The lists must be aligned and of equal length.

    Args:
        pred_masks: List of predicted binary masks, each a 2D numpy array (H, W) with dtype bool or uint8.
        gt_masks: List of ground-truth binary masks, same shape as corresponding prediction.

    Returns:
        The mean IoU over all object pairs. If the lists are empty, returns 0.0.
    """
    if len(pred_masks) == 0:
        return 0.0
    assert len(pred_masks) == len(gt_masks), "Mismatched number of predicted and ground-truth masks."

    ious = []
    for pred, gt in zip(pred_masks, gt_masks):
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        intersection = np.sum(pred & gt)
        union = np.sum(pred | gt)
        if union > 0:
            ious.append(intersection / union)
        else:
            # Both masks empty: perfect prediction.
            ious.append(1.0)

    return float(np.mean(ious))
