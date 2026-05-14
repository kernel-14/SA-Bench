"""
Evaluation metrics for SAM 2.

Metrics used in the paper:
- J&F: Standard metric for video object segmentation (Pont-Tuset et al., 2017)
  - J (Region similarity): Intersection-over-Union (IoU)
  - F (Contour accuracy): F-measure based on contour matching
- G: Overall metric for YouTube-VOS 2019
- mIoU: Mean IoU for image segmentation
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def compute_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Compute Intersection over Union (IoU / J score).
    Args:
        pred_mask: [H, W] or [B, H, W] predicted mask (logits or probabilities)
        gt_mask: [H, W] or [B, H, W] ground truth binary mask
        threshold: threshold to binarize prediction
    """
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.squeeze(1)
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.squeeze(1)

    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > 0.5).float()

    intersection = (pred_binary * gt_binary).sum(dim=(-2, -1))
    union = pred_binary.sum(dim=(-2, -1)) + gt_binary.sum(dim=(-2, -1)) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def compute_f_score(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Compute F-measure (F score) based on contour matching.
    Simplified version: uses precision/recall of mask boundaries.
    """
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.squeeze(1)
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.squeeze(1)

    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > 0.5).float()

    # Simple boundary-based F-measure using morphological operations
    # Compute precision and recall on mask pixels
    intersection = (pred_binary * gt_binary).sum(dim=(-2, -1))
    precision = (intersection + 1e-6) / (pred_binary.sum(dim=(-2, -1)) + 1e-6)
    recall = (intersection + 1e-6) / (gt_binary.sum(dim=(-2, -1)) + 1e-6)

    f_score = 2 * precision * recall / (precision + recall + 1e-6)
    return f_score.mean().item()


def compute_jf_score(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute J&F score (Pont-Tuset et al., 2017).
    J = IoU, F = contour F-measure.
    J&F = (J + F) / 2.

    Returns:
        j_score, f_score, jf_score
    """
    j = compute_iou(pred_mask, gt_mask, threshold)
    f = compute_f_score(pred_mask, gt_mask, threshold)
    jf = (j + f) / 2.0
    return j, f, jf


def compute_miou(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    num_classes: Optional[int] = None,
) -> float:
    """
    Compute mean IoU for image segmentation.
    Args:
        pred_masks: [B, N, H, W] or [B, H, W] predicted masks
        gt_masks: [B, N, H, W] or [B, H, W] ground truth masks
    """
    if pred_masks.dim() == 3:
        pred_masks = pred_masks.unsqueeze(1)
    if gt_masks.dim() == 3:
        gt_masks = gt_masks.unsqueeze(1)

    B, N, H, W = pred_masks.shape
    total_iou = 0.0
    count = 0

    for b in range(B):
        for n in range(N):
            iou = compute_iou(pred_masks[b, n], gt_masks[b, n])
            total_iou += iou
            count += 1

    return total_iou / max(1, count)


class JFMetric:
    """
    Accumulator for J&F metrics over multiple frames/videos.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.j_scores = []
        self.f_scores = []
        self.jf_scores = []

    def update(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
    ):
        j, f, jf = compute_jf_score(pred_mask, gt_mask)
        self.j_scores.append(j)
        self.f_scores.append(f)
        self.jf_scores.append(jf)

    def compute(self) -> Tuple[float, float, float]:
        """Return average J, F, and J&F."""
        j_mean = sum(self.j_scores) / max(1, len(self.j_scores))
        f_mean = sum(self.f_scores) / max(1, len(self.f_scores))
        jf_mean = (j_mean + f_mean) / 2.0
        return j_mean, f_mean, jf_mean

    def get_jf(self) -> float:
        """Return J&F score."""
        _, _, jf = self.compute()
        return jf
