"""
Loss functions for SAM 2 training.

From the paper (Section D.2):
- Mask prediction: linear combination of focal loss (weight 20) and dice loss (weight 1)
- IoU prediction: L1 loss (weight 1) - more aggressive supervision than SAM's BCE
- Occlusion prediction: cross-entropy loss (weight 1)

Total loss ratio: 20:1:1:1 (focal:dice:iou:occlusion)

Key differences from SAM:
- L1 loss for IoU (instead of BCE) with sigmoid activation on IoU logits
- For multi-mask predictions, only supervise the mask with lowest segmentation loss
- Supervise IoU predictions of all masks to encourage better learning
- No supervision on mask logits when ground-truth has no mask (but always supervise occlusion)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Sigmoid focal loss for binary segmentation.

    Args:
        inputs: [B, ...] raw logits
        targets: [B, ...] binary targets in {0, 1}
        alpha: weighting factor for positive class
        gamma: focusing parameter
        reduction: 'mean', 'sum', or 'none'

    Returns:
        focal loss value
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Dice loss for binary segmentation.

    Args:
        inputs: [B, ...] raw logits
        targets: [B, ...] binary targets in {0, 1}
        smooth: smoothing factor to avoid division by zero

    Returns:
        dice loss value (1 - dice coefficient)
    """
    inputs = torch.sigmoid(inputs)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    intersection = (inputs * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + smooth)
    return (1 - dice).mean()


def mask_loss(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    focal_weight: float = 20.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """
    Combined focal + dice loss for mask prediction.

    Args:
        pred_masks: [B, H, W] predicted mask logits
        gt_masks: [B, H, W] ground-truth binary masks
        focal_weight: weight for focal loss
        dice_weight: weight for dice loss

    Returns:
        combined mask loss
    """
    focal = sigmoid_focal_loss(pred_masks, gt_masks.float(), reduction="mean")
    dice = dice_loss(pred_masks, gt_masks.float())
    return focal_weight * focal + dice_weight * dice


def iou_loss(
    pred_iou: torch.Tensor,
    gt_iou: torch.Tensor,
) -> torch.Tensor:
    """
    L1 loss for IoU prediction.

    From the paper: "we found it beneficial to use an L1 loss to more aggressively
    supervise the IoU predictions and to apply a sigmoid activation to the IoU logits"

    Args:
        pred_iou: [B] predicted IoU scores (after sigmoid)
        gt_iou: [B] ground-truth IoU values

    Returns:
        L1 loss
    """
    return F.l1_loss(pred_iou, gt_iou)


def occlusion_loss(
    pred_occlusion: torch.Tensor,
    gt_occlusion: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss for occlusion prediction.

    Args:
        pred_occlusion: [B, 1] predicted occlusion logits
        gt_occlusion: [B] binary ground-truth (1 = occluded, 0 = visible)

    Returns:
        binary cross-entropy loss
    """
    return F.binary_cross_entropy_with_logits(
        pred_occlusion.squeeze(-1),
        gt_occlusion.float(),
    )


def compute_iou(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Compute IoU between predicted and ground-truth masks.

    Args:
        pred_masks: [B, H, W] predicted mask logits
        gt_masks: [B, H, W] ground-truth binary masks
        threshold: threshold for binarizing predictions

    Returns:
        iou: [B] IoU values
    """
    pred_binary = (pred_masks > threshold).float()
    gt_binary = gt_masks.float()

    intersection = (pred_binary * gt_binary).sum(dim=(-2, -1))
    union = (pred_binary + gt_binary).clamp(0, 1).sum(dim=(-2, -1))

    iou = intersection / (union + 1e-6)
    return iou


class SAM2Loss(nn.Module):
    """
    Combined loss for SAM 2 training.

    Handles:
    - Multi-mask prediction: only supervise the mask with lowest segmentation loss
    - Occlusion: always supervise occlusion head, skip mask supervision when occluded
    - IoU: supervise all mask IoU predictions
    """

    def __init__(
        self,
        focal_weight: float = 20.0,
        dice_weight: float = 1.0,
        iou_weight: float = 1.0,
        occlusion_weight: float = 1.0,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.occlusion_weight = occlusion_weight

    def forward(
        self,
        pred_masks: torch.Tensor,
        pred_iou: torch.Tensor,
        pred_occlusion: torch.Tensor,
        gt_masks: torch.Tensor,
        gt_occlusion: Optional[torch.Tensor] = None,
        multimask: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute SAM 2 training loss.

        Args:
            pred_masks: [B, num_masks, H, W] predicted mask logits
            pred_iou: [B, num_masks] predicted IoU scores (after sigmoid)
            pred_occlusion: [B, 1] predicted occlusion logits
            gt_masks: [B, H, W] ground-truth binary masks (or None if occluded)
            gt_occlusion: [B] ground-truth occlusion labels (1=occluded, 0=visible)
            multimask: whether multi-mask prediction was used

        Returns:
            total_loss: scalar loss
            loss_dict: dict with individual loss components
        """
        B, num_masks, H, W = pred_masks.shape

        # Compute occlusion loss (always supervised)
        if gt_occlusion is None:
            gt_occlusion = torch.zeros(B, device=pred_masks.device)
        occ_loss = occlusion_loss(pred_occlusion, gt_occlusion)

        # For frames where object is occluded, skip mask supervision
        visible_mask = (gt_occlusion == 0)  # [B]

        if visible_mask.sum() == 0:
            # All frames are occluded, no mask supervision
            total_loss = self.occlusion_weight * occ_loss
            return total_loss, {
                'mask_loss': torch.tensor(0.0),
                'iou_loss': torch.tensor(0.0),
                'occlusion_loss': occ_loss.item(),
                'total_loss': total_loss.item(),
            }

        # Only compute mask loss for visible frames
        vis_pred_masks = pred_masks[visible_mask]  # [B_vis, num_masks, H, W]
        vis_pred_iou = pred_iou[visible_mask]  # [B_vis, num_masks]
        vis_gt_masks = gt_masks[visible_mask]  # [B_vis, H, W]

        if multimask and num_masks > 1:
            # For multi-mask: find the mask with lowest segmentation loss
            seg_losses = []
            for i in range(num_masks):
                loss_i = mask_loss(
                    vis_pred_masks[:, i],
                    vis_gt_masks,
                    focal_weight=self.focal_weight,
                    dice_weight=self.dice_weight,
                )
                seg_losses.append(loss_i)

            seg_losses_tensor = torch.stack(seg_losses)  # [num_masks]
            best_mask_idx = seg_losses_tensor.argmin()
            seg_loss = seg_losses_tensor[best_mask_idx]

            # Compute IoU for all masks and supervise all
            gt_iou_all = []
            for i in range(num_masks):
                gt_iou_i = compute_iou(vis_pred_masks[:, i].detach(), vis_gt_masks)
                gt_iou_all.append(gt_iou_i)
            gt_iou_all = torch.stack(gt_iou_all, dim=1)  # [B_vis, num_masks]
            iou_l = iou_loss(vis_pred_iou, gt_iou_all)
        else:
            # Single mask prediction
            seg_loss = mask_loss(
                vis_pred_masks[:, 0],
                vis_gt_masks,
                focal_weight=self.focal_weight,
                dice_weight=self.dice_weight,
            )
            gt_iou = compute_iou(vis_pred_masks[:, 0].detach(), vis_gt_masks)
            iou_l = iou_loss(vis_pred_iou[:, 0], gt_iou)

        total_loss = (
            seg_loss
            + self.iou_weight * iou_l
            + self.occlusion_weight * occ_loss
        )

        return total_loss, {
            'mask_loss': seg_loss.item(),
            'iou_loss': iou_l.item(),
            'occlusion_loss': occ_loss.item(),
            'total_loss': total_loss.item(),
        }
