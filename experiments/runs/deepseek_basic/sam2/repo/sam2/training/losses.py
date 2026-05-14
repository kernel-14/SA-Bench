"""
Loss functions for SAM 2 training.

From Appendix D.2:
- Linear combination of focal and dice losses for mask prediction
- Mean-absolute-error (MAE/L1) loss for IoU prediction
- Cross-entropy loss for object (occlusion) prediction
- Loss ratio of 20:1:1:1 for mask_focal : mask_dice : iou_mae : object_ce

For multi-mask predictions:
- Supervise IoU predictions of ALL masks (to learn when a mask is bad)
- Only supervise mask logits with the LOWEST segmentation loss
- If ground-truth doesn't contain a mask for a frame, don't supervise any mask outputs
  (but always supervise the occlusion prediction head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss for segmentation masks.
    Args:
        pred: [B, C, H, W] predicted masks (logits)
        target: [B, C, H, W] ground truth masks
    """
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for segmentation masks.
    Args:
        pred: [B, C, H, W] predicted masks (logits)
        target: [B, C, H, W] ground truth masks
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = torch.exp(-bce)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def sigmoid_mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss with sigmoid on predictions (for IoU supervision).
    Args:
        pred: [B, N] logits
        target: [B, N] target IoU values
    """
    pred = torch.sigmoid(pred)
    return F.l1_loss(pred, target)


class SAM2Loss(nn.Module):
    """
    Loss function for SAM 2 training.

    Losses:
    - Mask loss: focal + dice (weighted 20:1)
    - IoU loss: L1/MAE (weight 1)
    - Object loss: cross-entropy for occlusion (weight 1)
    """

    def __init__(
        self,
        focal_weight: float = 20.0,
        dice_weight: float = 1.0,
        iou_weight: float = 1.0,
        object_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.object_weight = object_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.bce_loss = nn.BCEWithLogitsLoss()

    def mask_loss(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        mask_to_supervise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mask losses.
        Args:
            pred_masks: [B, N_masks, H, W] predicted mask logits
            gt_masks: [B, N_masks, H, W] ground truth masks
            mask_to_supervise: [B] indices of which mask to supervise (-1 for none)
        Returns:
            focal_loss_value, dice_loss_value
        """
        B, N, H, W = pred_masks.shape

        if mask_to_supervise is not None:
            # Only supervise the mask with the lowest loss
            total_losses = []
            for i in range(N):
                fl = focal_loss(
                    pred_masks[:, i:i+1], gt_masks[:, i:i+1],
                    self.focal_alpha, self.focal_gamma
                )
                dl = dice_loss(pred_masks[:, i:i+1], gt_masks[:, i:i+1])
                total_losses.append(
                    self.focal_weight * fl + self.dice_weight * dl
                )
            total_losses = torch.stack(total_losses, dim=1)  # [B, N]

            # For each sample, pick the mask with lowest loss
            best_idx = total_losses.argmin(dim=1)

            fl = 0.0
            dl = 0.0
            for b in range(B):
                if mask_to_supervise[b] >= 0:
                    idx = best_idx[b]
                    fl = fl + focal_loss(
                        pred_masks[b:b+1, idx:idx+1],
                        gt_masks[b:b+1, idx:idx+1],
                        self.focal_alpha, self.focal_gamma
                    )
                    dl = dl + dice_loss(
                        pred_masks[b:b+1, idx:idx+1],
                        gt_masks[b:b+1, idx:idx+1],
                    )
            fl = fl / B
            dl = dl / B
        else:
            # Supervise all masks
            fl = focal_loss(
                pred_masks, gt_masks,
                self.focal_alpha, self.focal_gamma
            )
            dl = dice_loss(pred_masks, gt_masks)

        return fl, dl

    def iou_loss(
        self,
        pred_iou: torch.Tensor,
        gt_iou: torch.Tensor,
    ) -> torch.Tensor:
        """L1/MAE loss for IoU predictions. Supervise ALL masks."""
        return sigmoid_mae_loss(pred_iou, gt_iou)

    def object_loss(
        self,
        pred_occlusion: torch.Tensor,
        gt_present: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss for object presence prediction.
        Args:
            pred_occlusion: [B] logits (higher = more likely occluded)
            gt_present: [B] 1 = object present, 0 = not present
        """
        # occlusion_pred: higher = occluded (object NOT present)
        # gt_present: 1 = present, 0 = absent
        gt_occlusion = 1.0 - gt_present.float()  # 1 = occluded/absent
        return self.bce_loss(pred_occlusion, gt_occlusion)

    def forward(
        self,
        pred_masks: torch.Tensor,
        pred_iou: torch.Tensor,
        pred_occlusion: torch.Tensor,
        gt_masks: torch.Tensor,
        gt_present: torch.Tensor,
        mask_to_supervise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total SAM 2 loss.

        Args:
            pred_masks: [B, N, H, W] predicted mask logits
            pred_iou: [B, N] predicted IoU values
            pred_occlusion: [B] occlusion prediction logits
            gt_masks: [B, N, H, W] ground truth masks (can be all zeros for absent objects)
            gt_present: [B] whether object is present (1) or absent (0)
            mask_to_supervise: [B] which mask to supervise (-1 = don't supervise masks)

        Returns:
            total_loss, loss_dict
        """
        # Compute ground truth IoU for each predicted mask
        with torch.no_grad():
            pred_masks_sigmoid = torch.sigmoid(pred_masks)
            gt_iou = torch.zeros_like(pred_iou)
            for i in range(pred_masks_sigmoid.shape[1]):
                intersection = (pred_masks_sigmoid[:, i] * gt_masks[:, i]).sum(dim=(-2, -1))
                union = pred_masks_sigmoid[:, i].sum(dim=(-2, -1)) + gt_masks[:, i].sum(dim=(-2, -1))
                gt_iou[:, i] = intersection / (union + 1e-6)

        # Mask loss (focal + dice)
        if mask_to_supervise is not None:
            # Don't supervise masks for absent objects
            valid = mask_to_supervise >= 0
            if valid.any():
                focal_l, dice_l = self.mask_loss(
                    pred_masks[valid], gt_masks[valid],
                    mask_to_supervise[valid],
                )
            else:
                focal_l = torch.tensor(0.0, device=pred_masks.device)
                dice_l = torch.tensor(0.0, device=pred_masks.device)
        else:
            focal_l, dice_l = self.mask_loss(pred_masks, gt_masks)

        # IoU loss
        iou_l = self.iou_loss(pred_iou, gt_iou)

        # Object presence loss (always supervised)
        obj_l = self.object_loss(pred_occlusion, gt_present)

        total_loss = (
            self.focal_weight * focal_l +
            self.dice_weight * dice_l +
            self.iou_weight * iou_l +
            self.object_weight * obj_l
        )

        loss_dict = {
            "focal_loss": focal_l,
            "dice_loss": dice_l,
            "iou_loss": iou_l,
            "object_loss": obj_l,
            "total_loss": total_loss,
        }

        return total_loss, loss_dict
