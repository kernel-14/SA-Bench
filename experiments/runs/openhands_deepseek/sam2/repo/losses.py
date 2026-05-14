"""Loss functions for SAM 2 training.

Losses used:
- Focal loss + Dice loss for mask prediction (ratio 20:1)
- MAE (L1) loss for IoU prediction
- Cross-entropy loss for occlusion prediction
- Combined with ratio 20:1:1:1

During multi-mask prediction:
- Only supervise the mask with the lowest segmentation loss
- Supervise IoU predictions of ALL masks (to learn when a mask is bad)
- If ground truth has no mask for a frame, don't supervise mask outputs
  (but always supervise occlusion prediction)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for binary classification (mask prediction).

    Args:
        inputs: [B, N, H, W] predicted logits (not activated)
        targets: [B, N, H, W] ground truth binary masks
        alpha: balancing parameter for positive/negative
        gamma: focusing parameter

    Returns:
        scalar loss
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss for mask prediction.

    Args:
        inputs: [B, N, H, W] predicted logits
        targets: [B, N, H, W] ground truth binary masks
        eps: small constant for numerical stability

    Returns:
        scalar loss
    """
    prob = inputs.sigmoid()
    intersection = (prob * targets).sum(dim=[2, 3])
    union = prob.sum(dim=[2, 3]) + targets.sum(dim=[2, 3])
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


class MaskLoss(nn.Module):
    """Combined mask loss: focal + dice.

    During multi-mask training, only the mask with the lowest
    segmentation loss is supervised.
    """
    def __init__(self, focal_weight: float = 20.0, dice_weight: float = 1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor,
                is_multi_mask: bool = False) -> torch.Tensor:
        """Compute mask loss.

        Args:
            pred_masks: [B, N_masks, H, W] predicted masks (logits)
            gt_masks: [B, 1, H, W] ground truth mask
            is_multi_mask: if True, only supervise the best mask

        Returns:
            scalar loss
        """
        B, N_masks, H, W = pred_masks.shape
        gt = gt_masks.expand(-1, N_masks, -1, -1)

        f_loss = torch.zeros(B, N_masks, device=pred_masks.device)
        d_loss = torch.zeros(B, N_masks, device=pred_masks.device)

        for i in range(N_masks):
            f_loss[:, i] = focal_loss(pred_masks[:, i:i+1, :, :], gt[:, i:i+1, :, :]).detach() \
                if is_multi_mask else focal_loss(pred_masks[:, i:i+1, :, :], gt[:, i:i+1, :, :])
            d_loss[:, i] = dice_loss(pred_masks[:, i:i+1, :, :], gt[:, i:i+1, :, :]).detach() \
                if is_multi_mask else dice_loss(pred_masks[:, i:i+1, :, :], gt[:, i:i+1, :, :])

        total_per_mask = self.focal_weight * f_loss + self.dice_weight * d_loss  # [B, N_masks]

        if is_multi_mask:
            best_idx = total_per_mask.argmin(dim=1)  # [B]
            # Recompute losses only for best mask
            best_f_loss = torch.zeros(B, device=pred_masks.device)
            best_d_loss = torch.zeros(B, device=pred_masks.device)
            for b in range(B):
                best_f_loss[b] = focal_loss(
                    pred_masks[b:b+1, best_idx[b]:best_idx[b]+1, :, :],
                    gt[b:b+1, best_idx[b]:best_idx[b]+1, :, :]
                )
                best_d_loss[b] = dice_loss(
                    pred_masks[b:b+1, best_idx[b]:best_idx[b]+1, :, :],
                    gt[b:b+1, best_idx[b]:best_idx[b]+1, :, :]
                )
            return (self.focal_weight * best_f_loss + self.dice_weight * best_d_loss).mean()
        else:
            return (self.focal_weight * f_loss + self.dice_weight * d_loss).mean()


class IoULoss(nn.Module):
    """L1 (MAE) loss for IoU prediction.

    Following SAM 2: apply sigmoid to restrict IoU output to [0, 1],
    and use L1 loss for more aggressive supervision.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred_iou: torch.Tensor, gt_iou: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss.

        Args:
            pred_iou: [B, N_masks] predicted IoU (with sigmoid applied)
            gt_iou: [B, N_masks] ground truth IoU (between 0 and 1)

        Returns:
            scalar loss
        """
        return self.weight * F.l1_loss(pred_iou, gt_iou)


class OcclusionLoss(nn.Module):
    """Binary cross-entropy loss for occlusion prediction.

    Predicts whether the object of interest is present in the current frame.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred_occlusion: torch.Tensor, gt_occlusion: torch.Tensor) -> torch.Tensor:
        """Compute occlusion loss.

        Args:
            pred_occlusion: [B, 1] predicted occlusion probability (0=occluded, 1=visible)
            gt_occlusion: [B, 1] ground truth (0=occluded, 1=visible)

        Returns:
            scalar loss
        """
        return self.weight * F.binary_cross_entropy(pred_occlusion, gt_occlusion)


class SAM2Loss(nn.Module):
    """Combined loss for SAM 2 training.

    Weights: mask_loss : iou_loss : occlusion_loss = 20 : 1 : 1
    (where mask_loss internally has focal:dice = 20:1)
    """
    def __init__(self, focal_weight: float = 20.0, dice_weight: float = 1.0,
                 iou_weight: float = 1.0, occlusion_weight: float = 1.0):
        super().__init__()
        self.mask_loss_fn = MaskLoss(focal_weight, dice_weight)
        self.iou_loss_fn = IoULoss(iou_weight)
        self.occlusion_loss_fn = OcclusionLoss(occlusion_weight)

    def forward(self, model_outputs: dict, gt_masks: torch.Tensor,
                gt_iou: Optional[torch.Tensor] = None,
                gt_occlusion: Optional[torch.Tensor] = None,
                is_multi_mask: bool = True,
                frame_has_object: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """Compute total training loss.

        Args:
            model_outputs: dict from SAM2.forward() with keys:
                - masks: [B, N_masks, H, W]
                - iou_pred: [B, N_masks]
                - occlusion_pred: [B, 1] or None
            gt_masks: [B, 1, H, W] ground truth masks
            gt_iou: [B, N_masks] ground truth IoU, computed from gt vs pred
            gt_occlusion: [B, 1] ground truth occlusion (0/1)
            is_multi_mask: whether using multi-mask output
            frame_has_object: [B] boolean, which frames have valid objects.
                              Frames without objects skip mask loss but keep occlusion loss.

        Returns:
            total_loss: scalar
            loss_dict: breakdown of individual losses
        """
        pred_masks = model_outputs["masks"]
        pred_iou = model_outputs["iou_pred"]
        pred_occlusion = model_outputs.get("occlusion_pred")

        loss_dict = {}

        # Mask loss: only for frames with objects
        if frame_has_object is not None:
            mask_loss = torch.tensor(0.0, device=pred_masks.device)
            for b in range(pred_masks.shape[0]):
                if frame_has_object[b]:
                    mask_loss = mask_loss + self.mask_loss_fn(
                        pred_masks[b:b+1], gt_masks[b:b+1], is_multi_mask
                    )
            mask_loss = mask_loss / max(frame_has_object.sum(), 1)
        else:
            mask_loss = self.mask_loss_fn(pred_masks, gt_masks, is_multi_mask)
        loss_dict["mask_loss"] = mask_loss

        # IoU loss
        if gt_iou is not None:
            iou_loss = self.iou_loss_fn(pred_iou, gt_iou)
            loss_dict["iou_loss"] = iou_loss
        else:
            iou_loss = torch.tensor(0.0, device=pred_masks.device)

        # Occlusion loss
        if pred_occlusion is not None and gt_occlusion is not None:
            occ_loss = self.occlusion_loss_fn(pred_occlusion, gt_occlusion)
            loss_dict["occlusion_loss"] = occ_loss
        else:
            occ_loss = torch.tensor(0.0, device=pred_masks.device)

        total_loss = mask_loss + iou_loss + occ_loss
        loss_dict["total_loss"] = total_loss

        return total_loss, loss_dict


def compute_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute IoU between predicted and target masks.

    Args:
        pred: [B, N, H, W] predicted masks (after sigmoid/binarization)
        target: [B, 1, H, W] ground truth masks

    Returns:
        [B, N] IoU scores
    """
    pred = (pred.sigmoid() > 0.5).float()
    target = target.float()

    intersection = (pred * target).sum(dim=[2, 3])
    union = pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) - intersection

    return (intersection + eps) / (union + eps)
