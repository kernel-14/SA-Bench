
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # For binary segmentation, inputs are logits
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == "mean":
            return F_loss.mean()
        elif self.reduction == "sum":
            return F_loss.sum()
        else:
            return F_loss

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    """
    def __init__(self, smooth: float = 1e-5, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs are logits, apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        if self.reduction == "mean":
            return (1 - dice).mean()
        elif self.reduction == "sum":
            return (1 - dice).sum()
        else:
            return (1 - dice)

class MaskLoss(nn.Module):
    """
    Combines Focal Loss and Dice Loss for mask prediction.
    """
    def __init__(self, focal_weight: float = 20.0, dice_weight: float = 1.0):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice

class IoULoss(nn.Module):
    """
    Mean Absolute Error (MAE) Loss for IoU prediction.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Paper states sigmoid activation on IoU logits, so inputs should be [0,1]
        # and targets should be [0,1]
        loss = F.l1_loss(inputs, targets, reduction=self.reduction)
        return loss

class ObjectnessLoss(nn.Module):
    """
    Cross-Entropy Loss for objectness/occlusion prediction.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Inputs are logits, targets are 0 or 1
        loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction=self.reduction)
        return loss

class SAM2Loss(nn.Module):
    """
    Combines all loss components for SAM2 training.
    """
    def __init__(self, config):
        super().__init__()
        self.mask_loss = MaskLoss(focal_weight=config.MASK_LOSS_WEIGHT_FOCAL, dice_weight=config.MASK_LOSS_WEIGHT_DICE)
        self.iou_loss = IoULoss()
        self.objectness_loss = ObjectnessLoss()

        self.iou_loss_weight = config.IOU_LOSS_WEIGHT
        self.object_loss_weight = config.OBJECT_LOSS_WEIGHT

    def forward(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor,
                pred_ious: torch.Tensor, gt_ious: torch.Tensor,
                pred_objectness: torch.Tensor, gt_objectness: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # Calculate mask loss (for multiple masks, take the one with lowest loss)
        # gt_masks should be (B, 1, H, W)
        # pred_masks should be (B, N_masks, H, W)
        num_masks = pred_masks.shape[1]
        
        if num_masks > 1:
            # For each predicted mask, calculate loss against GT
            all_mask_losses = []
            for i in range(num_masks):
                # Ensure pred_masks[:, i, :, :] is (B, H, W) and gt_masks.squeeze(1) is (B, H, W)
                loss_i = self.mask_loss(pred_masks[:, i, :, :].unsqueeze(1), gt_masks) # unsqueeze to (B,1,H,W) for mask_loss
                all_mask_losses.append(loss_i)
            all_mask_losses = torch.stack(all_mask_losses, dim=1) # (B, N_masks)
            
            # Select the mask with the lowest loss for supervision (for mask and IoU)
            best_mask_loss, best_mask_idx = torch.min(all_mask_losses, dim=1)
            mask_loss = best_mask_loss.mean()

            # Select corresponding IoU prediction
            pred_ious_best = pred_ious[torch.arange(pred_ious.size(0)), best_mask_idx]
            iou_loss = self.iou_loss(pred_ious_best, gt_ious) # gt_ious needs to be (B,)
        else:
            mask_loss = self.mask_loss(pred_masks, gt_masks) # Both (B,1,H,W)
            iou_loss = self.iou_loss(pred_ious.squeeze(1), gt_ious) # Both (B,)

        objectness_loss = self.objectness_loss(pred_objectness, gt_objectness)

        total_loss = mask_loss + self.iou_loss_weight * iou_loss + self.object_loss_weight * objectness_loss

        losses = {
            "total_loss": total_loss,
            "mask_loss": mask_loss,
            "iou_loss": iou_loss,
            "objectness_loss": objectness_loss,
        }
        return total_loss, losses

