
import torch
import numpy as np

def calculate_iou_from_masks(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Intersection over Union (IoU) for a batch of masks.
    Args:
        pred_masks (torch.Tensor): Predicted masks (logits). Shape (B, 1, H, W).
        gt_masks (torch.Tensor): Ground truth masks (binary). Shape (B, 1, H, W).
    Returns:
        torch.Tensor: IoU scores for each mask in the batch. Shape (B,).
    """
    if pred_masks.shape != gt_masks.shape:
        raise ValueError(f"Shape mismatch: pred_masks {pred_masks.shape} vs gt_masks {gt_masks.shape}")

    pred_masks = (torch.sigmoid(pred_masks) > 0.5).float()
    
    intersection = (pred_masks * gt_masks).sum(dim=(-1, -2))
    union = (pred_masks + gt_masks).sum(dim=(-1, -2)) - intersection
    
    # Add a small epsilon to avoid division by zero
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.squeeze(1) # Remove channel dimension

def calculate_tf_metric(ious: List[torch.Tensor]) -> float:
    """
    Calculates the T&F metric (Temporal & F-measure) for video object segmentation.
    This is a simplified version; a full T&F calculation involves F-measure and Jaccard.
    For now, assume it's the average IoU over time.
    Args:
        ious (List[torch.Tensor]): List of IoU scores for each frame. Each tensor is (B,).
    Returns:
        float: Average T&F score.
    """
    # Simple average of IoU across all frames and batch items
    all_ious = torch.cat(ious)
    return all_ious.mean().item()

def calculate_miou(ious: List[torch.Tensor]) -> float:
    """
    Calculates the Mean IoU (mIoU) for image segmentation.
    Args:
        ious (List[torch.Tensor]): List of IoU scores for each mask in the batch. Each tensor is (B,).
    Returns:
        float: Mean IoU score.
    """
    all_ious = torch.cat(ious)
    return all_ious.mean().item()

# Placeholder for G metric as described in YTVOS 2019
def calculate_g_metric(jaccard_scores: List[torch.Tensor], f_scores: List[torch.Tensor]) -> float:
    """
    Calculates the G metric for YTVOS 2019, which is the average of Jaccard and F-measure.
    This is a placeholder as actual F-measure calculation is complex.
    For now, assumes jaccard_scores are IoU.
    """
    all_jaccard = torch.cat(jaccard_scores)
    all_f = torch.cat(f_scores) if f_scores else all_jaccard # If no specific F-scores, use Jaccard
    return (all_jaccard.mean() + all_f.mean()).item() / 2.0

