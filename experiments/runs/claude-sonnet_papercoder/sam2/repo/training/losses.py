```python
## training/losses.py
"""Loss functions for SAM 2 training.

Implements SAM2Losses, the unified loss computation class for SAM 2. Handles
multi-task supervision across mask prediction (focal + dice), IoU prediction
(L1/MAE), and occlusion prediction (binary cross-entropy).

Supervision rules (Appendix D.2.2, config.yaml):
    - Focal loss (weight=20): visible frames only, lowest-loss mask only
    - Dice loss (weight=1): visible frames only, lowest-loss mask only
    - IoU loss (weight=1): visible frames only, ALL N mask predictions
    - Occlusion loss (weight=1): ALWAYS, every frame unconditionally

Config references (config.yaml training.losses):
    training.losses.focal_weight: 20
    training.losses.dice_weight: 1
    training.losses.iou_loss_type: "l1"
    training.losses.iou_weight: 1
    training.losses.occlusion_loss_type: "cross_entropy"
    training.losses.occlusion_weight: 1
    training.multimask_supervision.supervise_all_iou_predictions: true
    training.multimask_supervision.supervise_lowest_loss_mask_only: true
    training.occlusion_supervision.always_supervise_occlusion_head: true
    training.occlusion_supervision.skip_mask_supervision_when_occluded: true

Paper references:
    Appendix D.2.1: "we found it beneficial to use an ℓ1 loss to more
        aggressively supervise the IoU predictions"
    Appendix D.2.1: "for multi-mask predictions, we supervise the IoU
        predictions of all masks to encourage better learning of when a
        mask might be bad, but only supervise the mask logits with the
        lowest segmentation loss"
    Appendix D.2.2: "Losses and optimization. We supervise the model's
        predictions using a linear combination of focal and dice losses for
        the mask prediction, mean-absolute-error (MAE) loss for the IoU
        prediction, and cross-entropy loss for object prediction with a
        ratio of 20:1:1:1 respectively."
    Appendix D.2.2: "If the ground-truth does not contain a mask for a
        frame, we do not supervise any of the mask outputs (but always
        supervise the occlusion prediction head)"
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numerical stability constants
# ---------------------------------------------------------------------------

# Epsilon for dice and IoU denominators to prevent division by zero
_EPS: float = 1e-6

# Clamp range for focal loss probability to prevent log(0)
_FOCAL_PROB_MIN: float = 1e-7
_FOCAL_PROB_MAX: float = 1.0 - 1e-7

# Standard focal loss gamma parameter (not explicitly stated in paper;
# follows SAM's implementation and standard focal loss convention)
_FOCAL_GAMMA: float = 2.0


class SAM2Losses(nn.Module):
    """Unified loss computation for SAM 2 training.

    Computes the combined loss across three prediction heads:
        1. Mask prediction: focal loss + dice loss (ratio 20:1)
        2. IoU prediction: L1/MAE loss
        3. Occlusion prediction: binary cross-entropy

    The total loss ratio is 20:1:1:1 (focal:dice:iou:occlusion) as stated
    in Appendix D.2.2 of the paper.

    Key supervision rules:
        - Mask losses (focal + dice): only on visible frames, only on the
          mask with the lowest combined segmentation loss
        - IoU loss: only on visible frames, but on ALL N mask predictions
        - Occlusion loss: always computed, regardless of frame visibility

    This class is used by both Pretrainer (SA-1B pre-training) and Trainer
    (joint image/video training). The gt_occ tensor controls which frames
    receive mask supervision — for SA-1B images, gt_occ is always zero.

    Config references (config.yaml):
        training.losses.focal_weight: 20
        training.losses.dice_weight: 1
        training.losses.iou_weight: 1
        training.losses.occlusion_weight: 1

    Args:
        focal_weight: Weight for focal loss component. Defaults to 20.0
            (config: training.losses.focal_weight).
        dice_weight: Weight for dice loss component. Defaults to 1.0
            (config: training.losses.dice_weight).
        iou_weight: Weight for IoU L1 loss component. Defaults to 1.0
            (config: training.losses.iou_weight).
        occlusion_weight: Weight for occlusion BCE loss component.
            Defaults to 1.0 (config: training.losses.occlusion_weight).

    Example:
        losses = SAM2Losses(
            focal_weight=20.0,
            dice_weight=1.0,
            iou_weight=1.0,
            occlusion_weight=1.0,
        )
        result = losses.compute_total_loss(
            pred_masks=pred_masks,    # [B, N, H, W]
            gt_masks=gt_masks,        # [B, H, W]
            pred_iou=pred_iou,        # [B, N]
            pred_occ=pred_occ,        # [B]
            gt_occ=gt_occ,            # [B]
        )
        total_loss = result["total"]
        result["total"].backward()
    """

    def __init__(
        self,
        focal_weight: float = 20.0,
        dice_weight: float = 1.0,
        iou_weight: float = 1.0,
        occlusion_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.focal_weight: float = focal_weight
        self.dice_weight: float = dice_weight
        self.iou_weight: float = iou_weight
        self.occlusion_weight: float = occlusion_weight

        logger.info(
            "SAM2Losses initialized: focal_weight=%.1f, dice_weight=%.1f, "
            "iou_weight=%.1f, occlusion_weight=%.1f (ratio %d:1:1:1)",
            focal_weight,
            dice_weight,
            iou_weight,
            occlusion_weight,
            int(focal_weight),
        )

    # ------------------------------------------------------------------
    # Individual loss functions
    # ------------------------------------------------------------------

    def focal_loss(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute focal loss for mask prediction.

        Penalizes hard-to-classify pixels more heavily than easy ones.
        Applied to the selected (lowest-loss) mask logits only.

        Formula:
            p = sigmoid(pred)
            pt = target * p + (1 - target) * (1 - p)
            bce = -[target * log(p) + (1-target) * log(1-p)]
            focal = (1 - pt)^gamma * bce
            loss = mean(focal)

        Paper reference: Appendix D.2.2 — "linear combination of focal and
        dice losses for the mask prediction"

        Args:
            pred: Raw mask logits of shape [B, H, W] or [B, N, H, W].
                Values are unbounded (before sigmoid).
            target: Binary GT mask of shape [B, H, W], values in {0.0, 1.0}.
                Must be broadcastable to pred's shape.

        Returns:
            Scalar focal loss tensor (unweighted — weight applied in
            compute_total_loss). Returns zero tensor if pred is empty.
        """
        if pred.numel() == 0:
            return torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        # Apply sigmoid to get probabilities, clamped for numerical stability
        p: Tensor = torch.sigmoid(pred)
        p = torch.clamp(p, min=_FOCAL_PROB_MIN, max=_FOCAL_PROB_MAX)

        # Ensure target is float32 and broadcastable
        target_float: Tensor = target.float()

        # Binary cross-entropy per pixel
        # bce = -[y * log(p) + (1-y) * log(1-p)]
        bce: Tensor = -(
            target_float * torch.log(p)
            + (1.0 - target_float) * torch.log(1.0 - p)
        )

        # Focal modulation: pt = probability of the correct class
        # pt = p when target=1, pt = (1-p) when target=0
        pt: Tensor = target_float * p + (1.0 - target_float) * (1.0 - p)

        # Focal factor: (1 - pt)^gamma
        focal_factor: Tensor = (1.0 - pt) ** _FOCAL_GAMMA

        # Focal loss per pixel
        focal: Tensor = focal_factor * bce

        # Reduce: mean over all spatial and batch dimensions
        return focal.mean()

    def dice_loss(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute dice loss for mask prediction.

        Measures global shape overlap between predicted and GT masks.
        Complementary to focal loss — focal handles per-pixel accuracy,
        dice handles global shape overlap.

        Formula:
            p = sigmoid(pred)
            numerator = 2 * sum(p * target) over H, W
            denominator = sum(p) + sum(target) over H, W + eps
            dice_coeff = numerator / denominator
            loss = 1 - dice_coeff
            loss = mean over batch

        Paper reference: Appendix D.2.2 — "linear combination of focal and
        dice losses for the mask prediction"

        Args:
            pred: Raw mask logits of shape [B, H, W].
                Values are unbounded (before sigmoid).
            target: Binary GT mask of shape [B, H, W], values in {0.0, 1.0}.

        Returns:
            Scalar dice loss tensor (unweighted). Returns zero tensor if
            pred is empty.
        """
        if pred.numel() == 0:
            return torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        # Apply sigmoid to get soft probabilities
        p: Tensor = torch.sigmoid(pred)

        # Ensure target is float32
        target_float: Tensor = target.float()

        # Flatten spatial dimensions for sum: [B, H*W]
        p_flat: Tensor = p.flatten(start_dim=1)
        t_flat: Tensor = target_float.flatten(start_dim=1)

        # Numerator: 2 * sum(p * target) per batch element
        numerator: Tensor = 2.0 * (p_flat * t_flat).sum(dim=1)  # [B]

        # Denominator: sum(p) + sum(target) per batch element
        denominator: Tensor = p_flat.sum(dim=1) + t_flat.sum(dim=1)  # [B]

        # Dice coefficient per batch element
        dice_coeff: Tensor = numerator / (denominator + _EPS)  # [B]

        # Dice loss: 1 - dice_coeff, averaged over batch
        return (1.0 - dice_coeff).mean()

    def iou_loss(
        self,
        pred_iou: Tensor,
        target_iou: Tensor,
    ) -> Tensor:
        """Compute L1/MAE loss for IoU prediction.

        Supervises the model's predicted IoU score to match the actual IoU
        between the predicted binary mask and the GT mask.

        From Appendix D.2.1: "we found it beneficial to use an ℓ1 loss to
        more aggressively supervise the IoU predictions and to apply a sigmoid
        activation to the IoU logits to restrict the output into the range
        between 0 and 1."

        Config reference: training.losses.iou_loss_type: "l1"

        Args:
            pred_iou: Model's IoU predictions after sigmoid activation,
                shape [B, N] where N = num_multimask_outputs + 1.
                Values in [0, 1] (sigmoid-activated per Appendix D.2.1).
            target_iou: Actual IoU between predicted binary mask and GT mask,
                shape [B, N]. Values in [0, 1].
                Computed by _compute_target_iou() for each of the N masks.

        Returns:
            Scalar L1 loss tensor (unweighted). Returns zero tensor if
            pred_iou is empty.
        """
        if pred_iou.numel() == 0:
            return torch.zeros(1, device=pred_iou.device, dtype=pred_iou.dtype).squeeze()

        # L1/MAE loss: mean absolute error between predicted and actual IoU
        return F.l1_loss(pred_iou, target_iou, reduction="mean")

    def occlusion_loss(
        self,
        pred_occ: Tensor,
        target_occ: Tensor,
    ) -> Tensor:
        """Compute binary cross-entropy loss for occlusion prediction.

        Supervises the binary occlusion head that predicts whether the object
        is visible in the current frame.

        From Appendix D.2.2: "cross-entropy loss for object prediction"
        Config reference: training.losses.occlusion_loss_type: "cross_entropy"

        This loss is ALWAYS computed, regardless of whether the frame has a
        valid GT mask. Per config:
            training.occlusion_supervision.always_supervise_occlusion_head: true

        Uses BCE with logits (numerically stable: combines sigmoid + BCE in
        log-sum-exp form) rather than applying sigmoid first.

        Args:
            pred_occ: Raw occlusion logit (before sigmoid), shape [B] or [B, 1].
                Positive values indicate the model predicts occlusion.
            target_occ: Binary occlusion label, shape [B].
                1 = occluded (object not visible), 0 = visible.

        Returns:
            Scalar BCE loss tensor (unweighted). Returns zero tensor if
            pred_occ is empty.
        """
        if pred_occ.numel() == 0:
            return torch.zeros(1, device=pred_occ.device, dtype=pred_occ.dtype).squeeze()

        # Normalize pred_occ to shape [B]
        pred_flat: Tensor = pred_occ.view(-1)

        # Ensure target is float32 with shape [B]
        target_float: Tensor = target_occ.float().view(-1)

        # Validate shapes match
        if pred_flat.shape[0] != target_float.shape[0]:
            raise ValueError(
                f"pred_occ batch size {pred_flat.shape[0]} != "
                f"target_occ batch size {target_float.shape[0]}."
            )

        # BCE with logits: numerically stable sigmoid + BCE
        return F.binary_cross_entropy_with_logits(
            pred_flat,
            target_float,
            reduction="mean",
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_target_iou(
        self,
        pred_mask: Tensor,
        gt_mask: Tensor,
    ) -> Tensor:
        """Compute actual IoU between binarized predicted mask and GT mask.

        This is the supervision target for the IoU prediction head. Computed
        with no_grad since it is a target value, not part of the gradient graph.

        Binarization threshold: 0.0 for logits (sigmoid(0) = 0.5), which is
        equivalent to thresholding sigmoid output at 0.5.

        Args:
            pred_mask: Raw logit or probability mask, shape [B, H, W].
                Binarized at threshold 0.0 (logit > 0 ↔ sigmoid > 0.5).
            gt_mask: Binary GT mask, shape [B, H, W], values in {0.0, 1.0}.

        Returns:
            IoU tensor of shape [B], values in [0.0, 1.0].
            Returns 1.0 for batch elements where both masks are empty
            (perfect match convention).
        """
        # Binarize predicted mask: logit > 0 is equivalent to sigmoid > 0.5
        binary_pred: Tensor = (pred_mask > 0.0).float()
        gt_float: Tensor = gt_mask.float()

        # Flatten spatial dimensions: [B, H*W]
        pred_flat: Tensor = binary_pred.flatten(start_dim=1)
        gt_flat: Tensor = gt_float.flatten(start_dim=1)

        # Intersection: sum(pred * gt) per batch element
        intersection: Tensor = (pred_flat * gt_flat).sum(dim=1)  # [B]

        # Union: sum(pred) + sum(gt) - intersection per batch element
        union: Tensor = pred_flat.sum(dim=1) + gt_flat.sum(dim=1) - intersection  # [B]

        # IoU: intersection / union, with epsilon for numerical stability
        # When union == 0 (both masks empty), IoU = 1.0 (perfect match)
        iou: Tensor = torch.where(
            union > 0,
            intersection / (union + _EPS),
            torch.ones_like(intersection),
        )

        return iou  # [B]

    def _select_best_mask(
        self,
        pred_masks: Tensor,
        gt_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Select the mask with the lowest combined segmentation loss.

        From Appendix D.2.1: "we only supervise the mask logits with the
        lowest segmentation loss (linear combination of focal and dice loss)."

        The selection uses the unweighted sum of focal + dice for comparison
        purposes (weights are applied in compute_total_loss, not here).

        Args:
            pred_masks: Raw logits for all N masks, shape [B, N, H, W].
                N = num_multimask_outputs + 1 (typically 4 = 3 multi + 1 single).
            gt_mask: Binary GT mask, shape [B, H, W], values in {0.0, 1.0}.

        Returns:
            Tuple of:
                - best_mask_logits: Tensor[B, H, W] — logits of the selected
                  (lowest-loss) mask for each batch element.
                - best_mask_idx: Tensor[B] long — index of the selected mask
                  for each batch element. Used by Trainer to determine which
                  mask to pass to MemoryEncoder.

        Raises:
            ValueError: If pred_masks has fewer than 1 mask (N < 1).
        """
        B, N, H, W = pred_masks.shape

        if N == 0:
            raise ValueError(
                "pred_masks must have at least 1 mask (N >= 1), got N=0."
            )

        if N == 1:
            # Only one mask — trivially select it
            best_logits: Tensor = pred_masks[:, 0, :, :]  # [B, H, W]
            best_idx: Tensor = torch.zeros(B, dtype=torch.long, device=pred_masks.device)
            return best_logits, best_idx

        # Compute unweighted focal + dice loss for each of the N masks
        # Shape: [N] scalar losses (averaged over batch and spatial dims)
        # We compute per-mask loss to find the best mask index
        per_mask_losses: Tensor = torch.zeros(
            B, N, device=pred_masks.device, dtype=pred_masks.dtype
        )

        for i in range(N):
            mask_i: Tensor = pred_masks[:, i, :, :]  # [B, H, W]

            # Compute focal loss per batch element (not reduced over batch)
            focal_i: Tensor = self._focal_loss_per_sample(mask_i, gt_mask)  # [B]
            dice_i: Tensor = self._dice_loss_per_sample(mask_i, gt_mask)    # [B]

            per_mask_losses[:, i] = focal_i + dice_i

        # Find the mask index with minimum loss for each batch element
        best_mask_idx = torch.argmin(per_mask_losses, dim=1)  # [B]

        # Gather the best mask logits for each batch element
        # best_mask_idx: [B] → expand to [B, 1, H, W] for gather
        idx_expanded: Tensor = best_mask_idx.view(B, 1, 1, 1).expand(B, 1, H, W)
        best_logits = pred_masks.gather(dim=1, index=idx_expanded).squeeze(1)  # [B, H, W]

        return best_logits, best_mask_idx

    def _focal_loss_per_sample(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute focal loss per batch element (not reduced over batch).

        Used internally by _select_best_mask for per-sample mask selection.
        Identical to focal_loss() but returns shape [B] instead of scalar.

        Args:
            pred: Raw mask logits of shape [B, H, W].
            target: Binary GT mask of shape [B, H, W].

        Returns:
            Focal loss tensor of shape [B], one value per batch element.
        """
        p: Tensor = torch.sigmoid(pred)
        p = torch.clamp(p, min=_FOCAL_PROB_MIN, max=_FOCAL_PROB_MAX)

        target_float: Tensor = target.float()

        bce: Tensor = -(
            target_float * torch.log(p)
            + (1.0 - target_float) * torch.log(1.0 - p)
        )

        pt: Tensor = target_float * p + (1.0 - target_float) * (1.0 - p)
        focal_factor: Tensor = (1.0 - pt) ** _FOCAL_GAMMA
        focal: Tensor = focal_factor * bce

        # Mean over spatial dimensions H, W; keep batch dimension
        return focal.flatten(start_dim=1).mean(dim=1)  # [B]

    def _dice_loss_per_sample(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute dice loss per batch element (not reduced over batch).

        Used internally by _select_best_mask for per-sample mask selection.
        Identical to dice_loss() but returns shape [B] instead of scalar.

        Args:
            pred: Raw mask logits of shape [B, H, W].
            target: Binary GT mask of shape [B, H, W].

        Returns:
            Dice loss tensor of shape [B], one value per batch element.
        """
        p: Tensor = torch.sigmoid(pred)
        target_float: Tensor = target.float()

        p_flat: Tensor = p.flatten(start_dim=1)
        t_flat: Tensor = target_float.flatten(start_dim=1)

        numerator: Tensor = 2.0 * (p_flat * t_flat).sum(dim=1)  # [B]
        denominator: Tensor = p_flat.sum(dim=1) + t_flat.sum(dim=1)  # [B]

        dice_coeff: Tensor = numerator / (denominator + _EPS)  # [B]
        return 1.0 - dice_coeff  # [B]

    # ------------------------------------------------------------------
    # Core method: compute_total_loss
    # ------------------------------------------------------------------

    def compute_total_loss(
        self,
        pred_masks: Tensor,
        gt_masks: Optional[Tensor],
        pred_iou: Tensor,
        pred_occ: Tensor,
        gt_occ: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute the combined SAM 2 training loss.

        Implements the full supervision logic from Appendix D.2.2:
            - Focal + dice losses: visible frames only, lowest-loss mask only
            - IoU loss: visible frames only, ALL N mask predictions
            - Occlusion loss: ALWAYS, every frame unconditionally

        Total loss ratio: 20:1:1:1 (focal:dice:iou:occlusion)

        Config references:
            training.multimask_supervision.supervise_all_iou_predictions: true
            training.multimask_supervision.supervise_lowest_loss_mask_only: true
            training.occlusion_supervision.always_supervise_occlusion_head: true
            training.occlusion_supervision.skip_mask_supervision_when_occluded: true

        Args:
            pred_masks: Raw logits for all N masks, shape [B, N, H, W].
                N = num_multimask_outputs + 1 (typically 4).
                Values are unbounded (before sigmoid).
            gt_masks: Binary GT masks, shape [B, H, W], values in {0.0, 1.0}.
                May be None if all frames in the batch are occluded.
                For occluded frames (gt_occ == 1), the corresponding GT mask
                should be all zeros or None.
            pred_iou: Sigmoid-activated IoU predictions for all N masks,
                shape [B, N]. Values in [0, 1].
                Per Appendix D.2.1: sigmoid is applied before this function.
            pred_occ: Raw occlusion logit (before sigmoid), shape [B] or [B, 1].
                Positive values indicate predicted occlusion.
            gt_occ: Binary occlusion label, shape [B].
                1 = occluded (object not visible), 0 = visible.
                For SA-1B pre-training, this is always zeros.

        Returns:
            Dict[str, Tensor] with keys:
                - "total": Scalar total loss (weighted sum of all components).
                  This is the tensor to call .backward() on.
                - "focal": Scalar unweighted focal loss (for logging).
                - "dice": Scalar unweighted dice loss (for logging).
                - "iou": Scalar unweighted IoU loss (for logging).
                - "occlusion": Scalar unweighted occlusion loss (for logging).
                - "best_mask_idx": Tensor[B] long — index of the selected mask
                  for each batch element. Used by Trainer to determine which
                  mask to pass to MemoryEncoder for memory creation.

        Raises:
            ValueError: If pred_masks and gt_masks have incompatible shapes.
        """
        device: torch.device = pred_masks.device
        dtype: torch.dtype = pred_masks.dtype
        B: int = pred_masks.shape[0]
        N: int = pred_masks.shape[1]

        # ------------------------------------------------------------------
        # Step 1: Always compute occlusion loss (unconditional supervision)
        # Config: training.occlusion_supervision.always_supervise_occlusion_head: true
        # Paper: "always supervise the occlusion prediction head"
        # ------------------------------------------------------------------
        occ_loss: Tensor = self.occlusion_loss(pred_occ, gt_occ)

        # ------------------------------------------------------------------
        # Step 2: Determine which frames have valid GT masks (visible frames)
        # Config: training.occlusion_supervision.skip_mask_supervision_when_occluded: true
        # Paper: "If the ground-truth does not contain a mask for a frame,
        #         we do not supervise any of the mask outputs"
        # ------------------------------------------------------------------
        # gt_occ: [B], 0 = visible (has GT mask), 1 = occluded (no GT mask)
        gt_occ_flat: Tensor = gt_occ.view(B).long()
        visible_mask: Tensor = (gt_occ_flat == 0)  # [B] bool, True = visible frame

        num_visible: int = int(visible_mask.sum().item())

        # Initialize mask and IoU losses as zero (will remain zero if no visible frames)
        f_loss: Tensor = torch.zeros(1, device=device, dtype=dtype).squeeze()
        d_loss: Tensor = torch.zeros(1, device=device, dtype=dtype).squeeze()
        iou_l: Tensor = torch.zeros(1, device=device, dtype=dtype).squeeze()

        # Default best_mask_idx: zeros for all batch elements
        best_mask_idx: Tensor = torch.zeros(B, dtype=torch.long, device=device)

        if num_visible > 0 and gt_masks is not None:
            # ------------------------------------------------------------------
            # Step 3: Extract visible frames' predictions and GT masks
            # ------------------------------------------------------------------
            pred_masks_visible: Tensor = pred_masks[visible_mask]   # [V, N, H, W]
            gt_masks_visible: Tensor = gt_masks[visible_mask].float()  # [V, H, W]
            pred_iou_visible: Tensor = pred_iou[visible_mask]       # [V, N]

            # Validate spatial dimensions match
            if pred_masks_visible.shape[-2:] != gt_masks_visible.shape[-2:]:
                raise ValueError(
                    f"pred_masks spatial size {pred_masks_visible.shape[-2:]} != "
                    f"gt_masks spatial size {gt_masks_visible.shape[-2:]}. "
                    "Ensure masks are at the same resolution before computing loss."
                )

            # ------------------------------------------------------------------
            # Step 4: Select the best mask (lowest focal + dice loss)
            # Config: training.multimask_supervision.supervise_lowest_loss_mask_only: true
            # Paper: "only supervise the mask logits with the lowest segmentation loss"
            # ------------------------------------------------------------------
            best_logits_visible, best_idx_visible = self._select_best_mask(
                pred_masks_visible, gt_masks_visible
            )
            # best_logits_visible: [V, H, W]
            # best_idx_visible: [V] long

            # Store best_mask_idx for visible frames back into the full-batch tensor
            best_mask_idx[visible_mask] = best_idx_visible

            # ------------------------------------------------------------------
            # Step 5: Compute mask losses on the best mask only
            # ------------------------------------------------------------------
            f_loss = self.focal_loss(best_logits_visible, gt_masks_visible)
            d_loss = self.dice_loss(best_logits_visible, gt_masks_visible)

            # ------------------------------------------------------------------
            # Step 6: Compute IoU targets for ALL N masks on visible frames
            # Config: training.multimask_supervision.supervise_all_iou_predictions: true
            # Paper: "we supervise the IoU predictions of all masks"
            # ------------------------------------------------------------------
            V: int = pred_masks_visible.shape[0]
            target_iou: Tensor = torch.zeros(
                V, N, device=device, dtype=dtype
            )

            for i in range(N):
                mask_i: Tensor = pred_masks_visible[:, i, :, :]  # [V, H, W]
                target_iou[:, i] = self._compute_target_iou(
                    mask_i, gt_masks_visible
                )  # [V]

            # ------------------------------------------------------------------
            # Step 7: Compute IoU loss on all N masks for visible frames
            # ------------------------------------------------------------------
            iou_l = self.iou_loss(pred_iou_visible, target_iou)

        elif num_visible > 0 and gt_masks is None:
            # gt_masks is None but there are visible frames — this is a data error
            logger.warning(
                "compute_total_