# training/pretrain.py
"""
Pretraining on SA-1B images (image‑only, interactive prompts).

This module implements the function :func:`pretrain_on_sa1b` which trains the
SAM 2 model on static images using simulated interactive prompts, exactly as
described in the SAM 2 paper (Appendix D.2.1).  The process includes:

- Filtering oversized masks,
- Random sampling up to ``max_masks_per_image``,
- Simulating an initial prompt (50% mask, 25% click, 25% box) followed by
  up to ``correction_points`` corrective clicks (default 7),
- Computing a combination of focal, dice, and L1 IoU losses,
- Using gradient accumulation to reach the effective global batch size,
- Applying AdamW with layer‑wise decay and a reciprocal square‑root LR schedule.

Typical usage::

    from config import Config
    from data.image_dataset import ImageDataset
    from model.sam2 import SAM2Model
    from training.pretrain import pretrain_on_sa1b

    cfg = Config("config.yaml")
    model = SAM2Model(cfg.to_dict())
    dataset = ImageDataset(cfg.data.sa1b_root, cfg)
    model = pretrain_on_sa1b(model, dataset, cfg)
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# Import project modules (assuming they are in the path)
from model.sam2 import SAM2Model
from data.image_dataset import ImageDataset
from evaluation.click_simulator import ClickSimulator


# ---------------------------------------------------------------------------
#  Loss helpers
# ---------------------------------------------------------------------------

def _focal_loss(
    pred_logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """Focal loss for binary segmentation.

    Args:
        pred_logits: (B, H, W) raw logits.
        target: (B, H, W) ground‑truth binary mask.
        alpha, gamma: focal loss parameters.

    Returns:
        scalar loss averaged over batch.
    """
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    pt = torch.exp(-bce)  # pt = 1 - p_t (where p_t = sigmoid(logits) for target=1 else 1-sigmoid)
    focal_weight = (1 - pt) ** gamma
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        focal_weight *= alpha_t
    loss = focal_weight * bce
    return loss.mean()


def _dice_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft Dice loss.

    Args:
        pred_logits: (B, H, W) raw logits.
        target: (B, H, W) ground‑truth mask.

    Returns:
        scalar loss.
    """
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    return (1.0 - dice).mean()


def _compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """Compute per‑mask IoU.

    Args:
        pred_mask: (B, H, W) binary (after sigmoid > 0.5) or soft.
        gt_mask: (B, H, W) binary.

    Returns:
        (B,) IoU values.
    """
    intersection = (pred_mask * gt_mask).sum(dim=(1, 2))
    union = pred_mask.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou


def compute_pretrain_losses(
    masks_logits: torch.Tensor,
    iou_pred: torch.Tensor,
    gt_mask: torch.Tensor,
    loss_weights: Dict[str, float],
    multi_mask: bool,
) -> torch.Tensor:
    """
    Compute the total loss for a single interactive step.

    Args:
        masks_logits: (num_masks, H, W) tensor of mask logits.
        iou_pred: (num_masks,) tensor of predicted IoU values (sigmoid‑activated).
        gt_mask: (H, W) binary ground‑truth mask.
        loss_weights: dictionary with keys 'focal', 'dice', 'iou_l1'.
        multi_mask: bool, if True, ignore the mask loss for all but the best mask.

    Returns:
        scalar total loss.
    """
    num_masks = masks_logits.shape[0]
    gt = gt_mask.unsqueeze(0).expand(num_masks, -1, -1)  # (num_masks, H, W)

    # Compute mask losses for each candidate
    focal_list = []
    dice_list = []
    for k in range(num_masks):
        logits_k = masks_logits[k].unsqueeze(0)  # (1, H, W)
        gt_k = gt[k].unsqueeze(0)
        focal_list.append(_focal_loss(logits_k, gt_k, alpha=0.25, gamma=2.0))
        dice_list.append(_dice_loss(logits_k, gt_k))
    focal = torch.stack(focal_list)
    dice = torch.stack(dice_list)
    mask_losses = loss_weights["focal"] * focal + loss_weights["dice"] * dice

    # If multi_mask, only supervise the mask that yields the lowest segmentation loss
    if multi_mask and num_masks > 1:
        best_idx = torch.argmin(mask_losses)
        # Zero out the contribution of the other masks
        mask_loss = mask_losses[best_idx]  # keep only best
    else:
        mask_loss = mask_losses.mean()  # average equally (could also sum)

    # IoU loss: apply to all masks
    with torch.no_grad():
        # Binarize predictions for IoU calculation
        pred_bin = (torch.sigmoid(masks_logits) > 0.5).float()
        actual_ious = _compute_iou(pred_bin, gt)  # (num_masks,)
    iou_l1 = F.l1_loss(iou_pred, actual_ious)

    total = mask_loss + loss_weights["iou_l1"] * iou_l1
    return total


# ---------------------------------------------------------------------------
#  Interactive simulation for a single mask
# ---------------------------------------------------------------------------

def _simulate_interactive_sequence(
    model: SAM2Model,
    image: torch.Tensor,          # (C, H, W) float32 [0,1]
    gt_mask: torch.Tensor,        # (H, W) binary float32
    click_sim: ClickSimulator,
    config: Any,
) -> torch.Tensor:
    """
    Run the full interactive prompt sequence for one ground‑truth mask and
    return the sum of losses over 1 initial + N correction steps.

    This function handles multi‑mask selection after the first ambiguous
    prompt and accumulates gradients for every step.
    """
    cfg_pretrain = config.training.pretrain
    initial_probs = config.training.full_training.initial_prompt_probs
    correction_clicks = cfg_pretrain.correction_points
    resolution = config.model.image_encoder.resolution  # typically 1024
    H, W = resolution, resolution  # assume square
    device = next(model.parameters()).device

    # Convert image to batch form (1, C, H, W)
    img_batch = image.unsqueeze(0).to(device)
    gt_mask = gt_mask.to(device)

    # Store list of (prompt_dict, multi_mask_flag) for each step
    steps_prompts: List[Dict[str, Any]] = []
    multi_mask_flags: List[bool] = []

    # Step 0: initial prompt
    rand = random.random()
    if rand < initial_probs["mask"]:
        # Use ground‑truth mask as prompt
        init_prompt = {
            "is_prompted": True,
            "masks": gt_mask.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        }
        multi_mask = False
    elif rand < initial_probs["mask"] + initial_probs["click"]:
        # Single positive click
        coords, labels = click_sim.generate_initial_clicks(gt_mask.cpu().numpy())
        # coords: (N,2) pixel coords, labels: (N,) 1/0
        init_prompt = {
            "is_prompted": True,
            "coords": torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0),
            "labels": torch.tensor(labels, dtype=torch.int64, device=device).unsqueeze(0),
        }
        multi_mask = True  # ambiguous first click
    else:
        # Bounding box prompt
        # Compute bounding box from mask
        mask_np = gt_mask.cpu().numpy()
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if not rows.any():
            # Edge case: empty mask; fallback to click
            coords, labels = click_sim.generate_initial_clicks(mask_np)
            init_prompt = {
                "is_prompted": True,
                "coords": torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0),
                "labels": torch.tensor(labels, dtype=torch.int64, device=device).unsqueeze(0),
            }
            multi_mask = True
        else:
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # boxes are in (x1,y1,x2,y2) order and must be in pixel coordinates
            box = torch.tensor([[cmin, rmin, cmax, rmax]], dtype=torch.float32, device=device)
            init_prompt = {
                "is_prompted": True,
                "boxes": box,
            }
            multi_mask = False

    steps_prompts.append(init_prompt)
    multi_mask_flags.append(multi_mask)

    # Accumulate loss for all steps
    total_loss = 0.0

    # Forward pass for initial step
    model_output = model.forward(
        frames=img_batch,
        prompts=[init_prompt],
        memory_bank_state=None,
    )
    # model_output contains masks_logits (1, num_masks, H, W) -> squeeze batch
    masks_logits = model_output["masks_logits"].squeeze(0)  # (num_masks, H, W)
    iou_pred = model_output["iou_pred"].squeeze(0)           # (num_masks,)
    loss_weights = {
        "focal": cfg_pretrain.loss_weights.focal,
        "dice": cfg_pretrain.loss_weights.dice,
        "iou_l1": cfg_pretrain.loss_weights.iou_l1,
    }
    step_loss = compute_pretrain_losses(
        masks_logits, iou_pred, gt_mask, loss_weights, multi_mask
    )
    total_loss += step_loss

    # Determine the predicted mask (best after loss computation) for subsequent correction clicks
    # We'll compute binary masks for all candidates and select the one with minimal mask loss
    # (similar to the loss selection logic). Reuse the focal+dice per mask.
    focal_per_mask = []
    dice_per_mask = []
    for k in range(masks_logits.shape[0]):
        logits_k = masks_logits[k].unsqueeze(0)
        gt_k = gt_mask.unsqueeze(0)
        focal_per_mask.append(_focal_loss(logits_k, gt_k, alpha=0.25, gamma=2.0))
        dice_per_mask.append(_dice_loss(logits_k, gt_k))
    mask_losses = (
        loss_weights["focal"] * torch.stack(focal_per_mask)
        + loss_weights["dice"] * torch.stack(dice_per_mask)
    )
    best_idx = torch.argmin(mask_losses).item()
    best_mask = (torch.sigmoid(masks_logits[best_idx]) > 0.5).float()

    # Track the current set of prompts (cumulative clicks/boxes/mask)
    current_prompt = _merge_prompts(init_prompt, None)
    # We'll maintain a list of clicks for correction steps
    all_clicks_coords = []
    all_clicks_labels = []

    # Correction steps
    for step_idx in range(correction_clicks):
        # Generate one new click based on error between best_mask and gt_mask
        if not isinstance(click_sim, ClickSimulator):
            raise TypeError("click_sim must be an instance of ClickSimulator")
        new_coord, new_label = click_sim.generate_correction_clicks(
            best_mask.cpu().numpy(), gt_mask.cpu().numpy()
        )
        # Accumulate
        all_clicks_coords.append(new_coord[0])  # (2,)
        all_clicks_labels.append(new_label[0])

        # Build prompt for this correction step: include all previous clicks (initial + corrections)
        # For simplicity, we'll build a list of prompts but the model expects a single prompt dict per frame.
        # So we'll create a single combined prompt dict with all accumulated clicks.
        merged_prompt = _build_cumulative_click_prompt(
            initial_prompt=init_prompt,
            additional_coords=torch.tensor(all_clicks_coords, dtype=torch.float32, device=device),
            additional_labels=torch.tensor(all_clicks_labels, dtype=torch.int64, device=device),
        )
        # No multi_mask after first step
        multi_mask_step = False

        # Forward pass with the accumulated prompts
        model_output = model.forward(
            frames=img_batch,
            prompts=[merged_prompt],
            memory_bank_state=None,
        )
        masks_logits = model_output["masks_logits"].squeeze(0)
        iou_pred = model_output["iou_pred"].squeeze(0)
        step_loss = compute_pretrain_losses(
            masks_logits, iou_pred, gt_mask, loss_weights, multi_mask_step
        )
        total_loss += step_loss

        # Update best_mask for next step (select best according to current mask losses)
        focal_per_mask = []
        dice_per_mask = []
        for k in range(masks_logits.shape[0]):
            logits_k = masks_logits[k].unsqueeze(0)
            gt_k = gt_mask.unsqueeze(0)
            focal_per_mask.append(_focal_loss(logits_k, gt_k, alpha=0.25, gamma=2.0))
            dice_per_mask.append(_dice_loss(logits_k, gt_k))
        mask_losses_now = (
            loss_weights["focal"] * torch.stack(focal_per_mask)
            + loss_weights["dice"] * torch.stack(dice_per_mask)
        )
        best_idx = torch.argmin(mask_losses_now).item()
        best_mask = (torch.sigmoid(masks_logits[best_idx]) > 0.5).float()

    return total_loss


def _merge_prompts(
    existing: Dict[str, Any],
    new: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge two prompt dicts, combining click coordinates and masks."""
    if new is None:
        return existing
    merged = {}
    # For sparse prompts: if both have coords, concatenate
    if "coords" in existing and "coords" in new:
        merged["coords"] = torch.cat([existing["coords"], new["coords"]], dim=1)
        merged["labels"] = torch.cat([existing["labels"], new["labels"]], dim=1)
    elif "coords" in existing:
        merged["coords"] = existing["coords"]
        merged["labels"] = existing["labels"]
    elif "coords" in new:
        merged["coords"] = new["coords"]
        merged["labels"] = new["labels"]
    # For boxes: last one wins (unlikely in correction)
    if "boxes" in existing:
        merged["boxes"] = existing["boxes"]
    if "boxes" in new:
        merged["boxes"] = new["boxes"]
    # For masks: keep existing if present
    if "masks" in existing:
        merged["masks"] = existing["masks"]
    if "masks" in new:
        merged["masks"] = new["masks"]
    merged["is_prompted"] = True
    return merged


def _build_cumulative_click_prompt(
    initial_prompt: Dict[str, Any],
    additional_coords: torch.Tensor,  # (num_new, 2)
    additional_labels: torch.Tensor,  # (num_new,)
) -> Dict[str, Any]:
    """Build a prompt dict containing all clicks from initial and subsequent steps."""
    prompt = {"is_prompted": True}
    # Start with initial clicks if present
    if "coords" in initial_prompt:
        base_coords = initial_prompt["coords"]  # (1, N0, 2)
        base_labels = initial_prompt["labels"]  # (1, N0)
        # remove batch dim
        base_coords = base_coords.squeeze(0) if base_coords.dim() == 3 else base_coords
        base_labels = base_labels.squeeze(0) if base_labels.dim() == 2 else base_labels
    else:
        base_coords = torch.empty(0, 2, device=additional_coords.device, dtype=additional_coords.dtype)
        base_labels = torch.empty(0, device=additional_labels.device, dtype=additional_labels.dtype)

    # Concatenate additional clicks
    all_coords = torch.cat([base_coords, additional_coords], dim=0)
    all_labels = torch.cat([base_labels, additional_labels], dim=0)
    # Add batch dimension: (1, total_clicks, 2)
    prompt["coords"] = all_coords.unsqueeze(0)
    prompt["labels"] = all_labels.unsqueeze(0)

    # Preserve mask if initial was a mask
    if "masks" in initial_prompt:
        prompt["masks"] = initial_prompt["masks"]

    return prompt


# ---------------------------------------------------------------------------
#  Layer‑wise learning rate decay
# ---------------------------------------------------------------------------

def _get_param_groups_with_layer_decay(
    model: nn.Module,
    base_lr: float,
    layer_decay: float,
) -> List[Dict[str, Any]]:
    """
    Group parameters for layer‑wise learning rate decay.

    The Hiera image encoder parameters are assigned a multiplier that
    decays exponentially with depth, while all other parameters get the
    base learning rate.

    Args:
        model: SAM2Model instance.
        base_lr: base learning rate.
        layer_decay: decay factor per layer (e.g. 0.9 for B+).

    Returns:
        list of param group dicts suitable for AdamW.
    """
    param_groups = []
    # Collect image encoder parameters with depth information
    # Hiera's blocks are organised in stages; each stage contains a list of TransformerBlocks.
    # The paper applies layer‑wise decay to image encoder only.
    # We'll recursively walk the image_encoder submodule.
    def _assign_depth(module, prefix: str = ""):
        # Heuristic: depth is determined by number of TransformerBlock modules traversed.
        depth = 0
        if isinstance(module, nn.ModuleList) or isinstance(module, nn.Sequential):
            for name, child in module.named_children():
                sub_depth = _assign_depth(child, f"{prefix}.{name}" if prefix else name)
                depth = max(depth, sub_depth)
        else:
            for name, child in module.named_children():
                sub_depth = _assign_depth(child, f"{prefix}.{name}" if prefix else name)
                depth = max(depth, sub_depth)
        # If this module is a TransformerBlock, count it.
        classname = module.__class__.__name__
        if "TransformerBlock" in classname or "HieraStage" in classname:
            depth += 1
        return depth

    # Get maximum depth of image encoder
    max_depth = _assign_depth(model.image_encoder)
    if max_depth == 0:
        max_depth = 1  # fallback

    # Walk again and assign multiplier based on depth
    param_to_lr = {}

    def _walk_and_assign(module, prefix: str = "", current_depth: int = 0):
        # Assign current module parameters
        for n, p in module.named_parameters(recurse=False):
            if p.requires_grad:
                lr_mult = layer_decay ** (max_depth - current_depth)
                param_to_lr[p] = base_lr * lr_mult
        # Recurse
        for name, child in module.named_children():
            child_depth = current_depth
            classname = child.__class__.__name__
            if "TransformerBlock" in classname or "HieraStage" in classname:
                child_depth += 1
            _walk_and_assign(child, f"{prefix}.{name}" if prefix else name, child_depth)

    _walk_and_assign(model.image_encoder)

    # All other parameters (non‑image‑encoder) get base_lr
    other_params = []
    for n, p in model.named_parameters():
        if p.requires_grad and p not in param_to_lr:
            param_to_lr[p] = base_lr

    # Build groups: group by learning rate multiplier to reduce number of param groups
    lr_to_params: Dict[float, List[torch.nn.Parameter]] = {}
    for p, lr in param_to_lr.items():
        lr_to_params.setdefault(lr, []).append(p)

    for lr, params in lr_to_params.items():
        param_groups.append({"params": params, "lr": lr})

    return param_groups


# ---------------------------------------------------------------------------
#  Reciprocal square‑root schedule with warmup and cooldown
# ---------------------------------------------------------------------------

def _create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    cooldown_steps: int,
    timescale: float,
) -> LambdaLR:
    """
    Create a LambdaLR scheduler that follows the reciprocal square‑root
    schedule with linear warmup and cooldown.

    The returned scheduler expects a step() call after each optimizer step.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # linear warmup
            return step / max(1, warmup_steps)
        elif step > total_steps - cooldown_steps:
            # linear cooldown
            remaining = total_steps - step
            return max(0.0, remaining / cooldown_steps)
        else:
            # reciprocal sqrt
            return math.sqrt(timescale) / math.sqrt(step)

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
#  Main pretraining function
# ---------------------------------------------------------------------------

def pretrain_on_sa1b(
    model: SAM2Model,
    dataset: ImageDataset,
    config: Any,  # Config object
) -> SAM2Model:
    """
    Pre‑train SAM 2 on static images from SA‑1B.

    This function implements the full pre‑training stage as described in the
    SAM 2 paper, with the exact hyper‑parameters taken from ``config.yaml``.

    Args:
        model: A fresh (or partially trained) SAM2Model instance.
        dataset: ImageDataset providing SA‑1B samples.
        config: A Config object (or AttrDict) holding all hyper‑parameters.

    Returns:
        The trained SAM2Model (latest checkpoint also saved to disk).
    """
    pretrain_cfg = config.training.pretrain
    total_steps = pretrain_cfg.steps
    effective_batch_size = pretrain_cfg.batch_size
    base_lr = pretrain_cfg.learning_rate
    layer_decay = pretrain_cfg.layer_decay
    weight_decay = pretrain_cfg.weight_decay
    grad_clip_norm = pretrain_cfg.grad_clip_norm
    betas = tuple(pretrain_cfg.betas)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Create ClickSimulator (provides click generation algorithm)
    click_sim = ClickSimulator(strategy="centroid")  # default to centroid per paper

    # Dataloader: batch_size=1 because we iterate masks internally
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # Parameter groups with layer‑wise decay
    param_groups = _get_param_groups_with_layer_decay(model, base_lr, layer_decay)
    optimizer = AdamW(
        param_groups,
        betas=betas,
        weight_decay=weight_decay,
    )
    scheduler = _create_lr_scheduler(
        optimizer,
        base_lr,
        total_steps,
        pretrain_cfg.lr_schedule.warmup_steps,
        pretrain_cfg.lr_schedule.cooldown_steps,
        pretrain_cfg.lr_schedule.timescale,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=False)  # bfloat16 doesn't need scaling
    # We'll use autocast for mixed precision.

    global_step = 0
    accumulated_loss = 0.0
    optimizer.zero_grad()
    data_iter = iter(dataloader)

    # Training loop
    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        image: torch.Tensor = batch["image"].squeeze(0)  # (C, H, W)
        masks_tensor: torch.Tensor = batch["masks"].squeeze(0)  # (num_initial_masks, H, W)
        # masks_tensor is a tensor of shape (N_masks, H, W) from the dataset; it already
        # has filtered masks (area < threshold) and sampled up to max_masks_per_image.
        # However, the dataset may return all masks if max_masks_per_image > total; we'll
        # trust the dataset to do the random sampling. If not, we can sample here.

        num_masks = masks_tensor.shape[0]
        if num_masks == 0:
            continue

        # For each mask, run interactive sequence
        for m_idx in range(num_masks):
            gt_mask = masks_tensor[m_idx].float()
            # Skip empty mask (shouldn't happen)
            if gt_mask.sum() == 0:
                continue
            with autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
                loss = _simulate_interactive_sequence(
                    model, image, gt_mask, click_sim, config
                )
            # Scale loss by effective_batch_size (since we accumulate)
            (loss / effective_batch_size).backward()
            accumulated_loss += loss.item() / effective_batch_size
            global_step += 1

            if global_step % effective_batch_size == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging
                if global_step % (50 * effective_batch_size) == 0:
                    print(
                        f"Pretrain step {global_step}/{total_steps}, "
                        f"loss = {accumulated_loss:.4f}, "
                        f"lr = {optimizer.param_groups[0]['lr']:.6f}"
                    )
                accumulated_loss = 0.0

            if global_step >= total_steps:
                break

    # Save final checkpoint
    torch.save(model.state_dict(), "pretrained_sam2.pth")
    print("Pre‑training completed. Model saved to pretrained_sam2.pth")
    return model

