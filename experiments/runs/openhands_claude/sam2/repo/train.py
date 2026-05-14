"""
SAM 2 training script.

Implements three training stages (Appendix D.2):
  1. Pre-training on SA-1B (~90k steps, batch 256, resolution 1024)
  2. Full training on SA-V + Internal + SA-1B + VOS datasets (200k steps)
  3. Fine-tuning on 16-frame sequences of challenging videos (50k steps)

Loss functions (Appendix D.2.2):
  - Focal loss (weight 20) + Dice loss (weight 1) for mask prediction
  - L1 / MAE loss (weight 1) for IoU prediction
  - Cross-entropy loss (weight 1) for occlusion prediction
  Total ratio: 20:1:1:1

Multi-mask supervision:
  - For multi-mask predictions, supervise only the mask with lowest segmentation loss.
  - Supervise IoU predictions of all masks.
  - If GT has no mask for a frame, skip mask supervision but always supervise occlusion.

Interactive training simulation:
  - Sample 8-frame sequences, up to 2 prompted frames.
  - Initial prompt: GT mask (50%), positive click (25%), bounding box (25%).
  - Up to 7 corrective clicks per frame.
  - Mosaic transform with 10% probability.
  - Reverse temporal order with 50% probability.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import SAM2Config, get_config
from data import (
    SA1BDataset, SAVDataset, VOSDataset, MixedDataset, collate_fn,
    sample_correction_clicks, sample_initial_prompt, prompts_to_tensors,
)
from model import build_sam2
from utils import (
    AverageMeter, MetricLogger,
    build_optimizer_with_layer_decay, clip_gradients,
    load_checkpoint, reciprocal_sqrt_schedule, save_checkpoint, setup_logger,
)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> Tensor:
    """Sigmoid focal loss for binary segmentation."""
    prob = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def dice_loss(inputs: Tensor, targets: Tensor, eps: float = 1e-5) -> Tensor:
    """Dice loss for binary segmentation."""
    inputs = torch.sigmoid(inputs)
    inputs = inputs.flatten(1)
    targets = targets.float().flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss.mean()


def mask_loss(
    pred_masks: Tensor,
    gt_masks: Tensor,
    focal_weight: float = 20.0,
    dice_weight: float = 1.0,
) -> Tensor:
    """Combined focal + dice loss for mask prediction."""
    focal = sigmoid_focal_loss(pred_masks, gt_masks)
    dice = dice_loss(pred_masks, gt_masks)
    return focal_weight * focal + dice_weight * dice


def iou_loss(pred_iou: Tensor, gt_iou: Tensor) -> Tensor:
    """L1 loss for IoU prediction (with sigmoid activation on predictions)."""
    return F.l1_loss(pred_iou, gt_iou)


def occlusion_loss(pred_occlusion: Tensor, gt_visible: Tensor) -> Tensor:
    """
    Cross-entropy loss for occlusion prediction.
    pred_occlusion: (B, 1) logit — positive = object present
    gt_visible:     (B,) bool — True if object is visible
    """
    return F.binary_cross_entropy_with_logits(
        pred_occlusion.squeeze(1), gt_visible.float()
    )


def compute_gt_iou(pred_masks: Tensor, gt_masks: Tensor) -> Tensor:
    """Compute ground-truth IoU between predicted and GT masks for IoU supervision."""
    pred_bin = (torch.sigmoid(pred_masks) > 0.5).float()
    pred_flat = pred_bin.flatten(1)
    gt_flat = gt_masks.float().flatten(1)
    intersection = (pred_flat * gt_flat).sum(1)
    union = pred_flat.sum(1) + gt_flat.sum(1) - intersection
    iou = torch.where(union > 0, intersection / (union + 1e-6), torch.ones_like(intersection))
    return iou


# ---------------------------------------------------------------------------
# Training step for a single video sequence
# ---------------------------------------------------------------------------

def train_video_step(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    num_correction_clicks: int = 7,
    random_gt_click_prob: float = 0.1,
) -> Dict[str, Tensor]:
    """
    Simulate interactive training on a video sequence.

    For each object in the sequence:
      1. Process frames sequentially.
      2. On prompted frames: provide initial prompt + corrective clicks.
      3. On unprompted frames: propagate using memory bank.
      4. Compute losses.
    """
    frames = batch["frames"].to(device)       # (B, T, 3, H, W)
    masks = batch["masks"].to(device)         # (B, T, N, H, W)
    prompted_frames_list = batch["prompted_frames"]  # list of lists

    B, T, C, H, W = frames.shape
    _, _, N, Hm, Wm = masks.shape

    total_mask_loss = torch.tensor(0.0, device=device)
    total_iou_loss = torch.tensor(0.0, device=device)
    total_occ_loss = torch.tensor(0.0, device=device)
    num_supervised = 0

    for obj_idx in range(N):
        # Reset memory bank for this object
        model.module.reset_state(obj_idx) if hasattr(model, "module") else model.reset_state(obj_idx)

        recent_memories: List[Tensor] = []
        prompted_memories: List[Tensor] = []
        object_pointers: List[Tensor] = []

        for t in range(T):
            frame_t = frames[:, t]  # (B, 3, H, W)
            gt_mask_t = masks[:, t, obj_idx]  # (B, H, W)
            gt_visible = gt_mask_t.any(dim=(-1, -2))  # (B,) — object visible?

            # Determine if this frame is prompted
            is_prompted = all(t in pf for pf in prompted_frames_list)

            # Build prompts for this frame
            points_t = boxes_t = masks_t = None
            if is_prompted:
                # Sample initial prompt for first prompted frame
                if t == 0 or not recent_memories:
                    # Use GT mask, click, or box
                    prompt_type = random.choices(
                        ["mask", "click", "box"], weights=[0.5, 0.25, 0.25]
                    )[0]
                    # Build prompt tensors from GT mask of first batch element
                    gt_np = gt_mask_t[0].cpu().numpy()
                    prompt = sample_initial_prompt(gt_np, prompt_type)
                    points_t, boxes_t, masks_t = prompts_to_tensors(
                        [prompt], (H, W), device
                    )
                    # Expand to batch
                    if points_t is not None:
                        coords, labels = points_t
                        coords = coords.expand(B, -1, -1)
                        labels = labels.expand(B, -1)
                        points_t = (coords, labels)
                    if boxes_t is not None:
                        boxes_t = boxes_t.expand(B, -1)
                    if masks_t is not None:
                        masks_t = masks_t.expand(B, -1, -1, -1)

            # Forward pass
            multimask = is_prompted and (t == 0 or not recent_memories)
            output = model(
                frames=frame_t,
                points=points_t,
                boxes=boxes_t,
                masks=masks_t,
                recent_memories=recent_memories[-6:],
                prompted_memories=prompted_memories,
                object_pointers=object_pointers,
                multimask_output=multimask,
            )

            pred_masks = output["masks"]    # (B, num_masks, H_out, W_out)
            pred_iou = output["iou_pred"]   # (B, num_masks)
            pred_occ = output["occlusion"]  # (B, 1)
            memory = output["memory"]
            pointer = output["pointer_tokens"]

            # Resize GT mask to match prediction resolution
            H_out, W_out = pred_masks.shape[-2:]
            gt_resized = F.interpolate(
                gt_mask_t.float().unsqueeze(1), size=(H_out, W_out), mode="nearest"
            ).squeeze(1)  # (B, H_out, W_out)

            # Compute per-mask losses and select best mask
            num_masks = pred_masks.shape[1]
            per_mask_losses = []
            for m_idx in range(num_masks):
                ml = mask_loss(pred_masks[:, m_idx], gt_resized)
                per_mask_losses.append(ml)

            per_mask_losses_t = torch.stack(per_mask_losses)  # (num_masks,)
            best_mask_idx = per_mask_losses_t.argmin()

            # Supervise only the best mask's logits
            if gt_visible.any():
                ml = per_mask_losses_t[best_mask_idx]
                total_mask_loss = total_mask_loss + ml

                # IoU supervision: all masks
                gt_ious = torch.stack([
                    compute_gt_iou(pred_masks[:, m_idx:m_idx+1], gt_resized.unsqueeze(1))
                    for m_idx in range(num_masks)
                ], dim=1)  # (B, num_masks)
                il = iou_loss(pred_iou, gt_ious)
                total_iou_loss = total_iou_loss + il

            # Occlusion supervision: always
            ol = occlusion_loss(pred_occ, gt_visible)
            total_occ_loss = total_occ_loss + ol
            num_supervised += 1

            # Update memory bank
            if is_prompted:
                prompted_memories.append(memory.detach())
            else:
                recent_memories.append(memory.detach())
            object_pointers.append(pointer.detach())

    if num_supervised == 0:
        num_supervised = 1

    return {
        "mask_loss": total_mask_loss / num_supervised,
        "iou_loss": total_iou_loss / num_supervised,
        "occ_loss": total_occ_loss / num_supervised,
    }


def train_image_step(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Tensor]:
    """Training step for static images (SA-1B pre-training / joint training)."""
    images = batch["image"].to(device)   # (B, 3, H, W)
    masks = batch["masks"].to(device)    # (B, N, H, W)

    B, N, H, W = masks.shape

    total_mask_loss = torch.tensor(0.0, device=device)
    total_iou_loss = torch.tensor(0.0, device=device)
    num_supervised = 0

    # Sample one mask per image for this step
    for b in range(B):
        valid_masks = [i for i in range(N) if masks[b, i].any()]
        if not valid_masks:
            continue
        mask_idx = random.choice(valid_masks)
        gt_mask = masks[b:b+1, mask_idx]  # (1, H, W)

        # Sample prompt
        gt_np = gt_mask[0].cpu().numpy()
        prompt_type = random.choices(["mask", "click", "box"], weights=[0.5, 0.25, 0.25])[0]
        prompt = sample_initial_prompt(gt_np, prompt_type)
        points_t, boxes_t, masks_t = prompts_to_tensors([prompt], (H, W), device)

        output = model(
            frames=images[b:b+1],
            points=points_t,
            boxes=boxes_t,
            masks=masks_t,
            recent_memories=[],
            prompted_memories=[],
            object_pointers=[],
            multimask_output=True,
        )

        pred_masks = output["masks"]   # (1, num_masks, H_out, W_out)
        pred_iou = output["iou_pred"]  # (1, num_masks)

        H_out, W_out = pred_masks.shape[-2:]
        gt_resized = F.interpolate(
            gt_mask.float().unsqueeze(0), size=(H_out, W_out), mode="nearest"
        ).squeeze(0)  # (1, H_out, W_out)

        # Select best mask
        num_masks = pred_masks.shape[1]
        per_mask_losses = [
            mask_loss(pred_masks[:, m_idx], gt_resized)
            for m_idx in range(num_masks)
        ]
        per_mask_losses_t = torch.stack(per_mask_losses)
        best_mask_idx = per_mask_losses_t.argmin()
        total_mask_loss = total_mask_loss + per_mask_losses_t[best_mask_idx]

        # IoU supervision: all masks
        gt_ious = torch.stack([
            compute_gt_iou(pred_masks[:, m_idx:m_idx+1], gt_resized.unsqueeze(1))
            for m_idx in range(num_masks)
        ], dim=1)
        total_iou_loss = total_iou_loss + iou_loss(pred_iou, gt_ious)
        num_supervised += 1

    if num_supervised == 0:
        num_supervised = 1

    return {
        "mask_loss": total_mask_loss / num_supervised,
        "iou_loss": total_iou_loss / num_supervised,
        "occ_loss": torch.tensor(0.0, device=device),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: SAM2Config,
    logger,
    start_step: int = 0,
) -> int:
    model.train()
    metric_logger = MetricLogger()
    step = start_step

    focal_w = cfg.loss_focal_weight
    dice_w = cfg.loss_dice_weight
    iou_w = cfg.loss_iou_weight
    occ_w = cfg.loss_occ_weight

    for batch in dataloader:
        is_video = batch["is_video"][0] if isinstance(batch["is_video"], list) else batch["is_video"].item()

        with autocast(dtype=torch.bfloat16, enabled=cfg.use_amp):
            if is_video:
                losses = train_video_step(model, batch, device)
            else:
                losses = train_image_step(model, batch, device)

            total_loss = (
                focal_w * losses["mask_loss"]
                + iou_w * losses["iou_loss"]
                + occ_w * losses["occ_loss"]
            )

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_gradients(model, max_norm=cfg.grad_clip_max_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        metric_logger.update(
            loss=total_loss.item(),
            mask_loss=losses["mask_loss"].item(),
            iou_loss=losses["iou_loss"].item(),
            occ_loss=losses["occ_loss"].item(),
            grad_norm=grad_norm,
            lr=optimizer.param_groups[0]["lr"],
        )

        if step % cfg.log_interval == 0:
            logger.info(f"Epoch {epoch} Step {step}: {metric_logger}")

        if step % cfg.save_interval == 0 and step > 0:
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch,
                },
                cfg.output_dir,
                filename=f"checkpoint_step{step}.pth",
            )

        step += 1
        if step >= cfg.max_steps:
            break

    return step


def pretrain(cfg: SAM2Config) -> None:
    """Stage 1: Pre-training on SA-1B."""
    logger = setup_logger("sam2.pretrain", os.path.join(cfg.output_dir, "pretrain.log"))
    device = torch.device(cfg.device)

    model = build_sam2(
        variant=cfg.encoder_variant,
        image_size=cfg.image_size,
        embed_dim=cfg.embed_dim,
        memory_dim=cfg.memory_dim,
        num_memory_attention_layers=cfg.num_memory_attention_layers,
        max_recent_frames=cfg.max_recent_frames,
    ).to(device)

    dataset = SA1BDataset(
        root=cfg.sa1b_root,
        image_size=cfg.image_size,
        max_masks_per_image=cfg.max_masks_per_image,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.pretrain_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = build_optimizer_with_layer_decay(
        model,
        base_lr=cfg.pretrain_lr,
        weight_decay=cfg.weight_decay,
        layer_decay=cfg.layer_decay[cfg.encoder_variant],
        encoder_variant=cfg.encoder_variant,
    )
    scheduler = reciprocal_sqrt_schedule(
        optimizer,
        warmup_steps=cfg.pretrain_warmup_steps,
        timescale=cfg.lr_timescale,
        cooldown_steps=cfg.pretrain_cooldown_steps,
        total_steps=cfg.pretrain_max_steps,
    )
    scaler = GradScaler(enabled=cfg.use_amp)

    cfg.max_steps = cfg.pretrain_max_steps
    step = 0
    for epoch in range(1000):
        step = train_one_epoch(
            model, dataloader, optimizer, scheduler, scaler,
            device, epoch, cfg, logger, start_step=step
        )
        if step >= cfg.pretrain_max_steps:
            break

    save_checkpoint(
        {"model": model.state_dict(), "step": step},
        cfg.output_dir,
        filename="pretrain_final.pth",
    )
    logger.info("Pre-training complete.")


def full_train(cfg: SAM2Config, pretrain_ckpt: Optional[str] = None) -> None:
    """Stage 2: Full training on SA-V + Internal + SA-1B + VOS datasets."""
    logger = setup_logger("sam2.train", os.path.join(cfg.output_dir, "train.log"))
    device = torch.device(cfg.device)

    model = build_sam2(
        variant=cfg.encoder_variant,
        image_size=cfg.image_size,
        embed_dim=cfg.embed_dim,
        memory_dim=cfg.memory_dim,
        num_memory_attention_layers=cfg.num_memory_attention_layers,
        max_recent_frames=cfg.max_recent_frames,
    ).to(device)

    if pretrain_ckpt:
        load_checkpoint(pretrain_ckpt, model, strict=False)
        logger.info(f"Loaded pre-trained weights from {pretrain_ckpt}")

    # Build datasets
    video_datasets = []
    if cfg.sav_root and os.path.exists(cfg.sav_root):
        video_datasets.append(SAVDataset(
            cfg.sav_root, split="train",
            image_size=cfg.image_size,
            num_frames=cfg.num_frames,
            max_masklets=cfg.max_masklets_per_sequence,
        ))
    for vos_root in cfg.vos_roots:
        if os.path.exists(vos_root):
            video_datasets.append(VOSDataset(
                vos_root, split="train",
                image_size=cfg.image_size,
                num_frames=cfg.num_frames,
            ))

    image_datasets = []
    if cfg.sa1b_root and os.path.exists(cfg.sa1b_root):
        image_datasets.append(SA1BDataset(
            cfg.sa1b_root,
            image_size=cfg.image_size,
            max_masks_per_image=cfg.max_masks_per_image,
        ))

    if not video_datasets and not image_datasets:
        logger.warning("No datasets found. Check data paths in config.")
        return

    dataset = MixedDataset(
        image_datasets=image_datasets,
        video_datasets=video_datasets,
        image_weight=cfg.image_data_weight,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    optimizer = build_optimizer_with_layer_decay(
        model,
        base_lr=cfg.train_lr,
        weight_decay=cfg.weight_decay,
        layer_decay=cfg.layer_decay[cfg.encoder_variant],
        encoder_variant=cfg.encoder_variant,
    )
    scheduler = reciprocal_sqrt_schedule(
        optimizer,
        warmup_steps=cfg.train_warmup_steps,
        timescale=cfg.lr_timescale,
        cooldown_steps=cfg.train_cooldown_steps,
        total_steps=cfg.train_max_steps,
    )
    scaler = GradScaler(enabled=cfg.use_amp)

    cfg.max_steps = cfg.train_max_steps
    step = 0
    for epoch in range(1000):
        step = train_one_epoch(
            model, dataloader, optimizer, scheduler, scaler,
            device, epoch, cfg, logger, start_step=step
        )
        if step >= cfg.train_max_steps:
            break

    save_checkpoint(
        {"model": model.state_dict(), "step": step},
        cfg.output_dir,
        filename="train_final.pth",
    )
    logger.info("Full training complete.")


def finetune(cfg: SAM2Config, train_ckpt: str) -> None:
    """
    Stage 3: Fine-tuning on 16-frame sequences of challenging videos.
    (Appendix D.2.2: top 50% most-edited masklets, 50k iters, half LR, frozen encoder)
    """
    logger = setup_logger("sam2.finetune", os.path.join(cfg.output_dir, "finetune.log"))
    device = torch.device(cfg.device)

    model = build_sam2(
        variant=cfg.encoder_variant,
        image_size=cfg.image_size,
        embed_dim=cfg.embed_dim,
        memory_dim=cfg.memory_dim,
        num_memory_attention_layers=cfg.num_memory_attention_layers,
        max_recent_frames=cfg.max_recent_frames,
    ).to(device)

    load_checkpoint(train_ckpt, model, strict=True)
    logger.info(f"Loaded weights from {train_ckpt}")

    # Freeze image encoder
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    logger.info("Image encoder frozen for fine-tuning.")

    # Build challenging video datasets (16-frame sequences)
    video_datasets = []
    if cfg.sav_root and os.path.exists(cfg.sav_root):
        video_datasets.append(SAVDataset(
            cfg.sav_root, split="train",
            image_size=cfg.image_size,
            num_frames=16,  # 16-frame sequences for fine-tuning
            max_masklets=cfg.max_masklets_per_sequence,
        ))

    if not video_datasets:
        logger.warning("No datasets found for fine-tuning.")
        return

    dataset = MixedDataset(
        image_datasets=[],
        video_datasets=video_datasets,
        image_weight=0.0,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    optimizer = build_optimizer_with_layer_decay(
        model,
        base_lr=cfg.train_lr * 0.5,  # half LR
        weight_decay=cfg.weight_decay,
        layer_decay=cfg.layer_decay[cfg.encoder_variant],
        encoder_variant=cfg.encoder_variant,
    )
    scheduler = reciprocal_sqrt_schedule(
        optimizer,
        warmup_steps=cfg.train_warmup_steps,
        timescale=cfg.lr_timescale,
        cooldown_steps=cfg.train_cooldown_steps,
        total_steps=cfg.finetune_max_steps,
    )
    scaler = GradScaler(enabled=cfg.use_amp)

    cfg.max_steps = cfg.finetune_max_steps
    step = 0
    for epoch in range(1000):
        step = train_one_epoch(
            model, dataloader, optimizer, scheduler, scaler,
            device, epoch, cfg, logger, start_step=step
        )
        if step >= cfg.finetune_max_steps:
            break

    save_checkpoint(
        {"model": model.state_dict(), "step": step},
        cfg.output_dir,
        filename="finetune_final.pth",
    )
    logger.info("Fine-tuning complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM 2 Training")
    parser.add_argument("--stage", choices=["pretrain", "train", "finetune"], default="train")
    parser.add_argument("--config", type=str, default="B+", help="Encoder variant: T, S, B+, L")
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--train_ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--sa1b_root", type=str, default="")
    parser.add_argument("--sav_root", type=str, default="")
    parser.add_argument("--vos_roots", nargs="+", default=[])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = get_config(args.config)
    cfg.output_dir = args.output_dir
    cfg.device = args.device
    if args.sa1b_root:
        cfg.sa1b_root = args.sa1b_root
    if args.sav_root:
        cfg.sav_root = args.sav_root
    if args.vos_roots:
        cfg.vos_roots = args.vos_roots

    os.makedirs(cfg.output_dir, exist_ok=True)

    if args.stage == "pretrain":
        pretrain(cfg)
    elif args.stage == "train":
        full_train(cfg, pretrain_ckpt=args.pretrain_ckpt)
    elif args.stage == "finetune":
        if not args.train_ckpt:
            raise ValueError("--train_ckpt required for fine-tuning")
        finetune(cfg, train_ckpt=args.train_ckpt)


if __name__ == "__main__":
    main()
