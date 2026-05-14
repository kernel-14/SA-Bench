"""Training script for SAM 2.

Implements the training protocol from the paper:
1. Pre-training on SA-1B (static images)
2. Full training on mixed image+video data with interactive prompting simulation
3. Optional fine-tuning on 16-frame sequences

Training simulates interactive prompting:
- Sample 8-frame sequences
- Randomly select up to 2 frames for corrective clicks
- Initial prompts: mask (50%), click (25%), box (25%)
- Correction clicks sampled using ground-truth masklet and model predictions
- With 10% probability, sample random clicks from ground truth
"""

import argparse
import math
import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from config import get_config, SAM2Config, TrainingConfig
from sam2_model import SAM2, build_sam2
from losses import SAM2Loss, compute_iou
from data import build_dataloaders, SAM2VideoAugmentation, MixedBatchSampler
from prompt_encoder import sample_prompts_from_ground_truth


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer(model: nn.Module, config: TrainingConfig, encoder_size: str) -> torch.optim.Optimizer:
    """Create AdamW optimizer with layer-wise decay (Clark et al., 2020)."""
    layer_decay = config.layer_decay

    # Separate image encoder from other parameters for layer decay
    image_encoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "image_encoder" in name or "hiera" in name:
            # Apply layer decay based on depth
            depth = sum(1 for _ in name.split(".") if _.startswith("block") or _.startswith("stage"))
            decay = layer_decay ** depth if "image_encoder" in name else layer_decay
            image_encoder_params.append({"params": param, "lr_scale": decay})
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "lr_scale": 1.0},
    ] + image_encoder_params

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )
    return optimizer


class ReciprocalSqrtScheduler:
    """Reciprocal square-root learning rate schedule with warmup and cooldown."""

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float,
                 timescale: int = 1000, warmup_steps: int = 1000,
                 cooldown_steps: int = 5000, total_steps: int = 200000):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.timescale = timescale
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.total_steps = total_steps
        self.step_count = 0

    def get_lr(self) -> float:
        """Get current learning rate."""
        if self.step_count < self.warmup_steps:
            return self.base_lr * self.step_count / self.warmup_steps
        elif self.step_count >= self.total_steps - self.cooldown_steps:
            progress = (self.total_steps - self.step_count) / self.cooldown_steps
            return self.base_lr * progress
        else:
            return self.base_lr * self.timescale ** 0.5 / (self.timescale + self.step_count) ** 0.5

    def step(self):
        """Update learning rate."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr * param_group.get("lr_scale", 1.0)
        self.step_count += 1


def train_step_image(model: SAM2, batch: dict, loss_fn: SAM2Loss,
                     device: torch.device, scaler: Optional[GradScaler] = None) -> Tuple[torch.Tensor, dict]:
    """Single training step on image data (SA-1B).

    Follows SAM training: iterative point sampling, multi-mask on first click.
    """
    images = batch["image"].to(device)
    gt_masks = batch["masks"].to(device)  # [B, N_masks, H, W]

    B = images.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    total_masks_processed = 0

    for b in range(B):
        image = images[b:b+1]  # [1, 3, H, W]
        gt_b = gt_masks[b]     # [N, H, W]

        # Select a random mask
        mask_idx = random.randint(0, gt_b.shape[0] - 1)
        gt_mask = gt_b[mask_idx:mask_idx+1]  # [1, H, W]

        # Sample initial prompt
        coords, labels, boxes, masks = sample_prompts_from_ground_truth(
            gt_mask.unsqueeze(0), mask_prompt_prob=0.5, click_prompt_prob=0.25, box_prompt_prob=0.25
        )

        # Forward pass
        with autocast(enabled=scaler is not None):
            output = model.forward_image(
                image, coords=coords, labels=labels, boxes=boxes, masks=masks,
                multimask_output=True,
            )

            # Compute IoU between pred and gt
            pred_masks = output["masks"]  # [1, 3, H, W]
            gt_expanded = gt_mask.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1, -1)
            gt_iou = compute_iou(pred_masks, gt_mask.unsqueeze(0).unsqueeze(0))

            loss, loss_dict = loss_fn(
                output, gt_mask.unsqueeze(0).unsqueeze(0),
                gt_iou=gt_iou, is_multi_mask=True,
            )

        total_loss = total_loss + loss
        total_masks_processed += 1

    total_loss = total_loss / max(total_masks_processed, 1)
    return total_loss, {"image_loss": total_loss.item()}


def train_step_video(model: SAM2, batch: dict, loss_fn: SAM2Loss,
                     device: torch.device, max_correction_clicks: int = 3,
                     scaler: Optional[GradScaler] = None) -> Tuple[torch.Tensor, dict]:
    """Single training step on video data (SA-V, Internal, VOS).

    Simulates interactive video segmentation:
    - Sample 8-frame sequences
    - Up to 2 prompted frames with correction clicks
    """
    frames = batch["frames"].to(device)    # [B, T, 3, H, W]
    masklets = batch["masklets"].to(device)  # [B, T, H, W]

    B, T, C, H, W = frames.shape
    total_loss = torch.tensor(0.0, device=device)

    for b in range(B):
        video_frames = frames[b]     # [T, 3, H, W]
        gt_masklet = masklets[b]     # [T, H, W]

        model.reset_memory()

        # Randomly select up to 2 frames to prompt (including first frame)
        num_prompt_frames = random.randint(1, min(2, T))
        prompt_frame_indices = sorted(random.sample(range(T), num_prompt_frames))
        if 0 not in prompt_frame_indices:
            prompt_frame_indices[0] = 0  # First frame always gets an initial prompt
            prompt_frame_indices = sorted(set(prompt_frame_indices))

        frame_losses = []
        current_prompt_frames = set(prompt_frame_indices)

        for t in range(T):
            frame = video_frames[t:t+1]  # [1, 3, H, W]
            gt = gt_masklet[t:t+1]       # [1, H, W]

            has_object = gt.sum() > 0

            if t in current_prompt_frames and has_object:
                # Sample initial or correction prompts
                coords, labels, boxes, masks = sample_prompts_from_ground_truth(
                    gt.unsqueeze(0), mask_prompt_prob=0.5, click_prompt_prob=0.25, box_prompt_prob=0.25
                )
            else:
                coords, labels, boxes, masks = None, None, None, None

            with autocast(enabled=scaler is not None):
                output = model(
                    frame,
                    coords=coords, labels=labels, boxes=boxes, masks=masks,
                    is_first_frame=(t == 0),
                    multimask_output=(t == 0),
                )

                pred_masks = output["masks"]  # [1, N_masks, H, W] or [1, 1, H, W]
                gt_exp = gt.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

                # Compute ground-truth IoU
                if pred_masks.shape[1] > 1:
                    gt_iou = compute_iou(pred_masks, gt_exp)
                else:
                    gt_iou = None

                gt_occlusion = torch.tensor([[float(has_object)]], device=device)

                loss, loss_dict = loss_fn(
                    output, gt_exp,
                    gt_iou=gt_iou,
                    gt_occlusion=gt_occlusion,
                    is_multi_mask=(pred_masks.shape[1] > 1),
                    frame_has_object=torch.tensor([has_object], device=device),
                )
                frame_losses.append(loss)

        total_loss = total_loss + torch.stack(frame_losses).mean()

    total_loss = total_loss / B
    return total_loss, {"video_loss": total_loss.item()}


def train_epoch(model: SAM2, train_loader, loss_fn: SAM2Loss,
                optimizer: torch.optim.Optimizer, scheduler: ReciprocalSqrtScheduler,
                device: torch.device, config: TrainingConfig,
                epoch: int, scaler: Optional[GradScaler] = None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_steps = 0

    if hasattr(train_loader, "batch_sampler"):
        train_loader.batch_sampler.set_epoch(epoch)

    for batch_idx, batch in enumerate(train_loader):
        data_type = batch.get("type", "video") if isinstance(batch, dict) else "mixed"

        optimizer.zero_grad()

        if data_type == "image":
            loss, loss_dict = train_step_image(model, batch, loss_fn, device, scaler)
        else:
            loss, loss_dict = train_step_video(
                model, batch, loss_fn, device,
                max_correction_clicks=config.max_correction_clicks_video,
                scaler=scaler,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            if config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        total_steps += 1

        if batch_idx % 100 == 0:
            current_lr = scheduler.get_lr()
            print(f"Epoch {epoch} | Step {batch_idx} | Loss {loss.item():.4f} | LR {current_lr:.6f}")

    return total_loss / max(total_steps, 1)


def validate(model: SAM2, val_loader, loss_fn: SAM2Loss, device: torch.device) -> Dict[str, float]:
    """Validation loop."""
    model.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            data_type = batch.get("type", "video") if isinstance(batch, dict) else "video"

            if data_type == "image":
                loss, _ = train_step_image(model, batch, loss_fn, device)
            else:
                loss, _ = train_step_video(model, batch, loss_fn, device)

            val_loss += loss.item()
            num_batches += 1

    return {"val_loss": val_loss / max(num_batches, 1)}


def pretrain_sa1b(model: SAM2, config: TrainingConfig,
                  pretrain_loader: DataLoader, device: torch.device):
    """Pre-train on SA-1B following SAM protocol.

    Settings from Table 12 (a):
    - Steps: ~90k
    - Batch size: 256
    - Resolution: 1024
    - LR: 4e-4 with reciprocal sqrt schedule
    - 7 correction clicks
    """
    print("Starting SA-1B pre-training...")

    loss_fn = SAM2Loss()
    optimizer = get_optimizer(model, config, "b_plus")
    scheduler = ReciprocalSqrtScheduler(
        optimizer, config.pretrain_learning_rate, timescale=1000,
        warmup_steps=1000, cooldown_steps=5000, total_steps=config.pretrain_steps,
    )
    scaler = GradScaler() if config.precision == "float16" else None

    for epoch in range(10000):  # Will break when steps reached
        avg_loss = train_epoch(
            model, pretrain_loader, loss_fn, optimizer, scheduler,
            device, config, epoch, scaler,
        )
        print(f"Pre-training Epoch {epoch}: avg_loss = {avg_loss:.4f}")

        if scheduler.step_count >= config.pretrain_steps:
            break

    # Save pre-trained checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "sam2_pretrained_sa1b.pth")

    print("SA-1B pre-training complete.")


def train_sam2(model: SAM2, config: TrainingConfig,
               train_loader: DataLoader, val_loader: Optional[DataLoader],
               device: torch.device, resume_from: Optional[str] = None):
    """Full SAM 2 training on mixed image+video data.

    Settings from Table 12 (b):
    - Alternating training between video and image data
    - 8-frame sequences
    - Up to 2 prompted frames
    - Batch sizes proportional to data source sizes
    """
    print("Starting SAM 2 full training...")

    loss_fn = SAM2Loss()
    optimizer = get_optimizer(model, config, "b_plus")
    scheduler = ReciprocalSqrtScheduler(
        optimizer, config.learning_rate, timescale=config.lr_timescale,
        warmup_steps=config.warmup_steps, cooldown_steps=config.cooldown_steps,
        total_steps=config.train_steps,
    )
    scaler = GradScaler() if config.precision == "float16" else None

    start_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from {resume_from} at epoch {start_epoch}")

    best_val_loss = float("inf")

    for epoch in range(start_epoch, 10000):
        avg_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler,
            device, config, epoch, scaler,
        )
        print(f"Training Epoch {epoch}: avg_loss = {avg_loss:.4f}")

        if val_loader is not None:
            val_metrics = validate(model, val_loader, loss_fn, device)
            print(f"Validation Epoch {epoch}: {val_metrics}")

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                }, "sam2_best.pth")

        if scheduler.step_count >= config.train_steps:
            break

    # Final checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "sam2_final.pth")

    print("Full training complete.")


def finetune_long_video(model: SAM2, config: TrainingConfig,
                        train_loader: DataLoader, device: torch.device):
    """Fine-tune on 16-frame sequences for better long-video performance.

    - Sample 16-frame sequences from challenging videos (top 50% most edited)
    - Half of original learning rate
    - Freeze image encoder
    - 50k iterations (1/3 of original schedule)
    """
    print("Starting 16-frame fine-tuning...")

    # Freeze image encoder
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    loss_fn = SAM2Loss()
    finetune_lr = config.learning_rate * config.finetune_learning_rate_multiplier
    optimizer = get_optimizer(model, config, "b_plus")
    optimizer.param_groups[0]["lr"] = finetune_lr

    scheduler = ReciprocalSqrtScheduler(
        optimizer, finetune_lr, timescale=500,
        warmup_steps=500, cooldown_steps=2500,
        total_steps=config.finetune_steps,
    )

    for epoch in range(10000):
        avg_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler,
            device, config, epoch,
        )
        print(f"Fine-tuning Epoch {epoch}: avg_loss = {avg_loss:.4f}")

        if scheduler.step_count >= config.finetune_steps:
            break

    torch.save({
        "model_state_dict": model.state_dict(),
    }, "sam2_finetuned.pth")

    print("Fine-tuning complete.")


def main():
    parser = argparse.ArgumentParser(description="SAM 2 Training")
    parser.add_argument("--encoder-size", type=str, default="b_plus",
                       choices=["t", "s", "b_plus", "l"])
    parser.add_argument("--mode", type=str, default="full",
                       choices=["pretrain", "full", "finetune", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--sa1b-root", type=str, default="data/sa1b")
    parser.add_argument("--sav-root", type=str, default="data/sav")
    parser.add_argument("--davis-root", type=str, default=None)
    parser.add_argument("--mose-root", type=str, default=None)
    parser.add_argument("--ytvos-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build model config
    model_config = get_config(args.encoder_size)
    train_config = model_config.training_config
    assert train_config is not None, "Training config not set in model config"

    # Build SAM 2 model
    model = build_sam2(args.encoder_size)
    model = model.to(device)

    print(f"Built SAM 2 ({args.encoder_size}) with {sum(p.numel() for p in model.parameters()):,} parameters")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in ("pretrain", "all"):
        # SA-1B pre-training dataloader
        pretrain_cfg = {
            "sa1b_root": args.sa1b_root,
            "image_size": train_config.pretrain_image_size,
            "max_masks_per_image": 64,
            "max_mask_area_ratio": 0.9,
            "image_batch_size": train_config.pretrain_batch_size,
            "image_prob": 1.0,
            "num_workers": 8,
        }
        pretrain_loader, _ = build_dataloaders(pretrain_cfg)
        pretrain_sa1b(model, train_config, pretrain_loader, device)

    if args.mode in ("full", "all"):
        # Full training dataloaders
        full_cfg = {
            "sa1b_root": args.sa1b_root,
            "sav_root": args.sav_root,
            "davis_root": args.davis_root,
            "mose_root": args.mose_root,
            "ytvos_root": args.ytvos_root,
            "image_size": train_config.train_image_size,
            "num_frames": train_config.train_num_frames,
            "max_masks_per_image": 64,
            "max_mask_area_ratio": 0.9,
            "image_batch_size": train_config.train_batch_size_images,
            "video_batch_size": train_config.train_batch_size_video,
            "image_prob": train_config.sa1b_ratio,
            "num_workers": 8,
            "use_horizontal_flip": train_config.use_horizontal_flip,
            "use_affine_transform": train_config.use_affine_transform,
            "use_color_jitter": train_config.use_color_jitter,
            "use_grayscale": train_config.use_grayscale,
            "mosaic_prob": train_config.mosaic_prob,
            "reverse_time_prob": train_config.reverse_time_prob,
        }
        train_loader, data_info = build_dataloaders(full_cfg)
        print(f"Data info: {data_info}")

        train_sam2(model, train_config, train_loader, None, device, args.resume)

    if args.mode in ("finetune", "all"):
        # Fine-tuning dataloader with 16 frames
        finetune_cfg = {
            "sa1b_root": args.sa1b_root,
            "sav_root": args.sav_root,
            "image_size": train_config.train_image_size,
            "num_frames": train_config.train_num_frames_finetune,
            "max_masks_per_image": 64,
            "max_mask_area_ratio": 0.9,
            "image_batch_size": train_config.train_batch_size_images,
            "video_batch_size": train_config.train_batch_size_video // 2,
            "image_prob": train_config.sa1b_ratio,
            "num_workers": 8,
        }
        finetune_loader, _ = build_dataloaders(finetune_cfg)
        finetune_long_video(model, train_config, finetune_loader, device)


if __name__ == "__main__":
    main()
