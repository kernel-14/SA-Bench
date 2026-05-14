"""
SAM 2 Training Script

Implements the training procedure from Section D.2 of the paper:

Pre-training (on SA-1B):
- ~90k steps, batch size 256
- AdamW optimizer with layer decay
- Reciprocal square-root learning rate schedule
- Resolution: 1024x1024

Full training (on SA-V + Internal + SA-1B + VOS datasets):
- Alternating between video and image batches
- 8-frame sequences for video
- Up to 2 prompted frames per sequence
- Interactive prompt simulation

Fine-tuning (16-frame sequences):
- 50k iterations (1/3 of original schedule)
- Half learning rate
- Frozen image encoder
- Only top 50% most edited masklets
"""

import argparse
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

from sam2.modeling.sam2_model import SAM2Model, build_sam2
from sam2.modeling.losses import SAM2Loss
from sam2.datasets.video_dataset import VideoSegmentationDataset, ImageSegmentationDataset


def get_reciprocal_sqrt_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 1000,
    timescale: int = 1000,
    cooldown_steps: int = 5000,
    total_steps: int = 90000,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Reciprocal square-root learning rate schedule with warmup and cooldown.
    From the paper: "follow a reciprocal square-root schedule (Zhai et al., 2022)"
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        elif step > total_steps - cooldown_steps:
            # Linear cooldown
            progress = (step - (total_steps - cooldown_steps)) / cooldown_steps
            base_lr = math.sqrt(timescale / max(step, timescale))
            return base_lr * (1 - progress)
        else:
            # Reciprocal square-root decay
            return math.sqrt(timescale / max(step, timescale))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_layer_decay_param_groups(
    model: nn.Module,
    base_lr: float,
    layer_decay: float,
    weight_decay: float = 0.1,
) -> List[Dict]:
    """
    Create parameter groups with layer-wise learning rate decay.
    Applied to the image encoder following Clark et al. (2020).
    """
    param_groups = []

    # Image encoder parameters with layer decay
    if hasattr(model, 'image_encoder'):
        encoder = model.image_encoder
        # Get all named parameters
        encoder_params = {}
        for name, param in encoder.named_parameters():
            if not param.requires_grad:
                continue
            # Determine layer depth for decay
            if 'patch_embed' in name or 'pos_embed' in name:
                layer_id = 0
            elif 'stages' in name:
                # Extract stage and block index
                parts = name.split('.')
                try:
                    stage_idx = int(parts[1])
                    layer_id = stage_idx + 1
                except (ValueError, IndexError):
                    layer_id = 1
            else:
                layer_id = len(encoder.stages) + 1

            lr_scale = layer_decay ** (len(encoder.stages) + 1 - layer_id)
            encoder_params[name] = (param, lr_scale)

        # Group by lr_scale
        lr_scale_groups = {}
        for name, (param, lr_scale) in encoder_params.items():
            if lr_scale not in lr_scale_groups:
                lr_scale_groups[lr_scale] = {'params': [], 'lr': base_lr * lr_scale}
            lr_scale_groups[lr_scale]['params'].append(param)

        for lr_scale, group in lr_scale_groups.items():
            param_groups.append({
                'params': group['params'],
                'lr': group['lr'],
                'weight_decay': weight_decay,
            })

    # All other parameters with base learning rate
    encoder_param_ids = set()
    if hasattr(model, 'image_encoder'):
        encoder_param_ids = {id(p) for p in model.image_encoder.parameters()}

    other_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in encoder_param_ids
    ]
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
        })

    return param_groups


def simulate_interactive_prompts(
    model: SAM2Model,
    frames: torch.Tensor,
    gt_masks: torch.Tensor,
    has_mask: torch.Tensor,
    prompt_frames: List[int],
    num_correction_clicks: int = 7,
    device: torch.device = torch.device('cuda'),
) -> Tuple[Dict, torch.Tensor]:
    """
    Simulate interactive prompting during training.

    From the paper:
    - Sample sequences of 8 frames
    - Randomly select up to 2 frames to prompt
    - Probabilistically receive corrective clicks
    - Initial prompts: GT mask (50%), positive click (25%), bounding box (25%)
    - Corrective clicks: sampled from center of error region
    - With 10% probability, randomly sample clicks from GT mask

    Returns:
        prompts: dict of prompts for each frame
        total_loss: accumulated loss
    """
    B, T, C, H, W = frames.shape
    memory_bank = {
        'recent_feats': [],
        'recent_pos': [],
        'prompted_feats': [],
        'prompted_pos': [],
        'object_ptrs': [],
    }

    total_loss = torch.tensor(0.0, device=device)
    loss_fn = SAM2Loss()

    for t in range(T):
        frame = frames[:, t].to(device)
        gt_mask = gt_masks[:, t].to(device)
        frame_has_mask = has_mask[:, t].to(device)

        # Determine if this frame has a prompt
        is_prompt_frame = t in prompt_frames

        # Build prompts for this frame
        points = None
        boxes = None
        masks = None

        if is_prompt_frame:
            # Sample initial prompt type
            r = random.random()
            if r < 0.5:
                # GT mask prompt
                masks = gt_mask.unsqueeze(1)  # B 1 H W
            elif r < 0.75:
                # Positive click from GT mask center
                coords = []
                labels = []
                for b in range(B):
                    mask_np = gt_mask[b].cpu().numpy()
                    if mask_np.sum() > 0:
                        cy, cx = get_center_of_mass(mask_np)
                        coords.append([cx, cy])
                        labels.append(1)
                    else:
                        coords.append([W // 2, H // 2])
                        labels.append(0)
                coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(1)
                labels_tensor = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(1)
                points = (coords_tensor, labels_tensor)
            else:
                # Bounding box prompt
                boxes_list = []
                for b in range(B):
                    mask_np = gt_mask[b].cpu().numpy()
                    bbox = mask_to_bbox(mask_np)
                    if bbox is not None:
                        boxes_list.append(list(bbox))
                    else:
                        boxes_list.append([0, 0, W - 1, H - 1])
                boxes = torch.tensor(boxes_list, dtype=torch.float32, device=device)

        # Forward pass for this frame
        outputs, memory = model.forward_video_frame(
            img=frame,
            memory_bank=memory_bank if len(memory_bank['recent_feats']) > 0 else None,
            points=points,
            boxes=boxes,
            masks=masks,
            multimask_output=is_prompt_frame,
        )

        # Compute loss
        pred_masks = outputs['low_res_masks']
        pred_iou = outputs['iou_predictions']
        pred_occlusion = outputs['occlusion_pred']

        # Resize GT mask to match prediction size
        gt_mask_resized = F.interpolate(
            gt_mask.unsqueeze(1).float(),
            size=pred_masks.shape[-2:],
            mode='nearest',
        ).squeeze(1)

        gt_occlusion = (frame_has_mask == 0).float()

        frame_loss, _ = loss_fn(
            pred_masks=pred_masks,
            pred_iou=pred_iou,
            pred_occlusion=pred_occlusion,
            gt_masks=gt_mask_resized,
            gt_occlusion=gt_occlusion,
            multimask=is_prompt_frame,
        )
        total_loss = total_loss + frame_loss

        # Update memory bank
        B_f, C_m, H_m, W_m = memory.shape
        from sam2.modeling.prompt_encoder import PositionEmbeddingRandom
        pe_layer = PositionEmbeddingRandom(model.hidden_dim // 2)
        pos = pe_layer((H_m, W_m)).unsqueeze(0).expand(B_f, -1, -1, -1).to(device)

        memory_bank['recent_feats'].append(memory.detach())
        memory_bank['recent_pos'].append(pos.detach())

        if len(memory_bank['recent_feats']) > model.num_maskmem:
            memory_bank['recent_feats'].pop(0)
            memory_bank['recent_pos'].pop(0)

        if 'mask_tokens' in outputs:
            obj_ptr = outputs['mask_tokens'][:, 0, :].detach()
            memory_bank['object_ptrs'].append(obj_ptr)
            if len(memory_bank['object_ptrs']) > model.num_maskmem:
                memory_bank['object_ptrs'].pop(0)

    return total_loss / T


def get_center_of_mass(mask: np.ndarray) -> Tuple[int, int]:
    """Get center of mass of binary mask."""
    if mask.sum() == 0:
        h, w = mask.shape
        return h // 2, w // 2
    y_coords, x_coords = np.where(mask > 0)
    return int(y_coords.mean()), int(x_coords.mean())


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Convert binary mask to bounding box."""
    if mask.sum() == 0:
        return None
    y_coords, x_coords = np.where(mask > 0)
    return int(x_coords.min()), int(y_coords.min()), int(x_coords.max()), int(y_coords.max())


def train_one_epoch(
    model: SAM2Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    video_loader: Optional[DataLoader],
    image_loader: Optional[DataLoader],
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_steps = 0

    video_iter = iter(video_loader) if video_loader is not None else None
    image_iter = iter(image_loader) if image_loader is not None else None

    for step in range(args.steps_per_epoch):
        # Alternate between video and image batches
        use_video = (video_iter is not None) and (
            image_iter is None or random.random() < args.video_prob
        )

        try:
            if use_video and video_iter is not None:
                batch = next(video_iter)
                frames = batch['frames'].to(device)  # B T 3 H W
                gt_masks = batch['masks'].to(device)  # B T H W
                has_mask = batch['has_mask'].to(device)  # B T
                prompt_frames = batch['prompt_frames'][0] if isinstance(batch['prompt_frames'], list) else [0]

                optimizer.zero_grad()

                if scaler is not None:
                    with autocast(dtype=torch.bfloat16):
                        loss = simulate_interactive_prompts(
                            model=model,
                            frames=frames,
                            gt_masks=gt_masks,
                            has_mask=has_mask,
                            prompt_frames=prompt_frames,
                            device=device,
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = simulate_interactive_prompts(
                        model=model,
                        frames=frames,
                        gt_masks=gt_masks,
                        has_mask=has_mask,
                        prompt_frames=prompt_frames,
                        device=device,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

            elif image_iter is not None:
                batch = next(image_iter)
                images = batch['image'].to(device)  # B 3 H W
                masks = batch['masks'].to(device)  # B N H W

                optimizer.zero_grad()

                # Simple image training: pick one mask per image
                B = images.shape[0]
                loss_fn = SAM2Loss()
                total_img_loss = torch.tensor(0.0, device=device)

                for b in range(B):
                    num_masks = batch['num_masks'][b].item()
                    if num_masks == 0:
                        continue

                    mask_idx = random.randint(0, min(num_masks, masks.shape[1]) - 1)
                    gt_mask = masks[b, mask_idx]  # H W

                    # Sample a click from the mask
                    mask_np = gt_mask.cpu().numpy()
                    if mask_np.sum() == 0:
                        continue

                    cy, cx = get_center_of_mass(mask_np)
                    coords = torch.tensor([[[cx, cy]]], dtype=torch.float32, device=device)
                    labels = torch.tensor([[1]], dtype=torch.long, device=device)

                    with autocast(dtype=torch.bfloat16) if scaler else torch.no_grad():
                        outputs = model.forward_image(
                            img=images[b:b+1],
                            points=(coords, labels),
                            multimask_output=True,
                        )

                    pred_masks = outputs['low_res_masks']
                    pred_iou = outputs['iou_predictions']
                    pred_occlusion = outputs['occlusion_pred']

                    gt_mask_resized = F.interpolate(
                        gt_mask.unsqueeze(0).unsqueeze(0).float(),
                        size=pred_masks.shape[-2:],
                        mode='nearest',
                    ).squeeze(0).squeeze(0)

                    img_loss, _ = loss_fn(
                        pred_masks=pred_masks,
                        pred_iou=pred_iou,
                        pred_occlusion=pred_occlusion,
                        gt_masks=gt_mask_resized.unsqueeze(0),
                        multimask=True,
                    )
                    total_img_loss = total_img_loss + img_loss

                loss = total_img_loss / max(B, 1)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
            else:
                continue

            scheduler.step()
            total_loss += loss.item()
            num_steps += 1

            if step % args.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Step {step}/{args.steps_per_epoch}, "
                      f"Loss: {loss.item():.4f}, LR: {lr:.6f}")

        except StopIteration:
            # Restart iterator
            if use_video and video_iter is not None:
                video_iter = iter(video_loader)
            elif image_iter is not None:
                image_iter = iter(image_loader)

    return {'loss': total_loss / max(num_steps, 1)}


def main():
    parser = argparse.ArgumentParser(description='SAM 2 Training')

    # Model
    parser.add_argument('--model_size', type=str, default='base_plus',
                        choices=['tiny', 'small', 'base_plus', 'large'])
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--num_maskmem', type=int, default=7)

    # Data
    parser.add_argument('--video_dir', type=str, default='data/sa_v/videos')
    parser.add_argument('--video_ann_dir', type=str, default='data/sa_v/annotations')
    parser.add_argument('--image_dir', type=str, default='data/sa_1b/images')
    parser.add_argument('--image_ann_dir', type=str, default='data/sa_1b/annotations')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--video_prob', type=float, default=0.85,
                        help='Probability of sampling video batch vs image batch')

    # Training
    parser.add_argument('--mode', type=str, default='full',
                        choices=['pretrain', 'full', 'finetune'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--layer_decay', type=float, default=0.9)

    # LR schedule
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--timescale', type=int, default=1000)
    parser.add_argument('--cooldown_steps', type=int, default=5000)

    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=10)

    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', default=True)

    # Resume
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Build model
    print(f"Building SAM 2 ({args.model_size})...")
    model = build_sam2(
        model_size=args.model_size,
        image_size=args.image_size,
        num_maskmem=args.num_maskmem,
    )
    model = model.to(device)

    # For fine-tuning, freeze image encoder
    if args.mode == 'finetune':
        for param in model.image_encoder.parameters():
            param.requires_grad = False
        print("Image encoder frozen for fine-tuning")

    # Build optimizer with layer decay
    param_groups = get_layer_decay_param_groups(
        model=model,
        base_lr=args.lr,
        layer_decay=args.layer_decay,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
    )

    # Build LR scheduler
    total_steps = args.epochs * args.steps_per_epoch
    if args.mode == 'finetune':
        total_steps = 50000
        args.lr = args.lr / 2

    scheduler = get_reciprocal_sqrt_schedule(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        timescale=args.timescale,
        cooldown_steps=args.cooldown_steps,
        total_steps=total_steps,
    )

    # AMP scaler
    scaler = GradScaler() if args.use_amp and device.type == 'cuda' else None

    # Resume from checkpoint
    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Build datasets
    video_loader = None
    image_loader = None

    if os.path.exists(args.video_dir):
        video_dataset = VideoSegmentationDataset(
            video_dir=args.video_dir,
            annotation_dir=args.video_ann_dir,
            num_frames=args.num_frames if args.mode != 'finetune' else 16,
            image_size=args.image_size,
        )
        video_loader = DataLoader(
            video_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Video dataset: {len(video_dataset)} videos")

    if os.path.exists(args.image_dir) and args.mode != 'finetune':
        image_dataset = ImageSegmentationDataset(
            image_dir=args.image_dir,
            annotation_dir=args.image_ann_dir,
            image_size=args.image_size,
        )
        image_loader = DataLoader(
            image_dataset,
            batch_size=args.batch_size * 2,  # Larger batch for images
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Image dataset: {len(image_dataset)} images")

    # Training loop
    print(f"Starting training in {args.mode} mode...")
    for epoch in range(start_epoch, args.epochs):
        metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            video_loader=video_loader,
            image_loader=image_loader,
            device=device,
            epoch=epoch,
            args=args,
            scaler=scaler,
        )

        print(f"Epoch {epoch}: loss={metrics['loss']:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'args': vars(args),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, 'sam2_final.pth')
    torch.save({
        'epoch': args.epochs - 1,
        'model': model.state_dict(),
        'args': vars(args),
    }, final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == '__main__':
    main()
