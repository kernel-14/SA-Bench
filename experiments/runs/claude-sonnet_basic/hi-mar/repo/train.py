"""
Training script for Hi-MAR: Hierarchical Masked Autoregressive Models.

Supports:
- Class-conditional image generation on ImageNet 256x256
- Text-to-image generation on MS-COCO 256x256

Usage:
    # Class-conditional (ImageNet)
    python train.py --task imagenet --model hi_mar_b --data_path /path/to/imagenet

    # Text-to-image (MS-COCO)
    python train.py --task coco --model hi_mar_s --data_path /path/to/coco
"""

import os
import math
import time
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from models.hi_mar import HiMAR, HiMAR_B, HiMAR_L, HiMAR_H
from models.hi_mar_t2i import HiMARText
from utils.ema import EMA
from utils.logger import setup_logger


def get_args():
    parser = argparse.ArgumentParser('Hi-MAR Training')

    # Task
    parser.add_argument('--task', type=str, default='imagenet',
                        choices=['imagenet', 'coco'],
                        help='Training task')

    # Model
    parser.add_argument('--model', type=str, default='hi_mar_b',
                        choices=['hi_mar_b', 'hi_mar_l', 'hi_mar_h', 'hi_mar_s'],
                        help='Model variant')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--low_res_img_size', type=int, default=128,
                        help='Low-resolution image size for phase 1')

    # VAE
    parser.add_argument('--vae_path', type=str, default='pretrained/kl16.ckpt',
                        help='Path to pretrained VAE (KL-16)')
    parser.add_argument('--vae_stride', type=int, default=16,
                        help='VAE downsampling stride')

    # Training
    parser.add_argument('--epochs', type=int, default=800,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=100,
                        help='Number of warmup epochs')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clipping')

    # EMA
    parser.add_argument('--ema_momentum', type=float, default=0.9999,
                        help='EMA momentum')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA')

    # Masking
    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='Minimum masking ratio for phase 1')
    parser.add_argument('--mask_ratio_max', type=float, default=1.0,
                        help='Maximum masking ratio for phase 1')

    # Diffusion
    parser.add_argument('--num_sampling_steps', type=int, default=100,
                        help='Number of diffusion sampling steps')

    # CFG
    parser.add_argument('--class_dropout_prob', type=float, default=0.1,
                        help='Class dropout probability for CFG training')

    # Output
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Log every N steps')

    # Distributed
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)

    # Mixed precision
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use mixed precision training')

    # Gradient checkpointing
    parser.add_argument('--grad_checkpointing', action='store_true', default=False)

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


def build_model(args):
    """Build Hi-MAR model based on args."""
    common_kwargs = dict(
        img_size=args.img_size,
        low_res_img_size=args.low_res_img_size,
        patch_size=args.vae_stride,
        in_channels=16,  # KL-16 VAE latent channels
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
        num_sampling_steps=args.num_sampling_steps,
        grad_checkpointing=args.grad_checkpointing,
    )

    if args.task == 'imagenet':
        common_kwargs['num_classes'] = 1000
        common_kwargs['class_dropout_prob'] = args.class_dropout_prob

        if args.model == 'hi_mar_b':
            model = HiMAR_B(**common_kwargs)
        elif args.model == 'hi_mar_l':
            model = HiMAR_L(**common_kwargs)
        elif args.model == 'hi_mar_h':
            model = HiMAR_H(**common_kwargs)
        else:
            raise ValueError(f'Unknown model: {args.model}')
    else:  # coco
        model = HiMARText(
            img_size=args.img_size,
            low_res_img_size=args.low_res_img_size,
            patch_size=args.vae_stride,
            in_channels=16,
            text_dim=768,  # CLIP ViT-L/14 text embedding dim
            num_sampling_steps=args.num_sampling_steps,
            grad_checkpointing=args.grad_checkpointing,
        )

    return model


def build_optimizer(model, args):
    """Build AdamW optimizer."""
    # Separate parameters that should not have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'pos_embed' in name or 'mask_token' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {'params': decay_params, 'weight_decay': args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    return optimizer


def get_lr_schedule(optimizer, warmup_epochs, total_epochs, base_lr):
    """Constant LR with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_imagenet_dataset(args):
    """Build ImageNet dataset."""
    transform = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, 'train'),
        transform=transform
    )
    return train_dataset


def train_one_epoch(
    model, vae, optimizer, scheduler, scaler,
    data_loader, epoch, args, logger, ema=None
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_steps = 0

    for step, batch in enumerate(data_loader):
        if args.task == 'imagenet':
            images, class_labels = batch
            images = images.cuda(non_blocking=True)
            class_labels = class_labels.cuda(non_blocking=True)
            text_emb = None
        else:
            images, text_emb = batch
            images = images.cuda(non_blocking=True)
            text_emb = text_emb.cuda(non_blocking=True)
            class_labels = None

        # Encode images with VAE
        with torch.no_grad():
            # High-resolution encoding
            high_res_latents = vae.encode(images).latent_dist.sample()
            high_res_latents = high_res_latents * 0.18215  # VAE scaling factor

            # Low-resolution encoding (resize image first)
            low_res_images = torch.nn.functional.interpolate(
                images, size=(args.low_res_img_size, args.low_res_img_size),
                mode='bilinear', align_corners=False
            )
            low_res_latents = vae.encode(low_res_images).latent_dist.sample()
            low_res_latents = low_res_latents * 0.18215

        # Reshape latents to token sequences
        B = high_res_latents.shape[0]
        h_high = args.img_size // args.vae_stride
        w_high = args.img_size // args.vae_stride
        h_low = args.low_res_img_size // args.vae_stride
        w_low = args.low_res_img_size // args.vae_stride

        high_res_tokens = high_res_latents.permute(0, 2, 3, 1).reshape(B, h_high * w_high, -1)
        low_res_tokens = low_res_latents.permute(0, 2, 3, 1).reshape(B, h_low * w_low, -1)

        # Forward pass
        with autocast(enabled=args.fp16):
            if args.task == 'imagenet':
                loss, loss_dict = model(high_res_tokens, low_res_tokens, class_labels)
            else:
                loss, loss_dict = model(high_res_tokens, low_res_tokens, text_emb)

        # Backward pass
        optimizer.zero_grad()
        if args.fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        # Update EMA
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        num_steps += 1

        if step % args.log_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}] Step [{step}/{len(data_loader)}] '
                f'Loss: {loss.item():.4f} '
                f'Loss1: {loss_dict["loss_phase1"]:.4f} '
                f'Loss2: {loss_dict["loss_phase2"]:.4f} '
                f'LR: {lr:.6f}'
            )

    scheduler.step()
    return total_loss / num_steps


def main():
    args = get_args()

    # Setup distributed training
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        is_main = dist.get_rank() == 0
    else:
        local_rank = 0
        is_main = True

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(args.output_dir, is_main)

    if is_main:
        logger.info(f'Args: {args}')

    # Build model
    model = build_model(args)
    model = model.cuda()

    if is_main:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'Model parameters: {num_params / 1e6:.1f}M')

    # Load VAE
    from utils.vae import load_vae
    vae = load_vae(args.vae_path).cuda()
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Build optimizer
    optimizer = build_optimizer(model, args)

    # Build scheduler
    scheduler = get_lr_schedule(optimizer, args.warmup_epochs, args.epochs, args.lr)

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.fp16)

    # EMA
    ema = None
    if args.use_ema:
        ema = EMA(model, momentum=args.ema_momentum)

    # Distributed model
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Build dataset
    if args.task == 'imagenet':
        train_dataset = build_imagenet_dataset(args)
    else:
        from utils.coco_dataset import COCODataset
        train_dataset = COCODataset(
            args.data_path,
            img_size=args.img_size,
            split='train',
        )

    if dist.is_initialized():
        sampler = DistributedSampler(train_dataset)
    else:
        sampler = None

    data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        if ema and 'ema' in checkpoint:
            ema.load_state_dict(checkpoint['ema'])
        if is_main:
            logger.info(f'Resumed from epoch {start_epoch}')

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        avg_loss = train_one_epoch(
            model, vae, optimizer, scheduler, scaler,
            data_loader, epoch, args, logger, ema
        )

        if is_main:
            logger.info(f'Epoch [{epoch}] Average Loss: {avg_loss:.4f}')

            # Save checkpoint
            if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'args': vars(args),
                }
                if ema:
                    checkpoint['ema'] = ema.state_dict()

                save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch:04d}.pth')
                torch.save(checkpoint, save_path)
                logger.info(f'Saved checkpoint to {save_path}')

    if is_main:
        logger.info('Training complete!')


if __name__ == '__main__':
    main()
