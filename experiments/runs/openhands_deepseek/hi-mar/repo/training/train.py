"""
Main training script for Hi-MAR models.
Supports class-conditional (ImageNet) and text-to-image (MS-COCO) generation.
"""

import os
import sys
import argparse
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hi_mar import HiMAR
from data.dataset import ImageNetLatentDataset, MSCOCOLatentDataset, VAEEncoder, TextEmbedder
from training.trainer import HiMARTrainer, MaskSampler


def get_config(args):
    """Build configuration for each model scale."""
    configs = {
        'Hi-MAR-B': {
            'token_dim': 16,  # VAE latent channels (KL-16)
            'low_res_num_tokens': 256,   # 16x16 for 128x128 latent
            'high_res_num_tokens': 1024,  # 32x32 for 256x256 latent
            'transformer_dim': 768,
            'transformer_depth': 24,
            'transformer_num_heads': 12,
            'mlp_ratio': 4.0,
            'dropout': 0.0,
            'head1_hidden_dim': 1024,
            'head1_depth': 6,
            'head2_hidden_dim': 512,
            'head2_depth': 6,
            'head2_num_heads': 8,
            'num_classes': 1000,
            'num_train_timesteps': 1000,
            'beta_start': 1e-4,
            'beta_end': 0.02,
        },
        'Hi-MAR-L': {
            'token_dim': 16,
            'low_res_num_tokens': 256,
            'high_res_num_tokens': 1024,
            'transformer_dim': 1024,
            'transformer_depth': 32,
            'transformer_num_heads': 16,
            'mlp_ratio': 4.0,
            'dropout': 0.0,
            'head1_hidden_dim': 1280,
            'head1_depth': 8,
            'head2_hidden_dim': 512,
            'head2_depth': 8,
            'head2_num_heads': 8,
            'num_classes': 1000,
            'num_train_timesteps': 1000,
            'beta_start': 1e-4,
            'beta_end': 0.02,
        },
        'Hi-MAR-H': {
            'token_dim': 16,
            'low_res_num_tokens': 256,
            'high_res_num_tokens': 1024,
            'transformer_dim': 1280,
            'transformer_depth': 40,
            'transformer_num_heads': 16,
            'mlp_ratio': 4.0,
            'dropout': 0.0,
            'head1_hidden_dim': 1536,
            'head1_depth': 12,
            'head2_hidden_dim': 768,
            'head2_depth': 12,
            'head2_num_heads': 12,
            'num_classes': 1000,
            'num_train_timesteps': 1000,
            'beta_start': 1e-4,
            'beta_end': 0.02,
        },
    }
    base = configs[args.model_size]

    # Training hyperparameters
    training_config = {
        'lr': args.lr,
        'beta1': 0.9,
        'beta2': 0.95,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'total_epochs': args.epochs,
        'total_steps': args.total_steps,
        'ema_decay': args.ema_decay,
        'grad_clip': 1.0,
        'batch_size': args.batch_size,
    }
    base.update(training_config)
    return base


def train(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_config(args)

    # Distributed setup
    if args.distributed:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)

    # VAE encoder
    vae = VAEEncoder(vae_path=args.vae_path, device=device)

    # Create dataset
    if args.dataset == 'imagenet':
        dataset = ImageNetLatentDataset(
            root=args.data_root,
            split=args.split,
            image_size=256,
            low_res_size=128,
            vae=vae,
        )
    elif args.dataset == 'coco':
        dataset = MSCOCOLatentDataset(
            root=args.data_root,
            ann_file=args.ann_file,
            split=args.split,
            image_size=256,
            low_res_size=128,
            vae=vae,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # DataLoader
    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=(args.split == 'train'))
        shuffle = False
    else:
        sampler = None
        shuffle = (args.split == 'train')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Create model
    model = HiMAR(config)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)

    if args.distributed:
        model = DDP(model, device_ids=[local_rank])

    # Create trainer
    trainer = HiMARTrainer(model, config, device)

    # Mask sampling functions
    if args.dataset == 'imagenet':
        p1_mask_fn = lambda B, N, d: MaskSampler.uniform_mask_ratio(B, N, d, min_ratio=0.7, max_ratio=1.0)
        p2_mask_fn = lambda B, N, d: MaskSampler.cosine_mask_ratio(B, N, d)
    else:
        p1_mask_fn = lambda B, N, d: MaskSampler.uniform_mask_ratio(B, N, d, min_ratio=0.7, max_ratio=1.0)
        p2_mask_fn = lambda B, N, d: MaskSampler.beta_mask_ratio(B, N, d, alpha=4.0, beta=1.0)

    # Logger
    writer = SummaryWriter(args.log_dir) if args.local_rank in [0, -1] else None

    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)

        metrics = trainer.train_epoch(
            dataloader,
            epoch,
            p1_mask_fn,
            p2_mask_fn,
        )

        if args.local_rank in [0, -1]:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"P1 Loss: {metrics['phase1_loss']:.4f} | "
                f"P2 Loss: {metrics['phase2_loss']:.4f} | "
                f"LR: {metrics['lr']:.6f}"
            )
            if writer:
                writer.add_scalar('Loss/total', metrics['loss'], epoch)
                writer.add_scalar('Loss/phase1', metrics['phase1_loss'], epoch)
                writer.add_scalar('Loss/phase2', metrics['phase2_loss'], epoch)
                writer.add_scalar('LR', metrics['lr'], epoch)

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 and args.local_rank in [0, -1]:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            state = {
                'epoch': epoch + 1,
                'model': trainer.ema_model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'config': config,
                'global_step': trainer.global_step,
            }
            torch.save(state, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    if writer:
        writer.close()


def generate_samples(args):
    """Generate samples from a trained Hi-MAR model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_config(args)
    model = HiMAR(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    vae = VAEEncoder(vae_path=args.vae_path, device=device)

    # Generate
    class_ids = torch.arange(args.num_samples, device=device) % 1000
    with torch.no_grad():
        high_res_tokens = model.generate(
            class_ids=class_ids,
            phase1_steps=args.phase1_steps,
            phase2_steps=args.phase2_steps,
            cfg_scale=args.cfg_scale,
        )

    # Decode to images
    images = vae.decode(high_res_tokens, 256, 256)
    images = (images + 1) / 2  # [-1, 1] → [0, 1]

    # Save images
    from torchvision.utils import save_image
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.num_samples):
        save_image(images[i], os.path.join(args.output_dir, f'sample_{i:04d}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hi-MAR Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'])
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'coco'])
    parser.add_argument('--model_size', type=str, default='Hi-MAR-B', choices=['Hi-MAR-B', 'Hi-MAR-L', 'Hi-MAR-H'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--ann_file', type=str, default='')
    parser.add_argument('--vae_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--warmup_epochs', type=int, default=100)
    parser.add_argument('--total_steps', type=int, default=400000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--output_dir', type=str, default='./samples')
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--phase1_steps', type=int, default=32)
    parser.add_argument('--phase2_steps', type=int, default=4)
    parser.add_argument('--cfg_scale', type=float, default=1.0)
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        generate_samples(args)
