"""
Training script for Ca2-VDM.

Supports two training modes:
  1. Text-to-Video (T2V): Train on InternVid dataset with T5 text encoder.
  2. Video Prediction: Train on SkyTimelapse dataset without text conditioning.

Training procedure (from Section 4.1 and Appendix C):
  - T2V: Two-stage training
    Stage 1: Train causal modeling without clean prefix on 32-frame videos
             (batch_size=288, 32k steps)
    Stage 2: Train with clean prefix on 65-frame videos
             (l=16, P_max=49, batch_size=144, 21k steps)
  - Video Prediction: Single stage on SkyTimelapse
    (l=8, P_max=25, batch_size=8, 11k steps)

Optimizer: AdamW (lr=2e-5)
Noise schedule: DDPM linear (T=1000, beta_1=1e-4, beta_T=0.02)
"""

import os
import sys
import argparse
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ca2_vdm.models import Ca2VDM
from ca2_vdm.models.transformer import Ca2VDMTransformer
from ca2_vdm.data import SkyTimelapseDataset, MSRVTTDataset, UCF101Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Ca2-VDM")

    # Task
    parser.add_argument(
        "--task",
        type=str,
        default="video_prediction",
        choices=["video_prediction", "t2v_stage1", "t2v_stage2"],
        help="Training task",
    )

    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")

    # Model
    parser.add_argument("--hidden_size", type=int, default=1152, help="Transformer hidden size")
    parser.add_argument("--depth", type=int, default=28, help="Number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--patch_size", type=int, default=2, help="Spatial patch size")
    parser.add_argument("--prefix_len", type=int, default=3, help="P', spatial prefix length")

    # Training
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--max_steps", type=int, default=11000, help="Maximum training steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps")

    # Diffusion
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Beta start")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Beta end")

    # Video
    parser.add_argument("--resolution", type=int, default=256, help="Video resolution")
    parser.add_argument("--chunk_size", type=int, default=8, help="l, frames per AR chunk")
    parser.add_argument("--max_prefix_len", type=int, default=25, help="P_max")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")

    return parser.parse_args()


def build_model(args) -> Ca2VDM:
    """Build Ca2-VDM model."""
    resolution = args.resolution
    patch_size = args.patch_size
    max_height = resolution // (patch_size * 8)  # After VAE 8x downsampling
    max_width = resolution // (patch_size * 8)

    # For 256x256 with 8x VAE downsampling: 32x32 latent
    # With patch_size=2: 16x16 patches
    latent_h = resolution // 8
    latent_w = resolution // 8
    max_height = latent_h // patch_size
    max_width = latent_w // patch_size

    max_seq_len = args.max_prefix_len + args.chunk_size

    # Determine if text conditioning is needed
    use_text = args.task in ["t2v_stage1", "t2v_stage2"]
    context_dim = 4096 if use_text else None  # T5-XXL output dim

    transformer = Ca2VDMTransformer(
        in_channels=4,  # VAE latent channels
        out_channels=8,  # 2 * in_channels for learned variance
        patch_size=patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        context_dim=context_dim,
        prefix_len=args.prefix_len,
        max_seq_len=max_seq_len,
        max_height=max_height,
        max_width=max_width,
    )

    model = Ca2VDM(
        transformer=transformer,
        T=args.T,
        beta_schedule="linear",
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        chunk_size=args.chunk_size,
        max_prefix_len=args.max_prefix_len,
    )

    return model


def build_dataset(args):
    """Build training dataset."""
    resolution = (args.resolution, args.resolution)

    if args.task == "video_prediction":
        dataset = SkyTimelapseDataset(
            data_dir=args.data_dir,
            split="train",
            chunk_size=args.chunk_size,
            max_prefix_len=args.max_prefix_len,
            resolution=resolution,
        )
    elif args.task in ["t2v_stage1", "t2v_stage2"]:
        # InternVid dataset (generic video dataset with text)
        from ca2_vdm.data.dataset import VideoDataset
        dataset = VideoDataset(
            video_dir=args.data_dir,
            max_frames=args.max_prefix_len + args.chunk_size,
            resolution=resolution,
            chunk_size=args.chunk_size,
            max_prefix_len=args.max_prefix_len,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")

    return dataset


def train(args):
    """Main training loop."""
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build model
    model = build_model(args)
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pretrained weights if specified
    if args.pretrained:
        logger.info(f"Loading pretrained weights from {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    # Build dataset and dataloader
    dataset = build_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint.get("step", 0)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    step = start_step
    data_iter = iter(dataloader)

    logger.info(f"Starting training from step {step}")

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move to device
        frames = batch["frames"].to(device)  # (B, L, C, H, W)
        # frames are pixel-space; in practice we'd encode with VAE first
        # For this implementation, we use frames directly as "latents"
        # (In the full implementation, frames would be encoded by VAE)

        # For T2V, get text embeddings
        context = None
        context_mask = None
        if args.task in ["t2v_stage1", "t2v_stage2"]:
            # In practice, encode text with T5 here
            # For now, use None (no text conditioning in this simplified version)
            pass

        # Compute training loss
        loss_dict = model.training_loss(frames, context=context, context_mask=context_mask)
        loss = loss_dict["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        step += 1

        # Logging
        if step % args.log_every == 0:
            logger.info(
                f"Step {step}/{args.max_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"L_simple: {loss_dict['loss_simple'].item():.4f} | "
                f"L_vlb: {loss_dict['loss_vlb'].item():.4f}"
            )

        # Save checkpoint
        if step % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_{step:06d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_path = output_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training complete. Saved final model to {final_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
