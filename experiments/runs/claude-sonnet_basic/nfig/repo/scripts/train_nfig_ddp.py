"""
Distributed training script for NFIG Transformer using PyTorch DDP.

Usage:
    torchrun --nproc_per_node=8 scripts/train_nfig_ddp.py \
        --data-path /path/to/imagenet \
        --tokenizer-path output/fr_vae/fr_vae_final.pt \
        --output-dir output/nfig \
        --model-size 310m \
        --batch-size 96 \
        --epochs 350 \
        --lr 8e-5

Note: batch_size is per-GPU. Total batch size = batch_size * nproc_per_node.
For the paper's batch_size=768 with 8 GPUs: --batch-size 96
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from typing import List

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.fr_vae import FRVAE
from models.nfig_transformer import NFIGTransformer, nfig_310m, nfig_600m


def get_args():
    parser = argparse.ArgumentParser(description="Train NFIG Transformer (DDP)")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./output/nfig")
    parser.add_argument("--model-size", type=str, default="310m", choices=["310m", "600m"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=96,
                        help="Per-GPU batch size. Total = batch_size * n_gpus")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--n-classes", type=int, default=1000)
    parser.add_argument("--codebook-size", type=int, default=4096)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    return parser.parse_args()


def setup_ddp():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    import math
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = get_args()
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    # Data
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, "train"), transform=transform
    )
    sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    # Load tokenizer
    tokenizer = FRVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        scale_factors=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
    ).to(device)

    if os.path.isfile(args.tokenizer_path):
        state = torch.load(args.tokenizer_path, map_location=device)
        if "model" in state:
            state = state["model"]
        tokenizer.load_state_dict(state)
        if is_main:
            print(f"Loaded tokenizer from {args.tokenizer_path}")

    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    # Compute token counts
    token_counts = tokenizer.get_token_counts()
    if is_main:
        print(f"Token counts: {token_counts}, Total: {sum(token_counts)}")

    # Build model
    if args.model_size == "310m":
        model = nfig_310m(codebook_size=args.codebook_size, n_classes=args.n_classes,
                          token_counts=token_counts)
    else:
        model = nfig_600m(codebook_size=args.codebook_size, n_classes=args.n_classes,
                          token_counts=token_counts)

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params/1e6:.1f}M")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    # LR schedule
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    start_epoch = 0
    global_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        if is_main:
            print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        tokenizer.eval()

        total_loss = 0.0
        n_batches = 0

        for batch_idx, (images, class_labels) in enumerate(train_loader):
            images = images.to(device)
            class_labels = class_labels.to(device)

            with torch.no_grad():
                token_sequences, _ = tokenizer.encode(images)

            logits = model(token_sequences, class_labels)
            target = torch.cat(token_sequences, dim=1)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), target.reshape(B * T))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            global_step += 1

            if is_main and batch_idx % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} LR: {lr:.6f}")

        if is_main:
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

            if (epoch + 1) % args.save_every == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "token_counts": token_counts,
                }
                torch.save(checkpoint, os.path.join(args.output_dir, f"nfig_epoch{epoch}.pt"))

    if is_main:
        torch.save(model.module.state_dict(), os.path.join(args.output_dir, "nfig_final.pt"))
        print("Training complete.")

    cleanup_ddp()


if __name__ == "__main__":
    main()
