"""
Training script for NFIG Transformer (Next-Frequency Image Generation).

Training details from paper:
- Optimizer: Adam, lr=8e-5, batch_size=768
- Epochs: 350 (for 310M model)
- CFG dropout probability: 0.1
- Inference: CFG=4.5, top_k=990
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.fr_vae import FRVAE
from models.nfig_transformer import NFIGTransformer, nfig_310m, nfig_600m


def get_args():
    parser = argparse.ArgumentParser(description="Train NFIG Transformer")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True,
                        help="Path to trained FR-VAE checkpoint")
    parser.add_argument("--output-dir", type=str, default="./output/nfig")
    parser.add_argument("--model-size", type=str, default="310m",
                        choices=["310m", "600m"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--n-classes", type=int, default=1000)
    parser.add_argument("--codebook-size", type=int, default=4096)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine LR schedule with linear warmup."""
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def tokenize_batch(
    tokenizer: FRVAE,
    images: torch.Tensor,
    latent_H: int,
    latent_W: int,
) -> List[torch.Tensor]:
    """Tokenize a batch of images using the FR-VAE."""
    tokenizer.eval()
    all_indices, _ = tokenizer.encode(images)
    return all_indices


def train_one_epoch(
    model: NFIGTransformer,
    tokenizer: FRVAE,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    latent_H: int,
    latent_W: int,
    log_every: int,
    global_step: int,
):
    model.train()
    tokenizer.eval()

    total_loss = 0.0
    n_batches = 0

    for batch_idx, (images, class_labels) in enumerate(train_loader):
        images = images.to(device)
        class_labels = class_labels.to(device)

        # Tokenize images
        with torch.no_grad():
            token_sequences = tokenize_batch(tokenizer, images, latent_H, latent_W)

        # Forward pass: predict tokens
        # For training, we shift: input is tokens[:-1], target is tokens[1:]
        # But for next-frequency prediction, we predict each band given previous bands
        # The model takes all tokens as input and predicts the next token at each position
        
        # Build input: for each band, the input is the previous bands + current band tokens
        # shifted by one (teacher forcing within each band)
        # Actually, following VAR: we predict all tokens in band i given bands 0..i-1
        # So the input to predict band i is: [band_0, ..., band_{i-1}]
        # and the target is band_i
        
        # For simplicity, we use the full sequence and compute cross-entropy loss
        # The model outputs logits for each position, and we compare with the target tokens
        
        logits = model(token_sequences, class_labels)  # (B, total_tokens, codebook_size)

        # Build target sequence
        target = torch.cat(token_sequences, dim=1)  # (B, total_tokens)

        # Cross-entropy loss
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), target.reshape(B * T))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        global_step += 1

        if batch_idx % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} LR: {lr:.6f}")

    return total_loss / n_batches, global_step


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
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
        print(f"Loaded tokenizer from {args.tokenizer_path}")
    else:
        print(f"Warning: tokenizer checkpoint not found at {args.tokenizer_path}")

    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    # Compute token counts
    latent_H = args.image_size // 16
    latent_W = args.image_size // 16
    token_counts = tokenizer.get_token_counts(latent_H, latent_W)
    total_tokens = sum(token_counts)
    print(f"Token counts per band: {token_counts}")
    print(f"Total tokens: {total_tokens}")

    # Build transformer model
    if args.model_size == "310m":
        model = nfig_310m(
            codebook_size=args.codebook_size,
            n_classes=args.n_classes,
            token_counts=token_counts,
        )
    else:
        model = nfig_600m(
            codebook_size=args.codebook_size,
            n_classes=args.n_classes,
            token_counts=token_counts,
        )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # Optimizer (Adam, lr=8e-5 from paper)
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
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        avg_loss, global_step = train_one_epoch(
            model, tokenizer, train_loader, optimizer, scheduler,
            device, epoch, latent_H, latent_W, args.log_every, global_step
        )
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "token_counts": token_counts,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f"nfig_epoch{epoch}.pt"))
            print(f"Saved checkpoint at epoch {epoch}")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "nfig_final.pt"))
    print("Training complete.")


if __name__ == "__main__":
    main()
