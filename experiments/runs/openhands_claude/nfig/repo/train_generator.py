"""
NFIG Transformer training script.

Trains the Next-Frequency Image Generation transformer with:
- Class-conditional generation via AdaLN
- Classifier-free guidance (CFG) training with 10% label drop
- Cross-entropy loss on frequency token sequences
- Adam optimizer, lr=8e-5, batch_size=768, 350 epochs
"""

import argparse
import math
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from config import NFIGConfig, TransformerConfig
from data import build_dataloader, build_imagenet_dataset
from losses import NFIGTransformerLoss
from models.fr_vae import FRVAE
from models.transformer import NFIGTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NFIG transformer")
    parser.add_argument("--data-root", type=str, default="/data/imagenet")
    parser.add_argument("--tokenizer-ckpt", type=str, required=True,
                        help="Path to trained FR-VAE checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/transformer")
    parser.add_argument("--log-dir", type=str, default="./logs/transformer")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--model-size", type=str, default="310M",
                        choices=["310M", "600M"])
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--num-epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--distributed", action="store_true")
    return parser.parse_args()


def build_transformer(cfg: TransformerConfig, device: torch.device) -> NFIGTransformer:
    model = NFIGTransformer(
        vocab_size=cfg.vocab_size,
        num_classes=cfg.num_classes,
        depth=cfg.depth,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        attn_dropout=cfg.attn_dropout,
        scale_factors=cfg.scale_factors,
    )
    return model.to(device)


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: NFIGTransformer,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    path: str,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def load_checkpoint(path: str, model: NFIGTransformer, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["global_step"]


@torch.no_grad()
def encode_batch(
    tokenizer: FRVAE,
    images: torch.Tensor,
    device: torch.device,
) -> List[torch.Tensor]:
    """Encode a batch of images to frequency token indices using frozen FR-VAE."""
    tokenizer.eval()
    images = images.to(device)
    indices_list, _, _ = tokenizer.encode(images)
    return indices_list


def train_one_epoch(
    model: NFIGTransformer,
    tokenizer: FRVAE,
    loader,
    optimizer: optim.Optimizer,
    scheduler,
    criterion: NFIGTransformerLoss,
    device: torch.device,
    epoch: int,
    global_step: int,
    cfg_drop_prob: float = 0.1,
    grad_clip: float = 2.0,
    log_every: int = 100,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
) -> int:
    model.train()
    tokenizer.eval()

    for batch_idx, (images, class_labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        class_labels = class_labels.to(device, non_blocking=True)

        # Encode images to frequency tokens (no gradient through tokenizer)
        with torch.no_grad():
            indices_list, _, _ = tokenizer.encode(images)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(indices_list, class_labels, cfg_drop_prob=cfg_drop_prob)
            loss, loss_dict = criterion(logits, indices_list)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        global_step += 1

        if global_step % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch} | Step {global_step} | "
                f"CE Loss: {loss_dict['ce_loss']:.4f} | "
                f"LR: {lr:.2e}"
            )

    return global_step


def main():
    args = parse_args()
    cfg = NFIGConfig()

    # Select model size
    if args.model_size == "600M":
        from config import config_600m
        transformer_cfg = config_600m.transformer
    else:
        transformer_cfg = cfg.transformer

    transformer_cfg.batch_size = args.batch_size
    transformer_cfg.num_epochs = args.num_epochs
    transformer_cfg.lr = args.lr
    transformer_cfg.weight_decay = args.weight_decay
    transformer_cfg.grad_clip = args.grad_clip
    transformer_cfg.warmup_epochs = args.warmup_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Load frozen tokenizer
    tokenizer = FRVAE(
        image_size=cfg.tokenizer.image_size,
        in_channels=cfg.tokenizer.in_channels,
        z_channels=cfg.tokenizer.z_channels,
        ch=cfg.tokenizer.ch,
        ch_mult=cfg.tokenizer.ch_mult,
        num_res_blocks=cfg.tokenizer.num_res_blocks,
        attn_resolutions=cfg.tokenizer.attn_resolutions,
        codebook_size=cfg.tokenizer.codebook_size,
        scale_factors=cfg.tokenizer.scale_factors,
        feature_map_size=cfg.tokenizer.feature_map_size,
    ).to(device)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location=device))
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False
    print(f"Loaded tokenizer from {args.tokenizer_ckpt}")

    # Build transformer
    model = build_transformer(transformer_cfg, device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Transformer parameters: {num_params / 1e6:.1f}M")

    # Optimizer with weight decay (exclude bias and norm params)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in name or "norm" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = optim.Adam(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    # Data
    train_dataset = build_imagenet_dataset(args.data_root, cfg.tokenizer.image_size, "train")
    train_loader, _ = build_dataloader(
        train_dataset, args.batch_size, args.num_workers, is_train=True
    )

    # Scheduler: cosine with warmup
    steps_per_epoch = len(train_loader)
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Loss
    criterion = NFIGTransformerLoss()

    # AMP
    scaler = GradScaler() if args.use_amp else None

    start_epoch = 0
    global_step = 0

    if args.resume:
        start_epoch, global_step = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        global_step = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            global_step=global_step,
            cfg_drop_prob=args.cfg_drop_prob,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            use_amp=args.use_amp,
            scaler=scaler,
        )

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"nfig_{args.model_size}_epoch{epoch:04d}.pt"
            )
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, f"nfig_{args.model_size}_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
