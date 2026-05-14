"""
FR-VAE training script.

Trains the Frequency-guided Residual-quantized VAE with:
- VQGAN framework (encoder + decoder + discriminator)
- DINO discriminator
- Frequency-guided residual quantization
- Combined loss: reconstruction + frequency + perceptual + GAN
"""

import argparse
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from config import NFIGConfig, TokenizerConfig
from data import build_dataloader, build_imagenet_dataset
from losses import FRVAELoss
from models.discriminator import CombinedDiscriminator
from models.fr_vae import FRVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FR-VAE tokenizer")
    parser.add_argument("--data-root", type=str, default="/data/imagenet")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/tokenizer")
    parser.add_argument("--log-dir", type=str, default="./logs/tokenizer")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=4.5e-6)
    parser.add_argument("--disc-lr", type=float, default=4.5e-6)
    parser.add_argument("--disc-start", type=int, default=50001)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--use-dino-disc", action="store_true", default=True)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--distributed", action="store_true")
    return parser.parse_args()


def build_model(cfg: TokenizerConfig, device: torch.device) -> FRVAE:
    model = FRVAE(
        image_size=cfg.image_size,
        in_channels=cfg.in_channels,
        z_channels=cfg.z_channels,
        ch=cfg.ch,
        ch_mult=cfg.ch_mult,
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=cfg.attn_resolutions,
        dropout=cfg.dropout,
        codebook_size=cfg.codebook_size,
        scale_factors=cfg.scale_factors,
        feature_map_size=cfg.feature_map_size,
        commitment_cost=cfg.commitment_loss_weight,
    )
    return model.to(device)


def save_checkpoint(
    model: FRVAE,
    discriminator: CombinedDiscriminator,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: FRVAE,
    discriminator: CombinedDiscriminator,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    discriminator.load_state_dict(ckpt["discriminator"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    return ckpt["epoch"], ckpt["global_step"]


def train_one_epoch(
    model: FRVAE,
    discriminator: CombinedDiscriminator,
    loader,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    criterion: FRVAELoss,
    device: torch.device,
    epoch: int,
    global_step: int,
    log_every: int = 100,
    use_amp: bool = False,
    scaler_g: Optional[GradScaler] = None,
    scaler_d: Optional[GradScaler] = None,
) -> int:
    model.train()
    discriminator.train()

    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        # ---- Generator (VAE) update ----
        opt_g.zero_grad()
        with autocast(enabled=use_amp):
            x_rec, indices_list, vq_loss, components, f, f_tilde = model(images)

            # Get discriminator outputs on fake (for generator loss)
            disc_fake = discriminator(x_rec)

            g_loss, g_loss_dict = criterion.generator_loss(
                x=images,
                x_rec=x_rec,
                f=f,
                f_tilde=f_tilde,
                vq_loss=vq_loss,
                disc_outputs=disc_fake,
                global_step=global_step,
                last_layer_weight=model.get_last_layer_weight(),
            )

        if use_amp and scaler_g is not None:
            scaler_g.scale(g_loss).backward()
            scaler_g.unscale_(opt_g)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_g.step(opt_g)
            scaler_g.update()
        else:
            g_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_g.step()

        # ---- Discriminator update ----
        if global_step >= criterion.disc_start:
            opt_d.zero_grad()
            with autocast(enabled=use_amp):
                disc_real = discriminator(images.detach())
                disc_fake_d = discriminator(x_rec.detach())
                d_loss, d_loss_dict = criterion.discriminator_loss(
                    disc_real, disc_fake_d, global_step
                )

            if use_amp and scaler_d is not None:
                scaler_d.scale(d_loss).backward()
                scaler_d.unscale_(opt_d)
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                d_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                opt_d.step()
        else:
            d_loss_dict = {"d_loss": 0.0}

        global_step += 1

        if global_step % log_every == 0:
            print(
                f"Epoch {epoch} | Step {global_step} | "
                f"rec={g_loss_dict['rec_loss']:.4f} | "
                f"freq={g_loss_dict['freq_loss']:.4f} | "
                f"perc={g_loss_dict['perc_loss']:.4f} | "
                f"vq={g_loss_dict['vq_loss']:.4f} | "
                f"g={g_loss_dict['g_loss']:.4f} | "
                f"d={d_loss_dict['d_loss']:.4f}"
            )

    return global_step


def main():
    args = parse_args()
    cfg = NFIGConfig()
    cfg.tokenizer.batch_size = args.batch_size
    cfg.tokenizer.num_epochs = args.num_epochs
    cfg.tokenizer.lr = args.lr
    cfg.tokenizer.disc_lr = args.disc_lr
    cfg.tokenizer.disc_start = args.disc_start

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Build model and discriminator
    model = build_model(cfg.tokenizer, device)
    discriminator = CombinedDiscriminator(
        in_channels=3,
        use_dino=args.use_dino_disc,
        dino_model=cfg.tokenizer.dino_model,
        image_size=cfg.tokenizer.image_size,
    ).to(device)

    # Optimizers
    opt_g = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_d = optim.Adam(discriminator.parameters(), lr=args.disc_lr, betas=(0.5, 0.9))

    # Loss
    criterion = FRVAELoss(
        rec_loss_weight=cfg.tokenizer.rec_loss_weight,
        freq_loss_weight=cfg.tokenizer.freq_loss_weight,
        perceptual_loss_weight=cfg.tokenizer.perceptual_loss_weight,
        gan_loss_weight=cfg.tokenizer.gan_loss_weight,
        codebook_loss_weight=cfg.tokenizer.codebook_loss_weight,
        disc_start=args.disc_start,
        use_perceptual=True,
    )

    # Data
    train_dataset = build_imagenet_dataset(args.data_root, cfg.tokenizer.image_size, "train")
    train_loader, _ = build_dataloader(
        train_dataset, args.batch_size, args.num_workers, is_train=True
    )

    # AMP scalers
    scaler_g = GradScaler() if args.use_amp else None
    scaler_d = GradScaler() if args.use_amp else None

    start_epoch = 0
    global_step = 0

    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, discriminator, opt_g, opt_d
        )
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        global_step = train_one_epoch(
            model=model,
            discriminator=discriminator,
            loader=train_loader,
            opt_g=opt_g,
            opt_d=opt_d,
            criterion=criterion,
            device=device,
            epoch=epoch,
            global_step=global_step,
            log_every=args.log_every,
            use_amp=args.use_amp,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
        )

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"frvae_epoch{epoch:04d}.pt")
            save_checkpoint(model, discriminator, opt_g, opt_d, epoch, global_step, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "frvae_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
