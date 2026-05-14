"""
Train the Frequency-guided Residual-quantized VAE (FR-VAE) tokenizer.
Implements the loss function from Appendix B.1:
L = ||I - Î||_2^2 + ||f - f̂||_2^2 + L_p(I) + 0.5 * L_g(I)
"""

import os
import sys
import argparse
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FRVAEConfig, DataConfig
from models.fr_vae import FRVAE
from models.discriminator import DINODiscriminator, dino_discriminator_loss
from data import get_imagenet_loaders
from utils.setup import setup_logging, save_checkpoint, AverageMeter, LossTracker, EMAModel


def perceptual_loss_fn(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """
    LPIPS perceptual loss using VGG features.
    In practice, this should use the lpips library.
    This is a simplified VGG-based implementation.
    """
    from torchvision import models
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
    vgg = vgg.eval()
    for p in vgg.parameters():
        p.requires_grad = False

    def _normalize(img):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        return (img - mean) / std

    fake_n = _normalize(fake)
    real_n = _normalize(real)

    fake_feats = vgg(fake_n)
    real_feats = vgg(real_n)

    return F.mse_loss(fake_feats, real_feats)


def train_fr_vae(
    config: FRVAEConfig,
    data_config: DataConfig,
    output_dir: str = "./checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(output_dir)
    logger.info(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_imagenet_loaders(
        data_path=data_config.data_path,
        image_size=data_config.image_size,
        batch_size=config.vae_batch_size,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )

    # Model
    model = FRVAE(
        image_size=config.image_size,
        latent_channels=config.latent_channels,
        codebook_size=config.codebook_size,
        codebook_dim=config.codebook_dim,
        downsampling_factor=config.downsampling_factor,
        scale_factors=config.scale_factors,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"FR-VAE parameters: {total_params:,}")

    # Discriminator
    discriminator = DINODiscriminator(
        image_size=config.image_size,
    ).to(device)

    # Optimizers
    opt_gen = optim.Adam(
        model.parameters(),
        lr=config.vae_learning_rate,
        betas=config.vae_adam_betas,
    )
    opt_disc = optim.Adam(
        discriminator.parameters(),
        lr=config.vae_learning_rate,
        betas=config.vae_adam_betas,
    )

    # EMA
    ema = EMAModel(model, decay=config.ema_decay) if config.use_ema else None

    # AMP
    scaler_gen = GradScaler()
    scaler_disc = GradScaler()

    # Loss weights
    recon_weight = 1.0
    freq_recon_weight = 1.0
    perceptual_weight = config.perceptual_loss_weight
    gan_weight = config.gan_loss_weight
    commit_weight = config.commitment_loss_weight

    logger.info(f"Loss weights: recon={recon_weight}, "
                f"freq_recon={freq_recon_weight}, "
                f"perceptual={perceptual_weight}, "
                f"gan={gan_weight}, commit={commit_weight}")

    best_loss = float("inf")
    global_step = 0

    for epoch in range(config.vae_training_steps // len(train_loader) + 1):
        model.train()
        discriminator.train()

        tracker = LossTracker()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            images, _ = batch
            images = images.to(device)

            # ---- Train Generator ----
            opt_gen.zero_grad()
            with autocast():
                reconstructed, all_tokens, commit_loss = model(images)

                # Reconstruction loss (pixel)
                recon_loss = F.mse_loss(reconstructed, images)

                # Frequency-guided quantized loss: compare encoder features
                with torch.no_grad():
                    f_real = model.encoder(images)
                f_recon = model.encoder(reconstructed)
                freq_recon_loss = F.mse_loss(f_recon, f_real)

                # Perceptual loss
                percep_loss = perceptual_loss_fn(reconstructed, images)

                # GAN loss
                disc_out_fake, _ = discriminator(reconstructed)
                gan_loss = -disc_out_fake.mean()

                # Total generator loss
                gen_loss = (
                    recon_weight * recon_loss
                    + freq_recon_weight * freq_recon_loss
                    + perceptual_weight * percep_loss
                    + gan_weight * gan_loss
                    + commit_weight * commit_loss
                )

            scaler_gen.scale(gen_loss).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()

            # ---- Train Discriminator ----
            opt_disc.zero_grad()
            with autocast():
                disc_real, _ = discriminator(images)
                disc_fake, _ = discriminator(reconstructed.detach())

                disc_loss = (
                    F.relu(1.0 - disc_real).mean()
                    + F.relu(1.0 + disc_fake).mean()
                )

            scaler_disc.scale(disc_loss).backward()
            scaler_disc.step(opt_disc)
            scaler_disc.update()

            # EMA update
            if ema is not None:
                ema.update()

            # Track losses
            tracker.update({
                "recon": recon_loss.item(),
                "freq_recon": freq_recon_loss.item(),
                "perceptual": percep_loss.item(),
                "gan": gan_loss.item(),
                "commit": commit_loss.item(),
                "gen": gen_loss.item(),
                "disc": disc_loss.item(),
            })

            pbar.set_postfix(tracker.get_avg())
            global_step += 1

            # Logging
            if global_step % 100 == 0:
                avg_losses = tracker.get_avg()
                logger.info(
                    f"Step {global_step}: "
                    + " ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
                )

            # Validation
            if global_step % 1000 == 0:
                model.eval()
                val_losses = AverageMeter()
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_images, _ = val_batch
                        val_images = val_images.to(device)
                        val_recon, _, _ = model(val_images)
                        val_loss = F.mse_loss(val_recon, val_images)
                        val_losses.update(val_loss.item(), val_images.size(0))
                        if val_losses.count >= 5000:
                            break

                logger.info(f"Step {global_step}: Val MSE = {val_losses.avg:.4f}")

                if val_losses.avg < best_loss:
                    best_loss = val_losses.avg
                    save_checkpoint(
                        model,
                        opt_gen,
                        epoch,
                        global_step,
                        os.path.join(output_dir, "fr_vae_best.pt"),
                        extra_state={"val_loss": best_loss},
                    )
                    logger.info(f"Saved best model with val_loss={best_loss:.4f}")

                model.train()

            # Save periodic checkpoint
            if global_step % 5000 == 0:
                save_checkpoint(
                    model,
                    opt_gen,
                    epoch,
                    global_step,
                    os.path.join(output_dir, f"fr_vae_step{global_step}.pt"),
                )

        tracker.reset()

    # Save final model
    save_checkpoint(
        model,
        opt_gen,
        epoch,
        global_step,
        os.path.join(output_dir, "fr_vae_final.pt"),
    )
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_path", type=str, default="/datasets/ImageNet")
    args = parser.parse_args()

    config = FRVAEConfig()
    data_config = DataConfig(data_path=args.data_path)
    train_fr_vae(config, data_config, args.output_dir)
