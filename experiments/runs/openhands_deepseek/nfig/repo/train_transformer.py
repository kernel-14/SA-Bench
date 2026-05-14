"""
Train the NFIG Autoregressive Transformer.
Uses the pre-trained FR-VAE tokenizer to extract frequency tokens,
then trains the transformer to generate tokens autoregressively.

Training hyperparameters from the paper:
- Learning rate: 8e-5
- Batch size: 768
- Optimizer: Adam with betas (0.9, 0.95)
- Epochs: 350
- CFG probability: 0.1
"""

import os
import sys
import argparse
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NFIGTransformerConfig, FRVAEConfig, DataConfig
from models.fr_vae import FRVAE
from models.transformer import NFIGTransformer, create_blockwise_causal_mask
from data import get_imagenet_loaders, denormalize
from utils.setup import (
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    LossTracker,
    setup_training,
)
from utils.metrics import InceptionFeatureExtractor, evaluate_model


def extract_tokens(
    vae: FRVAE,
    images: torch.Tensor,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Extract frequency tokens from images using FR-VAE.
    Returns list of token tensors, one per frequency band.
    """
    vae.eval()
    with torch.no_grad():
        _, all_tokens, _ = vae.encode(images.to(device))
    return all_tokens


@torch.no_grad()
def sample_images(
    transformer: NFIGTransformer,
    vae: FRVAE,
    class_ids: torch.Tensor,
    cfg_scale: float = 4.5,
    top_k: int = 990,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Generate images from class labels.
    """
    transformer.eval()
    vae.eval()

    tokens = transformer.generate(
        class_ids=class_ids,
        cfg_scale=cfg_scale,
        top_k=top_k,
        temperature=temperature,
        use_cfg=True,
    )

    images = vae.decode_from_tokens(tokens)
    return images


def train_nfig_transformer(
    transformer_config: NFIGTransformerConfig,
    vae_config: FRVAEConfig,
    data_config: DataConfig,
    output_dir: str = "./checkpoints",
    vae_checkpoint: Optional[str] = None,
    resume_from: Optional[str] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(output_dir)
    logger.info(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_imagenet_loaders(
        data_path=data_config.data_path,
        image_size=data_config.image_size,
        batch_size=transformer_config.batch_size,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )

    # Load FR-VAE
    logger.info("Loading FR-VAE...")
    vae = FRVAE(
        image_size=vae_config.image_size,
        latent_channels=vae_config.latent_channels,
        codebook_size=vae_config.codebook_size,
        codebook_dim=vae_config.codebook_dim,
        downsampling_factor=vae_config.downsampling_factor,
        scale_factors=vae_config.scale_factors,
    ).to(device)

    if vae_checkpoint and os.path.exists(vae_checkpoint):
        load_checkpoint(vae_checkpoint, vae, device=device)
        logger.info(f"Loaded FR-VAE from {vae_checkpoint}")
    else:
        logger.warning("No VAE checkpoint provided; using untrained VAE.")

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Create transformer
    logger.info("Creating NFIG Transformer...")
    transformer = NFIGTransformer(
        vocab_size=transformer_config.vocab_size,
        hidden_dim=transformer_config.hidden_dim,
        num_heads=transformer_config.num_heads,
        num_layers=transformer_config.num_layers,
        num_classes=transformer_config.num_classes,
        scale_factors=transformer_config.scale_factors,
        feature_map_size=transformer_config.feature_map_size,
        dropout=transformer_config.dropout,
        use_adaln=transformer_config.use_adaln,
    ).to(device)

    total_params = sum(p.numel() for p in transformer.parameters())
    logger.info(f"NFIG Transformer parameters: {total_params:,}")

    # Optimizer
    optimizer = optim.AdamW(
        transformer.parameters(),
        lr=transformer_config.learning_rate,
        betas=transformer_config.adam_betas,
        weight_decay=transformer_config.weight_decay,
    )

    # Learning rate scheduler with warmup
    total_steps = transformer_config.max_epochs * len(train_loader)
    warmup_steps = transformer_config.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume training
    start_epoch = 0
    global_step = 0
    if resume_from and os.path.exists(resume_from):
        ckpt = load_checkpoint(resume_from, transformer, optimizer, device)
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        scheduler.last_epoch = global_step
        logger.info(f"Resumed from {resume_from} at epoch {start_epoch}")

    # AMP
    scaler = GradScaler()
    use_amp = True

    # Block-wise causal mask (precomputed)
    block_mask = create_blockwise_causal_mask(
        transformer.block_sizes, device
    )

    # Evaluation
    feature_extractor = InceptionFeatureExtractor(device)
    logger.info("Extracting real features for evaluation...")
    real_features = feature_extractor.extract_from_loader(val_loader, max_samples=50000)

    best_fid = float("inf")
    best_epoch = 0

    logger.info("Starting training...")
    logger.info(f"Total epochs: {transformer_config.max_epochs}")
    logger.info(f"Batch size: {transformer_config.batch_size}")
    logger.info(f"Learning rate: {transformer_config.learning_rate}")
    logger.info(f"CFG scale: {transformer_config.cfg_scale}")
    logger.info(f"Top-k: {transformer_config.top_k}")

    for epoch in range(start_epoch, transformer_config.max_epochs):
        transformer.train()
        tracker = LossTracker()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{transformer_config.max_epochs}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # Extract frequency tokens from VAE
            target_tokens = extract_tokens(vae, images, device)

            # Apply class dropout for CFG training
            if torch.rand(1).item() < transformer_config.cfg_prob:
                labels = torch.full_like(
                    labels, transformer_config.num_classes
                )

            # Forward pass
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = transformer(target_tokens, labels, block_mask)
                loss = transformer.compute_loss(logits, target_tokens)

            scaler.scale(loss).backward()

            # Gradient clipping
            if transformer_config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    transformer.parameters(), transformer_config.gradient_clip
                )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tracker.update({"loss": loss.item()})
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            global_step += 1

            # Logging
            if global_step % transformer_config.log_interval == 0:
                logger.info(
                    f"Step {global_step} | Loss: {loss.item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            # Evaluation
            if (
                global_step % transformer_config.eval_interval == 0
                and global_step > 0
            ):
                logger.info("Running evaluation...")

                # Generate sample images
                sample_labels = torch.randint(
                    0, transformer_config.num_classes, (min(16, transformer_config.batch_size),),
                    device=device,
                )
                sample_imgs = sample_images(
                    transformer,
                    vae,
                    sample_labels,
                    cfg_scale=transformer_config.cfg_scale,
                    top_k=transformer_config.top_k,
                )

                # Compute FID, IS, Pre, Rec
                results = evaluate_model(
                    [sample_imgs],
                    real_features=real_features,
                    inception=feature_extractor.inception,
                    batch_size=32,
                    device=device,
                )

                fid = results.get("fid", float("inf"))
                is_mean = results.get("is_mean", 0)
                precision = results.get("precision", 0)
                recall = results.get("recall", 0)

                logger.info(
                    f"Eval @ step {global_step}: "
                    f"FID={fid:.4f}, IS={is_mean:.2f}, "
                    f"Pre={precision:.4f}, Rec={recall:.4f}"
                )

                if fid < best_fid:
                    best_fid = fid
                    best_epoch = epoch
                    save_checkpoint(
                        transformer,
                        optimizer,
                        epoch,
                        global_step,
                        os.path.join(output_dir, "nfig_transformer_best.pt"),
                        extra_state={
                            "fid": best_fid,
                            "is_mean": is_mean,
                            "epoch": epoch,
                        },
                    )
                    logger.info(f"Saved best model with FID={best_fid:.4f}")

                transformer.train()

            # Save periodic checkpoint
            if global_step % transformer_config.save_interval == 0:
                save_checkpoint(
                    transformer,
                    optimizer,
                    epoch,
                    global_step,
                    os.path.join(output_dir, f"nfig_transformer_e{epoch}_s{global_step}.pt"),
                    extra_state={"epoch": epoch},
                )

        # End of epoch
        avg_loss = tracker.get_avg().get("loss", 0)
        logger.info(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")

        # Save epoch checkpoint
        save_checkpoint(
            transformer,
            optimizer,
            epoch,
            global_step,
            os.path.join(output_dir, f"nfig_transformer_epoch{epoch+1}.pt"),
            extra_state={"epoch": epoch, "loss": avg_loss},
        )

    logger.info(f"Training complete! Best FID={best_fid:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_path", type=str, default="/datasets/ImageNet")
    parser.add_argument("--vae_checkpoint", type=str, default="./checkpoints/fr_vae_best.pt")
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    transformer_config = NFIGTransformerConfig()
    vae_config = FRVAEConfig()
    data_config = DataConfig(data_path=args.data_path)

    train_nfig_transformer(
        transformer_config=transformer_config,
        vae_config=vae_config,
        data_config=data_config,
        output_dir=args.output_dir,
        vae_checkpoint=args.vae_checkpoint,
        resume_from=args.resume_from,
    )
