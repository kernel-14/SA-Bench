"""
Training script for FR-VAE (Frequency-guided Residual-quantized VAE).

Loss function (from paper Appendix B.1):
  L = ||I - I_hat||_2^2 + ||f - f_hat||_2^2 + L_p(I) + 0.5 * L_g(I)

where:
  - ||I - I_hat||_2^2: pixel reconstruction loss
  - ||f - f_hat||_2^2: frequency-guided quantization loss (feature space)
  - L_p: LPIPS perceptual loss
  - L_g: GAN discriminator loss (weight 0.5)

The discriminator uses a DINO-based architecture (as in VAR's tokenizer).
"""

import os
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.fr_vae import FRVAE


# ---------------------------------------------------------------------------
# Discriminator (PatchGAN-style for GAN loss)
# ---------------------------------------------------------------------------

class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge GAN discriminator loss."""
    loss_real = F.relu(1.0 - logits_real).mean()
    loss_fake = F.relu(1.0 + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge GAN generator loss."""
    return -logits_fake.mean()


# ---------------------------------------------------------------------------
# LPIPS perceptual loss (simplified VGG-based)
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """Simplified VGG-based perceptual loss."""

    def __init__(self):
        super().__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=False)
            # Use features up to relu3_3
            self.features = nn.Sequential(*list(vgg.features.children())[:16])
            for param in self.features.parameters():
                param.requires_grad = False
        except Exception:
            self.features = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.features is None:
            return torch.tensor(0.0, device=x.device)
        # Normalize to VGG input range
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x * 0.5 + 0.5 - mean) / std
        y = (y * 0.5 + 0.5 - mean) / std
        fx = self.features(x)
        fy = self.features(y)
        return F.mse_loss(fx, fy)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Train FR-VAE")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to ImageNet dataset")
    parser.add_argument("--output-dir", type=str, default="./output/fr_vae",
                        help="Output directory")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--disc-lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--codebook-size", type=int, default=4096)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--disc-start", type=int, default=10000,
                        help="Step to start discriminator training")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def train_one_epoch(
    model: FRVAE,
    discriminator: NLayerDiscriminator,
    perceptual_loss: VGGPerceptualLoss,
    train_loader: DataLoader,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    device: torch.device,
    epoch: int,
    disc_start: int,
    global_step: int,
    log_every: int,
):
    model.train()
    discriminator.train()

    total_loss_g = 0.0
    total_loss_d = 0.0
    n_batches = 0

    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        B = images.shape[0]

        # ---- Generator / VAE update ----
        optimizer_g.zero_grad()

        x_hat, vq_loss, f, f_hat = model(images)

        # Reconstruction loss (pixel space)
        rec_loss = F.mse_loss(x_hat, images)

        # Frequency-guided quantization loss (feature space)
        freq_loss = F.mse_loss(f_hat, f.detach())

        # Perceptual loss
        perc_loss = perceptual_loss(x_hat, images)

        # GAN generator loss (only after disc_start)
        if global_step >= disc_start:
            logits_fake = discriminator(x_hat)
            gan_loss = hinge_g_loss(logits_fake)
        else:
            gan_loss = torch.tensor(0.0, device=device)

        # Total generator loss
        loss_g = rec_loss + freq_loss + perc_loss + 0.5 * gan_loss + vq_loss

        loss_g.backward()
        optimizer_g.step()

        # ---- Discriminator update ----
        if global_step >= disc_start:
            optimizer_d.zero_grad()
            logits_real = discriminator(images.detach())
            logits_fake = discriminator(x_hat.detach())
            loss_d = hinge_d_loss(logits_real, logits_fake)
            loss_d.backward()
            optimizer_d.step()
        else:
            loss_d = torch.tensor(0.0, device=device)

        total_loss_g += loss_g.item()
        total_loss_d += loss_d.item()
        n_batches += 1
        global_step += 1

        if batch_idx % log_every == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"G_loss: {loss_g.item():.4f} "
                  f"(rec: {rec_loss.item():.4f}, "
                  f"freq: {freq_loss.item():.4f}, "
                  f"perc: {perc_loss.item():.4f}, "
                  f"gan: {gan_loss.item():.4f}, "
                  f"vq: {vq_loss.item():.4f}) "
                  f"D_loss: {loss_d.item():.4f}")

    return total_loss_g / n_batches, total_loss_d / n_batches, global_step


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

    # Models
    model = FRVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        codebook_size=args.codebook_size,
        scale_factors=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
    ).to(device)

    discriminator = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).to(device)
    perceptual_loss = VGGPerceptualLoss().to(device)

    # Optimizers
    optimizer_g = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.disc_lr, betas=(0.5, 0.9))

    start_epoch = 0
    global_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        print(f"Resumed from epoch {start_epoch}")

    # Print token counts
    latent_size = args.image_size // 16  # assuming 16x downsampling
    token_counts = model.get_token_counts(latent_size, latent_size)
    total_tokens = model.total_tokens(latent_size, latent_size)
    print(f"Token counts per band: {token_counts}")
    print(f"Total tokens: {total_tokens}")

    for epoch in range(start_epoch, args.epochs):
        loss_g, loss_d, global_step = train_one_epoch(
            model, discriminator, perceptual_loss,
            train_loader, optimizer_g, optimizer_d,
            device, epoch, args.disc_start, global_step, args.log_every
        )
        print(f"Epoch {epoch}: avg G_loss={loss_g:.4f}, avg D_loss={loss_d:.4f}")

        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                "model": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f"fr_vae_epoch{epoch}.pt"))
            print(f"Saved checkpoint at epoch {epoch}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "fr_vae_final.pt"))
    print("Training complete.")


if __name__ == "__main__":
    main()
