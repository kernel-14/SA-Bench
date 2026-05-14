from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator used as the base GAN discriminator.
    """

    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, 4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DINODiscriminator(nn.Module):
    """
    DINO-based discriminator as used in VAR's tokenizer.

    Uses DINOv2 ViT-B/14 as a feature extractor, then applies
    a lightweight head to produce real/fake predictions.
    The discriminator operates on multi-scale patch features.
    """

    def __init__(
        self,
        dino_model: str = "dinov2_vitb14",
        proj_dim: int = 512,
        patch_size: int = 14,
        image_size: int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2

        # DINOv2 feature extractor (frozen backbone)
        self.dino = torch.hub.load("facebookresearch/dinov2", dino_model, pretrained=True)
        dino_dim = self.dino.embed_dim  # 768 for ViT-B

        # Freeze DINO backbone
        for param in self.dino.parameters():
            param.requires_grad = False

        # Discriminator head
        self.head = nn.Sequential(
            nn.Linear(dino_dim, proj_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(proj_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image in [-1, 1]

        Returns:
            (B, num_patches) patch-level real/fake logits
        """
        # Normalize to DINO's expected range
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_norm = (x * 0.5 + 0.5 - mean) / std

        with torch.no_grad():
            features = self.dino.get_intermediate_layers(x_norm, n=1)[0]  # (B, num_patches+1, D)
        patch_features = features[:, 1:, :]  # remove CLS token: (B, num_patches, D)

        logits = self.head(patch_features)  # (B, num_patches, 1)
        return logits.squeeze(-1)  # (B, num_patches)


class CombinedDiscriminator(nn.Module):
    """
    Combined discriminator: PatchGAN + DINO discriminator.
    Used during FR-VAE training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_dino: bool = True,
        dino_model: str = "dinov2_vitb14",
        image_size: int = 256,
    ):
        super().__init__()
        self.patch_disc = NLayerDiscriminator(in_channels, ndf, n_layers)
        self.use_dino = use_dino
        if use_dino:
            self.dino_disc = DINODiscriminator(dino_model=dino_model, image_size=image_size)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = [self.patch_disc(x)]
        if self.use_dino:
            outputs.append(self.dino_disc(x))
        return outputs


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = F.relu(1.0 - logits_real).mean()
    loss_fake = F.relu(1.0 + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    return -logits_fake.mean()


def adopt_weight(
    weight: float, global_step: int, threshold: int = 0, value: float = 0.0
) -> float:
    if global_step < threshold:
        return value
    return weight


def calculate_adaptive_weight(
    nll_loss: torch.Tensor,
    g_loss: torch.Tensor,
    last_layer_weight: torch.Tensor,
    disc_weight: float = 0.5,
) -> torch.Tensor:
    """
    Adaptive GAN weight balancing from VQGAN.
    Scales GAN loss to match reconstruction loss gradient magnitude.
    """
    nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight * disc_weight
