"""
Evaluation metrics for generative models.
FID, KID, IS using TorchMetrics.
"""
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore


def compute_fid(
    generated_images: torch.Tensor,
    real_images: torch.Tensor,
    feature: int = 2048,
    reset_real_features: bool = True,
) -> float:
    """
    Compute Frechet Inception Distance (FID) between generated and real images.

    Args:
        generated_images: Tensor of shape (N, C, H, W) in range [-1, 1]
        real_images: Tensor of shape (N, C, H, W) in range [-1, 1]
        feature: Inception feature dimension
        reset_real_features: Whether to reset real features

    Returns:
        FID score (float)
    """
    fid = FrechetInceptionDistance(feature=feature, reset_real_features=reset_real_features, normalize=True)

    # Convert from [-1, 1] to [0, 255] uint8 for TorchMetrics Inception
    generated = ((generated_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    real = ((real_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)

    fid.update(real, real=True)
    fid.update(generated, real=False)

    return float(fid.compute())


def compute_kid(
    generated_images: torch.Tensor,
    real_images: torch.Tensor,
    subset_size: int = 1000,
) -> float:
    """
    Compute Kernel Inception Distance (KID).

    Args:
        generated_images: Tensor of shape (N, C, H, W) in range [-1, 1]
        real_images: Tensor of shape (N, C, H, W) in range [-1, 1]
        subset_size: Subset size for KID computation

    Returns:
        KID mean score (float)
    """
    kid = KernelInceptionDistance(subset_size=subset_size, normalize=True)

    generated = ((generated_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    real = ((real_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)

    kid.update(real, real=True)
    kid.update(generated, real=False)

    kid_mean, _ = kid.compute()
    return float(kid_mean)


def compute_is(
    generated_images: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute Inception Score (IS).

    Args:
        generated_images: Tensor of shape (N, C, H, W) in range [-1, 1]

    Returns:
        (IS mean, IS std)
    """
    is_metric = InceptionScore(normalize=True)

    generated = ((generated_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    is_metric.update(generated)

    is_mean, is_std = is_metric.compute()
    return float(is_mean), float(is_std)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_samples: int = 50_000,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Evaluate a consistency model with FID, KID, and IS.

    Args:
        model: Consistency model
        dataloader: Dataloader for real images
        num_samples: Number of samples to generate
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        batch_size: Batch size for generation
        device: Device

    Returns:
        Dictionary with FID, KID, IS scores
    """
    model.eval()

    # Collect real images
    real_images = []
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        real_images.append(images)
        if len(torch.cat(real_images)) >= num_samples:
            break
    real_images = torch.cat(real_images)[:num_samples]

    # Generate images
    generated_images = []
    num_generated = 0
    img_shape = real_images.shape[1:]

    while num_generated < num_samples:
        current_bs = min(batch_size, num_samples - num_generated)
        shape = (current_bs, *img_shape)
        z = torch.randn(*shape, device=device) * sigma_max
        with torch.no_grad():
            samples = model(z, torch.full((current_bs,), sigma_max, device=device))
        generated_images.append(samples.cpu())
        num_generated += current_bs

    generated_images = torch.cat(generated_images)[:num_samples]

    # Compute metrics
    fid = compute_fid(generated_images, real_images)
    kid = compute_kid(generated_images, real_images)
    is_mean, is_std = compute_is(generated_images)

    model.train()
    return {
        "fid": fid,
        "kid": kid,
        "is_mean": is_mean,
        "is_std": is_std,
    }
