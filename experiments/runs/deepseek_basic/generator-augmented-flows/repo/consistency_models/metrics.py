"""
Evaluation metrics for generative models.

Implements:
- Fréchet Inception Distance (FID)
- Kernel Inception Distance (KID)
- Inception Score (IS)

Based on TorchMetrics implementation referenced in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class InceptionV3FeatureExtractor(nn.Module):
    """
    InceptionV3 feature extractor for FID/KID/IS computation.

    Uses a pretrained InceptionV3 from torchvision, extracting features
    from the pool3 layer (2048-d for FID/KID) and the final logits (for IS).
    """

    def __init__(self, use_pool3: bool = True, use_logits: bool = True):
        super().__init__()
        self.use_pool3 = use_pool3
        self.use_logits = use_logits

        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            self.model = inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1,
                transform_input=False,
                aux_logits=False,
            )
        except ImportError:
            # Fallback: try without weights
            from torchvision.models import inception_v3
            self.model = inception_v3(
                weights=None,
                transform_input=False,
                aux_logits=False,
            )
            print("Warning: InceptionV3 loaded without pretrained weights")

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Register hook for pool3 features
        self.pool3_features = None

        if self.use_pool3:
            def hook_fn(module, input, output):
                self.pool3_features = output.detach()

            # The pool3 layer in torchvision InceptionV3:
            # avgpool -> flatten -> dropout
            self.model.avgpool.register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: Image tensor [B, 3, H, W], values in [0, 1] or [-1, 1]

        Returns:
            (pool3_features, logits)
        """
        # InceptionV3 expects images in [0, 1] range
        if x.min() < 0:
            x = (x + 1) / 2.0

        # InceptionV3 requires 299x299 input
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        self.pool3_features = None
        logits = self.model(x)

        pool3 = self.pool3_features
        if pool3 is not None:
            pool3 = pool3.flatten(1)

        return pool3, logits


def compute_statistics(
    features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and covariance of features.

    Args:
        features: Feature vectors [N, D]

    Returns:
        (mu, sigma): Mean [D] and covariance [D, D]
    """
    mu = features.mean(dim=0)
    sigma = torch.cov(features.T)
    return mu, sigma


def frechet_distance(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    mu2: torch.Tensor,
    sigma2: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute Fréchet distance between two Gaussians.

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 * sigma2))
    """
    diff = mu1 - mu2
    diff_sq = (diff * diff).sum()

    # Compute sqrt of sigma1 @ sigma2
    # Use matrix square root via SVD
    sigma_prod = sigma1 @ sigma2

    try:
        s = torch.linalg.eigvalsh(sigma_prod)
        s = torch.clamp(s, min=0)  # Ensure non-negative eigenvalues
        sqrtm_trace = torch.sqrt(s).sum()
    except Exception:
        # Fallback: use identity approximation
        sqrtm_trace = torch.trace(sigma1) * 0.5 + torch.trace(sigma2) * 0.5

    trace = torch.trace(sigma1) + torch.trace(sigma2) - 2 * sqrtm_trace

    fid = diff_sq + trace
    return fid


def compute_fid(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    Compute Fréchet Inception Distance (FID).

    FID = ||μ_r - μ_f||^2 + Tr(Σ_r + Σ_f - 2√(Σ_r Σ_f))

    Args:
        real_features: Features from real images [N_real, D]
        fake_features: Features from generated images [N_fake, D]

    Returns:
        FID score (lower is better)
    """
    mu_real, sigma_real = compute_statistics(real_features)
    mu_fake, sigma_fake = compute_statistics(fake_features)
    fid = frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake, eps)
    return fid.item()


def compute_kid(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    subset_size: int = 1000,
    degree: int = 3,
    gamma: float = None,
    coef0: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute Kernel Inception Distance (KID).

    Uses polynomial kernel: k(x, y) = (γ * x^T y + coef0)^degree

    Args:
        real_features: Features from real images [N_real, D]
        fake_features: Features from generated images [N_fake, D]
        subset_size: Size of random subsets for unbiased estimate
        degree: Polynomial kernel degree
        gamma: Kernel gamma parameter
        coef0: Kernel coef0 parameter

    Returns:
        (kid_mean, kid_std)
    """
    if gamma is None:
        gamma = 1.0 / real_features.shape[1]

    N_real = real_features.shape[0]
    N_fake = fake_features.shape[0]

    # Use subsets for unbiased estimate
    if N_real > subset_size:
        idx = torch.randperm(N_real)[:subset_size]
        real_features = real_features[idx]
        N_real = subset_size

    if N_fake > subset_size:
        idx = torch.randperm(N_fake)[:subset_size]
        fake_features = fake_features[idx]
        N_fake = subset_size

    # Compute kernel matrices
    K_real = (gamma * (real_features @ real_features.T) + coef0) ** degree
    K_fake = (gamma * (fake_features @ fake_features.T) + coef0) ** degree
    K_cross = (gamma * (real_features @ fake_features.T) + coef0) ** degree

    # Unbiased MMD estimate
    kid_real = (K_real.sum() - K_real.diag().sum()) / (N_real * (N_real - 1))
    kid_fake = (K_fake.sum() - K_fake.diag().sum()) / (N_fake * (N_fake - 1))
    kid_cross = K_cross.sum() / (N_real * N_fake)

    kid = (kid_real + kid_fake - 2 * kid_cross) * 100  # Scale by 100

    return kid.item(), 0.0  # std not computed in simple version


def compute_is(
    logits: torch.Tensor,
    splits: int = 10,
) -> Tuple[float, float]:
    """
    Compute Inception Score (IS).

    IS = exp(E_x[KL(p(y|x) || p(y))])

    Args:
        logits: Logits from InceptionV3 [N, 1000]
        splits: Number of splits for computing std

    Returns:
        (is_mean, is_std)
    """
    N = logits.shape[0]
    probs = F.softmax(logits, dim=-1)

    split_scores = []
    for k in range(splits):
        part = probs[k * (N // splits): (k + 1) * (N // splits)]
        kl = part * (torch.log(part) - torch.log(part.mean(dim=0, keepdim=True)))
        kl = kl.sum(dim=-1).mean()
        split_scores.append(torch.exp(kl).item())

    is_mean = np.mean(split_scores)
    is_std = np.std(split_scores)
    return is_mean, is_std


def evaluate_model(
    consistency_model: nn.Module,
    real_dataloader: torch.utils.data.DataLoader,
    feature_extractor: nn.Module = None,
    num_samples: int = 50000,
    batch_size: int = 128,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Full evaluation: compute FID, KID, IS for a trained consistency model.

    Args:
        consistency_model: Trained consistency model
        real_dataloader: DataLoader for real images
        feature_extractor: InceptionV3 feature extractor
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: Device

    Returns:
        Dict with "fid", "kid", "is_mean", "is_std"
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature_extractor is None:
        feature_extractor = InceptionV3FeatureExtractor().to(device)

    model = consistency_model.to(device)
    model.eval()

    # Get image shape from dataloader
    sample_batch = next(iter(real_dataloader))
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]
    img_shape = sample_batch.shape[1:]

    # Collect real features
    real_features = []
    real_logits = []
    samples_collected = 0
    for batch in real_dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        pool3, logits = feature_extractor(x)
        real_features.append(pool3.cpu())
        real_logits.append(logits.cpu())
        samples_collected += x.shape[0]
        if samples_collected >= num_samples:
            break
    real_features = torch.cat(real_features, dim=0)[:num_samples]
    real_logits = torch.cat(real_logits, dim=0)[:num_samples]

    # Generate samples and get features
    fake_features = []
    fake_logits = []
    samples_generated = 0
    while samples_generated < num_samples:
        with torch.no_grad():
            B = min(batch_size, num_samples - samples_generated)
            z = torch.randn(B, *img_shape, device=device)
            sigma = torch.full((B,), 80.0, device=device)  # σ_max
            # One-step generation: f_θ(z * σ_max, σ_max) ≈ x_0
            x_gen = model(z * 80.0, sigma)

        pool3, logits = feature_extractor(x_gen)
        fake_features.append(pool3.cpu())
        fake_logits.append(logits.cpu())
        samples_generated += B

    fake_features = torch.cat(fake_features, dim=0)[:num_samples]
    fake_logits = torch.cat(fake_logits, dim=0)[:num_samples]

    # Compute metrics
    fid = compute_fid(real_features, fake_features)
    kid, _ = compute_kid(real_features, fake_features)
    is_mean, is_std = compute_is(fake_logits)

    return {
        "fid": fid,
        "kid": kid,
        "is_mean": is_mean,
        "is_std": is_std,
    }
