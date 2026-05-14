"""
Frechet Video Distance (FVD) computation.

Based on: Unterthiner et al., "FVD: A new metric for video generation", ICLR 2019 Workshop.
Uses a pretrained I3D model to extract features, following StyleGAN-V's implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from scipy import linalg


def compute_fvd(
    real_videos: torch.Tensor,
    generated_videos: torch.Tensor,
    i3d_model: nn.Module,
    batch_size: int = 64,
    device: torch.device = None,
) -> float:
    """Compute FVD between real and generated videos.

    Args:
        real_videos: (N, T, C, H, W) tensor of real videos, normalized to [0, 1]
        generated_videos: (N, T, C, H, W) tensor of generated videos, normalized to [0, 1]
        i3d_model: pretrained I3D feature extractor
        batch_size: batch size for feature extraction
        device: compute device

    Returns:
        FVD score
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    i3d_model = i3d_model.to(device)
    i3d_model.eval()

    real_features = _extract_i3d_features(real_videos, i3d_model, batch_size, device)
    gen_features = _extract_i3d_features(generated_videos, i3d_model, batch_size, device)

    return _compute_fvd_from_features(real_features, gen_features)


def _extract_i3d_features(
    videos: torch.Tensor,
    i3d_model: nn.Module,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract I3D features from videos.

    I3D expects input in range [0, 1] or [-1, 1]. We use standard preprocessing.
    """
    features = []
    N = videos.shape[0]

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = videos[i:i + batch_size].to(device)
            # I3D expects (B, C, T, H, W)
            batch = batch.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)

            # Standard I3D preprocessing: resize to 224x224 and normalize
            batch = torch.nn.functional.interpolate(
                batch.view(-1, *batch.shape[2:]),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).view(batch.shape[0], batch.shape[1], batch.shape[2], 224, 224)

            # Extract features
            feat = i3d_model.extract_features(batch)
            features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def _compute_fvd_from_features(
    real_features: np.ndarray,
    gen_features: np.ndarray,
) -> float:
    """Compute FVD from pre-extracted features."""
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)

    return _compute_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


def _compute_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute Frechet distance between two multivariate Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    fvd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fvd)


def compute_chunk_fvd(
    real_videos: torch.Tensor,
    generated_videos: torch.Tensor,
    i3d_model: nn.Module,
    chunk_size: int = 16,
    batch_size: int = 64,
    device: torch.device = None,
) -> List[float]:
    """Compute FVD for each chunk of generated videos.

    Used for evaluating temporal consistency across AR steps (Tables 3, 4 in the paper).

    Args:
        real_videos: (N, T_real, C, H, W) ground-truth videos
        generated_videos: (N, T_gen, C, H, W) generated videos
        i3d_model: I3D feature extractor
        chunk_size: frames per chunk for evaluation
        batch_size: batch size for feature extraction
        device: compute device

    Returns:
        List of FVD scores per chunk
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T_gen = generated_videos.shape[1]
    num_chunks = T_gen // chunk_size
    fvd_scores = []

    for c in range(num_chunks):
        start = c * chunk_size
        end = start + chunk_size
        gen_chunk = generated_videos[:, start:end]
        # Use ground truth as reference for each chunk
        fvd = compute_fvd(real_videos, gen_chunk, i3d_model, batch_size, device)
        fvd_scores.append(fvd)

    return fvd_scores


def compute_ar_step_fvd(
    real_videos: torch.Tensor,
    generated_videos_per_step: List[torch.Tensor],
    i3d_model: nn.Module,
    batch_size: int = 64,
    device: torch.device = None,
) -> List[float]:
    """Compute FVD for each AR step vs the first step.

    Used in Table 3 of the paper.

    Args:
        real_videos: ground-truth videos
        generated_videos_per_step: list of videos from each AR step
        i3d_model: I3D model
        batch_size: batch size
        device: compute device

    Returns:
        FVD between step i and step 1 for each i
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_videos = generated_videos_per_step[0]
    fvd_scores = []

    for gen_videos in generated_videos_per_step[1:]:
        fvd = compute_fvd(ref_videos, gen_videos, i3d_model, batch_size, device)
        fvd_scores.append(fvd)

    return fvd_scores


class I3DFeatureExtractor(nn.Module):
    """Wrapper for I3D feature extraction.

    This is a placeholder — replace with actual pretrained I3D model.
    The paper uses the I3D model from StyleGAN-V's FVD codebase.
    Reference: https://github.com/universome/stylegan-v
    """

    def __init__(self, pretrained_path: Optional[str] = None):
        super().__init__()
        # Placeholder: in practice, load a pretrained I3D model
        # e.g., from torch.hub or a local path
        self.pretrained_path = pretrained_path
        self.model = None  # Replace with actual I3D model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through I3D."""
        if self.model is None:
            raise NotImplementedError("Load a pretrained I3D model first")
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the mixed_5c layer (or equivalent)."""
        if self.model is None:
            raise NotImplementedError("Load a pretrained I3D model first")
        return self.model.extract_features(x)
