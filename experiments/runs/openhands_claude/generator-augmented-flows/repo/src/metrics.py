from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class EvaluationMetrics:
    """
    Evaluation metrics for generative models: FID, KID, IS.

    Uses TorchMetrics (Skafte Detlefsen et al., 2022) for all metrics.
    Standard practice: compare 50,000 generated vs training images.
    """

    def __init__(
        self,
        device: torch.device,
        num_samples: int = 50000,
        batch_size: int = 256,
        feature_dim: int = 2048,
    ):
        self.device = device
        self.num_samples = num_samples
        self.batch_size = batch_size
        self._setup_metrics(feature_dim)

    def _setup_metrics(self, feature_dim: int):
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance
            from torchmetrics.image.inception import InceptionScore
        except ImportError:
            raise ImportError(
                "torchmetrics[image] is required for evaluation. "
                "Install with: pip install torchmetrics[image]"
            )

        self.fid = FrechetInceptionDistance(feature=feature_dim, normalize=True).to(self.device)
        self.kid = KernelInceptionDistance(feature=feature_dim, normalize=True, subset_size=1000).to(self.device)
        self.inception_score = InceptionScore(normalize=True).to(self.device)

    def _to_uint8(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images from [-1, 1] to uint8 [0, 255]."""
        images = (images.clamp(-1, 1) + 1) / 2  # [0, 1]
        images = (images * 255).byte()
        return images

    def update_real(self, images: torch.Tensor):
        """Update metrics with real images."""
        images_uint8 = self._to_uint8(images).to(self.device)
        self.fid.update(images_uint8, real=True)
        self.kid.update(images_uint8, real=True)

    def update_fake(self, images: torch.Tensor):
        """Update metrics with generated images."""
        images_uint8 = self._to_uint8(images).to(self.device)
        self.fid.update(images_uint8, real=False)
        self.kid.update(images_uint8, real=False)
        self.inception_score.update(images_uint8)

    def compute(self) -> dict:
        """Compute all metrics and return as a dictionary."""
        fid_score = self.fid.compute().item()
        kid_mean, kid_std = self.kid.compute()
        is_mean, is_std = self.inception_score.compute()

        return {
            "fid": fid_score,
            "kid_mean": kid_mean.item(),
            "kid_std": kid_std.item(),
            "is_mean": is_mean.item(),
            "is_std": is_std.item(),
        }

    def reset(self):
        """Reset all metric states."""
        self.fid.reset()
        self.kid.reset()
        self.inception_score.reset()

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        real_dataloader: DataLoader,
        sigma_max: float = 80.0,
        num_steps: int = 1,
    ) -> dict:
        """
        Full evaluation pipeline.

        Args:
            model: consistency model
            real_dataloader: dataloader for real images
            sigma_max: maximum noise level for sampling
            num_steps: number of sampling steps

        Returns:
            Dictionary with FID, KID, IS metrics
        """
        self.reset()
        model.eval()

        # Collect real images
        real_count = 0
        for batch in real_dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)
            self.update_real(batch)
            real_count += batch.shape[0]
            if real_count >= self.num_samples:
                break

        # Generate fake images
        fake_count = 0
        while fake_count < self.num_samples:
            current_batch = min(self.batch_size, self.num_samples - fake_count)
            noise = torch.randn(
                current_batch,
                model.network.in_channels,
                model.network.img_resolution,
                model.network.img_resolution,
                device=self.device,
            )
            samples = model.sample(noise, sigma_max=sigma_max, num_steps=num_steps)
            self.update_fake(samples)
            fake_count += current_batch

        return self.compute()
