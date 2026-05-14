"""
Evaluation metrics for image generation:
- FID (Fréchet Inception Distance)
- IS (Inception Score)
- Precision/Recall

Uses pre-computed InceptionV3 features following standard practice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg


class InceptionFeatureExtractor(nn.Module):
    """InceptionV3 feature extractor for FID and IS computation."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.fc = nn.Identity()
        inception.eval()
        self.inception = inception.to(device)
        for p in self.inception.parameters():
            p.requires_grad_(False)

        self.resize = transforms.Resize((299, 299))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.resize(images)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = (images - 0.5) * 2  # back to [-1, 1] range for inception
        return self.inception(images)


def compute_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
) -> float:
    """Compute Fréchet Inception Distance."""
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    mu2 = np.mean(fake_features, axis=0)
    sigma2 = np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid)


def compute_inception_score(
    features: np.ndarray,
    splits: int = 10,
) -> tuple[float, float]:
    """Compute Inception Score."""
    scores = []
    for i in range(splits):
        part = features[i * (len(features) // splits): (i + 1) * (len(features) // splits)]
        probs = F.softmax(torch.from_numpy(part), dim=1).numpy()
        kl = probs * (np.log(probs) - np.log(np.expand_dims(np.mean(probs, axis=0), 0)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


def compute_precision_recall(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    k: int = 3,
    num_samples: int = 10000,
) -> tuple[float, float]:
    """Compute Precision and Recall metrics (Kynkäänniemi et al. 2019)."""
    real_features = real_features[:num_samples]
    fake_features = fake_features[:num_samples]

    # Normalize features
    real_features = real_features / np.linalg.norm(real_features, axis=1, keepdims=True)
    fake_features = fake_features / np.linalg.norm(fake_features, axis=1, keepdims=True)

    # Compute pairwise distances
    real_distances = np.sort(np.sum(real_features ** 2, axis=1))
    fake_distances = np.sort(np.sum(fake_features ** 2, axis=1))

    # K-nearest neighbor distances
    real_knn = np.sort(np.linalg.norm(
        real_features[:, None] - real_features[None], axis=-1
    ), axis=1)[:, k]
    fake_knn = np.sort(np.linalg.norm(
        fake_features[:, None] - fake_features[None], axis=-1
    ), axis=1)[:, k]

    # Precision: fraction of fake samples within real manifold
    dist_real_to_fake = np.linalg.norm(
        real_features[:, None] - fake_features[None], axis=-1
    )
    precision = np.mean(np.min(dist_real_to_fake, axis=0) <= real_knn)

    # Recall: fraction of real samples within fake manifold
    dist_fake_to_real = dist_real_to_fake.T
    recall = np.mean(np.min(dist_fake_to_real, axis=1) <= fake_knn)

    return float(precision), float(recall)


class Evaluator:
    """Combined evaluator for all metrics."""

    def __init__(self, device: torch.device, num_generated: int = 50000):
        self.device = device
        self.num_generated = num_generated
        self.extractor = InceptionFeatureExtractor(device)
        self.batch_size = 64

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract Inception features from images."""
        features = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size].to(self.device)
            feat = self.extractor(batch)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)

    def evaluate_all(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> dict:
        """Compute all metrics: FID, IS, Precision, Recall."""
        real_features = self.extract_features(real_images[:self.num_generated])
        fake_features = self.extract_features(fake_images[:self.num_generated])

        fid = compute_fid(real_features, fake_features)
        inception_model = inception_v3(pretrained=True, transform_input=True).to(self.device)
        inception_model.eval()

        # IS needs softmax probs, use different feature extraction
        is_mean, is_std = self._compute_is(fake_images, inception_model)
        precision, recall = compute_precision_recall(real_features, fake_features)

        return {
            'FID': fid,
            'IS_mean': is_mean,
            'IS_std': is_std,
            'Precision': precision,
            'Recall': recall,
        }

    @torch.no_grad()
    def _compute_is(self, images: torch.Tensor, inception_model: nn.Module) -> tuple[float, float]:
        """Compute Inception Score using classifier logits."""
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
        ])
        probs = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size].to(self.device)
            batch = transform(batch)
            batch = (batch - 0.5) * 2
            logits = inception_model(batch)
            prob = F.softmax(logits, dim=1)
            probs.append(prob.cpu().numpy())
        probs = np.concatenate(probs, axis=0)
        return compute_inception_score(probs)
