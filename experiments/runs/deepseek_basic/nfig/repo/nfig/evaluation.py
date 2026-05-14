"""
Evaluation metrics for NFIG: FID, IS, Precision, Recall.

Follows standard evaluation protocols on ImageNet 256x256.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg


def compute_inception_stats(images: torch.Tensor, model, batch_size: int = 50,
                            device: torch.device = torch.device('cpu')):
    """
    Compute Inception features for FID/IS computation.
    
    Args:
        images: (N, 3, H, W) in range [-1, 1] or [0, 1]
        model: InceptionV3 model
        batch_size: batch size for processing
        device: device
    
    Returns:
        features: (N, 2048) inception features
    """
    model = model.to(device)
    model.eval()
    
    # Normalize to [-1, 1] if needed
    if images.min() >= 0:
        images = 2.0 * images - 1.0
    
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        with torch.no_grad():
            feat = model(batch)[0]  # pool3 features
        
        # If output is (N, 2048, 1, 1), squeeze
        if feat.dim() == 4:
            feat = feat.squeeze(-1).squeeze(-1)
        
        features.append(feat.cpu())
    
    return torch.cat(features, dim=0)


def compute_fid(features_real: torch.Tensor, features_fake: torch.Tensor) -> float:
    """
    Compute Fréchet Inception Distance.
    
    FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*(Sigma_r*Sigma_f)^(1/2))
    
    Args:
        features_real: (N_real, D) inception features of real images
        features_fake: (N_fake, D) inception features of generated images
    
    Returns:
        fid: scalar FID value
    """
    mu_real = features_real.mean(dim=0).numpy()
    mu_fake = features_fake.mean(dim=0).numpy()
    
    sigma_real = np.cov(features_real.numpy(), rowvar=False)
    sigma_fake = np.cov(features_fake.numpy(), rowvar=False)
    
    # Compute squared difference of means
    diff = mu_real - mu_fake
    mean_diff = diff @ diff
    
    # Compute sqrt of product of covariance matrices
    cov_sqrt = linalg.sqrtm(sigma_real @ sigma_fake)
    
    # Handle numerical issues
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    
    fid = mean_diff + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
    
    return float(fid)


def compute_inception_score(features: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
    """
    Compute Inception Score.
    
    IS = exp(E_x[KL(p(y|x) || p(y))])
    
    Args:
        features: (N, 1000) softmax probabilities from Inception model
        splits: number of splits for computing std
    
    Returns:
        mean_is: mean inception score
        std_is: standard deviation
    """
    N = features.shape[0]
    split_scores = []
    
    for k in range(splits):
        part = features[k * (N // splits): (k + 1) * (N // splits)]
        py = part.mean(dim=0)
        
        scores = []
        for i in range(part.shape[0]):
            kl = part[i] * (torch.log(part[i]) - torch.log(py))
            scores.append(kl.sum())
        
        split_scores.append(torch.exp(torch.stack(scores).mean()))
    
    mean_is = torch.stack(split_scores).mean().item()
    std_is = torch.stack(split_scores).std().item()
    
    return mean_is, std_is


def compute_precision_recall(features_real: torch.Tensor, features_fake: torch.Tensor,
                              k: int = 3) -> Tuple[float, float]:
    """
    Compute Precision and Recall using k-nearest neighbors.
    
    Precision: fraction of fake images that are within the manifold of real images
    Recall: fraction of real images that are within the manifold of fake images
    
    Args:
        features_real: (N_real, D) real features
        features_fake: (N_fake, D) fake features
        k: number of neighbors
    
    Returns:
        precision, recall
    """
    # Normalize features
    features_real = F.normalize(features_real, p=2, dim=1)
    features_fake = F.normalize(features_fake, p=2, dim=1)
    
    # Compute pairwise distances
    # For precision: is each fake within the real manifold?
    # Compute k-th nearest real neighbor distance for each real sample
    dist_real_real = torch.cdist(features_real, features_real)
    real_knn_dist = dist_real_real.topk(k + 1, dim=1, largest=False).values[:, -1]  # exclude self
    
    # For each fake, check if it's within the real manifold
    dist_fake_real = torch.cdist(features_fake, features_real)
    fake_min_dist = dist_fake_real.min(dim=1).values
    
    # A fake sample is in the real manifold if its distance to nearest real
    # is less than that real's k-th nearest neighbor distance
    precision = 0.0
    for i in range(features_fake.shape[0]):
        nearest_real = dist_fake_real[i].argmin()
        if fake_min_dist[i] <= real_knn_dist[nearest_real]:
            precision += 1.0
    precision /= features_fake.shape[0]
    
    # For recall: is each real within the fake manifold?
    dist_fake_fake = torch.cdist(features_fake, features_fake)
    fake_knn_dist = dist_fake_fake.topk(k + 1, dim=1, largest=False).values[:, -1]
    
    dist_real_fake = torch.cdist(features_real, features_fake)
    real_min_dist = dist_real_fake.min(dim=1).values
    
    recall = 0.0
    for i in range(features_real.shape[0]):
        nearest_fake = dist_real_fake[i].argmin()
        if real_min_dist[i] <= fake_knn_dist[nearest_fake]:
            recall += 1.0
    recall /= features_real.shape[0]
    
    return precision, recall


class Evaluator:
    """
    Evaluator for image generation quality.
    
    Handles FID, IS, Precision, Recall computation.
    """
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        
        # Lazy load Inception model
        self._inception = None
    
    def _get_inception(self):
        if self._inception is None:
            try:
                from torchvision.models import inception_v3
                self._inception = inception_v3(pretrained=True, transform_input=False)
                self._inception.eval()
            except Exception:
                print("Warning: Could not load InceptionV3. Metrics will not be available.")
                self._inception = None
        return self._inception
    
    def compute_all_metrics(self, real_images: torch.Tensor, fake_images: torch.Tensor,
                            batch_size: int = 50) -> dict:
        """
        Compute all metrics.
        
        Args:
            real_images: (N, 3, H, W)
            fake_images: (M, 3, H, W)
        
        Returns:
            dict with 'fid', 'is_mean', 'is_std', 'precision', 'recall'
        """
        model = self._get_inception()
        if model is None:
            return {}
        
        # Compute features
        real_features = compute_inception_stats(real_images, model, batch_size, self.device)
        fake_features = compute_inception_stats(fake_images, model, batch_size, self.device)
        
        # FID
        fid = compute_fid(real_features, fake_features)
        
        # IS (using the Inception model's softmax output)
        # We need probabilities, not features
        fake_probs = []
        for i in range(0, len(fake_images), batch_size):
            batch = fake_images[i:i+batch_size].to(self.device)
            if batch.min() >= 0:
                batch = 2.0 * batch - 1.0
            with torch.no_grad():
                logits = model(batch)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = F.softmax(logits, dim=1)
            fake_probs.append(probs.cpu())
        fake_probs = torch.cat(fake_probs, dim=0)
        
        is_mean, is_std = compute_inception_score(fake_probs)
        
        # Precision & Recall
        precision, recall = compute_precision_recall(real_features, fake_features)
        
        return {
            'fid': fid,
            'is_mean': is_mean,
            'is_std': is_std,
            'precision': precision,
            'recall': recall,
        }
