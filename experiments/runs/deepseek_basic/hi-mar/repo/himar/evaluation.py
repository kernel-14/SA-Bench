"""
Evaluation utilities for Hi-MAR.

Computes standard metrics:
- FID (Fréchet Inception Distance)
- IS (Inception Score)
- Precision and Recall

Also supports T2I-CompBench evaluation for text-to-image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from typing import List, Tuple, Optional
import os


def compute_inception_stats(images, model, batch_size=50, device='cuda'):
    """
    Compute Inception features for a set of images.
    
    Args:
        images: torch.Tensor (N, 3, H, W) in range [-1, 1] or [0, 1]
        model: InceptionV3 model
        batch_size: batch size for processing
        device: device to use
    
    Returns:
        mu: mean of features (d,)
        sigma: covariance of features (d, d)
    """
    model.eval()
    model.to(device)
    
    features = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            # Inception expects [0, 1] range
            if batch.min() < 0:
                batch = (batch + 1) / 2.0
            
            # Resize to 299x299
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            feat = model(batch)[0]  # (B, 2048)
            features.append(feat.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma


def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute Fréchet Inception Distance.
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_inception_score(images, model, batch_size=50, splits=10, device='cuda'):
    """
    Compute Inception Score.
    
    IS = exp(E[KL(p(y|x) || p(y))])
    """
    model.eval()
    model.to(device)
    
    preds = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            if batch.min() < 0:
                batch = (batch + 1) / 2.0
            
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            logits = model(batch)
            preds.append(F.softmax(logits, dim=1).cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # Compute IS with splits
    scores = []
    N = len(preds)
    split_size = N // splits
    
    for k in range(splits):
        part = preds[k*split_size:(k+1)*split_size]
        py = np.mean(part, axis=0)
        kl = part * (np.log(part) - np.log(py))
        kl_mean = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl_mean))
    
    return np.mean(scores), np.std(scores)


def compute_precision_recall(real_features, fake_features, k=3):
    """
    Compute Precision and Recall using k-NN.
    
    Based on "Improved Precision and Recall Metric for Assessing Generative Models"
    (Kynkäänniemi et al., 2019)
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Fit NN on real features
    nn_real = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(real_features)
    dist_real, _ = nn_real.kneighbors(real_features)
    threshold_real = np.max(dist_real)
    
    # Fit NN on fake features
    nn_fake = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(fake_features)
    dist_fake, _ = nn_fake.kneighbors(fake_features)
    threshold_fake = np.max(dist_fake)
    
    # Precision: fraction of fake samples within real manifold
    dist_fake_to_real, _ = nn_real.kneighbors(fake_features)
    precision = np.mean(dist_fake_to_real.max(axis=1) <= threshold_real)
    
    # Recall: fraction of real samples within fake manifold
    dist_real_to_fake, _ = nn_fake.kneighbors(real_features)
    recall = np.mean(dist_real_to_fake.max(axis=1) <= threshold_fake)
    
    return float(precision), float(recall)


class HiMAREvaluator:
    """
    Evaluator for Hi-MAR models.
    
    Computes FID, IS, Precision, Recall, and T2I-CompBench metrics.
    """
    def __init__(
        self,
        model,
        vae,
        inception_model=None,
        device='cuda',
    ):
        self.model = model
        self.vae = vae
        self.device = device
        
        if inception_model is None:
            try:
                from torchvision.models import inception_v3
                self.inception = inception_v3(pretrained=True, transform_input=False)
                self.inception.fc = nn.Identity()  # Remove classification head for features
            except Exception:
                print("Warning: Could not load Inception model.")
                self.inception = None
        else:
            self.inception = inception_model
    
    @torch.no_grad()
    def generate_samples(
        self,
        num_samples,
        batch_size=16,
        class_idx=None,
        context_embeds=None,
        phase1_steps=32,
        phase2_steps=4,
        cfg_scale=1.0,
    ):
        """
        Generate samples using the Hi-MAR model.
        Returns decoded images.
        """
        all_images = []
        
        for i in range(0, num_samples, batch_size):
            bs = min(batch_size, num_samples - i)
            
            if class_idx is not None:
                batch_class = class_idx[i:i+bs].to(self.device)
            else:
                batch_class = None
            
            if context_embeds is not None:
                batch_context = context_embeds[i:i+bs].to(self.device)
            else:
                batch_context = None
            
            # Generate latents
            x_low, x_high = self.model.generate(
                batch_size=bs,
                class_idx=batch_class,
                context_embeds=batch_context,
                phase1_steps=phase1_steps,
                phase2_steps=phase2_steps,
                cfg_scale=cfg_scale,
                device=self.device,
            )
            
            # Decode high-res latent to image
            # Reshape from (B, N, C) to (B, C, H, W)
            H = W = int(x_high.shape[1] ** 0.5)
            latent = x_high.permute(0, 2, 1).reshape(x_high.shape[0], -1, H, W)
            
            images = self.vae.decode(latent).sample
            
            # Clamp to valid range
            images = torch.clamp(images, -1, 1)
            
            all_images.append(images.cpu())
        
        return torch.cat(all_images, dim=0)
    
    def evaluate_fid_is(
        self,
        num_samples=50000,
        batch_size=16,
        real_stats=None,
        class_idx=None,
        phase1_steps=32,
        phase2_steps=4,
        cfg_scale=1.0,
    ):
        """
        Evaluate FID and IS.
        
        Args:
            num_samples: number of samples to generate
            batch_size: batch size for generation
            real_stats: tuple of (mu, sigma) for real data, or path to .npz file
            class_idx: optional class conditioning
            phase1_steps: steps for phase 1
            phase2_steps: steps for phase 2
            cfg_scale: classifier-free guidance scale
        
        Returns:
            dict with FID, IS_mean, IS_std, precision, recall
        """
        if self.inception is None:
            raise RuntimeError("Inception model not available for evaluation.")
        
        # Generate samples
        print(f"Generating {num_samples} samples...")
        fake_images = self.generate_samples(
            num_samples=num_samples,
            batch_size=batch_size,
            class_idx=class_idx,
            phase1_steps=phase1_steps,
            phase2_steps=phase2_steps,
            cfg_scale=cfg_scale,
        )
        
        # Compute fake stats
        print("Computing fake Inception features...")
        fake_mu, fake_sigma = compute_inception_stats(fake_images, self.inception, device=self.device)
        
        # Load or compute real stats
        if real_stats is None:
            print("Warning: No real stats provided. Using placeholder.")
            # Create dummy stats
            real_mu = np.zeros_like(fake_mu)
            real_sigma = np.eye(len(fake_mu))
        elif isinstance(real_stats, str) and real_stats.endswith('.npz'):
            data = np.load(real_stats)
            real_mu = data['mu']
            real_sigma = data['sigma']
        elif isinstance(real_stats, tuple):
            real_mu, real_sigma = real_stats
        else:
            # Compute from provided tensor
            real_mu, real_sigma = compute_inception_stats(real_stats, self.inception, device=self.device)
        
        # Compute FID
        fid = compute_fid(real_mu, real_sigma, fake_mu, fake_sigma)
        
        # Compute IS
        is_mean, is_std = compute_inception_score(fake_images, self.inception, device=self.device)
        
        # Compute Precision and Recall
        # Need raw features
        fake_features = compute_inception_stats(fake_images, self.inception, device=self.device)[0]
        fake_features = fake_features.reshape(1, -1)  # We need per-sample features
        
        # Better: get per-sample features
        from torch.utils.data import DataLoader, TensorDataset
        self.inception.to(self.device)
        all_feats = []
        loader = DataLoader(TensorDataset(fake_images), batch_size=50)
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                if batch.min() < 0:
                    batch = (batch + 1) / 2.0
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                feats = self.inception(batch)
                if isinstance(feats, tuple):
                    feats = feats[0]
                all_feats.append(feats.cpu().numpy())
        fake_features = np.concatenate(all_feats, axis=0)
        
        # Get real features similarly
        if isinstance(real_stats, str):
            data = np.load(real_stats)
            real_features = data['features'] if 'features' in data else None
        else:
            real_features = None
        
        if real_features is not None:
            precision, recall = compute_precision_recall(real_features, fake_features)
        else:
            precision, recall = None, None
        
        results = {
            'FID': fid,
            'IS_mean': is_mean,
            'IS_std': is_std,
            'Precision': precision,
            'Recall': recall,
        }
        
        return results
    
    def evaluate_t2i_compbench(
        self,
        benchmark_data,
        batch_size=16,
        phase1_steps=32,
        phase2_steps=4,
    ):
        """
        Evaluate on T2I-CompBench.
        
        Args:
            benchmark_data: dict with prompts and evaluation config
            batch_size: batch size for generation
        
        Returns:
            dict with scores for attribute binding, object relationships, complex
        """
        # This requires the T2I-CompBench evaluation setup
        # which uses BLIP-VQA and other models for scoring
        # Placeholder structure
        results = {
            'Color': 0.0,
            'Shape': 0.0,
            'Texture': 0.0,
            'Spatial': 0.0,
            'Non-Spatial': 0.0,
            'Complex': 0.0,
        }
        
        print("T2I-CompBench evaluation requires external evaluation models.")
        print("This is a placeholder. In practice, use the official T2I-CompBench code.")
        
        return results


def compute_image_net_stats(dataloader, inception_model, device='cuda', cache_path=None):
    """
    Pre-compute ImageNet statistics for FID computation.
    
    Args:
        dataloader: DataLoader for ImageNet validation set
        inception_model: InceptionV3 model
        device: device to use
        cache_path: path to save computed stats
    
    Returns:
        mu, sigma
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['mu'], data['sigma']
    
    inception_model.eval()
    inception_model.to(device)
    
    features = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            
            imgs = imgs.to(device)
            if imgs.min() < 0:
                imgs = (imgs + 1) / 2.0
            
            imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
            feat = inception_model(imgs)
            if isinstance(feat, tuple):
                feat = feat[0]
            features.append(feat.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    if cache_path:
        np.savez(cache_path, mu=mu, sigma=sigma, features=features)
    
    return mu, sigma
