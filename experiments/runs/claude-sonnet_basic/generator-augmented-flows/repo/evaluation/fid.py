"""
FID, KID, and IS evaluation for consistency models.
Uses TorchMetrics for metric computation.
"""

import os
import math
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms


def generate_samples(model, num_samples, batch_size, device, class_labels=None):
    """
    Generate samples from a consistency model.
    
    Args:
        model: ConsistencyModel instance
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: Device for generation
        class_labels: Optional class labels for conditional generation
        
    Returns:
        Tensor of generated samples of shape (num_samples, C, H, W)
    """
    model.eval()
    samples = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Sample noise
            z = torch.randn(current_batch_size, *model.network.img_resolution_shape, device=device)
            
            # Generate samples
            if class_labels is not None:
                labels = class_labels[i:i + current_batch_size]
            else:
                labels = None
            
            x = model.sample(z, class_labels=labels)
            samples.append(x.cpu())
    
    return torch.cat(samples, dim=0)


def compute_metrics(generated_samples, real_samples, device='cpu', num_features=2048):
    """
    Compute FID, KID, and IS metrics.
    
    Args:
        generated_samples: Tensor of generated samples in [-1, 1]
        real_samples: Tensor of real samples in [-1, 1]
        device: Device for computation
        num_features: Number of Inception features
        
    Returns:
        Dictionary with FID, KID, and IS values
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        from torchmetrics.image.inception import InceptionScore
    except ImportError:
        raise ImportError("Please install torchmetrics: pip install torchmetrics[image]")
    
    # Convert from [-1, 1] to [0, 255] uint8
    def to_uint8(x):
        x = ((x + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        return x
    
    gen_uint8 = to_uint8(generated_samples)
    real_uint8 = to_uint8(real_samples)
    
    # FID
    fid_metric = FrechetInceptionDistance(feature=num_features).to(device)
    
    batch_size = 256
    for i in range(0, len(real_uint8), batch_size):
        batch = real_uint8[i:i + batch_size].to(device)
        fid_metric.update(batch, real=True)
    
    for i in range(0, len(gen_uint8), batch_size):
        batch = gen_uint8[i:i + batch_size].to(device)
        fid_metric.update(batch, real=False)
    
    fid = fid_metric.compute().item()
    
    # KID
    kid_metric = KernelInceptionDistance(feature=num_features, subset_size=1000).to(device)
    
    for i in range(0, len(real_uint8), batch_size):
        batch = real_uint8[i:i + batch_size].to(device)
        kid_metric.update(batch, real=True)
    
    for i in range(0, len(gen_uint8), batch_size):
        batch = gen_uint8[i:i + batch_size].to(device)
        kid_metric.update(batch, real=False)
    
    kid_mean, kid_std = kid_metric.compute()
    kid = kid_mean.item()
    
    # IS
    is_metric = InceptionScore(feature=num_features).to(device)
    
    for i in range(0, len(gen_uint8), batch_size):
        batch = gen_uint8[i:i + batch_size].to(device)
        is_metric.update(batch)
    
    is_mean, is_std = is_metric.compute()
    is_score = is_mean.item()
    
    return {
        'FID': fid,
        'KID': kid * 100,  # Report as x10^2 as in the paper
        'IS': is_score,
    }


def evaluate_model(model, dataset, num_samples=50000, batch_size=256, device='cpu'):
    """
    Evaluate a consistency model on a dataset.
    
    Args:
        model: ConsistencyModel instance
        dataset: Dataset to evaluate on
        num_samples: Number of samples to generate and compare
        batch_size: Batch size for generation and evaluation
        device: Device for computation
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Get real samples
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    real_samples = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            real_samples.append(batch[0])
        else:
            real_samples.append(batch)
        if len(real_samples) * batch_size >= num_samples:
            break
    real_samples = torch.cat(real_samples, dim=0)[:num_samples]
    
    # Get model's image shape
    sample_batch = real_samples[:1]
    C, H, W = sample_batch.shape[1:]
    
    # Generate samples
    model.eval()
    generated = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            z = torch.randn(current_batch_size, C, H, W, device=device)
            sigma_max = torch.tensor(model.sigma_max, dtype=z.dtype, device=device)
            x = model(z, sigma_max.expand(current_batch_size))
            generated.append(x.cpu())
    
    generated = torch.cat(generated, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(generated, real_samples, device=device)
    return metrics
