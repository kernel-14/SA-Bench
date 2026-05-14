"""
Frequency analysis utilities for NFIG.

Implements:
- Power Spectral Density (PSD) computation
- Frequency Keep Score (FKS) computation
- Frequency band visualization
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


def compute_psd(image: torch.Tensor) -> torch.Tensor:
    """
    Compute the Power Spectral Density of an image.
    
    Args:
        image: (B, C, H, W) or (C, H, W) tensor in [-1, 1]
    Returns:
        psd: (H, W) averaged PSD
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    B, C, H, W = image.shape
    # Average over batch and channels
    psd_sum = torch.zeros(H, W, device=image.device)
    
    for b in range(B):
        for c in range(C):
            f = torch.fft.fft2(image[b, c])
            f_shifted = torch.fft.fftshift(f)
            psd = f_shifted.abs() ** 2
            psd_sum += psd
    
    return psd_sum / (B * C)


def compute_radial_psd(image: torch.Tensor, n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radially-averaged PSD (1D power spectrum).
    
    Args:
        image: (B, C, H, W) tensor
        n_bins: number of radial bins
    Returns:
        freqs: radial frequency values
        power: power at each frequency
    """
    psd = compute_psd(image)
    H, W = psd.shape
    cy, cx = H // 2, W // 2
    
    ys = torch.arange(H, device=psd.device).float() - cy
    xs = torch.arange(W, device=psd.device).float() - cx
    r = torch.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
    
    max_r = math.sqrt(cy ** 2 + cx ** 2)
    r_norm = r / max_r
    
    bins = torch.linspace(0, 1, n_bins + 1)
    power = []
    freqs = []
    
    for i in range(n_bins):
        mask = (r_norm >= bins[i]) & (r_norm < bins[i + 1])
        if mask.sum() > 0:
            power.append(psd[mask].mean().item())
            freqs.append(((bins[i] + bins[i + 1]) / 2).item())
    
    return np.array(freqs), np.array(power)


def compute_frequency_keep_score(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    weights: Tuple[float, float, float] = (0.57, 0.28, 0.15),
) -> dict:
    """
    Compute Frequency Keep Score (FKS) between real and generated images.
    
    FKS measures weighted similarity across Low/Mid/High frequency bands.
    Weights from paper: Low=0.57, Mid=0.28, High=0.15 (emphasizing structural info).
    
    Args:
        real_images: (B, C, H, W) real images
        generated_images: (B, C, H, W) generated images
        weights: (low_weight, mid_weight, high_weight)
    Returns:
        dict with FKS, Low, Mid, High scores
    """
    def get_band_psd(images, low_r, high_r):
        B, C, H, W = images.shape
        cy, cx = H // 2, W // 2
        ys = torch.arange(H, device=images.device).float() - cy
        xs = torch.arange(W, device=images.device).float() - cx
        r = torch.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
        max_r = math.sqrt(cy ** 2 + cx ** 2) + 1e-6
        r_norm = r / max_r
        mask = (r_norm >= low_r) & (r_norm < high_r)
        
        total_power = 0.0
        for b in range(B):
            for c in range(C):
                f = torch.fft.fft2(images[b, c])
                f_shifted = torch.fft.fftshift(f)
                psd = f_shifted.abs() ** 2
                total_power += psd[mask].mean().item()
        return total_power / (B * C)
    
    # Define band boundaries
    bands = [(0.0, 0.33), (0.33, 0.67), (0.67, 1.0)]
    band_names = ["Low", "Mid", "High"]
    
    scores = {}
    fks = 0.0
    
    for (low, high), name, w in zip(bands, band_names, weights):
        real_power = get_band_psd(real_images, low, high)
        gen_power = get_band_psd(generated_images, low, high)
        
        # Similarity: 1 - |log(gen/real)| / log(max_ratio)
        if real_power > 0 and gen_power > 0:
            ratio = gen_power / real_power
            similarity = max(0.0, 1.0 - abs(math.log(ratio)) / math.log(10))
        else:
            similarity = 0.0
        
        scores[name] = similarity * 100  # as percentage
        fks += w * similarity
    
    scores["FKS"] = fks * 100
    return scores


def visualize_frequency_spectrum(image: torch.Tensor) -> torch.Tensor:
    """
    Create a log-scale frequency spectrum visualization.
    
    Args:
        image: (C, H, W) or (B, C, H, W) tensor
    Returns:
        spectrum: (H, W) log-magnitude spectrum for visualization
    """
    if image.dim() == 4:
        image = image[0]  # Take first image
    
    # Average over channels
    img_gray = image.mean(0)  # (H, W)
    
    f = torch.fft.fft2(img_gray)
    f_shifted = torch.fft.fftshift(f)
    magnitude = f_shifted.abs()
    
    # Log scale for visualization
    log_magnitude = torch.log1p(magnitude)
    
    # Normalize to [0, 1]
    log_magnitude = (log_magnitude - log_magnitude.min()) / (
        log_magnitude.max() - log_magnitude.min() + 1e-8
    )
    
    return log_magnitude


def compute_vq_loss_per_scale(
    model_frvae,
    images: torch.Tensor,
) -> List[float]:
    """
    Compute VQ loss per frequency band for analysis (Figure 5 in paper).
    
    Args:
        model_frvae: trained FRVAE model
        images: (B, C, H, W) input images
    Returns:
        vq_losses: list of VQ losses per band
    """
    import torch.nn.functional as F
    
    model_frvae.eval()
    with torch.no_grad():
        f = model_frvae.encoder(images)
        B, C, H, W = f.shape
        components = model_frvae.decomposer(f)
        
        vq_losses = []
        R_prev = None
        
        for i, (f_band, scale) in enumerate(zip(components, model_frvae.scale_factors)):
            hi = max(1, H // scale)
            wi = max(1, W // scale)
            v_i = F.interpolate(f_band, size=(hi, wi), mode="bilinear", align_corners=False)
            
            if i == 0:
                target = v_i
            else:
                R_prev_down = F.interpolate(R_prev, size=(hi, wi),
                                            mode="bilinear", align_corners=False)
                target = R_prev_down + v_i
            
            v_q, _, vq_loss = model_frvae.quantizer.quantizer(target)
            vq_losses.append(vq_loss.item())
            
            v_q_up = F.interpolate(v_q, size=(H, W), mode="bilinear", align_corners=False)
            if i == 0:
                R_prev = f_band - v_q_up
            else:
                R_prev = R_prev + (f_band - v_q_up)
    
    return vq_losses
